import os
import math
from functools import partial
from pathlib import Path
import contextlib as ctl

import wandb
import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import numpy as np
import torch
import torch.nn.functional as F
import torch.special
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from torch import nn

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg, Normalizer, NanToNum, RearrangeWrap
from train_probe import ProbeModule, build_probe
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack, torch_printoptions, split_dataset
from einops import repeat, rearrange, reduce
from einops.layers.torch import EinMix, Rearrange


def _reset_parameters(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()


class LearningRateMonitor(pl.pytorch.callbacks.LearningRateMonitor):
    def _extract_lr(self, param_group, name):
        lr = super()._extract_lr(param_group, name)
        if len(lr) != 1:
            raise RuntimeError()
        lr = {f"{key}-lr": value for key, value in lr.items()}
        return lr


class EMAMetricsSummaryCallback(pl.Callback):
    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self._ema: dict[str, float] = {}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        ema_logs = {}
        for k, v in trainer.callback_metrics.items():
            # TODO: maybe use a regex here
            if not "fit" in k:
                continue
            val = v.item() if hasattr(v, "item") else float(v)
            if k not in self._ema:
                self._ema[k] = val
            else:
                self._ema[k] = self.alpha * self._ema[k] + (1 - self.alpha) * val
            ema_logs[f"ema/{k}"] = self._ema[k]
        if ema_logs and trainer.logger and trainer.is_global_zero:
            trainer.logger.experiment.summary.update(ema_logs)


class GridMaskProbeCallback(pl.Callback):
    def __init__(self, full_cfg, run_dir):
        self.run_dir = run_dir
        probe_cfg = OmegaConf.select(full_cfg, "grid_mask_probe", default=OmegaConf.create({}))
        grid_cells = probe_cfg.grid_width * probe_cfg.grid_height
        probe_net = build_probe(full_cfg.wm.embed_dim, grid_cells, probe_cfg.model)
        self.probe_module = ProbeModule(
            probe=probe_net,
            cfg=probe_cfg,
            prefix="probe/",
            manual=True,
        )

    def on_fit_start(self, trainer, pl_module):
        self.probe_module = self.probe_module.to(pl_module.device)
        self.probe_module.setup_optimizer()

        # welcome to hacks
        self.probe_module.log_dict = lambda *args, **kwargs: pl_module.log_dict(*args, **kwargs)
        self.probe_module.trainer = trainer

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        probe_batch = (
            rearrange(outputs["emb"].detach().float(), "b t ... -> (b t) ..."),
            rearrange(batch["tetris_grid_mask"].float(), "b t ... -> (b t) ..."),
        )
        self.probe_module._optimizer.zero_grad()
        loss = self.probe_module.training_step(probe_batch, batch_idx)
        loss.backward()
        self.probe_module._optimizer.step()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        probe_batch = (
            rearrange(outputs["emb"].detach().float(), "b t ... -> (b t) ..."),
            rearrange(batch["tetris_grid_mask"].float(), "b t ... -> (b t) ..."),
        )
        self.probe_module.validation_step(probe_batch, batch_idx)

    def on_train_epoch_end(self, trainer, pl_module):
        interval = OmegaConf.select(self.probe_module.cfg, "checkpoint_interval")
        if interval is None:
            return
        epoch = trainer.current_epoch
        if (epoch + 1) % interval == 0:
            path = self.run_dir / f"grid_mask_probe_epoch_{epoch:04d}.pt"
            torch.save(self.probe_module.probe, path)

    def on_train_epoch_start(self, trainer, pl_module):
        interval = OmegaConf.select(self.probe_module.cfg, "reset_interval")
        if interval is None:
            return
        if trainer.current_epoch % interval == 0:
            self.probe_module.probe.apply(_reset_parameters)


class QueueProbeModule(pl.LightningModule):
    def __init__(self, probe, cfg, *, prefix="", manual=False):
        super().__init__()
        self.probe = probe
        self.cfg = cfg
        self.prefix = prefix
        self.manual = manual

    def _step(self, stage, batch, batch_idx):
        logs = {}
        latents, queue = batch  # (B, D), (B, queue_len)
        b = latents.shape[0]

        if not self.manual:
            samples_key = f"_{stage}_samples_num"
            setattr(self, samples_key, getattr(self, samples_key, 0) + b)
            logs[f"{stage}/samples"] = getattr(self, samples_key)
            self.samples_num = getattr(self, "samples_num", 0) + b
            logs["samples"] = self.samples_num

        logits = self.probe(latents)  # (B, queue_len * num_classes)
        num_classes = self.cfg.num_classes
        queue_len = self.cfg.queue_len
        logits = rearrange(logits, "b (q c) -> (b q) c", q=queue_len, c=num_classes)
        targets = rearrange(queue.long(), "b q -> (b q)")

        loss = F.cross_entropy(logits, targets)
        logs[self.prefix + f"{stage}/ce_loss"] = loss.detach()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == targets).float().mean()
            logs[self.prefix + f"{stage}/acc"] = acc

        self.log_dict(logs, on_step=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step("fit", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step("validate", batch, batch_idx)

    def setup_optimizer(self):
        self._optimizer = torch.optim.AdamW(
            self.probe.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.probe.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )


class QueueProbeCallback(pl.Callback):
    def __init__(self, full_cfg, run_dir):
        self.run_dir = run_dir
        probe_cfg = OmegaConf.select(full_cfg, "queue_probe", default=OmegaConf.create({}))
        out_dim = probe_cfg.queue_len * probe_cfg.num_classes
        probe_net = build_probe(full_cfg.wm.embed_dim, out_dim, probe_cfg.model)
        self.probe_module = QueueProbeModule(
            probe=probe_net,
            cfg=probe_cfg,
            prefix="queue_probe/",
            manual=True,
        )

    def on_fit_start(self, trainer, pl_module):
        self.probe_module = self.probe_module.to(pl_module.device)
        self.probe_module.setup_optimizer()
        self.probe_module.log_dict = lambda *args, **kwargs: pl_module.log_dict(*args, **kwargs)
        self.probe_module.trainer = trainer

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        probe_batch = (
            rearrange(outputs["emb"].detach().float(), "b t ... -> (b t) ..."),
            rearrange(batch["tetris_queue"], "b t ... -> (b t) ..."),
        )
        self.probe_module._optimizer.zero_grad()
        loss = self.probe_module.training_step(probe_batch, batch_idx)
        loss.backward()
        self.probe_module._optimizer.step()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        probe_batch = (
            rearrange(outputs["emb"].detach().float(), "b t ... -> (b t) ..."),
            rearrange(batch["tetris_queue"], "b t ... -> (b t) ..."),
        )
        self.probe_module.validation_step(probe_batch, batch_idx)

    def on_train_epoch_end(self, trainer, pl_module):
        interval = OmegaConf.select(self.probe_module.cfg, "checkpoint_interval")
        if interval is None:
            return
        epoch = trainer.current_epoch
        if (epoch + 1) % interval == 0:
            path = self.run_dir / f"queue_probe_epoch_{epoch:04d}.pt"
            torch.save(self.probe_module.probe, path)

    def on_train_epoch_start(self, trainer, pl_module):
        interval = OmegaConf.select(self.probe_module.cfg, "reset_interval")
        if interval is None:
            return
        if trainer.current_epoch % interval == 0:
            self.probe_module.probe.apply(_reset_parameters)


class ClampSchedulerCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        for config in trainer.lr_scheduler_configs:
            sched = config.scheduler
            if hasattr(sched, "max_steps") and sched.last_epoch >= sched.max_steps:
                eta_min = getattr(sched, "eta_min", 0.0)
                for pg in sched.optimizer.param_groups:
                    pg["lr"] = eta_min


@ctl.contextmanager
def set_eval(model: nn.Module):
    training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(training)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    logs = {}

    b, *_ = batch["pixels"].shape
    samples_key = f"_{stage}_samples_num"
    setattr(self, samples_key, getattr(self, samples_key, 0) + b)
    logs[f"{stage}/samples"] = getattr(self, samples_key)

    self.samples_num = getattr(self, "samples_num", 0) + b
    logs["samples"] = self.samples_num

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    # batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]  # (B, ctx_len, D_ctx_emb)
    ctx_act = act_emb[:, :ctx_len]  # (B, ctx_len, D_act_emb)

    tgt_emb = emb[:, n_preds:]  # label (B, n_preds, D_ctx_emb)
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred (B, n_preds, D_ctx_emb)

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    if lambd > 0:
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))

    # Loss balancing
    pred_loss_mul = 1
    if OmegaConf.select(cfg, "loss_balancing.enabled", default=False):
        alpha = cfg.loss_balancing.alpha
        mul = cfg.loss_balancing.mul

        pred_loss_val = output["pred_loss"].detach()
        if not hasattr(self, "_ema_pred_loss"):
            self._ema_pred_loss = pred_loss_val
        else:
            self._ema_pred_loss = alpha * self._ema_pred_loss + (1 - alpha) * pred_loss_val

        if lambd > 0:
            sigreg_loss_val = output["sigreg_loss"].detach()
            if not hasattr(self, "_ema_sigreg_loss"):
                self._ema_sigreg_loss = sigreg_loss_val
            else:
                self._ema_sigreg_loss = alpha * self._ema_sigreg_loss + (1 - alpha) * sigreg_loss_val

            pred_loss_mul = ((self._ema_sigreg_loss * lambd) / (self._ema_pred_loss + 1e-8)) * mul

    # Total loss
    output["loss"] = 0
    output["loss"] += pred_loss_mul * output["pred_loss"]
    if lambd > 0:
        output["loss"] += lambd * output["sigreg_loss"]

    if OmegaConf.select(cfg, "track_collapse.enabled", default=False):
        emb_flat = rearrange(output["emb"], "b t d -> (b t) d")
        emb_flat_mean = reduce(emb_flat, "bt d -> 1 d", "mean")

        logs[f"{stage}/emb_l2"] = (emb_flat ** 2).mean().detach()
        logs[f"{stage}/mean_emb_l2"] = ((emb_flat - emb_flat_mean) ** 2).mean().detach()

    # Action ranking
    with torch.no_grad(), set_eval(self.model):
        if (   (stage == "validate" and OmegaConf.select(cfg, "action_rank.enabled_val", default=False))
            or (stage == "fit"      and OmegaConf.select(cfg, "action_rank.enabled_train", default=False))
        ):
            ar_acts = torch.tensor(OmegaConf.to_container(cfg.action_rank.actions, resolve=True), dtype=torch.float32)
            ar_act_num = len(ar_acts)

            ar_ctx_act_base = batch["action"][:,:ctx_len]  # (B, T, D_act*frameskip)

            ar_pred_embs = []  # list of (B, n_preds, D_ctx_emb)
            ar_pred_embs.append(self.model.predict(ctx_emb, ctx_act))
            for ar_final_act in ar_acts:
                ar_ctx_act = ar_ctx_act_base.clone()
                ar_ctx_act[:, -1, -cfg.wm.action_dim:] = ar_final_act
                ar_ctx_act = self.model.action_trans(ar_ctx_act)
                ar_ctx_act_emb = self.model.action_encoder(ar_ctx_act)

                ar_pred_emb = self.model.predict(ctx_emb, ar_ctx_act_emb)
                ar_pred_embs.append(ar_pred_emb)
            ar_pred_embs = rearrange(ar_pred_embs, "n b n_preds d -> b n n_preds d")

            tgt_emb = rearrange(tgt_emb, "b n_preds d_ctx_emb -> b 1 n_preds d_ctx_emb")
            err_emb = reduce((ar_pred_embs - tgt_emb).pow(2), "b n n_preds d -> b n", "mean")

            # error for gt action
            gt_err = err_emb[:, (0,)]
            # error for ranking actions
            ar_err = err_emb[:, 1:]

            gt_rank_disc = reduce(gt_err >= ar_err, "b n -> b", "sum")
            gt_rank_cont = gt_rank_disc / ar_act_num

            logs[f"{stage}/gt_rank"] = gt_rank_cont.mean().detach()

    for k, v in output.items():
        if "loss" not in k:
            continue
        logs[f"{stage}/{k}"] = v.detach()

    self.log_dict(logs, on_step=True, sync_dist=True)

    # NOTE: This has to happen _after_ logging to wandb.
    output["loss"] = self.rescale_loss_for_grad_acc(output["loss"])

    return output

@hydra.main(version_base=None, config_path="../conf", config_name="train_lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            # normalizer = get_column_normalizer(dataset, col, col)
            # transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    # TODO: log which episodes/windows were sorted into train/validation
    train_set, val_set = split_dataset(cfg.train_split, dataset, rnd_gen=rnd_gen)

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=(len(train_set) > cfg.loader.batch_size),
        generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    with open_dict(cfg):
        cfg["total_steps"] = len(train) * cfg.trainer.max_epochs

    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    if cfg.img_size % cfg.patch_size != 0:
        raise RuntimeError()
    patch_num = (cfg.img_size // cfg.patch_size) ** 2

    enc_hidden_dim = encoder.config.hidden_size
    hidden_dim = cfg.wm.hidden_dim
    embed_dim = cfg.wm.embed_dim
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    projector_input_dim = enc_hidden_dim
    patch_proj = None
    if OmegaConf.select(cfg, "wm.patch_projector.enabled", default=False):
        projector_input_dim += cfg.wm.patch_projector.dim * patch_num
        match cfg.wm.patch_projector.type:
            case "linear":
                match cfg.wm.patch_projector.weights:
                    case "indep":
                        patch_proj = nn.Sequential(
                            EinMix(
                                "bt n d_in -> bt n d_out",
                                weight_shape="n d_in d_out",
                                bias_shape="n d_out",
                                n=patch_num,
                                d_in=enc_hidden_dim,
                                d_out=cfg.wm.patch_projector.dim,
                            ),
                            Rearrange("bt n d_out -> bt (n d_out)"),
                        )
                    case "shared":
                        patch_proj = RearrangeWrap(
                            nn.Linear(enc_hidden_dim, cfg.wm.patch_projector.dim),
                            pattern_in="bt n d_in -> (bt n) d_in",
                            pattern_out="(bt n) d_out -> bt (n d_out)",
                            capture="bt _ _",
                        )
                    case _:
                        raise RuntimeError(f"Unexpected {cfg.wm.patch_projector.weights=}")
            case "mlp":
                # TODO: fix this up
                raise NotImplemented()
                match cfg.wm.patch_projector.weights:
                    case "indep":
                        raise RuntimeError(f"Unsupported {cfg.wm.patch_projector.weights=}")
                    case "shared":
                        pass
                    case _:
                        raise RuntimeError(f"Unexpected {cfg.wm.patch_projector.weights=}")
                patch_proj = MLP(
                    input_dim=enc_hidden_dim,
                    output_dim=cfg.wm.patch_projector.dim,
                    hidden_dim=cfg.wm.patch_projector.hidden_dim,
                    # norm_fn=torch.nn.BatchNorm1d,
                )
            case _:
                raise RuntimeError(f"Unexpected {cfg.wm.patch_projector.type=}")

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_trans = nn.Sequential(
        RearrangeWrap(
            Normalizer.from_data(
                torch.from_numpy(np.array(dataset.get_col_data("action"))),
                dim=(0,),
            ),
            pattern_in="b t (fs d) -> (b t fs) d",
            pattern_out="(b t fs) d -> b t (fs d)",
            capture="b t _",
            fs=cfg.data.dataset.frameskip,
            d=cfg.wm.action_dim,
        ),
        # Replace NaN values with 0 (occurs at sequence boundaries)
        NanToNum(0.0),
    )
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    projector = MLP(
        input_dim=projector_input_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_trans=action_trans,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        patch_proj=patch_proj,
    )

    if OmegaConf.select(cfg, "overfit", default=False):
        # disable dropout for overfitting
        for module in world_model.modules():
            if isinstance(
                module,
                ( torch.nn.Dropout
                , torch.nn.Dropout1d
                , torch.nn.Dropout2d
                , torch.nn.Dropout3d
                )
            ):
                module.p = 0

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=OmegaConf.to_container(cfg.optimizers, resolve=True),
    )

    with open_dict(cfg):
        cfg["num_params"] = sum(p.numel() for p in world_model.parameters(0))

    ##########################
    ##       training       ##
    ##########################

    run_id = OmegaConf.select(cfg, "subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        wandb.finish()
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams({
            **OmegaConf.to_container(cfg, resolve=True),
            "raw": OmegaConf.to_container(cfg, resolve=False),
        })

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name,
        epoch_interval=cfg.checkpoint_interval,
    )

    callbacks = [
        object_dump_callback,
        EMAMetricsSummaryCallback(),
        LearningRateMonitor(logging_interval="step"),
        ClampSchedulerCallback(),
    ]
    if OmegaConf.select(cfg, "grid_mask_probe.enabled", default=False):
        callbacks.append(GridMaskProbeCallback(cfg, run_dir))
    if OmegaConf.select(cfg, "queue_probe.enabled", default=False):
        callbacks.append(QueueProbeCallback(cfg, run_dir))

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
