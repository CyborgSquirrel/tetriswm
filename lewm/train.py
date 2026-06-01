import os
from functools import partial
from pathlib import Path

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
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack, torch_printoptions
from einops import repeat, rearrange, reduce
import contextlib as ctl


def build_grid_mask_probe(latent_dim, out_dim, model_cfg):
    kind = model_cfg.type
    if kind == "linear":
        return torch.nn.Linear(latent_dim, out_dim)
    elif kind == "mlp":
        hidden_dim = model_cfg.hidden_dim
        num_layers = model_cfg.num_layers
        layers = [torch.nn.Linear(latent_dim, hidden_dim), torch.nn.GELU()]
        for _ in range(num_layers - 1):
            layers += [torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.GELU()]
        layers.append(torch.nn.Linear(hidden_dim, out_dim))
        return torch.nn.Sequential(*layers)
    else:
        raise ValueError(f"grid_mask_probe.model.type must be 'linear' or 'mlp', got '{kind}'")


def _reset_parameters(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()


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
        self.full_cfg = full_cfg
        self.cfg = OmegaConf.select(full_cfg, "grid_mask_probe", default=OmegaConf.create({}))
        self.run_dir = run_dir
        grid_cells = self.cfg.grid_width * self.cfg.grid_height
        self.probe = build_grid_mask_probe(full_cfg.wm.embed_dim, grid_cells, self.cfg.model)
        self.probe_opt = None

    def on_fit_start(self, trainer, pl_module):
        self.probe = self.probe.to(pl_module.device)
        self.probe_opt = torch.optim.AdamW(
            self.probe.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )

    def _probe_step(self, pl_module, outputs, batch, stage):
        if "tetris_grid_mask" not in batch:
            return None
        gmp_latents = rearrange(outputs["emb"].detach().float(), "b t d -> (b t) d")
        gmp_masks = rearrange(batch["tetris_grid_mask"].float().to(gmp_latents.device),
                              "b t w h -> (b t) (w h)")
        gmp_logits = self.probe(gmp_latents)
        gmp_loss = F.binary_cross_entropy_with_logits(gmp_logits, gmp_masks)

        if stage == "fit":
            self.probe_opt.zero_grad()
            gmp_loss.backward()
            self.probe_opt.step()

        with torch.no_grad():
            gmp_preds = gmp_logits.sigmoid() > 0.5
            gmp_acc = (gmp_preds == gmp_masks.bool()).float().mean()

        pl_module.log_dict({
            f"probe/{stage}/bce_loss": gmp_loss.detach(),
            f"probe/{stage}/pixel_acc": gmp_acc,
        }, on_step=True, sync_dist=True)

        return gmp_masks, gmp_logits

    def _log_mask_preview(self, trainer, gmp_masks, gmp_logits, stage):
        n = min(self.cfg.preview_images, gmp_masks.size(0))
        w, h = self.cfg.grid_width, self.cfg.grid_height
        gt   = rearrange(gmp_masks[:n], "n (w h) -> n 1 h w", w=w, h=h)
        pred = rearrange((gmp_logits[:n].sigmoid() > 0.5).float(), "n (w h) -> n 1 h w", w=w, h=h)
        gt_imgs   = F.interpolate(gt,   scale_factor=10, mode="nearest")
        pred_imgs = F.interpolate(pred, scale_factor=10, mode="nearest")
        trainer.logger.log_image(
            key=f"probe/{stage}/masks",
            images=[*gt_imgs, *pred_imgs],
            caption=[f"gt_{i}" for i in range(n)] + [f"pred_{i}" for i in range(n)],
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        result = self._probe_step(pl_module, outputs, batch, "fit")
        if batch_idx == 0 and trainer.is_global_zero and trainer.logger and result is not None:
            self._log_mask_preview(trainer, *result, "fit")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        result = self._probe_step(pl_module, outputs, batch, "validate")
        if batch_idx == 0 and trainer.is_global_zero and trainer.logger and result is not None:
            self._log_mask_preview(trainer, *result, "validate")

    def on_train_epoch_end(self, trainer, pl_module):
        interval = OmegaConf.select(self.cfg, "checkpoint_interval")
        if interval is None:
            return
        epoch = trainer.current_epoch
        if (epoch + 1) % interval == 0:
            path = self.run_dir / f"grid_mask_probe_epoch_{epoch:04d}.pt"
            torch.save(self.probe, path)

    def on_train_epoch_start(self, trainer, pl_module):
        interval = OmegaConf.select(self.cfg, "reset_interval")
        if interval is None:
            return
        if trainer.current_epoch % interval == 0:
            self.probe.apply(_reset_parameters)


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
        output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    
    output["loss"] = 0
    output["loss"] += output["pred_loss"]
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
            ar_pred_embs.append(pred_emb)
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
    match cfg.train_split.by:
        case "window":
            train_set, val_set = spt.data.random_split(
                dataset,
                lengths=[cfg.train_split.p, 1 - cfg.train_split.p],
                generator=rnd_gen,
            )
        case "episode":
            n_eps = len(dataset.lengths)
            ep_order = torch.randperm(n_eps, generator=rnd_gen).tolist()
            n_train = int(n_eps * cfg.train_split.p)
            train_eps = set(ep_order[:n_train])
            val_eps = set(ep_order[n_train:])
            if n_eps <= 1000:
                with torch_printoptions(threshold=torch.inf):
                    print(f"{train_eps=}")
                    print(f"{val_eps=}")
            train_idx = [i for i, (ep, _) in enumerate(dataset.clip_indices) if ep in train_eps]
            val_idx   = [i for i, (ep, _) in enumerate(dataset.clip_indices) if ep in val_eps]
            train_set = torch.utils.data.Subset(dataset, train_idx)
            val_set   = torch.utils.data.Subset(dataset, val_idx)
        case "episode_idx":
            train_eps = set(cfg.train_split.idx)
            train_idx = [i for i, (ep, _) in enumerate(dataset.clip_indices) if ep     in train_eps]
            val_idx   = [i for i, (ep, _) in enumerate(dataset.clip_indices) if ep not in train_eps]
            train_set = torch.utils.data.Subset(dataset, train_idx)
            val_set   = torch.utils.data.Subset(dataset, val_idx)
        case _:
            raise ValueError(f"Unexpected {cfg.train_split.by=}")

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=(len(train_set) > cfg.loader.batch_size),
        generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

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

    patch_project_dim = OmegaConf.select(cfg, "wm.patch_project_dim", default=None)
    projector_input_dim = enc_hidden_dim
    patch_proj = None
    if patch_project_dim is not None:
        projector_input_dim += patch_project_dim * patch_num
        patch_proj = nn.Linear(enc_hidden_dim, patch_project_dim)
        # patch_proj = MLP(
        #     input_dim=projector_input_dim,
        #     output_dim=embed_dim,
        #     hidden_dim=2048,
        #     # norm_fn=torch.nn.BatchNorm1d,
        # )

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

    if cfg.get("overfit", False):
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

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": dict(cfg.scheduler),
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        wandb.finish()
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams({
            "res": OmegaConf.to_container(cfg, resolve=True),
            "raw": OmegaConf.to_container(cfg, resolve=False),
        })

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name,
        epoch_interval=cfg.checkpoint_interval,
    )

    callbacks = [object_dump_callback, EMAMetricsSummaryCallback()]
    if OmegaConf.select(cfg, "grid_mask_probe.enabled", default=False):
        callbacks.append(GridMaskProbeCallback(cfg, run_dir))

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
