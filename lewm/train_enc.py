import os
from functools import partial
from pathlib import Path

import wandb
import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from torch import nn

from module import MLP, SIGReg
from train_probe import build_probe
from utils import get_img_preprocessor, ModelObjectCallBack, split_dataset
from einops import rearrange, reduce


class EncModel(nn.Module):
    def __init__(self, encoder, projector, probe):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.probe = probe

    def encode_pixels(self, pixels):
        pixels = pixels.float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        cls_token = output.last_hidden_state[:, 0]
        emb = self.projector(cls_token)
        return rearrange(emb, "(b t) d -> b t d", b=b)


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


class ClampSchedulerCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        for config in trainer.lr_scheduler_configs:
            sched = config.scheduler
            if hasattr(sched, "max_steps") and sched.last_epoch >= sched.max_steps:
                eta_min = getattr(sched, "eta_min", 0.0)
                for pg in sched.optimizer.param_groups:
                    pg["lr"] = eta_min


def enc_forward(self, batch, stage, cfg):
    logs = {}

    b, *_ = batch["pixels"].shape
    samples_key = f"_{stage}_samples_num"
    setattr(self, samples_key, getattr(self, samples_key, 0) + b)
    logs[f"{stage}/samples"] = getattr(self, samples_key)
    self.samples_num = getattr(self, "samples_num", 0) + b
    logs["samples"] = self.samples_num

    lambd = cfg.loss.sigreg.weight

    emb = self.model.encode_pixels(batch["pixels"])  # (B, T, D)

    flat_emb = rearrange(emb, "b t d -> (b t) d").float()
    grid_logits = self.model.probe(flat_emb)  # (B*T, grid_cells)
    grid_targets = rearrange(batch["tetris_grid_mask"].float(), "b t w h -> (b t) (w h)")
    grid_loss = F.binary_cross_entropy_with_logits(grid_logits, grid_targets)

    output = {"emb": emb, "grid_loss": grid_loss}

    if lambd > 0:
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))

    if OmegaConf.select(cfg, "track_collapse.enabled", default=False):
        emb_flat = rearrange(emb, "b t d -> (b t) d")
        emb_flat_mean = reduce(emb_flat, "bt d -> 1 d", "mean")
        logs[f"{stage}/emb_l2"] = (emb_flat ** 2).mean().detach()
        logs[f"{stage}/mean_emb_l2"] = ((emb_flat - emb_flat_mean) ** 2).mean().detach()

    output["loss"] = grid_loss
    if lambd > 0:
        output["loss"] = output["loss"] + lambd * output["sigreg_loss"]

    for k, v in output.items():
        if "loss" not in k:
            continue
        logs[f"{stage}/{k}"] = v.detach()

    self.log_dict(logs, on_step=True, sync_dist=True)
    output["loss"] = self.rescale_loss_for_grad_acc(output["loss"])
    return output


@hydra.main(version_base=None, config_path="../conf", config_name="enc_train_tetris")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = []

    if OmegaConf.select(cfg, "normalize_pixels"):
        transforms.append(get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size))

    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
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

    enc_hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.embed_dim

    projector = MLP(
        input_dim=enc_hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    probe_cfg = cfg.grid_mask_probe
    grid_cells = probe_cfg.grid_width * probe_cfg.grid_height
    probe = build_probe(embed_dim, grid_cells, probe_cfg.model)

    enc_model = EncModel(encoder=encoder, projector=projector, probe=probe)

    if OmegaConf.select(cfg, "overfit", default=False):
        for module in enc_model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                module.p = 0

    data_module = spt.data.DataModule(train=train, val=val)
    module = spt.Module(
        model=enc_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(enc_forward, cfg=cfg),
        optim=OmegaConf.to_container(cfg.optimizers, resolve=True),
    )

    with open_dict(cfg):
        cfg["num_params"] = sum(p.numel() for p in module.parameters(0))

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

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
