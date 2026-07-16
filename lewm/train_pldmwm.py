"""PLDM World Model training (IMPALA CNN encoder + GRU predictor).

Training objective faithfully ported from PLDM:
  - VICReg: sim loss + variance reg + covariance reg (batch & temporal variants)
  - IDM: inverse dynamics model — predict action from (emb_t, emb_{t+1})

Patterned on train_dinowm.py.
"""

from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from einops import rearrange
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from pldmwm_model import ImpalaEncoder, PLDMWM
from train import (
    ClampSchedulerCallback,
    EMAMetricsSummaryCallback,
    GridMaskProbeCallback,
    LearningRateMonitor,
    QueueProbeCallback,
)
from utils import ModelObjectCallBack, get_img_preprocessor, split_dataset


# ---------------------------------------------------------------------------
# VICReg loss components — ported from pldm/objectives/vicreg.py
# ---------------------------------------------------------------------------

def _std_loss(x: torch.Tensor, std_margin: float = 1.0) -> torch.Tensor:
    """Variance regularisation across dim 1.
    x: (T_or_B, B_or_T, D)  — works for both batch-wise and time-wise."""
    x = x - x.mean(dim=1, keepdim=True)
    std = torch.sqrt(x.var(dim=1) + 1e-4)          # (..., D)
    return F.relu(std_margin - std).mean()


def _cov_loss(x: torch.Tensor) -> torch.Tensor:
    """Off-diagonal covariance regularisation across dim 1.
    x: (T_or_B, B_or_T, D)"""
    N = x.shape[1]
    D = x.shape[-1]
    x = x - x.mean(dim=1, keepdim=True)
    cov = torch.einsum("tbi,tbj->tij", x, x) / (N - 1)        # (..., D, D)
    diag_sq = cov.diagonal(dim1=-2, dim2=-1).pow(2).sum(-1)
    return ((cov.pow(2).sum(dim=[-2, -1]) - diag_sq) / D / (D - 1)).mean()


def vicreg_loss(emb, pred_embs, cfg):
    """Full VICReg loss matching PLDM's VICRegObjective.

    emb:       (B, T, D) — all encoder outputs; emb[:,0,:] is the context frame
    pred_embs: (B, T-1, D) — GRU predictions for frames 1..T-1
    """
    vc = cfg.loss.vicreg

    # Rearrange to (T, B, D) to match PLDM convention
    enc = rearrange(emb, "b t d -> t b d")
    pred = rearrange(pred_embs, "b t d -> t b d")

    # sim loss: MSE between predictions and target encodings
    sim_loss = (enc[1:] - pred).pow(2).mean()

    # temporal sim loss: MSE between consecutive encoder outputs
    sim_loss_t = (enc[1:] - enc[:-1]).pow(2).mean() if vc.sim_coeff_t else enc.new_zeros(1)

    # std / cov on context frame (across batch)
    ctx = enc[:1]                               # (1, B, D)
    std_loss = _std_loss(ctx, vc.std_margin)
    cov_loss = _cov_loss(ctx)

    # std / cov on target frames (across time, within each batch sample)
    tgt_t = enc[1:].permute(1, 0, 2)           # (B, T-1, D)
    std_loss_t = _std_loss(tgt_t, vc.std_margin_t) if vc.std_coeff_t else enc.new_zeros(1)
    cov_loss_t = _cov_loss(tgt_t) if vc.cov_coeff_t else enc.new_zeros(1)

    total = (
        vc.sim_coeff   * sim_loss
        + vc.std_coeff   * std_loss
        + vc.cov_coeff   * cov_loss
        + vc.sim_coeff_t * sim_loss_t
        + vc.std_coeff_t * std_loss_t
        + vc.cov_coeff_t * cov_loss_t
    )
    return total, {
        "sim_loss": sim_loss, "std_loss": std_loss, "cov_loss": cov_loss,
        "sim_loss_t": sim_loss_t, "std_loss_t": std_loss_t, "cov_loss_t": cov_loss_t,
    }


# ---------------------------------------------------------------------------
# Training forward
# ---------------------------------------------------------------------------

def pldmwm_forward(self, batch, stage, cfg):
    logs = {}
    b = batch["pixels"].shape[0]
    samples_key = f"_{stage}_samples_num"
    setattr(self, samples_key, getattr(self, samples_key, 0) + b)
    logs[f"{stage}/samples"] = getattr(self, samples_key)
    self.samples_num = getattr(self, "samples_num", 0) + b
    logs["samples"] = self.samples_num

    pixels = batch["pixels"]                              # (B, T, C, H, W)
    action = batch["action"].to(pixels.dtype)             # (B, T, act_dim_raw)
    T = pixels.shape[1]

    # Encode all frames → (B, T, D)
    b_size, t_size = pixels.shape[:2]
    emb = rearrange(
        self.model.encoder(rearrange(pixels, "b t c h w -> (b t) c h w")),
        "(b t) d -> b t d", b=b_size, t=t_size,
    )

    # Action embeddings → (B, T, A)
    act_emb = self.model.action_encoder(action)

    # Teacher-forced GRU rollout: predict emb[t+1] from (emb[t], act[t])
    preds = []
    for t in range(T - 1):
        preds.append(self.model.gru_step_teacher(act_emb[:, t, :], emb[:, t, :]))
    pred_embs = torch.stack(preds, dim=1)                 # (B, T-1, D)

    # VICReg
    vc_loss, vc_parts = vicreg_loss(emb, pred_embs, cfg)

    # IDM: predict action[t] from (emb[t], emb[t+1])
    idm_cfg = cfg.loss.idm
    pairs = rearrange(
        torch.cat([emb[:, :-1, :], emb[:, 1:, :]], dim=-1),
        "b t d -> (b t) d",
    )                                                      # (B*(T-1), 2D)
    act_target = rearrange(action[:, :-1, :], "b t d -> (b t) d")
    act_pred = self.idm(pairs)
    idm_loss = F.mse_loss(act_pred, act_target)

    loss = vc_loss + idm_cfg.coeff * idm_loss

    output = {"emb": emb, "loss": loss, "idm_loss": idm_loss, **vc_parts}

    for k, v in output.items():
        if "loss" in k:
            logs[f"{stage}/{k}"] = v.detach()

    self.log_dict(logs, on_step=True, sync_dist=True)
    output["loss"] = self.rescale_loss_for_grad_acc(output["loss"])
    return output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../conf", config_name="pldmwm_train_tetris")
def run(cfg):
    ##########################
    ##       dataset        ##
    ##########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    dataset.transform = spt.data.transforms.Compose(
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    )

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = split_dataset(cfg.train_split, dataset, rnd_gen=rnd_gen)

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True,
        drop_last=(len(train_set) > cfg.loader.batch_size), generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    with open_dict(cfg):
        cfg["total_steps"] = len(train) * cfg.trainer.max_epochs

    ##############################
    ##       model / optim      ##
    ##############################

    action_dim_raw = dataset.get_dim("action")
    action_emb_dim = cfg.wm.action_emb_dim
    embed_dim = cfg.wm.embed_dim

    encoder = ImpalaEncoder(
        input_channels=cfg.wm.input_channels,
        width=cfg.wm.width,
        stack_sizes=tuple(cfg.wm.stack_sizes),
        num_blocks=cfg.wm.num_blocks,
        img_size=cfg.img_size,
        embed_dim=embed_dim,
        final_ln=cfg.wm.final_ln,
    )

    world_model = PLDMWM(
        encoder=encoder,
        action_encoder=nn.Linear(action_dim_raw, action_emb_dim),
        action_emb_dim=action_emb_dim,
        frameskip=cfg.data.dataset.frameskip,
    )

    # IDM: Linear(2D→512, BN, ReLU) → Linear(512→action_dim)  (matches PLDM arch='512')
    idm = nn.Sequential(
        nn.Linear(embed_dim * 2, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Linear(512, action_dim_raw),
    )

    with open_dict(cfg):
        cfg["num_params"] = sum(p.numel() for p in world_model.parameters() if p.requires_grad)

    data_module = spt.data.DataModule(train=train, val=val)
    module = spt.Module(
        model=world_model,
        idm=idm,
        forward=partial(pldmwm_forward, cfg=cfg),
        optim=OmegaConf.to_container(cfg.optimizers, resolve=True),
    )

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

    callbacks = [
        ModelObjectCallBack(
            dirpath=run_dir, filename=cfg.output_model_name,
            epoch_interval=cfg.checkpoint_interval,
        ),
        EMAMetricsSummaryCallback(),
        LearningRateMonitor(logging_interval="step"),
        ClampSchedulerCallback(),
    ]
    if OmegaConf.select(cfg, "grid_mask_probe.enabled", default=False):
        callbacks.append(GridMaskProbeCallback(cfg, run_dir))
    if OmegaConf.select(cfg, "queue_probe.enabled", default=False):
        callbacks.append(QueueProbeCallback(cfg, run_dir))

    trainer = pl.Trainer(
        **cfg.trainer, callbacks=callbacks,
        num_sanity_val_steps=1, logger=logger, enable_checkpointing=True,
    )

    resume_from = OmegaConf.select(cfg, "resume_from")
    ckpt_path = Path(resume_from) if resume_from is not None else None

    spt.Manager(trainer=trainer, module=module, data=data_module, ckpt_path=ckpt_path)()


if __name__ == "__main__":
    run()
