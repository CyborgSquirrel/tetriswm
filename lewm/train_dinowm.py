"""DINO-WM training integrated into the lewm Lightning infrastructure.

Uses the native ViTPredictor from dinowm with a frozen DINO encoder.
Loss: MSE between predicted patch tokens and actual DINO patch tokens of the next frame.
No SIGReg needed — frozen encoder prevents embedding collapse.
"""

import sys
import contextlib as ctl
from functools import partial
from pathlib import Path

# Make dinowm models importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dinowm" / "models"))

import wandb
import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from torchvision import transforms

from dino import DinoV2Encoder
from vit import ViTPredictor
from proprio import ProprioceptiveEmbedding

from train import (
    LearningRateMonitor,
    EMAMetricsSummaryCallback,
    GridMaskProbeCallback,
    QueueProbeCallback,
    ClampSchedulerCallback,
)
from utils import get_img_preprocessor, ModelObjectCallBack, split_dataset


class DinoWM(nn.Module):
    """Wraps frozen DINO encoder + ViTPredictor + action encoder."""

    def __init__(self, encoder, predictor, action_encoder):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder


def dinowm_forward(self, batch, stage, cfg):
    """Predict next-frame DINO patch tokens; MSE in embedding space."""

    logs = {}
    b = batch["pixels"].shape[0]
    samples_key = f"_{stage}_samples_num"
    setattr(self, samples_key, getattr(self, samples_key, 0) + b)
    logs[f"{stage}/samples"] = getattr(self, samples_key)
    self.samples_num = getattr(self, "samples_num", 0) + b
    logs["samples"] = self.samples_num

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds

    pixels = batch["pixels"]  # (B, T, C, H, W)
    action = batch["action"].to(pixels.dtype)  # (B, T, act_dim * frameskip)
    emb = None

    # Encode all T frames with frozen DINO → (B, T, P, D)
    with torch.no_grad():
        t_size = pixels.shape[1]
        pixels_flat = rearrange(pixels, "b t c h w -> (b t) c h w")
        patch_embs_flat = self.model.encoder(pixels_flat)  # (B*T, P, D)
        patch_embs = rearrange(patch_embs_flat, "(b t) p d -> b t p d", b=b, t=t_size)

        if self.patch_projector is not None:
            dims = {}
            dims["b"], dims["t"], dims["p"], *_ = patch_embs.shape

            emb = patch_embs
            emb = rearrange(emb, "b t p d -> (b t p) d")
            emb = self.patch_projector(emb)
            emb = rearrange(emb, "(b t p) d -> b t (p d)", **dims)

    # Action embedding → (B, T, action_emb_dim)
    act_emb = self.model.action_encoder(action)

    concat_dim = cfg.wm.concat_dim
    if concat_dim == 1:
        # Tile action across patches, concat along feature dim → (B, T, P, dino_dim + action_emb_dim)
        act_repeated = act_emb.unsqueeze(2).expand(-1, -1, patch_embs.shape[2], -1)
        z = torch.cat([patch_embs, act_repeated], dim=3)
    else:
        # Append action as an extra token per frame → (B, T, P+1, dino_dim)
        z = torch.cat([patch_embs, act_emb.unsqueeze(2)], dim=2)

    z_src = z[:, :ctx_len]
    z_tgt = patch_embs[:, n_preds:]  # (B, ctx_len, P, dino_dim)

    # Flatten T*P for ViTPredictor, then unflatten
    z_src_flat = rearrange(z_src, "b t p d -> b (t p) d")
    z_pred_flat = self.model.predictor(z_src_flat)
    z_pred = rearrange(z_pred_flat, "b (t p) d -> b t p d", t=ctx_len)

    # MSE on visual patch tokens only
    if concat_dim == 1:
        dino_dim = cfg.wm.dino_dim
        z_pred_visual = z_pred[:, :, :, :dino_dim]  # (B, ctx_len, P, dino_dim)
    else:
        z_pred_visual = z_pred[:, :, :-1, :]  # exclude action token
    pred_loss = F.mse_loss(z_pred_visual, z_tgt.detach())

    output = {
        "emb": emb,
        "pred_loss": pred_loss,
        "loss": pred_loss,
    }

    for k, v in output.items():
        if "loss" in k:
            logs[f"{stage}/{k}"] = v.detach()

    self.log_dict(logs, on_step=True, sync_dist=True)
    output["loss"] = self.rescale_loss_for_grad_acc(output["loss"])
    return output


@hydra.main(version_base=None, config_path="../conf", config_name="dinowm_train_tetris")
def run(cfg):
    ##########################
    ##       dataset        ##
    ##########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)

    transforms_list = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
    dataset.transform = spt.data.transforms.Compose(*transforms_list)

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

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

    encoder = DinoV2Encoder(name=cfg.wm.dino_model, feature_key="x_norm_patchtokens")
    for param in encoder.parameters():
        param.requires_grad = False

    # Infer num_patches from image size and DINO patch size
    patch_size = encoder.base_model.patch_size
    num_patches = (cfg.img_size // patch_size) ** 2
    dino_dim = encoder.emb_dim
    num_patches_with_action = num_patches + 1  # +1 for the action token

    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
    concat_dim = OmegaConf.select(cfg, "wm.concat_dim", default=0)
    if concat_dim == 1:
        action_emb_dim = cfg.wm.action_emb_dim
        predictor_dim = dino_dim + action_emb_dim
        num_patches_for_pred = num_patches
    else:
        action_emb_dim = dino_dim
        predictor_dim = dino_dim
        num_patches_for_pred = num_patches_with_action

    action_encoder = ProprioceptiveEmbedding(
        num_frames=cfg.wm.history_size,
        in_chans=effective_act_dim,
        emb_dim=action_emb_dim,
    )

    predictor = ViTPredictor(
        num_patches=num_patches_for_pred,
        num_frames=cfg.wm.history_size,
        dim=predictor_dim,
        **cfg.predictor,
    )

    world_model = DinoWM(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
    )

    patch_projector = None
    if OmegaConf.select(cfg, "wm.patch_projector.enabled", default=False):
        proj_dim = cfg.wm.patch_projector.dim
        patch_projector = nn.Linear(dino_dim, proj_dim)
        probe_embed_dim = num_patches * proj_dim
    else:
        probe_embed_dim = dino_dim

    with open_dict(cfg):
        cfg.wm.embed_dim = probe_embed_dim
        cfg.wm.dino_dim = dino_dim
        cfg.wm.concat_dim = concat_dim
        cfg["num_params"] = sum(p.numel() for p in world_model.parameters() if p.requires_grad)

    data_module = spt.data.DataModule(train=train, val=val)
    module = spt.Module(
        model=world_model,
        patch_projector=patch_projector,
        forward=partial(dinowm_forward, cfg=cfg),
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
        ModelObjectCallBack(dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=cfg.checkpoint_interval),
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

    resume_from = OmegaConf.select(cfg, "resume_from")
    ckpt_path = Path(resume_from) if resume_from is not None else None

    spt.Manager(trainer=trainer, module=module, data=data_module, ckpt_path=ckpt_path)()


if __name__ == "__main__":
    run()
