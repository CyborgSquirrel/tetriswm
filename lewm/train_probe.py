from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def build_dataloaders(cfg):
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels") or col == "tetris_grid_mask":
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )
    return spt.data.DataModule(train=train, val=val)


def load_world_model(path):
    if path is None:
        raise ValueError("Set probe.world_model_path to a saved LeWM object checkpoint.")

    try:
        model = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(Path(path), map_location="cpu")
    if hasattr(model, "model"):
        model = model.model
    return model.eval().requires_grad_(False)


class ProbeModule(pl.LightningModule):
    def __init__(self, world_model, probe, cfg):
        super().__init__()
        self.world_model = world_model
        self.probe = probe
        self.model = probe
        self.cfg = cfg
        self.save_hyperparameters(
            {"cfg": OmegaConf.to_container(cfg, resolve=True)}
        )

    def _get_latents(self, pixels):
        b = pixels.size(0)
        pixels_flat = rearrange(pixels.float(), "b t ... -> (b t) ...")
        output = self.world_model.encoder(
            pixels_flat, interpolate_pos_encoding=True
        )
        cls = output.last_hidden_state[:, 0]

        if self.cfg.probe.latent_source == "projected":
            cls = self.world_model.projector(cls)
        elif self.cfg.probe.latent_source != "encoder_cls":
            raise ValueError(
                "probe.latent_source must be 'encoder_cls' or 'projected'"
            )

        return cls  # (B*T, D)

    def _step(self, batch, stage):
        pixels = batch["pixels"]           # (B, T, C, H, W)
        masks = batch["tetris_grid_mask"]  # (B, T, W, H)
        b, t = pixels.shape[:2]

        with torch.no_grad():
            latents = self._get_latents(pixels)  # (B*T, D)

        targets = rearrange(masks, "b t w h -> (b t) (w h)").float()

        logits = self.probe(latents)  # (B*T, W*H)
        loss = F.binary_cross_entropy_with_logits(logits, targets)

        with torch.no_grad():
            preds = logits.sigmoid() > 0.5
            acc = (preds == targets.bool()).float().mean()

        self.log(f"{stage}/bce_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/pixel_acc", acc, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
        if batch_idx == 0 and self.trainer.is_global_zero and self.logger:
            self._log_mask_preview(batch)
        return loss

    def _log_mask_preview(self, batch):
        pixels = batch["pixels"]
        masks = batch["tetris_grid_mask"]  # (B, T, W, H)
        b, t = pixels.shape[:2]

        with torch.no_grad():
            latents = self._get_latents(pixels)
            logits = self.probe(latents)

        n = min(self.cfg.probe.preview_images, b * t)
        w, h = self.cfg.probe.grid_width, self.cfg.probe.grid_height

        gt = rearrange(masks, "b t w h -> (b t) 1 h w").float()[:n]
        pred = (logits.sigmoid() > 0.5).float().reshape(b * t, 1, w, h).permute(0, 1, 3, 2)[:n]

        # scale to [0, 1] and resize for visibility
        gt_imgs = F.interpolate(gt, scale_factor=10, mode="nearest")
        pred_imgs = F.interpolate(pred, scale_factor=10, mode="nearest")

        captions_gt = [f"gt_{i}" for i in range(n)]
        captions_pred = [f"pred_{i}" for i in range(n)]
        self.logger.log_image(
            key="probe/masks",
            images=[*gt_imgs, *pred_imgs],
            caption=captions_gt + captions_pred,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.probe.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )


@hydra.main(version_base=None, config_path="./config/train", config_name="probe")
def run(cfg):
    data_module = build_dataloaders(cfg)
    world_model = load_world_model(cfg.probe.world_model_path)

    latent_dim = (
        world_model.encoder.config.hidden_size
        if cfg.probe.latent_source == "encoder_cls"
        else cfg.wm.embed_dim
    )
    grid_cells = cfg.probe.grid_width * cfg.probe.grid_height
    probe = torch.nn.Linear(latent_dim, grid_cells)

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open_dict(cfg):
        cfg.run_dir = str(run_dir)

    with open(run_dir / "probe_config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    module = ProbeModule(world_model=world_model, probe=probe, cfg=cfg)
    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        logger=logger,
        enable_checkpointing=True,
    )
    trainer.fit(module, datamodule=data_module, ckpt_path=cfg.get("resume_ckpt"))


if __name__ == "__main__":
    run()
