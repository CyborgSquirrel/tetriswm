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


def cache_embeddings(world_model, dataloader, cfg, device):
    """Run the frozen world model once over the dataloader and return cached (latents, masks)."""
    world_model = world_model.to(device)
    all_latents, all_masks = [], []
    with torch.no_grad():
        for batch in dataloader:
            pixels = batch["pixels"].to(device)
            masks = batch["tetris_grid_mask"]
            b, t = pixels.shape[:2]

            pixels_flat = rearrange(pixels.float(), "b t ... -> (b t) ...")
            output = world_model.encoder(pixels_flat, interpolate_pos_encoding=True)
            cls = output.last_hidden_state[:, 0]
            if cfg.probe.latent_source == "projected":
                cls = world_model.projector(cls)
            elif cfg.probe.latent_source == "encoder_cls":
                pass
            else:
                raise ValueError("probe.latent_source must be 'encoder_cls' or 'projected'")

            all_latents.append(cls.cpu())
            all_masks.append(rearrange(masks, "b t w h -> (b t) (w h)").float())

    return torch.cat(all_latents), torch.cat(all_masks)


def build_cached_dataloaders(world_model, raw_dm, cfg):
    cache_path = cfg.probe.get("latents_cache_path")
    cache_path = cache_path and Path(cache_path)

    if cache_path is None or not cache_path.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Caching train embeddings...")
        train_latents, train_masks = cache_embeddings(world_model, raw_dm.train_dataloader(), cfg, device)
        print("Caching val embeddings...")
        val_latents, val_masks = cache_embeddings(world_model, raw_dm.val_dataloader(), cfg, device)

        if cache_path is not None:
            print(f"Saving cached embeddings to {cache_path}")
            torch.save(
                {"train_latents": train_latents, "train_masks": train_masks,
                 "val_latents": val_latents, "val_masks": val_masks},
                cache_path,
            )
    else:
        print(f"Loading cached embeddings from {cache_path}")
        saved = torch.load(cache_path, map_location="cpu", weights_only=True)
        train_latents, train_masks = saved["train_latents"], saved["train_masks"]
        val_latents, val_masks = saved["val_latents"], saved["val_masks"]

    loader_kwargs = dict(cfg.loader)
    loader_kwargs.pop("prefetch_factor", None)
    loader_kwargs.pop("pin_memory", None)
    loader_kwargs.pop("persistent_workers", None)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_ds = torch.utils.data.TensorDataset(train_latents, train_masks)
    val_ds = torch.utils.data.TensorDataset(val_latents, val_masks)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=loader_kwargs["batch_size"], shuffle=True,
        drop_last=True, generator=rnd_gen,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=loader_kwargs["batch_size"], shuffle=False,
        drop_last=False, num_workers=0,
    )
    return spt.data.DataModule(train=train_loader, val=val_loader)


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
    def __init__(self, probe, cfg):
        super().__init__()
        self.probe = probe
        self.model = probe
        self.cfg = cfg
        self.save_hyperparameters(
            {"cfg": OmegaConf.to_container(cfg, resolve=True)}
        )

    def _step(self, batch, stage):
        latents, targets = batch  # (N, D), (N, W*H)

        logits = self.probe(latents)
        loss = F.binary_cross_entropy_with_logits(logits, targets)

        with torch.no_grad():
            preds = logits.sigmoid() > 0.5
            acc = (preds == targets.bool()).float().mean()

        self.log(f"{stage}/bce_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/pixel_acc", acc, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")
        interval = self.cfg.probe.train_preview_interval
        if interval is not None and batch_idx % interval == 0 and self.trainer.is_global_zero and self.logger:
            self._log_mask_preview(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
        if batch_idx == 0 and self.trainer.is_global_zero and self.logger:
            self._log_mask_preview(batch)
        return loss

    def _log_mask_preview(self, batch):
        latents, masks = batch  # (N, D), (N, W*H)
        n = min(self.cfg.probe.preview_images, latents.size(0))
        w, h = self.cfg.probe.grid_width, self.cfg.probe.grid_height

        with torch.no_grad():
            logits = self.probe(latents)

        gt = masks[:n].reshape(n, 1, w, h).permute(0, 1, 3, 2)
        pred = (logits[:n].sigmoid() > 0.5).float().reshape(n, 1, w, h).permute(0, 1, 3, 2)

        gt_imgs = F.interpolate(gt, scale_factor=10, mode="nearest")
        pred_imgs = F.interpolate(pred, scale_factor=10, mode="nearest")

        self.logger.log_image(
            key="probe/masks",
            images=[*gt_imgs, *pred_imgs],
            caption=[f"gt_{i}" for i in range(n)] + [f"pred_{i}" for i in range(n)],
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.probe.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )


@hydra.main(version_base=None, config_path="./config/train", config_name="probe")
def run(cfg):
    world_model = load_world_model(cfg.probe.world_model_path)
    raw_dm = build_dataloaders(cfg)
    data_module = build_cached_dataloaders(world_model, raw_dm, cfg)

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

    module = ProbeModule(probe=probe, cfg=cfg)
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
