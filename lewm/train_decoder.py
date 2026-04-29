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
from torchvision.utils import save_image

from module import ImageDecoder
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


def build_dataloaders(cfg):
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
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
        raise ValueError("Set decoder.world_model_path to a saved LeWM object checkpoint.")

    try:
        model = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(Path(path), map_location="cpu")
    if hasattr(model, "model"):
        model = model.model
    return model.eval().requires_grad_(False)


class DecoderModule(pl.LightningModule):
    def __init__(self, world_model, decoder, cfg):
        super().__init__()
        self.world_model = world_model
        self.decoder = decoder
        self.model = decoder
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

        if self.cfg.decoder.latent_source == "projected":
            cls = self.world_model.projector(cls)
        elif self.cfg.decoder.latent_source != "encoder_cls":
            raise ValueError(
                "decoder.latent_source must be 'encoder_cls' or 'projected'"
            )

        return cls, pixels_flat

    def _step(self, batch, stage):
        with torch.no_grad():
            latents, targets = self._get_latents(batch["pixels"])

        recon = self.decoder(latents)
        loss = F.mse_loss(recon, targets)
        self.log(f"{stage}/recon_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
        if batch_idx == 0 and self.trainer.is_global_zero:
            self._save_preview(batch)
        return loss

    def _save_preview(self, batch):
        preview_dir = Path(self.cfg.run_dir) / "decoder_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            latents, targets = self._get_latents(batch["pixels"])
            recon = self.decoder(latents)

        n = min(self.cfg.decoder.preview_images, targets.size(0))
        imgs = torch.cat([targets[:n], recon[:n]], dim=0)
        imgs = unnormalize_imagenet(imgs).clamp(0, 1)
        save_image(
            imgs,
            preview_dir / f"epoch_{self.current_epoch + 1:04d}.png",
            nrow=n,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.decoder.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )


def unnormalize_imagenet(x):
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return x * std + mean


@hydra.main(version_base=None, config_path="./config/train", config_name="decoder")
def run(cfg):
    data_module = build_dataloaders(cfg)
    world_model = load_world_model(cfg.decoder.world_model_path)

    latent_dim = (
        world_model.encoder.config.hidden_size
        if cfg.decoder.latent_source == "encoder_cls"
        else cfg.wm.embed_dim
    )
    decoder = ImageDecoder(
        latent_dim=latent_dim,
        image_size=cfg.img_size,
        **cfg.decoder.model,
    )

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open_dict(cfg):
        cfg.run_dir = str(run_dir)

    with open(run_dir / "decoder_config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    module = DecoderModule(world_model=world_model, decoder=decoder, cfg=cfg)
    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        logger=logger,
        enable_checkpointing=True,
    )
    trainer.fit(module, datamodule=data_module)


if __name__ == "__main__":
    run()
