from pathlib import Path
import os.path

import hdf5plugin  # noqa: F401
import wandb
from tqdm import tqdm
import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.data.utils import get_cache_dir
import torch
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from utils import get_img_preprocessor, ModelObjectCallBack, split_dataset


class GraftedLatentsDataset(torch.utils.data.Dataset):
    """Wraps an HDF5Dataset and injects pre-computed latents per window."""

    def __init__(self, hdf5_dataset, latents):
        self.inner = hdf5_dataset
        self.latents = latents  # (N_total_frames, D)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        item = self.inner[idx]
        ep_idx, start = self.inner.clip_indices[idx]
        g_start = int(self.inner.offsets[ep_idx]) + start
        span = self.inner.span
        frameskip = self.inner.frameskip
        item["latents"] = self.latents[g_start:g_start + span:frameskip]  # (T, D)
        return item

    @property
    def clip_indices(self):
        return self.inner.clip_indices

    @property
    def lengths(self):
        return self.inner.lengths


def encode_and_save_latents(world_model, cfg):
    """Encode all pixels (T=1) and save latents as a .pt file next to the world model."""
    latents_path = Path(os.path.splitext(cfg.world_model_path)[0] + "_latents.pt")

    if latents_path.exists():
        print(f"Loading cached latents from {latents_path}")
        return torch.load(latents_path, map_location="cpu", weights_only=True)

    pixel_dataset = swm.data.HDF5Dataset(
        name=cfg.data.dataset.name,
        num_steps=1,
        frameskip=cfg.data.dataset.frameskip,
        keys_to_load=["pixels"],
        transform=spt.data.transforms.Compose(
            get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
        ),
    )
    loader = torch.utils.data.DataLoader(
        pixel_dataset, **cfg.wm_loader,
        shuffle=False, drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_model = world_model.to(device)

    all_latents = []
    print("Encoding all frames...")
    with torch.no_grad():
        for batch in tqdm(loader):
            pixels = batch["pixels"].to(device)  # (B, 1, C, H, W)
            info = {"pixels": pixels}
            world_model.encode(info)
            all_latents.append(info["emb"][:, 0].float().cpu())  # (B, D)

    latents = torch.cat(all_latents)  # (N_total, D)
    torch.save(latents, latents_path)
    print(f"Saved latents to {latents_path}")
    return latents


def build_dataloaders(cfg, latents):
    hdf5_dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    dataset = GraftedLatentsDataset(hdf5_dataset, latents)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = split_dataset(cfg.train_split, dataset, rnd_gen=rnd_gen)

    train = torch.utils.data.DataLoader(
        train_set, **cfg.wm_loader,
        shuffle=True, drop_last=True, generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.wm_loader,
        shuffle=False, drop_last=False,
    )
    return spt.data.DataModule(train=train, val=val)


def load_world_model(path):
    if path is None:
        raise ValueError("Set world_model_path in the config.")
    try:
        model = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(Path(path), map_location="cpu")
    if hasattr(model, "model"):
        model = model.model
    return model.eval().requires_grad_(False)


def _macro_f1(preds, labels, num_classes):
    f1_sum = 0.0
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).float().sum()
        fp = ((preds == c) & (labels != c)).float().sum()
        fn = ((preds != c) & (labels == c)).float().sum()
        f1_sum += tp / (tp + 0.5 * (fp + fn) + 1e-8)
    return f1_sum / num_classes


class LinesClearedProbeModule(pl.LightningModule):
    def __init__(self, probe, cfg):
        super().__init__()
        self.probe = probe
        self.model = probe
        self.cfg = cfg
        self.num_classes = cfg.num_classes

    def _step(self, stage, batch, batch_idx):
        latents = rearrange(batch["latents"], "b t d -> b (t d)")  # (B, 2*D)
        labels = batch["last_cleared_rows"][:, 1].long()           # (B,)

        logits = self.probe(latents)  # (B, num_classes)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean()
            macro_f1 = _macro_f1(preds, labels, self.num_classes)

        self.log_dict(
            {
                f"probe/{stage}/ce_loss": loss.detach(),
                f"probe/{stage}/acc": acc,
                f"probe/{stage}/macro_f1": macro_f1,
            },
            on_step=True, sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step("fit", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step("validate", batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.probe.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )


@hydra.main(version_base=None, config_path="../conf", config_name="lines_cleared_probe")
def run(cfg):
    world_model = load_world_model(cfg.world_model_path)

    latents = encode_and_save_latents(world_model, cfg)
    data_module = build_dataloaders(cfg, latents)

    probe = torch.nn.Linear(2 * cfg.lines_cleared_probe.embed_dim, cfg.lines_cleared_probe.num_classes)

    run_id = cfg.get("subdir") or ""
    run_dir = Path(get_cache_dir(), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open_dict(cfg):
        cfg.run_dir = str(run_dir)

    with open(run_dir / "lines_cleared_probe_config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enabled:
        wandb.finish()
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams({
            **OmegaConf.to_container(cfg, resolve=True),
            "raw": OmegaConf.to_container(cfg, resolve=False),
        })

    module = LinesClearedProbeModule(probe=probe, cfg=cfg.lines_cleared_probe)
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
