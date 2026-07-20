from pathlib import Path
import os.path

import wandb
from tqdm import tqdm
import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from utils import get_img_preprocessor, ModelObjectCallBack, split_dataset


def build_dataloaders(cfg):
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]
    dataset.transform = spt.data.transforms.Compose(*transforms)

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


def cache_embeddings(world_model, dataloader, cfg, device):
    """Run the frozen world model over consecutive pairs and return (paired_latents, labels)."""
    world_model = world_model.to(device)
    all_latents, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pixels = batch["pixels"].to(device)   # (B, T=2, C, H, W)
            # last_cleared_rows at index 1: lines cleared transitioning from step 0 to step 1
            labels = batch["last_cleared_rows"][:, 1].long()  # (B,)

            info = {"pixels": pixels}
            world_model.encode(info)
            emb = info["emb"]  # (B, T=2, D)

            paired = rearrange(emb, "b t d -> b (t d)").cpu()  # (B, 2*D)
            all_latents.append(paired)
            all_labels.append(labels.cpu())

    return torch.cat(all_latents), torch.cat(all_labels)


def build_cached_dataloaders(world_model, raw_dm, cfg):
    cache_path = Path(os.path.splitext(cfg.world_model_path)[0] + "_lines_cleared_emb.pt")

    if not cache_path.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Caching train embeddings...")
        train_latents, train_labels = cache_embeddings(world_model, raw_dm.train_dataloader(), cfg, device)
        print("Caching val embeddings...")
        val_latents, val_labels = cache_embeddings(world_model, raw_dm.val_dataloader(), cfg, device)

        print(f"Saving cached embeddings to {repr(str(cache_path))}")
        torch.save(
            {
                "train_latents": train_latents, "train_labels": train_labels,
                "val_latents": val_latents, "val_labels": val_labels,
            },
            cache_path,
        )
    else:
        print(f"Loading cached embeddings from {repr(str(cache_path))}")
        saved = torch.load(cache_path, map_location="cpu", weights_only=True)
        train_latents, train_labels = saved["train_latents"], saved["train_labels"]
        val_latents, val_labels = saved["val_latents"], saved["val_labels"]

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_ds = torch.utils.data.TensorDataset(train_latents, train_labels)
    val_ds = torch.utils.data.TensorDataset(val_latents, val_labels)
    loader_cfg = OmegaConf.to_container(cfg.probe_loader, resolve=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, **loader_cfg,
        shuffle=True, drop_last=True, generator=rnd_gen,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, **loader_cfg,
        shuffle=False, drop_last=False,
    )
    return spt.data.DataModule(train=train_loader, val=val_loader)


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
        self.cfg = cfg
        self.num_classes = cfg.num_classes

    def _step(self, stage, batch, batch_idx):
        latents, labels = batch  # (B, 2*D), (B,)
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
    raw_dm = build_dataloaders(cfg)
    data_module = build_cached_dataloaders(world_model, raw_dm, cfg)

    probe = torch.nn.Linear(2 * cfg.lines_cleared_probe.embed_dim, cfg.lines_cleared_probe.num_classes)

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)
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
