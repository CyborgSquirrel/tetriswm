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

from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack, split_dataset


def build_dataloaders(cfg):
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    # with open_dict(cfg):
    #     for col in cfg.data.dataset.keys_to_load:
    #         if col.startswith("pixels") or col == "tetris_grid_mask":
    #             continue
    #         normalizer = get_column_normalizer(dataset, col, col)
    #         transforms.append(normalizer)

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
    """Run the frozen world model once over the dataloader and return cached (latents, masks)."""
    world_model = world_model.to(device)
    all_latents, all_masks = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pixels = batch["pixels"].to(device)
            masks = batch["tetris_grid_mask"]

            info = {}
            info["pixels"] = pixels
            world_model.encode(info)
            emb = info["emb"]

            all_latents.append(rearrange(emb.cpu(), "b t d -> (b t) d"))
            all_masks.append(rearrange(masks, "b t w h -> (b t) w h").float())

    return torch.cat(all_latents), torch.cat(all_masks)


def build_cached_dataloaders(world_model, raw_dm, cfg):
    cache_path = Path(os.path.splitext(cfg.world_model_path)[0] + "_emb.pt")

    # cache_path = cfg.grid_mask_probe.get("latents_cache_path")
    # cache_path = cache_path and Path(cache_path)

    if cache_path is None or not cache_path.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Caching train embeddings...")
        train_latents, train_masks = cache_embeddings(world_model, raw_dm.train_dataloader(), cfg, device)
        print("Caching val embeddings...")
        val_latents, val_masks = cache_embeddings(world_model, raw_dm.val_dataloader(), cfg, device)

        if cache_path is not None:
            print(f"Saving cached embeddings to {repr(str(cache_path))}")
            torch.save(
                {
                    "train_latents": train_latents, "train_masks": train_masks,
                    "val_latents": val_latents, "val_masks": val_masks,
                },
                cache_path,
            )
    else:
        print(f"Loading cached embeddings from {repr(str(cache_path))}")
        saved = torch.load(cache_path, map_location="cpu", weights_only=True)
        train_latents, train_masks = saved["train_latents"], saved["train_masks"]
        val_latents, val_masks = saved["val_latents"], saved["val_masks"]

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_ds = torch.utils.data.TensorDataset(train_latents, train_masks)
    val_ds = torch.utils.data.TensorDataset(val_latents, val_masks)
    train_loader = torch.utils.data.DataLoader(
        train_ds, *cfg.probe_loader,
        shuffle=True, drop_last=True, generator=rnd_gen,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, *cfg.probe_loader,
        shuffle=False, drop_last=False,
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
    def __init__(self, probe, cfg, *, prefix="", manual=False):
        super().__init__()
        self.probe = probe
        self.model = probe
        self.cfg = cfg
        self.prefix = prefix
        self.manual = manual

    def _step(self, stage, batch, batch_idx):
        logs = {}

        latents, masks = batch  # (B, D), (B, W, H)
        masks = rearrange(masks, "b w h -> b (w h)")

        logits = self.probe(latents)
        loss = F.binary_cross_entropy_with_logits(logits, masks)
        logs[self.prefix + f"{stage}/bce_loss"] = loss.detach()
        with torch.no_grad():
            preds = logits.sigmoid() > 0.5
            masks_bool = masks.bool()
            acc = (preds == masks_bool).float().mean()
            logs[self.prefix + f"{stage}/pixel_acc"] = acc

            tp = (preds & masks_bool).float().sum()
            fp = (preds & ~masks_bool).float().sum()
            fn = (~preds & masks_bool).float().sum()
            f1 = tp / (tp + 0.5 * (fp + fn) + 1e-8)
            logs[self.prefix + f"{stage}/f1"] = f1

        self.log_dict(logs, on_step=True, sync_dist=True)

        if batch_idx == 0 and (self.manual or self.trainer.is_global_zero):
            self._log_mask_preview(masks, logits, self.prefix + f"{stage}/masks")

        return loss

    def _log_mask_preview(self, masks, logits, key):
        n = min(self.cfg.preview_images, masks.size(0))
        w, h = self.cfg.grid_width, self.cfg.grid_height
        gt   = rearrange(masks[:n], "n (w h) -> n 1 h w", w=w, h=h)
        pred = rearrange((logits[:n].sigmoid() > 0.5).float(), "n (w h) -> n 1 h w", w=w, h=h)
        gt_imgs   = F.interpolate(gt,   scale_factor=10, mode="nearest")
        pred_imgs = F.interpolate(pred, scale_factor=10, mode="nearest")
        self.trainer.logger.log_image(
            key=key,
            images=[*gt_imgs, *pred_imgs],
            caption=[f"gt_{i}" for i in range(n)] + [f"pred_{i}" for i in range(n)],
        )

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
        ret = {}

        ret["optimizer"] = torch.optim.AdamW(
            self.probe.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )

        sched_cfg = self.cfg.get("scheduler", {})
        if sched_cfg.get("enabled", False):
            ret["lr_scheduler"] = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    ret["optimizer"],
                    T_max=self.trainer.max_epochs,
                    eta_min=sched_cfg.eta_min,
                ),
                "interval": "epoch",
            }

        return ret


def build_probe(latent_dim, out_dim, model_cfg):
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
        raise ValueError(f"probe.model.type must be 'linear' or 'mlp', got '{kind}'")


@hydra.main(version_base=None, config_path="./config/train", config_name="probe")
def run(cfg):
    world_model = load_world_model(cfg.world_model_path)
    raw_dm = build_dataloaders(cfg)
    data_module = build_cached_dataloaders(world_model, raw_dm, cfg)

    # latent_dim = cfg.wm.embed_dim
    latent_dim = cfg.grid_mask_probe.embed_dim
    grid_cells = cfg.grid_mask_probe.grid_width * cfg.grid_mask_probe.grid_height
    probe = build_probe(latent_dim, grid_cells, cfg.grid_mask_probe.model)

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open_dict(cfg):
        cfg.run_dir = str(run_dir)

    with open(run_dir / "probe_config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enabled:
        wandb.finish()
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams({
            **OmegaConf.to_container(cfg, resolve=True),
            "raw": OmegaConf.to_container(cfg, resolve=False),
        })

    module = ProbeModule(probe=probe, cfg=cfg.grid_mask_probe, prefix="probe/")
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
