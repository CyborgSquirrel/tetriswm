import os
from functools import partial
from pathlib import Path

import wandb
import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.special
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack
from einops import repeat, rearrange, reduce


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    logs = {}

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]  # (B, ctx_len, D_ctx_emb)
    ctx_act = act_emb[:, :ctx_len]  # (B, ctx_len, D_act_emb)

    tgt_emb = emb[:, n_preds:]  # label (B, n_preds, D_ctx_emb)
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred (B, n_preds, D_ctx_emb)

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    # Action ranking
    if stage == "validate" and OmegaConf.select(cfg, "val.action_rank.enabled", default=False):
        ar_acts = torch.tensor(OmegaConf.to_container(cfg.val.action_rank.actions, resolve=True), dtype=torch.float32)
        ar_act_num = len(ar_acts)

        ar_ctx_act_base = batch["action"][:,:ctx_len]  # (B, T, D_act*frameskip)

        ar_pred_embs = []  # list of (B, n_preds, D_ctx_emb)
        ar_pred_embs.append(pred_emb)
        for ar_final_act in ar_acts:
            ar_ctx_act = ar_ctx_act_base.clone()
            ar_ctx_act[:, -1, -cfg.wm.action_dim:] = ar_final_act
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
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
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

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    match cfg.train_split.by:
        case "window":
            train_set, val_set = spt.data.random_split(
                dataset, lengths=[cfg.train_split.p, 1 - cfg.train_split.p], generator=rnd_gen
            )
        case "episode":
            n_eps = len(dataset.lengths)
            ep_order = torch.randperm(n_eps, generator=rnd_gen).tolist()
            n_train = int(n_eps * cfg.train_split.p)
            train_eps = set(ep_order[:n_train])
            val_eps = set(ep_order[n_train:])
            train_idx = [i for i, (ep, _) in enumerate(dataset.clip_indices) if ep in train_eps]
            val_idx   = [i for i, (ep, _) in enumerate(dataset.clip_indices) if ep in val_eps]
            train_set = torch.utils.data.Subset(dataset, train_idx)
            val_set   = torch.utils.data.Subset(dataset, val_idx)

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

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
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
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
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
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name,
        epoch_interval=cfg.checkpoint_interval,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
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
