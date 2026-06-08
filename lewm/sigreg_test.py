import wandb
import hydra
import torch
import itertools as itt
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from module import SIGReg
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../conf", config_name="sigreg_test")
def main(cfg: DictConfig):
    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)
    torch.manual_seed(cfg.data.seed)
    torch.use_deterministic_algorithms(cfg.deterministic)
    device = torch.device(cfg.device)
    v = torch.randn((cfg.data.hist, cfg.data.len, cfg.data.dim), requires_grad=True, device=device)
    sigreg = SIGReg().to(device)

    opt = hydra.utils.instantiate(cfg.optimizer, params=[v])

    sched = None
    if cfg.scheduler:
        sched_fn = hydra.utils.instantiate(cfg.scheduler, _partial_=True)
        sched = sched_fn(optimizer=opt)

    loader = DataLoader(
        TensorDataset(torch.arange(cfg.data.len)),
        batch_size=cfg.loader.batch_size,
        shuffle=cfg.loader.shuffle,
        drop_last=cfg.loader.drop_last,
    )

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            config={
                **OmegaConf.to_container(cfg, resolve=True),
                "raw": OmegaConf.to_container(cfg, resolve=False),
            },
        )

    global_step = itt.count()
    for _ in tqdm(range(cfg.epochs)):
        for (idx,) in loader:
            opt.zero_grad()
            loss = sigreg(v[:, idx, :])
            loss.backward()
            opt.step()
            if sched:
                sched.step()

            if cfg.wandb.enabled:
                lr = opt.param_groups[0]["lr"]
                norms = v.detach().norm(dim=-1).cpu().flatten().numpy()
                log_entry = {"loss": loss.item(), "lr": lr}
                if cfg.get("histogram_enabled", False):
                    log_entry["norm_hist"] = wandb.Histogram(norms)
                wandb.log(log_entry, step=next(global_step))

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
