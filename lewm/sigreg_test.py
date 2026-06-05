import wandb
import hydra
import torch
from omegaconf import DictConfig

from module import SIGReg
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../conf", config_name="sigreg_test")
def main(cfg: DictConfig):
    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic)
    device = torch.device(cfg.device)
    v = torch.randn((cfg.t, cfg.b, cfg.d), requires_grad=True, device=device)
    sigreg = SIGReg().to(device)

    opt = hydra.utils.instantiate(cfg.optimizer, params=[v])

    sched = None
    if cfg.scheduler:
        sched_fn = hydra.utils.instantiate(cfg.scheduler, _partial_=True)
        sched = sched_fn(optimizer=opt)

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            config={"t": cfg.t, "b": cfg.b, "d": cfg.d, "epochs": cfg.epochs},
        )

    for step in tqdm(range(cfg.epochs)):
        opt.zero_grad()
        loss = sigreg(v)
        loss.backward()
        opt.step()
        if sched:
            sched.step()

        if cfg.wandb.enabled:
            lr = opt.param_groups[0]["lr"]
            norms = v.detach().norm(dim=-1).cpu().flatten().numpy()
            wandb.log({"loss": loss.item(), "lr": lr, "norm_hist": wandb.Histogram(norms)}, step=step)

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
