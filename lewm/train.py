import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import TetrisDataset
from .model import LeWM


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = TetrisDataset(args.data_dir, seq_len=args.seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Dataset: {len(dataset)} sub-trajectories")

    # Infer n_actions from the environment
    # tetris-gymnasium default: 5 actions (left, right, rotate, hard drop, nothing)
    model = LeWM(n_actions=args.n_actions, embed_dim=args.embed_dim,
                 lambda_reg=args.lambda_reg).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f}M")

    # AdamW with weight decay is standard for transformers
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Cosine LR schedule: smooth decay, no sharp drops
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for obs, actions in loader:
            obs = obs.to(device)
            actions = actions.to(device)

            out = model(obs, actions)

            optimizer.zero_grad()
            out["loss"].backward()
            # Gradient clipping: prevents exploding gradients in the transformer
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % args.log_every == 0:
                print(
                    f"epoch {epoch+1}/{args.epochs}  step {global_step}"
                    f"  loss={out['loss'].item():.4f}"
                    f"  pred={out['pred_loss'].item():.4f}"
                    f"  sigreg={out['sigreg_loss'].item():.4f}"
                    f"  lr={scheduler.get_last_lr()[0]:.2e}"
                )

        # Save checkpoint every epoch
        torch.save(
            {"epoch": epoch, "model": model.state_dict(),
             "optimizer": optimizer.state_dict()},
            save_dir / f"lewm_epoch{epoch+1:03d}.pt",
        )

    print("Training complete.")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     default="data/traj1")
    parser.add_argument("--save_dir",     default="checkpoints")
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--seq_len",      type=int,   default=4)
    parser.add_argument("--embed_dim",    type=int,   default=192)
    parser.add_argument("--n_actions",    type=int,   default=8)
    parser.add_argument("--lambda_reg",   type=float, default=0.1)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--log_every",    type=int,   default=50)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
