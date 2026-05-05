from tetris.tetris import Action, GameState, Tetris, WmDynamics
import argparse
import os

import numpy as np
import pygame
from omegaconf import OmegaConf

os.environ.setdefault("SDL_VIDEODRIVER", "wayland")

SCALE = 4

KEY_ACTION: dict[int, Action] = {
    pygame.K_LEFT:   Action.MOVE_LEFT,
    pygame.K_RIGHT:  Action.MOVE_RIGHT,
    pygame.K_DOWN:   Action.MOVE_DOWN,
    pygame.K_SPACE:  Action.HARD_DROP,
    pygame.K_z:      Action.ROTATE_LEFT,
    pygame.K_x:      Action.ROTATE_RIGHT,
    pygame.K_RETURN: Action.NOTHING,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Play Tetris")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p = subparsers.add_parser("tetris", help="Play regular Tetris")
    p.add_argument("--config", metavar="PATH", help="Path to a dataset Hydra config (e.g. config/tetris/overfit.yaml)")

    p = subparsers.add_parser("wm", help="Play Tetris via World Model")
    p.add_argument("--world-model", required=True, metavar="PATH", dest="world_model_path")
    p.add_argument("--grid-probe",  required=True, metavar="PATH", dest="grid_probe_path")
    p.add_argument("--config", required=True, metavar="PATH", help="Path to a dataset Hydra config (e.g. config/tetris/overfit.yaml)")

    return parser.parse_args()


def make_game(args):
    match args.mode:
        case "tetris":
            if args.config is not None:
                return Tetris.from_config(OmegaConf.load(args.config).tetris)
            else:
                return Tetris()
        case "wm":
            cfg = OmegaConf.load(args.config).tetris
            initial_tetris = Tetris.from_config(cfg)
            tetris_kwargs = dict(width=cfg.width, height=cfg.height)
            dynamics = WmDynamics(
                world_model_path=args.world_model_path,
                grid_probe_path=args.grid_probe_path,
                tetris=initial_tetris,
            )
            return Tetris(dynamics=dynamics, **tetris_kwargs)
        case _:
            raise ValueError(f"Unknown mode: {args.mode}")


def main():
    args = parse_args()
    pygame.init()

    game = make_game(args)

    surf = pygame.display.set_mode((224*SCALE, 224*SCALE))

    pygame.display.set_caption("tetris")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return
                action = KEY_ACTION.get(event.key)
                if action is not None:
                    game.step(action)

        img = game.render()
        tile_surf = pygame.surfarray.make_surface(img)
        pygame.transform.scale(tile_surf, (224*SCALE, 224*SCALE), surf)
        pygame.display.flip()


if __name__ == "__main__":
    main()
