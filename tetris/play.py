from tetris.tetris import Action, GameState, Tetris, WmDynamics
import argparse
import os

import numpy as np
import pygame
from omegaconf import OmegaConf

os.environ.setdefault("SDL_VIDEODRIVER", "wayland")

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
    parser.add_argument("--scale",        type=int, default=4)
    parser.add_argument("--config",      metavar="PATH",
                        help="Dataset Hydra config (e.g. config/tetris/overfit.yaml)")
    parser.add_argument("--gt",          action="store_true",
                        help="Add a GtDynamics game")
    parser.add_argument("--wm",          action="store_true",
                        help="Add a WmDynamics game (requires --world-model and --grid-probe)")
    parser.add_argument("--world-model", metavar="PATH", dest="world_model_path")
    parser.add_argument("--grid-probe",  metavar="PATH", dest="grid_probe_path")
    return parser.parse_args()


def make_games(args) -> list[Tetris]:
    cfg = OmegaConf.load(args.config).tetris if args.config else None
    games = []

    gt_tetris = Tetris.from_config(cfg) if cfg else Tetris()

    if args.gt:
        games.append(gt_tetris)

    if args.wm:
        dynamics = WmDynamics(
            world_model_path=args.world_model_path,
            grid_probe_path=args.grid_probe_path,
            gt_tetris=gt_tetris,
        )
        tetris_kwargs = dict(width=cfg.width, height=cfg.height) if cfg else {}
        games.append(Tetris(dynamics=dynamics, **tetris_kwargs))

    if not games:
        games.append(gt_tetris)

    return games


def main():
    args = parse_args()
    pygame.init()

    games = make_games(args)
    scale = args.scale

    surf = pygame.display.set_mode((224 * scale * len(games), 224 * scale))
    pygame.display.set_caption("tetris")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return
                if event.key == pygame.K_r:
                    games = make_games(args)
                    continue
                if event.key == pygame.K_t:
                    for game in games:
                        if isinstance(game.dynamics, WmDynamics):
                            game.dynamics._reanchor_to_gt()
                            game.dynamics._update_parent()
                action = KEY_ACTION.get(event.key)
                if action is not None:
                    for game in games:
                        game.step(action)

        frames = [game.render() for game in games]
        img = np.concatenate(frames, axis=0)
        tile_surf = pygame.surfarray.make_surface(img)
        scaled = pygame.transform.scale(tile_surf, (224 * scale * len(games), 224 * scale))
        surf.blit(scaled, (0, 0))
        pygame.display.flip()


if __name__ == "__main__":
    main()
