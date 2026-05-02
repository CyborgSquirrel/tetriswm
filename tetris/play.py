from tetris.tetris import Action, GameState, Tetris, WMTetris
import argparse
import os

import numpy as np
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "wayland")

SCALE = 4

KEY_ACTION: dict[int, Action] = {
    pygame.K_LEFT:  Action.MOVE_LEFT,
    pygame.K_RIGHT: Action.MOVE_RIGHT,
    pygame.K_DOWN:  Action.MOVE_DOWN,
    pygame.K_SPACE: Action.HARD_DROP,
    pygame.K_z:     Action.ROTATE_LEFT,
    pygame.K_x:     Action.ROTATE_RIGHT,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Play Tetris")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p = subparsers.add_parser("tetris", help="Play regular Tetris")
    p.add_argument("--width",  type=int, default=10)
    p.add_argument("--height", type=int, default=20)

    p = subparsers.add_parser("wm", help="Play Tetris via World Model")
    p.add_argument("--world-model", required=True, metavar="PATH", dest="world_model_path")
    p.add_argument("--grid-probe",  required=True, metavar="PATH", dest="grid_probe_path")

    return parser.parse_args()


def make_game(args):
    match args.mode:
        case "tetris":
            return Tetris(width=args.width, height=args.height)
        case "wm":
            return WMTetris(world_model_path=args.world_model_path, grid_probe_path=args.grid_probe_path)
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
