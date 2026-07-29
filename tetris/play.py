from tetris.tetris import Action, GameState, Tetris, WmDynamics
import argparse
import os

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
    # pygame.K_RETURN: Action.NOTHING,
}


class AppendGame(argparse.Action):
    def __init__(self, option_strings, dest, game_type, **kwargs):
        self.game_type = game_type
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        games = getattr(namespace, self.dest) or []
        games.append({"type": self.game_type})
        setattr(namespace, self.dest, games)


class SetGameParam(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        games = getattr(namespace, "games")
        if not games:
            parser.error(f"{option_string} must follow --gt or --wm")
        games[-1][self.dest] = values


def hex_color(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    try:
        return pygame.Color(f"#{s}")[:3]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected hex color, got {s!r}")


def parse_args():
    parser = argparse.ArgumentParser(description="Play Tetris")
    parser.add_argument("--scale",        type=int, default=4)
    parser.add_argument("--config",       metavar="PATH",
                        help="Dataset Hydra config (e.g. config/tetris/overfit.yaml)")
    parser.add_argument("--gt",           action=AppendGame, game_type="gt", dest="games")
    parser.add_argument("--wm",           action=AppendGame, game_type="wm", dest="games")
    parser.add_argument("--world-model",  action=SetGameParam, metavar="PATH")
    parser.add_argument("--grid-probe",   action=SetGameParam, metavar="PATH")
    parser.add_argument("--queue-probe",  action=SetGameParam, metavar="PATH")
    parser.add_argument("--reanchor-every", type=int, default=None, metavar="N",
                        help="Reanchor WM games to ground truth every N steps")
    parser.add_argument("--gap",          type=int, default=0, metavar="PX",
                        help="Gap in pixels between game views")
    parser.add_argument("--gap-color",    type=hex_color, default=(0, 0, 0), metavar="HEX",
                        help="Background color for the gap between game views (e.g. ff8800)")
    parser.set_defaults(games=None)
    return parser.parse_args()


def make_games(args) -> list[Tetris]:
    cfg = OmegaConf.load(args.config).tetris if args.config else None

    gt_tetris = Tetris.from_config(cfg) if cfg else Tetris()

    game_specs = args.games or [{"type": "gt"}]
    games = []
    for spec in game_specs:
        if spec["type"] == "gt":
            games.append(gt_tetris)
        elif spec["type"] == "wm":
            dynamics = WmDynamics(
                world_model_path=spec["world_model"],
                grid_probe_path=spec["grid_probe"],
                queue_probe_path=spec.get("queue_probe"),
                gt_tetris=gt_tetris,
            )
            games.append(Tetris.from_config(cfg, dynamics=dynamics))

    return games


def main():
    args = parse_args()
    pygame.init()

    games = make_games(args)
    scale = args.scale
    gap = args.gap
    step_count = 0

    game_size = 224 * scale
    win_width = game_size * len(games) + gap * (len(games) - 1)
    surf = pygame.display.set_mode((win_width, game_size))
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
                    step_count = 0
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
                    step_count += 1
                    if args.reanchor_every and step_count % args.reanchor_every == 0:
                        for game in games:
                            if isinstance(game.dynamics, WmDynamics):
                                game.dynamics._reanchor_to_gt()
                                game.dynamics._update_parent()

        surf.fill(args.gap_color)
        x = 0
        for game in games:
            tile_surf = pygame.surfarray.make_surface(game.render())
            scaled = pygame.transform.scale(tile_surf, (game_size, game_size))
            surf.blit(scaled, (x, 0))
            x += game_size + gap
        pygame.display.flip()


if __name__ == "__main__":
    main()
