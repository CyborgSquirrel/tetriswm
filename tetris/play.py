import os
os.environ.setdefault("SDL_VIDEODRIVER", "wayland")

import numpy as np
import pygame

from tetris.tetris import Action, GameState, Tetris

SCALE = 4

COLOR_EMPTY = (30,  30,  30)
COLOR_TILE  = (180, 180, 180)
COLOR_MINO  = (100, 200, 255)

KEY_ACTION: dict[int, Action] = {
    pygame.K_LEFT:  Action.MOVE_LEFT,
    pygame.K_RIGHT: Action.MOVE_RIGHT,
    pygame.K_DOWN:  Action.MOVE_DOWN,
    pygame.K_SPACE: Action.HARD_DROP,
    pygame.K_z:     Action.ROTATE_LEFT,
    pygame.K_x:     Action.ROTATE_RIGHT,
}


# def render(tetris: Tetris) -> np.ndarray:
#     grid = tetris.grid
#     w,h = grid.shape
#     img = np.where(grid[..., None], COLOR_TILE, COLOR_EMPTY).astype(np.uint8)

#     if tetris.current is not None:
#         for x, y in tetris.current.coords:
#             if 0 <= x < w and 0 <= y < h:
#                 img[x,y] = COLOR_MINO

#     return img


def main():
    pygame.init()

    game = Tetris()
    surf = pygame.display.set_mode((224*SCALE, 224*SCALE))

    pygame.display.set_caption("tetris")

    while game.state == GameState.RUNNING:
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
