import numpy as np
from omegaconf import OmegaConf
from typing import Any
from collections import deque as Deque
from tetris.tetris import Action, Mino, MinoIntersection, MinoRotation, GameState


class MixPolicy:
    def __init__(self, policies):
        self.policies = policies

    def _init(self, tetris, rng: np.random.Generator):
        self.rng = rng
        weights  = np.array([a["w"] for a in self.policies])
        policies = [a["p"] for a in self.policies]
        policy = rng.choice(policies, p=weights/np.sum(weights))
        self.policy = policy
        self.policy._init(tetris, rng)

    def step(self):
        return self.policy.step()


class RandomPolicy:
    ACTIONS = list(Action)

    def __init__(self):
        pass

    def _init(self, tetris, rng):
        self.tetris = tetris
        self.rng = rng

    def step(self) -> Action:
        action = self.rng.choice(self.ACTIONS)
        return action


class ElTetrisPolicy:
    """
    Python port of the El-Tetris heuristic AI[0] (originally implemented in Java).

    Each call to get_actions() evaluates all (rotation × column) placements for
    the current piece, scores each resulting board state with six weighted features,
    and returns a sequence of Actions that place the piece optimally:
        [ROTATE_RIGHT × n, MOVE_LEFT/RIGHT × m, HARD_DROP]

    Because HARD_DROP always drops to the same final position regardless of the
    piece's current row, intermediate gravity applied during rotation/shift steps
    does not affect the outcome.

    [0]: https://github.com/YuhanXiaoJY/Implementation-of-El-Tetris/tree/master
    """

    W_LANDING_HEIGHT  = -45
    W_ERODED_CELLS    =  34
    W_ROW_TRANSITIONS = -32
    W_COL_TRANSITIONS = -98
    W_BURIED_HOLES    = -79
    W_WELLS           = -34

    def __init__(self):
        pass

    def _init(self, tetris, rng):
        self.tetris = tetris
        self.queue = Deque()

    def step(self) -> Action:
        if len(self.queue) <= 0:
            self.queue.extend(self.get_actions())
        return self.queue.popleft()

    def get_actions(self) -> list[Action]:
        """Return the action sequence that places the current piece optimally."""
        dynamics = self.tetris.dynamics
        if dynamics.current is None or dynamics.state == GameState.OVER:
            return []

        grid = dynamics.grid.copy()
        curr_grid = np.empty_like(grid)
        width, height = grid.shape
        current = dynamics.current.copy()

        best_score = float('-inf')
        best_rot = 0
        best_dx = 0

        for rot in range(4):
            rotated = current.copy()
            for _ in range(rot):
                rotated = rotated.rotate(MinoRotation.RIGHT)

            xs = rotated.coords[:, 0]
            min_x = int(xs.min())
            max_x = int(xs.max())

            for dx in range(-min_x, width - max_x):
                shifted = rotated.copy().translate((dx, 0))

                # Skip if horizontally out of bounds or already colliding with tiles
                ix = shifted.intersect_grid(grid)
                if ix & (MinoIntersection.OOB_LEFT | MinoIntersection.OOB_RIGHT):
                    continue
                if ix & MinoIntersection.GRID_TILE:
                    continue

                dropped = self._drop(shifted, grid)
                if dropped is None:
                    continue

                np.copyto(curr_grid, grid)
                tile_id = self.tetris.tile_set.obj_to_oid[dropped.tile]
                for cx, cy in dropped.coords:
                    if 0 <= cx < width and 0 <= cy < height:
                        curr_grid[cx, cy] = tile_id

                score = self._score(curr_grid, dropped, width, height)
                if score > best_score:
                    best_score = score
                    best_rot = rot
                    best_dx = dx

        actions: list[Action] = []
        for _ in range(best_rot):
            actions.append(Action.ROTATE_RIGHT)
        if best_dx < 0:
            for _ in range(-best_dx):
                actions.append(Action.MOVE_LEFT)
        elif best_dx > 0:
            for _ in range(best_dx):
                actions.append(Action.MOVE_RIGHT)
        actions.append(Action.HARD_DROP)
        return actions

    def _drop(self, mino: Mino, grid: np.ndarray) -> Mino | None:
        """Drop mino straight down until it lands. Returns the resting mino, or
        None if the starting position is already a hard collision."""
        mino0 = mino.copy()
        mino1 = mino.copy()
        while True:
            mino0.translate((0, 1))
            if mino0.intersect_grid(grid) & MinoIntersection.COLLISION:
                if mino1.intersect_grid(grid) & MinoIntersection.COLLISION:
                    return None
                return mino1
            mino1.translate((0, 1))

    def _clear_rows(self, grid: np.ndarray, width: int, height: int) -> tuple[np.ndarray, int]:
        new_grid = grid.copy()
        cleared = 0
        dy = 0
        for y in range(height - 1, -1, -1):
            if np.all(new_grid[:, y] != 0):
                cleared += 1
                dy += 1
            elif dy > 0:
                new_grid[:, y + dy] = new_grid[:, y]
        for y in range(dy):
            new_grid[:, y] = 0
        return new_grid, cleared

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(self, grid_with_piece: np.ndarray, placed: Mino,
               width: int, height: int) -> int:
        cleared_grid, rows_cleared = self._clear_rows(grid_with_piece, width, height)

        lh  = self._landing_height(placed, height)
        epc = self._eroded_piece_cells(placed, grid_with_piece, rows_cleared, width, height)
        rt  = self._row_transitions(cleared_grid, width, height)
        ct  = self._col_transitions(cleared_grid, width, height)
        bh  = self._buried_holes(cleared_grid, width, height)
        ws  = self._wells(cleared_grid, width, height)

        return (self.W_LANDING_HEIGHT  * lh
              + self.W_ERODED_CELLS    * epc
              + self.W_ROW_TRANSITIONS * rt
              + self.W_COL_TRANSITIONS * ct
              + self.W_BURIED_HOLES    * bh
              + self.W_WELLS           * ws)

    def _landing_height(self, placed: Mino, height: int) -> int:
        # Height from the floor to the topmost cell of the placed piece.
        # In Python coords (y=0 top), "top of piece" = smallest y value.
        min_y = int(placed.coords[:, 1].min())
        return height - 1 - min_y

    def _eroded_piece_cells(self, placed: Mino, grid_with_piece: np.ndarray,
                             rows_cleared: int, width: int, height: int) -> int:
        if rows_cleared == 0:
            return 0
        piece_cells_cleared = sum(
            1 for cx, cy in placed.coords
            if 0 <= cx < width and 0 <= cy < height
            and np.all(grid_with_piece[:, cy] != 0)
        )
        return rows_cleared * piece_cells_cleared

    def _row_transitions(self, grid: np.ndarray, width: int, height: int) -> int:
        transitions = 0
        for y in range(height):
            prev = 1  # left border treated as filled
            for x in range(width):
                cur = 1 if grid[x, y] != 0 else 0
                if cur != prev:
                    transitions += 1
                prev = cur
            if prev == 0:  # right border treated as filled
                transitions += 1
        return transitions

    def _col_transitions(self, grid: np.ndarray, width: int, height: int) -> int:
        transitions = 0
        for x in range(width):
            prev = 0  # top is open sky (empty)
            for y in range(height):
                cur = 1 if grid[x, y] != 0 else 0
                if cur != prev:
                    transitions += 1
                prev = cur
            if prev == 0:  # floor treated as filled
                transitions += 1
        return transitions

    def _buried_holes(self, grid: np.ndarray, width: int, height: int) -> int:
        holes = 0
        for x in range(width):
            found_filled = False
            for y in range(height):
                if grid[x, y] != 0:
                    found_filled = True
                elif found_filled:
                    holes += 1
        return holes

    def _wells(self, grid: np.ndarray, width: int, height: int) -> int:
        # Scan top-to-bottom. Each consecutive empty cell flanked by filled
        # neighbours on both sides adds (depth) to the total, yielding a
        # quadratic penalty for deep wells.
        wells = 0
        for x in range(width):
            depth = 0
            for y in range(height):
                left  = (x == 0)          or (grid[x - 1, y] != 0)
                right = (x == width - 1)  or (grid[x + 1, y] != 0)
                if grid[x, y] == 0 and left and right:
                    depth += 1
                    wells += depth
                else:
                    depth = 0
        return wells
