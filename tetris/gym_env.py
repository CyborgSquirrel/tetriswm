import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, OmegaConf
from gymnasium import spaces
from gymnasium.vector.utils import batch_space, iterate
from einops import rearrange

from tetris.tetris import (
    Action, GameState, Tetris, TILE_SET,
)


class OneHotSpace(spaces.Space[np.ndarray]):
    """Action space whose elements are one-hot vectors of length n.

    When batched (n_envs > 1), shape becomes (n_envs, n) and sample/contains
    operate on stacked one-hot vectors.
    """

    def __init__(
        self, n: int,
        *,
        n_envs: int | None = None,
        seed: int | np.random.Generator | None = None,
    ):
        shape = (n,) if n_envs is None else (n_envs, n)
        self.dtype = np.float32
        super().__init__(shape, self.dtype, seed)
        self.n = n
        self.n_envs = n_envs

    def sample(self) -> np.ndarray:
        if self.n_envs is None:
            v = np.zeros(self.n, dtype=self.dtype)
            v[self.np_random.integers(self.n)] = 1
        else:
            v = np.zeros((self.n_envs, self.n), dtype=self.dtype)
            v[np.arange(self.n_envs), self.np_random.integers(self.n, size=self.n_envs)] = 1
        return v

    def __repr__(self) -> str:
        return f"OneHotSpace({self.n})"

    def __eq__(self, other) -> bool:
        return isinstance(other, OneHotSpace) and self.n == other.n

    def contains(self, x) -> bool:
        x = np.asarray(x)
        if self.n_envs is None:
            return (
                x.shape == (self.n,)
                and np.sum(np.isclose(x, 1.0)) == 1
                and np.sum(np.isclose(x, 0.0)) == self.n - 1
            )
        else:
            return (
                x.shape == (self.n_envs, self.n)
                and np.all(np.sum(np.isclose(x, 1.0), axis=1) == 1)
                and np.all(np.sum(np.isclose(x, 0.0), axis=1) == self.n - 1)
            )


@batch_space.register(OneHotSpace)
def _batch_onehot_space(space: OneHotSpace, n: int = 1) -> OneHotSpace:
    return OneHotSpace(space.n, n_envs=n)


@iterate.register(OneHotSpace)
def _iterate_onehot_space(space: OneHotSpace, items: np.ndarray):
    return iter(items)


class TetrisEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    def __init__(self, tetris_cfg: dict | DictConfig, render_mode='rgb_array'):
        super().__init__()
        self._tetris_cfg = OmegaConf.create(tetris_cfg)
        self.render_mode = render_mode

        w, h = self._tetris_cfg.width, self._tetris_cfg.height
        self.action_space = OneHotSpace(len(Action))
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(0, 7, (w, h), dtype=np.uint8),
        })

        self._tetris: Tetris | None = None
        self._goal_grid: np.ndarray | None = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self._tetris = Tetris.from_config(self._tetris_cfg, rng=rng)
        self._goal_grid = None
        return self._get_obs(), self._get_info()

    def step(self, action):
        self._tetris.step(Action(int(np.argmax(action)) + 1))
        goal_reached = self._check_success()
        game_over = self._tetris.dynamics.state == GameState.OVER
        terminated = goal_reached
        truncated = game_over and not goal_reached
        reward = 1.0 if goal_reached else 0.0
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        return rearrange(self._tetris.render(), "h w c -> w h c")

    def _get_obs(self):
        return {'grid': self._tetris.grid.copy()}

    def _get_info(self):
        return {}

    def _check_success(self) -> bool:
        if self._goal_grid is None:
            return False
        return np.array_equal(self._tetris.grid != 0, self._goal_grid)

    def _set_state(
        self,
        placed_grid,
        mino_coords,
        mino_half,
        mino_origin,
        mino_rotations,
        mino_tile_idx,
    ):
        from tetris.tetris import Mino
        dyn = self._tetris.dynamics
        dyn.grid = np.asarray(placed_grid).astype(np.uint8).copy()

        mino = object.__new__(Mino)
        mino.coords = np.asarray(mino_coords).astype(np.int32)
        mino.half = np.asarray(mino_half).astype(np.int32)
        mino.origin = np.asarray(mino_origin).astype(np.int32)
        mino.rotations = int(mino_rotations)
        mino.tile = TILE_SET.oid_to_obj[int(mino_tile_idx)]
        dyn.current = mino

        dyn._update_parent()

    def _set_goal_state(self, goal_grid):
        self._goal_grid = (np.asarray(goal_grid) != 0)


gym.register(
    id='tetris/Tetris-v1',
    entry_point='tetris.gym_env:TetrisEnv',
)
