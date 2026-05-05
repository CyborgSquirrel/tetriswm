import enum
import random
import dataclasses as dc
import stable_pretraining as spt
from einops import rearrange, pack
import sys
import torch
from typing import Optional
import numpy as np
from hydra.utils import instantiate
from tetris import layout
from tetris import draw
import lewm.jepa
import lewm.module
import lewm.utils
# Checkpoints were saved when lewm submodules were importable as top-level; remap so pickle can find them.
sys.modules.setdefault("jepa", lewm.jepa)
sys.modules.setdefault("module", lewm.module)


class MinoRotation(enum.IntEnum):
    NONE  = enum.auto()
    LEFT  = enum.auto()
    RIGHT = enum.auto()


class MinoIntersection(enum.Flag):
    """OOB = Out Of Bounds"""

    GRID_TILE = enum.auto()
    OOB_LEFT = enum.auto()
    OOB_RIGHT = enum.auto()
    OOB_TOP = enum.auto()
    OOB_BOTTOM = enum.auto()

    NOTHING = 0
    COLLISION = (0
        | GRID_TILE
        | OOB_LEFT
        | OOB_RIGHT
        | OOB_BOTTOM
    )


@dc.dataclass(eq=False)
class TileDef:
    color: tuple[int, int, int]


class Mino:
    """Tetromino"""

    def __init__(
        self,
        *,
        coords: np.ndarray | list[tuple[int, int]],
        half: tuple[bool, bool],
        tile: TileDef,
    ):
        self.coords    = np.array(coords, dtype=np.int32)
        self.half      = np.array(half, dtype=np.int32)
        self.origin    = np.zeros(2, dtype=np.int32)
        self.rotations = 0
        self.tile      = tile

    def copy(self) -> 'Mino':
        new = object.__new__(Mino)
        new.coords    = self.coords.copy()
        new.half      = self.half.copy()
        new.origin    = self.origin.copy()
        new.rotations = self.rotations
        new.tile      = self.tile
        return new

    def bounding_rect(self) -> tuple[int, int, int, int]:
        min_xy = self.coords.min(axis=0)
        max_xy = self.coords.max(axis=0)
        x, y = min_xy
        w, h = max_xy - min_xy + 1
        return int(x), int(y), int(w), int(h)

    def intersect_grid(self, grid: np.ndarray) -> MinoIntersection:
        ret = MinoIntersection.NOTHING
        w,h = grid.shape
        for x, y in self.coords:
            oob = False
            if not (0 <= x):
                ret |= MinoIntersection.OOB_LEFT
                oob = True
            if not (x < w):
                ret |= MinoIntersection.OOB_RIGHT
                oob = True
            if not (0 <= y):
                ret |= MinoIntersection.OOB_TOP
                oob = True
            if not (y < h):
                ret |= MinoIntersection.OOB_BOTTOM
                oob = True

            if not oob and grid[x,y] != 0:
                ret |= MinoIntersection.GRID_TILE
        return ret

    def center(self, grid_w: int) -> 'Mino':
        bx, by, bw, bh = self.bounding_rect()
        dx = (grid_w - bw) // 2 - bx
        dy = -by
        return self.translate((dx, dy))

    def rotate(self, direction: MinoRotation) -> 'Mino':
        match direction:
            case MinoRotation.NONE:
                return self
            case MinoRotation.LEFT:
                m = np.array([1, -1], dtype=np.int32)
                new_rotations = (self.rotations - 1) % 4
            case MinoRotation.RIGHT:
                m = np.array([-1, 1], dtype=np.int32)
                new_rotations = (self.rotations + 1) % 4
            case _:
                raise ValueError(f"Invalid rotation direction: {direction}")

        tiles = self.coords - self.origin
        tiles = tiles * 2 - self.half
        tiles = tiles[:, (1,0)]   # swap x, y
        tiles = tiles * m + 1
        tiles = tiles // 2
        tiles = tiles + self.origin

        new = self.copy()
        new.coords    = np.array(tiles, dtype=np.int32)
        new.rotations = new_rotations
        return new

    def translate(self, delta: tuple[int, int]) -> 'Mino':
        d = np.array(delta, dtype=np.int32)
        new = self.copy()
        new.coords += d
        new.origin += d
        return new


class ObjectSet:
    def __init__(self, objs=None):
        if objs is None:
            objs = []

        self.oid_to_obj = []
        self.obj_to_oid = {}

        for obj in objs:
            self.add(obj)

    def add(self, obj):
        oid = len(self.oid_to_obj)
        self.oid_to_obj.append(obj)
        self.obj_to_oid[obj] = oid
        return obj


TILE_SET = ObjectSet()

TILE_EMPTY = TILE_SET.add(TileDef( color=(0  , 0  , 0  ) ))
TILE_I     = TILE_SET.add(TileDef( color=(0  , 240, 240) ))
TILE_L     = TILE_SET.add(TileDef( color=(240, 160, 0  ) ))
TILE_J     = TILE_SET.add(TileDef( color=(0  , 0  , 240) ))
TILE_O     = TILE_SET.add(TileDef( color=(240, 240, 0  ) ))
TILE_Z     = TILE_SET.add(TileDef( color=(240, 0  , 0  ) ))
TILE_S     = TILE_SET.add(TileDef( color=(0  , 240, 0  ) ))
TILE_T     = TILE_SET.add(TileDef( color=(160, 0  , 240) ))

class Minoes(enum.Enum):
    I = Mino(coords=[(-1, 0), ( 0, 0), ( 1, 0), ( 2, 0)], half=(1, 1), tile=TILE_I)
    L = Mino(coords=[(-1, 0), ( 0, 0), ( 1, 0), ( 1,-1)], half=(0, 0), tile=TILE_L)
    J = Mino(coords=[(-1,-1), (-1, 0), ( 0, 0), ( 1, 0)], half=(0, 0), tile=TILE_J)
    O = Mino(coords=[( 0, 0), ( 1, 0), ( 0, 1), ( 1, 1)], half=(1, 1), tile=TILE_O)
    Z = Mino(coords=[(-1,-1), ( 0,-1), ( 0, 0), ( 1, 0)], half=(0, 0), tile=TILE_Z)
    S = Mino(coords=[( 1,-1), ( 0,-1), ( 0, 0), (-1, 0)], half=(0, 0), tile=TILE_S)
    T = Mino(coords=[(-1, 0), ( 0, 0), ( 1, 0), ( 0,-1)], half=(0, 0), tile=TILE_T)


class Action(enum.IntEnum):
    # NOTE: enum.auto() starts from 1
    NOTHING      = enum.auto()
    MOVE_LEFT    = enum.auto()
    MOVE_RIGHT   = enum.auto()
    MOVE_DOWN    = enum.auto()
    HARD_DROP    = enum.auto()
    ROTATE_LEFT  = enum.auto()
    ROTATE_RIGHT = enum.auto()


class GameState(enum.IntEnum):
    RUNNING = enum.auto()
    OVER = enum.auto()


def rng_derive_seed(rng: np.random.Generator | random.Random):
    if isinstance(rng, np.random.Generator):
        b = rng.bytes(4)
    elif isinstance(rng, random.Random):
        b = rng.randbytes(4)
    else:
        raise TypeError()
    seed = int.from_bytes(b)
    return seed


class BagGenerator:
    def __init__(self, minoes, *, rng: np.random.Generator):
        self.minoes = [Minoes[m].value if isinstance(m, str) else m for m in minoes]
        if len(self.minoes) <= 0:
            raise ValueError()

        self.bag = []
        self.rng = rng

    def generate(self):
        if len(self.bag) <= 0:
            for mino in self.minoes:
                self.bag.append(mino)
            self.rng.shuffle(self.bag)
        return self.bag.pop()


class GtDynamics:
    def __init__(
        self,
        *,
        mino_generator = None,
        rng = None,
    ):
        rng = rng or np.random.default_rng()

        self.current: Mino | None = None
        self.state = GameState.RUNNING
        self.cleared_rows = 0

        self.mino_generator = (
            mino_generator
            or BagGenerator([mino.value for mino in Minoes], rng=np.random.default_rng(rng_derive_seed(rng)))
        )

    def _init(self, tetris):
        self.tetris = tetris

        self.grid = np.full(
            (tetris.width, tetris.height),
            self.tetris.tile_set.obj_to_oid[TILE_EMPTY],
            dtype=np.uint8,
        )

        self._spawn_mino()

        self._update_parent()

    def _update_parent(self):
        grid = self.grid.copy()
        mino_mask = np.full((self.tetris.width, self.tetris.height), False)
        if self.current is not None:
            for x, y in self.current.coords:
                if 0 <= x < self.tetris.width and 0 <= y < self.tetris.height:
                    grid[x, y] = self.tetris.tile_set.obj_to_oid[self.current.tile]
                    mino_mask[x, y] = True
        self.tetris.grid = grid
        self.tetris.mino_mask = mino_mask

    def _spawn_mino(self):
        # Spawn mino if missing
        if self.current is None:
            mino = self.mino_generator.generate()
            mino = mino.center(self.tetris.width)
            self.current = mino

    def _rotate_mino(self, rotation: MinoRotation):
        mino = self.current.copy()

        rotated_mino = mino.copy().rotate(rotation)
        bx, by, bw, bh = rotated_mino.bounding_rect()

        def dx_gen_fn():
            yield 0
            for abs_dx in range(1, (bw+1)//2):
                yield abs_dx
                yield abs_dx * -1

        for dx in dx_gen_fn():
            new_mino = rotated_mino.copy().translate((dx, 0))
            if new_mino.intersect_grid(self.grid) & MinoIntersection.COLLISION:
                continue
            self.current = new_mino
            return

    def _step(self, action: Action):
        if self.state == GameState.OVER:
            return

        # Handle action
        match action:
            case Action.NOTHING:
                pass
            case Action.ROTATE_LEFT:
                self._rotate_mino(MinoRotation.LEFT)
            case Action.ROTATE_RIGHT:
                self._rotate_mino(MinoRotation.RIGHT)
            case Action.MOVE_LEFT:
                mino = self.current.copy()
                mino = mino.translate((-1, 0))
                if not (mino.intersect_grid(self.grid) & MinoIntersection.COLLISION):
                    self.current = mino
            case Action.MOVE_RIGHT:
                mino = self.current.copy()
                mino = mino.translate((1, 0))
                if not (mino.intersect_grid(self.grid) & MinoIntersection.COLLISION):
                    self.current = mino
            case Action.MOVE_DOWN:
                pass  # let gravity handle it
            case Action.HARD_DROP:
                mino0 = self.current.copy()
                while True:
                    mino0, mino1 = mino0.translate((0, 1)), mino0
                    if mino0.intersect_grid(self.grid) & MinoIntersection.COLLISION:
                        break
                self.current = mino1

        # Apply gravity
        mino = self.current.copy().translate((0, 1))
        if not (mino.intersect_grid(self.grid) & MinoIntersection.COLLISION):
            self.current = mino
        else:
            if self.current.intersect_grid(self.grid) != MinoIntersection.NOTHING:
                self.state = GameState.OVER
                return

            # Place mino in grid
            self.current, mino = None, self.current
            for x,y in mino.coords:
                self.grid[x,y] = self.tetris.tile_set.obj_to_oid[mino.tile]

        # Clear rows
        dy = 0
        for y in range(self.tetris.height-1, 0, -1):
            empty_tiles_num = np.sum(self.grid[:,y] == 0)
            if empty_tiles_num == 0:
                self.cleared_rows += 1
                dy += 1
            else:
                if dy > 0:
                    self.grid[:,y+dy] = self.grid[:,y]
        for y in range(dy-1, 0, -1):
            self.grid[:,y] = 0

        self._spawn_mino()

    def step(self, action: Action):
        self._step(action)
        self._update_parent()


class WmDynamics:
    def __init__(
        self,
        *,
        world_model_path: str,
        grid_probe_path: str,
        gt_tetris: 'Tetris',
    ):
        self.world_model = torch.load(world_model_path, weights_only=False, map_location="cpu").eval().requires_grad_(False)
        self.grid_probe = torch.load(grid_probe_path, weights_only=False, map_location="cpu").eval().requires_grad_(False)

        self.history_size = 3

        self.gt_tetris = gt_tetris

    def _reanchor_to_gt(self):
        info = {}

        pixels = torch.from_numpy(self.gt_tetris.render())
        pixels = rearrange(pixels, "w h c -> 1 1 c h w")
        info["pixels"] = pixels

        transforms = [lewm.utils.get_img_preprocessor(source='pixels', target='pixels', img_size=224)]
        transform = spt.data.transforms.Compose(*transforms)
        info = transform(info)

        self.world_model.encode(info)
        self.state_emb_hist = info["emb"]
        self.act_emb_hist = torch.empty((1, 0, 192))

    def _init(self, tetris):
        self.tetris = tetris
        self._reanchor_to_gt()
        self._update_parent()

    def _update_parent(self):
        grid = self.grid_probe(self.state_emb_hist[:, (-1,), :])
        grid = grid.sigmoid() > 0.5
        grid = grid.numpy()
        grid = rearrange(grid, "1 1 (w h) -> w h", w=self.tetris.width, h=self.tetris.height)
        grid = grid.astype(np.uint8)
        self.tetris.grid = grid

    def embed_action(self, action: Optional[Action]):
        action_enc = torch.zeros(len(Action), dtype=torch.float32)
        if action is not None:
            action_enc[action-1] = 1
        action_enc = rearrange(action_enc, "d -> 1 1 d")
        action_emb = self.world_model.action_encoder(action_enc)
        return action_emb

    def step(self, action: Action):
        HS = self.history_size

        act_emb = self.embed_action(action)
        self.act_emb_hist = pack([self.act_emb_hist, act_emb], "b * d")[0][:, -HS:, :]

        state_emb_hist = self.state_emb_hist
        pred_state_emb = self.world_model.predict(state_emb_hist, self.act_emb_hist)[:, -1:, :]
        state_emb_hist = pack([state_emb_hist, pred_state_emb], "b * d")[0][:, -HS:, :]
        self.state_emb_hist = state_emb_hist

        self._update_parent()


class Tetris:
    def __init__(
        self,
        *,
        width: int = 10,
        height: int = 20,
        dynamics = None,
    ):
        self.width = width
        self.height = height

        self.tile_set = TILE_SET
        self.tile_color_lut = np.array([ tile.color for tile in TILE_SET.oid_to_obj ])

        self.dynamics = dynamics or GtDynamics()
        self.dynamics._init(self)

        with layout.Frame(224, 224):
            with layout.Center():
                self.lay_grid = layout.Rect(width*10, height*10)

    @classmethod
    def from_config(cls, cfg, *, rng=None):
        mino_generator = instantiate(cfg.mino_generator, rng=rng or np.random.default_rng())
        dynamics = GtDynamics(mino_generator=mino_generator)
        return cls(width=cfg.width, height=cfg.height, dynamics=dynamics)

    def step(self, action: Action):
        self.dynamics.step(action)

    def render(self):
        COLOR_BG = (128, 128, 128)

        canvas = np.zeros((224, 224, 3), dtype=np.uint8)
        canvas[:,:] = COLOR_BG

        grid = self.tile_color_lut[self.grid]
        grid = draw.scale(grid, 10)

        draw.blit(canvas, grid, self.lay_grid.x, self.lay_grid.y)

        return canvas
