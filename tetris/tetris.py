import enum
import numpy as np
from tetris import layout
from tetris import draw


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


class Mino:
    """Tetromino"""

    def __init__(
        self,
        coords: np.ndarray | list[tuple[int, int]],
        half: tuple[bool, bool],
        rotations: int = 0,
    ):
        self.coords    = np.array(coords, dtype=np.int32)
        self.half      = np.array(half, dtype=np.int32)
        self.origin    = np.zeros(2, dtype=np.int32)
        self.rotations = rotations

    def copy(self) -> 'Mino':
        new = object.__new__(Mino)
        new.coords    = self.coords.copy()
        new.half      = self.half.copy()
        new.origin    = self.origin.copy()
        new.rotations = self.rotations
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

    def spawn(self, grid_w: int) -> 'Mino':
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


class MINOES:
    I = Mino([(-1, 0), ( 0, 0), ( 1, 0), ( 2, 0)], half=(1, 1))
    L = Mino([(-1, 0), ( 0, 0), ( 1, 0), ( 1,-1)], half=(0, 0))
    J = Mino([(-1,-1), (-1, 0), ( 0, 0), ( 1, 0)], half=(0, 0))
    O = Mino([( 0, 0), ( 1, 0), ( 0, 1), ( 1, 1)], half=(1, 1))
    Z = Mino([(-1,-1), ( 0,-1), ( 0, 0), ( 1, 0)], half=(0, 0))
    S = Mino([( 1,-1), ( 0,-1), ( 0, 0), (-1, 0)], half=(0, 0))
    T = Mino([(-1, 0), ( 0, 0), ( 1, 0), ( 0,-1)], half=(0, 0))


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


class Tetris:
    def __init__(
        self,
        width: int = 10,
        height: int = 20,
    ):
        self.width   = width
        self.height  = height
        self.grid    = np.zeros((width,height), dtype=np.uint8)
        self.current: Mino | None = None
        self.state = GameState.RUNNING
        self.cleared_rows = 0
        self._spawn_mino()

        with layout.Frame(224, 224):
            with layout.Center():
                self.lay_grid = layout.Rect(width*10, height*10)

    def _spawn_mino(self):
        # Spawn mino if missing
        if self.current is None:
            # TODO: pick from many minoes
            self.current = MINOES.I.spawn(self.width)

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
            new_mino = rotated_mino.copy().translate((dx,0))
            if new_mino.intersect_grid(self.grid) & MinoIntersection.COLLISION:
                continue
            self.current = new_mino
            return

    def step(self, action: Action):
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
                mino = mino.translate((-1,0))
                if not (mino.intersect_grid(self.grid) & MinoIntersection.COLLISION):
                    self.current = mino
            case Action.MOVE_RIGHT:
                mino = self.current.copy()
                mino = mino.translate((1,0))
                if not (mino.intersect_grid(self.grid) & MinoIntersection.COLLISION):
                    self.current = mino
            case Action.MOVE_DOWN:
                pass  # let gravity handle it
            case Action.HARD_DROP:
                mino0 = self.current.copy()
                while True:
                    mino0, mino1 = mino0.translate((0,1)), mino0
                    if mino0.intersect_grid(self.grid) & MinoIntersection.COLLISION:
                        break
                self.current = mino1

        # Apply gravity
        mino = self.current.copy().translate((0,1))
        if not (mino.intersect_grid(self.grid) & MinoIntersection.COLLISION):
            self.current = mino
        else:
            if self.current.intersect_grid(self.grid) != MinoIntersection.NOTHING:
                self.state = GameState.OVER
                return

            # Place mino in grid
            for x,y in self.current.coords:
                self.grid[x,y] = 1
            self.current = None

        # Clear rows
        dy = 0
        for y in range(self.height-1, 0, -1):
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

    def render(self):
        COLOR_BG = (128, 128, 128)
        COLOR_GRID_BG = (0, 0, 0)
        COLOR_TILE = (180, 180, 180)
        COLOR_MINO = (100, 200, 255)

        canvas = np.zeros((224, 224, 3), dtype=np.uint8)
        canvas[:,:] = COLOR_BG

        grid = np.where(self.grid[..., None], COLOR_TILE, COLOR_GRID_BG).astype(np.uint8)
        if self.current is not None:
            for x, y in self.current.coords:
                if 0 <= x < self.width and 0 <= y < self.height:
                    grid[x,y] = COLOR_MINO
        grid = draw.scale(grid, 10)

        draw.blit(canvas, grid, self.lay_grid.x, self.lay_grid.y)

        return canvas
