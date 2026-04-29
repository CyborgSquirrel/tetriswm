from __future__ import annotations
from contextvars import ContextVar

_current: ContextVar[Container | None] = ContextVar('_current', default=None)


class Node:
    def __init__(self):
        parent = _current.get()
        if parent is not None:
            parent._children.append(self)

    def _measure(self): pass

    def _arrange(self, x: int, y: int, aw: int, ah: int):
        self.x, self.y = x, y


class Rect(Node):
    def __init__(self, w: int, h: int):
        self.x = self.y = 0
        self.w, self.h = w, h
        super().__init__()


class Container(Node):
    def __init__(self, w: int = 0, h: int = 0):
        self.x = self.y = 0
        self.w, self.h = w, h
        self._children: list[Node] = []
        super().__init__()

    def __enter__(self):
        self._token = _current.set(self)
        return self

    def __exit__(self, *_):
        _current.reset(self._token)
        self._measure()
        if _current.get() is None:
            self._arrange(0, 0, self.w, self.h)
        return False


class Frame(Container):
    """Fixed-size root. Passes its full size down to each child."""
    def __init__(self, w: int, h: int):
        super().__init__(w, h)

    def _arrange(self, x, y, aw, ah):
        self.x, self.y = x, y
        for child in self._children:
            child._arrange(x, y, aw, ah)


class HStack(Container):
    def __init__(self, gap: int = 0):
        self.gap = gap
        super().__init__()

    def _measure(self):
        if not self._children:
            return
        self.w = sum(c.w for c in self._children) + self.gap * (len(self._children) - 1)
        self.h = max(c.h for c in self._children)

    def _arrange(self, x, y, aw, ah):
        self.x, self.y = x, y
        cx = x
        for child in self._children:
            child._arrange(cx, y, child.w, ah)
            cx += child.w + self.gap


class VStack(Container):
    def __init__(self, gap: int = 0):
        self.gap = gap
        super().__init__()

    def _measure(self):
        if not self._children:
            return
        self.h = sum(c.h for c in self._children) + self.gap * (len(self._children) - 1)
        self.w = max(c.w for c in self._children)

    def _arrange(self, x, y, aw, ah):
        self.x, self.y = x, y
        cy = y
        for child in self._children:
            child._arrange(x, cy, aw, child.h)
            cy += child.h + self.gap


class Center(Container):
    """Centers children within the space allocated by the parent."""
    def _measure(self):
        if not self._children:
            return
        self.w = max(c.w for c in self._children)
        self.h = max(c.h for c in self._children)

    def _arrange(self, x, y, aw, ah):
        self.x, self.y = x, y
        self.w, self.h = aw, ah
        for child in self._children:
            cx = x + (aw - child.w) // 2
            cy = y + (ah - child.h) // 2
            child._arrange(cx, cy, child.w, child.h)
