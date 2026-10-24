"""Place pieces on Letter pages: rotate each to its tightest box, then pack."""

import copy
import math
import numpy as np

from . import params as P
from . import geom2d as g2


def usable():
    """Usable box (x0, y0, w, h) in page mm, y up."""
    x0 = P.MARGIN
    y0 = P.MARGIN + P.FOOTER_H
    return x0, y0, P.PAGE_W - 2 * P.MARGIN, P.PAGE_H - 2 * P.MARGIN - P.HEADER_H - P.FOOTER_H


def orient(piece, W, H):
    pts = piece.points()
    opts = []
    for deg in np.arange(0.0, 180.0, 1.0):
        r = g2.rotate(pts, math.radians(deg))
        w, h = np.ptp(r[:, 0]), np.ptp(r[:, 1])
        for extra, (ww, hh) in ((0, (w, h)), (90, (h, w))):
            if ww <= W and hh <= H:
                opts.append((ww * hh, (deg + extra) % 180, ww, hh))
    if not opts:
        raise RuntimeError(f"{piece.name} does not fit on a page")
    least = min(o[0] for o in opts)
    # Among near-tightest boxes prefer the least rotation (keeps text level).
    near = [o for o in opts if o[0] <= least * 1.01]
    _, deg, w, h = min(near, key=lambda o: (min(o[1], 180 - o[1]), -o[3]))
    return deg, w, h


class MaxRects:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.free = [(0.0, 0.0, W, H)]

    def insert(self, w, h):
        """Place a w x h box (or rotated 90 degrees). Returns (x, y, turned) or None."""
        best = None
        for turned, (bw, bh) in ((False, (w, h)), (True, (h, w))):
            for fx, fy, fw, fh in self.free:
                if bw <= fw + 1e-9 and bh <= fh + 1e-9:
                    key = (min(fw - bw, fh - bh), max(fw - bw, fh - bh), fy, fx, turned)
                    if best is None or key < best[0]:
                        best = (key, fx, fy, bw, bh, turned)
        if best is None:
            return None
        _, x, y, bw, bh, turned = best
        self._split(x, y, bw, bh)
        return x, y, turned

    def _split(self, x, y, w, h):
        out = []
        for fx, fy, fw, fh in self.free:
            if x >= fx + fw or x + w <= fx or y >= fy + fh or y + h <= fy:
                out.append((fx, fy, fw, fh))
                continue
            if x > fx:
                out.append((fx, fy, x - fx, fh))
            if x + w < fx + fw:
                out.append((x + w, fy, fx + fw - x - w, fh))
            if y > fy:
                out.append((fx, fy, fw, y - fy))
            if y + h < fy + fh:
                out.append((fx, y + h, fw, fy + fh - y - h))
        self.free = [r for i, r in enumerate(out) if not any(
            j != i and _contains(o, r) for j, o in enumerate(out))]


def _contains(a, b):
    return (b[0] >= a[0] - 1e-9 and b[1] >= a[1] - 1e-9 and
            b[0] + b[2] <= a[0] + a[2] + 1e-9 and b[1] + b[3] <= a[1] + a[3] + 1e-9)


def _plan(todo, W, H, gap):
    pages = []
    while todo:
        bins = MaxRects(W, H)
        placed, left = [], []
        for t in todo:
            pos = bins.insert(t[2] + gap, t[3] + gap)
            if pos is None:
                left.append(t)
            else:
                placed.append((t, pos))
        if not placed:
            raise RuntimeError("packing made no progress")
        pages.append(placed)
        todo = left
    return pages


def pack(pieces, groups):
    """Pack every piece (all copies) onto as few pages as possible. groups:
    list of (title, predicate) used to name each page by what is on it.
    Returns pages: dict(title, items), items moved into page coordinates."""
    x0, y0, W, H = usable()
    gap = P.PIECE_GAP
    todo = []
    for pc in pieces:
        deg, w, h = orient(pc, W - gap, H - gap)
        for k in range(pc.count):
            todo.append((pc, deg, w, h, k))
    orders = [lambda t: -(t[2] * t[3]), lambda t: -t[3], lambda t: -max(t[2], t[3]),
              lambda t: -t[2], lambda t: (t[0].kind, -(t[2] * t[3]))]
    plans = [_plan(sorted(todo, key=lambda t, o=o: (o(t), t[0].order, t[4])), W, H, gap) for o in orders]
    plan = min(plans, key=len)
    pages = []
    for placed in plan:
        items = []
        for (pc, deg, w, h, k), (px, py, turned) in placed:
            cp = copy.deepcopy(pc)
            a = math.radians(deg + (90.0 if turned else 0.0))
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            cp.transform(R, np.zeros(2))
            lo = cp.points().min(axis=0)
            cp.transform(np.eye(2), np.array([x0 + px + gap / 2, y0 + py + gap / 2]) - lo)
            cp.copy_index = k
            items.append(cp)
        titles = [t for t, pred in groups if any(pred(it) for it in items)]
        pages.append(dict(title=" + ".join(titles), items=sorted(items, key=lambda c: c.order)))
    pages.sort(key=lambda pg: min(it.order for it in pg["items"]))
    return pages
