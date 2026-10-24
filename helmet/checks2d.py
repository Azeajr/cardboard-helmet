"""Checks on the flat pieces and on the printed PDF itself."""

import math
import re
import subprocess
import tempfile
import os
from collections import Counter
import numpy as np

from . import params as P
from . import geom2d as g2
from .layout import usable
from .pdf import PT


def unfold_checks(bodies, pieces, seams):
    """Every flattened face has exactly its 3D shape, every seam is cut on both
    sides with one tab, and nothing overlaps inside a piece."""
    meshes = {b.mesh.name: b.mesh for b in bodies}
    worst_len = 0.0
    worst_ang = 0.0
    overlaps = []
    for pc in pieces:
        if pc.body == "flat":
            continue
        m = meshes[pc.body]
        for f, poly in pc.poly.items():
            p3 = m.pts(m.faces[f])[::-1]          # printed polygons are mirrored + reversed
            for a in range(len(p3)):
                for b in range(a + 1, len(p3)):
                    worst_len = max(worst_len, abs(np.linalg.norm(p3[a] - p3[b]) - np.linalg.norm(poly[a] - poly[b])))
            # mirrored once: orientation must be reversed relative to the outside view
            worst_ang = max(worst_ang, 0.0 if g2.area(poly) > 0 else 1.0)
        polys = [pc.poly[f] for f in pc.faces] + [t["poly"] for t in pc.tabs]
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if g2.polys_overlap(polys[i], polys[j]):
                    overlaps.append((pc.name, i, j))
    # seams: each number cut on two pieces (or twice on one), lengths equal, one tab set
    cut_len = {}
    for pc in pieces:
        for c in pc.cuts:
            if c["seam"] is not None:
                cut_len.setdefault(c["seam"], []).append(float(np.linalg.norm(c["b"] - c["a"])))
    tab_count = Counter(t["seam"] for pc in pieces for t in pc.tabs)
    bad_seams = []
    for s in seams:
        n = s["seam"]
        lens = cut_len.get(n, [])
        half = len(lens) // 2
        if len(lens) % 2 or half == 0:
            bad_seams.append((n, "not cut on both sides"))
            continue
        if abs(sum(lens) / 2 - s["length"] * sum(b - a for a, b in s["solid"])) > 1e-6:
            bad_seams.append((n, "lengths differ"))
        need = 6.0 if len(s["solid"]) == 1 else 12.0   # same rule as the tab builder
        if tab_count.get(n, 0) == 0 and s["length"] * max(b - a for a, b in s["solid"]) >= need:
            bad_seams.append((n, "no glue tab"))
    # every mesh edge accounted for: a fold inside one piece, a numbered seam, or a free edge
    return dict(face_shape_error_mm=worst_len, orientation_errors=int(worst_ang), overlaps=overlaps,
                seams=len(seams), bad_seams=bad_seams, tabs=sum(tab_count.values()))


def page_checks(pages):
    x0, y0, W, H = usable()
    out_of_bounds, clashes = [], []
    for k, page in enumerate(pages, 1):
        boxes = []
        for it in page["items"]:
            pts = it.points()
            lo, hi = pts.min(0), pts.max(0)
            if lo[0] < x0 - 1e-6 or lo[1] < y0 - 1e-6 or hi[0] > x0 + W + 1e-6 or hi[1] > y0 + H + 1e-6:
                out_of_bounds.append((k, it.name))
            boxes.append((it.name, lo, hi, it))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if (a[1] < b[2]).all() and (b[1] < a[2]).all():
                    pa = [a[3].poly[f] for f in a[3].faces] + [t["poly"] for t in a[3].tabs]
                    pb = [b[3].poly[f] for f in b[3].faces] + [t["poly"] for t in b[3].tabs]
                    if any(g2.polys_overlap(p, q, shrink=-1.0) for p in pa for q in pb):
                        clashes.append((k, a[0], b[0]))
    return dict(pages=len(pages), out_of_bounds=out_of_bounds, clashes=clashes)


def expected_cut_lengths(page):
    """Lengths (mm) of every solid cut stroke drawn on a stencil page."""
    out = []
    for it in page["items"]:
        for c in it.cuts:
            out.append(float(np.linalg.norm(c["b"] - c["a"])))
        for a, b in it.holes:
            out.append(float(np.linalg.norm(b - a)))
        for t in it.tabs:
            p = t["poly"]
            for k in (1, 2, 3):
                out.append(float(np.linalg.norm(p[(k + 1) % 4] - p[k])))
    return out


_PATH = re.compile(r'<path[^>]*?/>', re.S)


def measure_pdf(pdf_path, first_page, pages, stroke_mm=0.18):
    """Render each stencil page to SVG with poppler (an independent PDF reader)
    and measure every solid stroke drawn at the cut-line width."""
    want_w = stroke_mm * PT
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, page in enumerate(pages):
            n = first_page + k
            svg = os.path.join(tmp, f"p{n}.svg")
            subprocess.run(["pdftocairo", "-svg", "-f", str(n), "-l", str(n), pdf_path, svg], check=True)
            body = open(svg).read().split("</defs>")[-1]
            got = []
            bar = []
            for m in _PATH.finditer(body):
                tag = m.group(0)
                if "stroke-dasharray" in tag:
                    continue
                w = re.search(r'stroke-width="([\d.]+)"', tag)
                if not w:
                    continue
                d = re.search(r' d="([^"]*)"', tag).group(1)
                pts = [(float(x), float(y)) for x, y in re.findall(r'[ML] ([-\d.]+) ([-\d.]+)', d)]
                tf = re.search(r'matrix\(([^)]*)\)', tag)
                if tf:
                    a, b, c, dd, e, f = map(float, tf.group(1).replace(",", " ").split())
                    pts = [(a * x + c * y + e, b * x + dd * y + f) for x, y in pts]
                    scale = math.sqrt(abs(a * dd - b * c))
                else:
                    scale = 1.0
                L = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)) / PT
                width = float(w.group(1)) * scale
                if abs(width - want_w) < 0.02:
                    got.append(L)
                elif abs(width - 0.3 * PT) < 0.02 and len(pts) == 2:
                    bar.append(L)
            exp = sorted(expected_cut_lengths(page))
            got = sorted(got)
            n_ok = len(exp) == len(got)
            err = max((abs(a - b) for a, b in zip(exp, got)), default=0.0) if n_ok else math.inf
            results.append(dict(page=n, expected=len(exp), found=len(got), max_error_mm=err,
                                scale_bar_mm=max(bar) if bar else None))
    return results
