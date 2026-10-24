"""Draw packed pieces onto stencil pages."""

import numpy as np

from . import params as P
from .pdf import text_width

CUT_W = 0.18          # cut lines (also what the PDF re-measure check looks for)
FOLD_W = 0.16
CAL_W = 0.3           # calibration square
LEGEND_W = 0.2        # legend samples (kept apart from real cut lines)
VALLEY = (2.4, 1.4)
MOUNTAIN = (3.0, 1.0, 0.5, 1.0)
TAB_FILL = (0.86, 0.86, 0.86)
INK = (0, 0, 0)
GREY = (0.35, 0.35, 0.35)
ACCENT = (0.0, 0.33, 0.55)


def draw_piece(pg, pc):
    for tb in pc.tabs:
        poly = tb["poly"]
        pg.polyline(poly, closed=True, fill=TAB_FILL, stroke=False)
        # the base of a tab is a fold, its other three sides are cuts
        pg.line(poly[0], poly[1], FOLD_W, INK, VALLEY)
        for k in (1, 2, 3):
            pg.line(poly[k], poly[(k + 1) % 4], CUT_W, INK)
    for c in pc.cuts:
        pg.line(c["a"], c["b"], CUT_W, INK)
    for f in pc.folds:
        pg.line(f["a"], f["b"], FOLD_W, INK, VALLEY if f["kind"] == "valley" else MOUNTAIN)
    for a, b in pc.holes:
        pg.line(a, b, CUT_W, INK)
    for lb in pc.labels:
        color = {"name": ACCENT, "paint": (0.75, 0.2, 0.15)}.get(lb["kind"], GREY)
        pg.text(lb["text"], lb["pos"], lb["size"], lb["angle"], bold=lb["bold"], color=color)


def calibration(pg, x, y):
    s = 25.4
    pg.rect(x, y, s, s, width=CAL_W, color=INK)
    pg.text("1 in", (x + s / 2, y + s / 2 + 2.5), 3.4, bold=True)
    pg.text("25.4 mm", (x + s / 2, y + s / 2 - 3.0), 2.4, color=GREY)


def header(pg, left, right, sub=""):
    y = P.PAGE_H - P.MARGIN - 5.0
    pg.text(left, (P.MARGIN, y), 4.0, bold=True, anchor="start")
    pg.text(right, (P.PAGE_W - P.MARGIN, y), 3.2, bold=True, anchor="end", color=ACCENT)
    if sub:
        pg.text(sub, (P.MARGIN, y - 5.2), 2.5, anchor="start", color=GREY)
    yl = P.PAGE_H - P.MARGIN - P.HEADER_H + 0.8
    pg.line((P.MARGIN, yl), (P.PAGE_W - P.MARGIN, yl), 0.15, (0.7, 0.7, 0.7))


def scale_bar(pg, x, y, length=100.0):
    pg.line((x, y), (x + length, y), CAL_W, INK)
    for k in range(int(length / 10) + 1):
        h = 2.6 if k % 5 == 0 else 1.5
        pg.line((x + 10 * k, y), (x + 10 * k, y + h), CAL_W if k % 5 == 0 else FOLD_W, INK)


def footer(pg, note="CUT PATTERN - ACTUAL SIZE"):
    x, y = P.MARGIN, P.MARGIN
    pg.line((x, y + P.FOOTER_H - 0.8), (P.PAGE_W - P.MARGIN, y + P.FOOTER_H - 0.8), 0.15, (0.7, 0.7, 0.7))
    scale_bar(pg, x, y + 10.2)
    pg.text("this bar must measure exactly 100 mm", (x + 103, y + 11.2), 2.4, bold=True, anchor="start")
    lx, ly = x, y + 5.6
    pg.line((lx, ly), (lx + 8, ly), LEGEND_W, INK)
    pg.text("cut", (lx + 9.5, ly), 2.3, anchor="start")
    pg.line((lx + 18, ly), (lx + 26, ly), FOLD_W, INK, VALLEY)
    pg.text("valley fold (towards you)", (lx + 27.5, ly), 2.3, anchor="start")
    pg.line((lx + 64, ly), (lx + 72, ly), FOLD_W, INK, MOUNTAIN)
    pg.text("mountain fold (away)", (lx + 73.5, ly), 2.3, anchor="start")
    pg.polyline([(lx + 105, ly - 1.4), (lx + 114, ly - 1.4), (lx + 112, ly + 1.4), (lx + 107, ly + 1.4)],
                closed=True, fill=TAB_FILL, width=LEGEND_W)
    pg.text("glue tab: goes under the edge with the same number", (lx + 115.5, ly), 2.3, anchor="start")
    pg.text("Print at 100% / Actual size. Printed side = INSIDE of the helmet.",
            (lx, y + 1.0), 2.3, anchor="start", color=GREY)
    pg.text(note, (P.PAGE_W - P.MARGIN, y + 1.0), 2.4, bold=True, anchor="end", color=ACCENT)


def stencil_pages(pdf, pages, first_no, total):
    for k, page in enumerate(pages):
        pg = pdf.page()
        names = []
        for it in page["items"]:
            label = it.name if it.count == 1 else f"{it.name} ({it.copy_index + 1}/{it.count})"
            if label not in names:
                names.append(label)
        header(pg, "CARDBOARD HELMET - STENCILS", f"{first_no + k} / {total}",
               "Pieces: " + ", ".join(names))
        for it in page["items"]:
            draw_piece(pg, it)
        footer(pg)
