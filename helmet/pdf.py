"""Minimal vector PDF writer. Coordinates are millimetres with the origin at the
bottom-left of the page and y up, written 1:1 so pieces print at true size."""

import math
import zlib

PT = 72.0 / 25.4   # points per millimetre

# Helvetica advance widths (1/1000 em) for ASCII 32..126.
_W = [278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
      556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
      1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
      667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
      333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
      556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584]


def text_width(s, size, bold=False):
    w = sum(_W[ord(c) - 32] if 32 <= ord(c) <= 126 else 556 for c in s)
    return w / 1000.0 * size * (1.06 if bold else 1.0)


def _esc(s):
    s = s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return s.encode("cp1252", "replace").decode("latin-1")


def _n(v):
    return f"{v:.4f}".rstrip("0").rstrip(".") if v != 0 else "0"


class Page:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.ops = []
        self.images = []

    def _xy(self, p):
        return f"{_n(p[0] * PT)} {_n(p[1] * PT)}"

    def _style(self, width, color, dash):
        o = [f"{_n(width * PT)} w"]
        o.append("{} {} {} RG".format(*map(_n, color)))
        if dash:
            o.append("[" + " ".join(_n(d * PT) for d in dash) + "] 0 d")
        else:
            o.append("[] 0 d")
        return " ".join(o)

    def line(self, a, b, width=0.18, color=(0, 0, 0), dash=None):
        self.ops.append(f"{self._style(width, color, dash)} {self._xy(a)} m {self._xy(b)} l S")

    def polyline(self, pts, width=0.18, color=(0, 0, 0), dash=None, closed=False, fill=None, stroke=True):
        pts = list(pts)
        if len(pts) < 2:
            return
        path = [f"{self._xy(pts[0])} m"] + [f"{self._xy(p)} l" for p in pts[1:]]
        if closed:
            path.append("h")
        op = "S"
        if fill is not None and stroke:
            op = "B"
        elif fill is not None:
            op = "f"
        pre = self._style(width, color, dash)
        if fill is not None:
            pre += " {} {} {} rg".format(*map(_n, fill))
        self.ops.append(pre + " " + " ".join(path) + " " + op)

    def rect(self, x, y, w, h, **kw):
        self.polyline([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], closed=True, **kw)

    def text(self, s, pos, size=3.0, angle=0.0, bold=False, anchor="middle", color=(0, 0, 0), vcenter=True):
        w = text_width(s, size, bold)
        dx = {"start": 0.0, "middle": -w / 2, "end": -w}[anchor]
        dy = -0.35 * size if vcenter else 0.0
        a = math.radians(angle)
        c, si = math.cos(a), math.sin(a)
        x = pos[0] + dx * c - dy * si
        y = pos[1] + dx * si + dy * c
        font = "F2" if bold else "F1"
        self.ops.append(
            "BT /{} {} Tf {} {} {} rg {} {} {} {} {} {} Tm ({}) Tj ET".format(
                font, _n(size * PT), *map(_n, color), _n(c), _n(si), _n(-si), _n(c),
                _n(x * PT), _n(y * PT), _esc(s)))

    def image(self, jpeg, px_w, px_h, x, y, w, h):
        name = f"Im{len(self.images)}"
        self.images.append((name, jpeg, px_w, px_h))
        self.ops.append(f"q {_n(w * PT)} 0 0 {_n(h * PT)} {_n(x * PT)} {_n(y * PT)} cm /{name} Do Q")


class PDF:
    def __init__(self, w, h, title="", author=""):
        self.w, self.h = w, h
        self.pages = []
        self.title, self.author = title, author

    def page(self):
        p = Page(self.w, self.h)
        self.pages.append(p)
        return p

    def save(self, path):
        objs = []

        def add(body):
            objs.append(body)
            return len(objs)

        cat = add(None)
        pages_id = add(None)
        f1 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        f2 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
        kids = []
        for pg in self.pages:
            xobj = []
            for name, jpeg, pw, ph in pg.images:
                iid = add(b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB "
                          b"/BitsPerComponent 8 /Filter /DCTDecode /Length %d >>\nstream\n" % (pw, ph, len(jpeg))
                          + jpeg + b"\nendstream")
                xobj.append(f"/{name} {iid} 0 R")
            data = zlib.compress("\n".join(pg.ops).encode("latin-1"))
            cid = add(b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(data) + data + b"\nendstream")
            res = f"<< /Font << /F1 {f1} 0 R /F2 {f2} 0 R >>"
            if xobj:
                res += " /XObject << " + " ".join(xobj) + " >>"
            res += " >>"
            pid = add(f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {_n(self.w * PT)} {_n(self.h * PT)}] "
                      f"/Resources {res} /Contents {cid} 0 R >>".encode())
            kids.append(pid)
        objs[cat - 1] = f"<< /Type /Catalog /Pages {pages_id} 0 R /ViewerPreferences << /PrintScaling /None >> >>".encode()
        objs[pages_id - 1] = (f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] "
                              f"/Count {len(kids)} >>").encode()
        info = add(f"<< /Title ({_esc(self.title)}) /Author ({_esc(self.author)}) /Producer (cardboard-helmet) >>".encode("latin-1"))
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for i, body in enumerate(objs, 1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
        for off in offsets:
            out += f"{off:010d} 00000 n \n".encode()
        out += (f"trailer\n<< /Size {len(objs) + 1} /Root {cat} 0 R /Info {info} 0 R >>\n"
                f"startxref\n{xref}\n%%EOF\n").encode()
        with open(path, "wb") as fh:
            fh.write(out)
