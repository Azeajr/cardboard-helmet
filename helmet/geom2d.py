"""Small 2D polygon toolkit (no external dependencies)."""

import math
import numpy as np

EPS = 1e-9


def area(poly):
    p = np.asarray(poly, float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def centroid(poly):
    p = np.asarray(poly, float)
    a = area(p)
    if abs(a) < EPS:
        return p.mean(axis=0)
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    c = x * yn - xn * y
    return np.array([np.sum((x + xn) * c), np.sum((y + yn) * c)]) / (6.0 * a)


def point_in_poly(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xi > x:
                inside = not inside
    return inside


def seg_intersect(p1, p2, q1, q2, eps=1e-9):
    """Proper intersection of two segments (touching endpoints don't count)."""
    d1 = np.subtract(p2, p1)
    d2 = np.subtract(q2, q1)
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < eps:
        return False
    w = np.subtract(q1, p1)
    t = (w[0] * d2[1] - w[1] * d2[0]) / den
    u = (w[0] * d1[1] - w[1] * d1[0]) / den
    return eps < t < 1 - eps and eps < u < 1 - eps


def offset_inward(poly, d):
    """Miter offset of a simple polygon towards its inside by d (small d)."""
    p = np.asarray(poly, float)
    if area(p) < 0:
        p = p[::-1]
    n = len(p)
    out = []
    for i in range(n):
        a, b, c = p[i - 1], p[i], p[(i + 1) % n]
        e1 = b - a
        e2 = c - b
        n1 = np.array([-e1[1], e1[0]]) / (np.linalg.norm(e1) + EPS)
        n2 = np.array([-e2[1], e2[0]]) / (np.linalg.norm(e2) + EPS)
        m = n1 + n2
        ml = np.linalg.norm(m)
        if ml < 1e-6:
            out.append(b + n1 * d)
            continue
        m /= ml
        cosh = max(np.dot(m, n1), 0.2)
        out.append(b + m * (d / cosh))
    return np.array(out)


def polys_overlap(a, b, shrink=0.02):
    """True if two simple polygons overlap by more than a hairline."""
    A = offset_inward(a, shrink)
    B = offset_inward(b, shrink)
    amin, amax = A.min(0), A.max(0)
    bmin, bmax = B.min(0), B.max(0)
    if (amin > bmax).any() or (bmin > amax).any():
        return False
    for i in range(len(A)):
        p1, p2 = A[i], A[(i + 1) % len(A)]
        for j in range(len(B)):
            if seg_intersect(p1, p2, B[j], B[(j + 1) % len(B)]):
                return True
    return point_in_poly(A[0], B) or point_in_poly(B[0], A)


def clip_halfplane(poly, p, n):
    """Keep the part of poly where dot(x - p, n) >= 0 (Sutherland-Hodgman)."""
    out = []
    m = len(poly)
    for i in range(m):
        a, b = np.asarray(poly[i]), np.asarray(poly[(i + 1) % m])
        da, db = np.dot(a - p, n), np.dot(b - p, n)
        if da >= 0:
            out.append(a)
        if (da >= 0) != (db >= 0):
            t = da / (da - db)
            out.append(a + t * (b - a))
    return out


def clip_rect(poly, x0, x1, y0, y1):
    for p, n in (((x0, 0), (1, 0)), ((x1, 0), (-1, 0)), ((0, y0), (0, 1)), ((0, y1), (0, -1))):
        if not poly:
            return []
        poly = clip_halfplane(poly, np.array(p, float), np.array(n, float))
    return poly


def seg_poly_intervals(a, b, poly):
    """Parameter intervals [t0, t1] of segment a->b that lie inside poly."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ts = [0.0, 1.0]
    d = b - a
    m = len(poly)
    for i in range(m):
        q1, q2 = np.asarray(poly[i], float), np.asarray(poly[(i + 1) % m], float)
        e = q2 - q1
        den = d[0] * e[1] - d[1] * e[0]
        if abs(den) < 1e-12:
            continue
        w = q1 - a
        t = (w[0] * e[1] - w[1] * e[0]) / den
        u = (w[0] * d[1] - w[1] * d[0]) / den
        if 0 <= t <= 1 and 0 <= u <= 1:
            ts.append(t)
    ts = sorted(set(round(t, 12) for t in ts))
    out = []
    for t0, t1 in zip(ts, ts[1:]):
        if t1 - t0 < 1e-9:
            continue
        mid = a + d * (t0 + t1) / 2
        if point_in_poly(mid, poly):
            if out and abs(out[-1][1] - t0) < 1e-9:
                out[-1] = (out[-1][0], t1)
            else:
                out.append((t0, t1))
    return out


def subtract_intervals(intervals, holes):
    """[0,1] minus hole intervals -> solid intervals."""
    solid = [(0.0, 1.0)]
    for h0, h1 in holes:
        nxt = []
        for s0, s1 in solid:
            if h1 <= s0 or h0 >= s1:
                nxt.append((s0, s1))
                continue
            if h0 > s0:
                nxt.append((s0, h0))
            if h1 < s1:
                nxt.append((h1, s1))
        solid = nxt
    return [(a, b) for a, b in solid if b - a > 1e-9]


def rotate(pts, ang):
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return np.asarray(pts, float) @ R.T


def densify(poly, step, closed=True):
    pts = [np.asarray(p, float) for p in poly]
    out = []
    n = len(pts) if closed else len(pts) - 1
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        k = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        for j in range(k):
            out.append(a + (b - a) * (j / k))
    if not closed:
        out.append(pts[-1])
    return out
