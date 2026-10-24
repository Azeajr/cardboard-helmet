"""Head proxy: a sampled stand-in for the wearer's head, scaled from the
measurements in params.py. The design is sized around it and every fit check
measures against it."""

import math
import numpy as np

from . import params as P

# Reference head (550 mm forehead circumference). Horizontal slices below the
# brow line: z, front depth, back depth, half width, front superellipse exponent.
REF_SLICES = [
    (0.0, 97.0, 97.0, 77.5, 2.0),
    (-20.0, 95.0, 96.0, 77.0, 2.3),
    (-40.0, 94.0, 92.0, 75.0, 2.5),
    (-60.0, 92.0, 86.0, 71.0, 2.5),
    (-80.0, 92.0, 76.0, 64.0, 2.3),
    (-100.0, 90.0, 64.0, 58.0, 2.0),
    (-120.0, 86.0, 58.0, 54.0, 1.8),
    (-135.0, 74.0, 55.0, 52.0, 1.6),
    (-150.0, 45.0, 55.0, 55.0, 2.0),
    (-200.0, 45.0, 56.0, 58.0, 2.0),
    (-240.0, 48.0, 60.0, 62.0, 2.0),
]
REF_TOP = 100.0          # crown height above the brow line
REF_LENGTH = 194.0
REF_BREADTH = 155.0
REF_HEIGHT = 233.0       # top of head to bottom of chin

REF_LANDMARKS = {
    "glabella": (0.0, -97.0, 0.0),
    "eye_l": (32.0, -86.0, -24.0),
    "eye_r": (-32.0, -86.0, -24.0),
    "nose_tip": (0.0, -118.0, -55.0),
    "mouth": (0.0, -93.0, -80.0),
    "pogonion": (0.0, -88.0, -118.0),
    "menton": (0.0, -76.0, -133.0),
    "vertex": (0.0, 0.0, REF_TOP),
    "occiput": (0.0, 97.0, 0.0),
}
REF_EAR = dict(x=(66.0, 90.0), y=(2.0, 40.0), z=(-58.0, 6.0))
REF_NOSE = [  # nasion, tip, subnasale, alar left, alar right
    (0.0, -92.0, -18.0), (0.0, -118.0, -55.0), (0.0, -100.0, -66.0),
    (17.0, -96.0, -60.0), (-17.0, -96.0, -60.0),
]


def _superellipse_ring(front, back, hw, nf, n=96):
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    ex = np.where(s < 0, 2.0 / nf, 1.0)
    x = hw * np.sign(c) * np.abs(c) ** ex
    y = np.where(s < 0, -front, back) * np.abs(s) ** ex
    return x, y


def _ref_slice(z):
    """front, back, half width, exponent of the reference head at height z."""
    if z >= 0.0:
        k = max(0.0, 1.0 - (z / REF_TOP) ** 2.3) ** (1.0 / 2.3)
        return 97.0 * k, 97.0 * k, 77.5 * k, 2.0
    zs = [s[0] for s in REF_SLICES]
    for a, b in zip(REF_SLICES, REF_SLICES[1:]):
        if b[0] <= z <= a[0]:
            w = (a[0] - z) / (a[0] - b[0])
            return tuple(a[i] + w * (b[i] - a[i]) for i in range(1, 5))
    return REF_SLICES[-1][1:] if z < zs[-1] else REF_SLICES[0][1:]


class Head:
    def __init__(self):
        c = P.HEAD_CIRCUMFERENCE / 550.0
        self.sx = (P.HEAD_BREADTH / REF_BREADTH) if P.HEAD_BREADTH else c
        self.sy = (P.HEAD_LENGTH / REF_LENGTH) if P.HEAD_LENGTH else c
        self.sz = (P.HEAD_HEIGHT / REF_HEIGHT) if P.HEAD_HEIGHT else c
        self.scale = np.array([self.sx, self.sy, self.sz])
        self.landmarks = {k: np.array(v) * self.scale for k, v in REF_LANDMARKS.items()}
        # Mid-line landmarks sit exactly on the proxy's front surface.
        for k in ("glabella", "mouth", "pogonion", "menton"):
            z = self.landmarks[k][2]
            self.landmarks[k][1] = -_ref_slice(z / self.sz)[0] * self.sy
        e = REF_EAR
        self.ear = {k: (e[k][0] * s, e[k][1] * s) for k, s in
                    (("x", self.sx), ("y", self.sy), ("z", self.sz))}
        self.ear_center = np.array([
            (self.ear["y"][0] + self.ear["y"][1]) / 2,
            (self.ear["z"][0] + self.ear["z"][1]) / 2])
        self._build()

    def _build(self):
        rings, zs = [], []
        for z in np.arange(-240.0, REF_TOP, 4.0):
            f, b, hw, nf = _ref_slice(z)
            x, y = _superellipse_ring(f, b, hw, nf)
            rings.append(np.column_stack([x, y, np.full_like(x, z)]))
            zs.append(z)
        rings.append(np.array([[0.0, 0.0, REF_TOP]]))
        self.rings = [r * self.scale for r in rings]
        skull = np.vstack(self.rings)
        # Outward surface normals: ring tangent x vertical tangent.
        normals = []
        R = self.rings
        for k in range(len(R) - 1):
            lo, hi = R[max(k - 1, 0)], R[k + 1] if k + 1 < len(R) - 1 else None
            r = R[k]
            t1 = np.roll(r, -1, 0) - np.roll(r, 1, 0)
            up = (hi if hi is not None else np.repeat(R[-1], len(r), 0)) - lo
            n = np.cross(t1, up)
            normals.append(n / np.linalg.norm(n, axis=1, keepdims=True))
        normals.append(np.array([[0.0, 0.0, 1.0]]))
        skull_n = np.vstack(normals)

        nose = np.array(REF_NOSE) * self.scale
        tris = [(0, 1, 3), (0, 1, 4), (1, 3, 2), (1, 4, 2)]
        pts, nrm = [], []
        for a, b, c in tris:
            n = np.cross(nose[b] - nose[a], nose[c] - nose[a])
            n /= np.linalg.norm(n)
            if n[1] > 0:          # nose faces point forward (-y)
                n = -n
            for i in range(13):
                for j in range(13 - i):
                    u, v = i / 12.0, j / 12.0
                    pts.append(nose[a] + u * (nose[b] - nose[a]) + v * (nose[c] - nose[a]))
                    nrm.append(n)
        nose_pts, nose_n = np.array(pts), np.array(nrm)

        ex, ey, ez = self.ear["x"], self.ear["y"], self.ear["z"]
        g = np.meshgrid(np.linspace(*ey, 10), np.linspace(*ez, 14))
        ear_face = np.column_stack([np.full(g[0].size, ex[1]), g[0].ravel(), g[1].ravel()])
        ears = np.vstack([ear_face, ear_face * [-1, 1, 1]])

        self.skull = skull
        self.nose = nose_pts
        self.ears = ears
        # Face points: the front of the head between brow and below the chin.
        fm = (skull[:, 1] < -20) & (skull[:, 2] < 10) & (skull[:, 2] > -150)
        self.face = np.vstack([skull[fm], nose_pts])
        self.face_n = np.vstack([skull_n[fm], nose_n])
        # Everything the shell must clear (ears handled with their own margin).
        cm = skull[:, 2] > -150
        self.cranium, self.cranium_n = skull[cm], skull_n[cm]

    def inflated(self, which, gap):
        """Points pushed out by `gap` along the surface: the shell must stay
        outside these to keep `gap` everywhere, not just straight out."""
        if which == "cranium":
            return self.cranium + self.cranium_n * gap
        if which == "face":
            # Sharp features (the nose) need every direction, not one normal.
            n_skull = len(self.face) - len(self.nose)
            return np.vstack([self.face[:n_skull] + self.face_n[:n_skull] * gap,
                              _ball(self.nose, gap)])
        return _ball(self.ears, gap)

    # --- measurements on the proxy ------------------------------------------
    def circumference(self, z=0.0):
        f, b, hw, nf = _ref_slice(z / self.sz)
        x, y = _superellipse_ring(f, b, hw, nf, 2000)
        pts = np.column_stack([x * self.sx, y * self.sy])
        return float(np.sum(np.linalg.norm(np.roll(pts, -1, 0) - pts, axis=1)))

    def midline(self):
        """Mid-line (y, z) points of the proxy, nose included."""
        pts = [r[np.abs(r[:, 0]) < 1e-6][:, 1:] for r in self.rings]
        mid = np.vstack(pts)
        return np.vstack([mid, self.nose[np.abs(self.nose[:, 0]) < 1e-6][:, 1:]])

    def tape_path(self):
        """The path a tape takes from the chin over the top and down the back of
        the neck: taut (convex hull) over the face and skull, then down the
        back of the neck."""
        mid = self.midline()
        chin = self.landmarks["menton"][1:]
        upper = np.vstack([mid[mid[:, 1] >= chin[1]], [chin]])
        hull = convex_hull(upper)
        i0 = int(np.argmin(np.linalg.norm(hull - chin, axis=1)))
        n = len(hull)
        # The hull is counter-clockwise in (y, z): stepping backwards from the
        # chin climbs the face, crosses the top and comes down the back.
        path = [hull[(i0 - k) % n] for k in range(n)]
        if path[1][0] > path[0][0]:
            path = [hull[(i0 + k) % n] for k in range(n)]
        while len(path) > 2 and path[-1][1] <= chin[1] + 1e-6 and path[-2][1] <= chin[1] + 1e-6:
            path.pop()  # drop the flat cut line under the chin
        neck = mid[(mid[:, 0] > 0) & (mid[:, 1] < path[-1][1])]
        neck = neck[np.argsort(-neck[:, 1])]
        return np.vstack([path, neck])

    def nape_from_tape(self, tape=None):
        """Where the measured tape (chin -> over the top -> nape) ends, (y, z)."""
        tape = P.CHIN_TOP_NAPE if tape is None else tape
        path = self.tape_path()
        left = tape
        for a, b in zip(path, path[1:]):
            d = float(np.linalg.norm(b - a))
            if d >= left:
                return a + (b - a) * (left / d)
            left -= d
        return path[-1]


def _ball(pts, gap):
    """Each point pushed out in 26 directions (a coarse Minkowski ball)."""
    g = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                  if (i, j, k) != (0, 0, 0)], float)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return (pts[:, None, :] + g[None, :, :] * gap).reshape(-1, 3)


def convex_hull(pts):
    pts = sorted(set(map(tuple, np.round(pts, 6))))
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])
