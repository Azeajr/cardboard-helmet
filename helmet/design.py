"""Helmet geometry, sized around the head proxy.

Shell: seams at x = +-SHELL_SEAMS_X run front-to-back over the head, rows of
seams run around the pivot axis (an axis through both ear pivots). Flat side
plates sit at x = +-SIDE_X. Between SHELL_FRONT_EDGE and the swing limit the
shell profile's corners lie on a circle about the pivot axis, so the
faceplate, which stays outside that circle, can swing over it at any angle.

Faceplate: the same construction (scaled copies of one profile) with flat
arms at x = +-ARM_X that pivot on Chicago screws through the side plates.

Every facet is its own flat panel of BOARD-thick cardboard. The meshes are the
panels' outer surfaces; the board lies inside them.
"""

import math
import numpy as np

from . import params as P
from . import geom2d as g2
from .head import Head, convex_hull
from .mesh import Mesh, Grid, point_at, polar, profile_radius, chord_min_dist


def _hull_profile(points_fu, phis):
    """Radii of the convex hull of (f, u) points along each angle in phis."""
    hull = convex_hull(np.vstack([points_fu, [[0.0, 0.0]]]))
    out = []
    for phi in phis:
        a = math.radians(phi)
        d = np.array([math.cos(a), math.sin(a)])
        best = 0.0
        for k in range(len(hull)):
            p, q = hull[k], hull[(k + 1) % len(hull)]
            e = q - p
            den = d[0] * e[1] - d[1] * e[0]
            if abs(den) < 1e-12:
                continue
            rho = (p[0] * e[1] - p[1] * e[0]) / den
            s = (p[0] * d[1] - p[1] * d[0]) / den
            if rho > 1e-9 and -1e-6 <= s <= 1 + 1e-6:
                best = max(best, rho)
        if best == 0.0:
            # A ray along the hull's own edge from the origin (the outermost
            # angle) misses every edge by rounding: take the farthest point on it.
            on = np.abs(np.arctan2(hull[:, 1], hull[:, 0]) - a) < 1e-5
            best = float(np.linalg.norm(hull[on], axis=1).max()) if on.any() else 0.0
        out.append(best)
    return np.array(out)


def _fu(r, phi):
    a = np.radians(phi)
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def _inflate(phis, radii, req_phi, req_r, fixed=None, iters=200):
    """Raise profile vertices until every required (phi, r) point is inside."""
    radii = np.array(radii, float)
    fixed = np.zeros(len(radii), bool) if fixed is None else fixed
    lo, hi = min(phis[0], phis[-1]), max(phis[0], phis[-1])
    m = (req_phi >= lo) & (req_phi <= hi)
    req_phi, req_r = req_phi[m], req_r[m]
    asc = phis[0] < phis[-1]
    key = np.asarray(phis) if asc else -np.asarray(phis)
    k = np.clip(np.searchsorted(key, req_phi if asc else -req_phi) - 1, 0, len(phis) - 2)
    for _ in range(iters):
        have = profile_radius(phis, radii, req_phi)
        ratio = req_r / have
        if (ratio <= 1 + 1e-9).all():
            break
        # One multiplicative step per vertex: the worst point on either chord.
        grow = np.ones(len(radii))
        for kk in (k, k + 1):
            np.maximum.at(grow, kk, ratio)
        grow[fixed] = 1.0
        if (grow <= 1 + 1e-12).all():
            break  # remaining shortfall sits on fixed chords
        radii *= np.minimum(grow, 1.05)
    return radii


def _fit_scales(knots, xs, need, iters=200):
    """Piecewise-linear scale over knots (first interval flat) that is at least
    `need` at every |x|: start from a safe solution, then lower each knot as
    far as its neighbours allow. Returns scales at the knots, made concave."""
    knots = np.asarray(knots, float)
    xs = np.clip(np.abs(xs), knots[0], knots[-1])
    k = np.clip(np.searchsorted(knots, xs, side="right") - 1, 0, len(knots) - 2)
    s = (xs - knots[k]) / (knots[k + 1] - knots[k])
    a = np.zeros(len(knots))
    for kk in range(len(knots)):
        m = (k == kk) | (k == kk - 1)
        a[kk] = need[m].max() if m.any() else 0.0
    for _ in range(iters):
        old = a.copy()
        for kk in range(len(knots)):
            lb = 0.0
            left = (k == kk - 1)            # kk is the right end: weight s
            if left.any():
                w = s[left]
                ok = w > 1e-3
                if ok.any():
                    lb = max(lb, np.max((need[left][ok] - (1 - w[ok]) * a[kk - 1]) / w[ok]))
            right = (k == kk)               # kk is the left end: weight 1 - s
            if right.any():
                w = 1 - s[right]
                ok = w > 1e-3
                if ok.any():
                    lb = max(lb, np.max((need[right][ok] - (1 - w[ok]) * a[kk + 1]) / w[ok]))
            a[kk] = lb
        if np.abs(a - old).max() < 1e-7:
            break
    have = np.interp(xs, knots, a)
    if (need - have).max() > 1e-6:
        a += (need - have).max()
    # Least concave majorant over the knots (smooth, dome-like front view).
    for _ in range(len(a)):
        for i in range(1, len(a) - 1):
            w = (knots[i] - knots[i - 1]) / (knots[i + 1] - knots[i - 1])
            chord = a[i - 1] + w * (a[i + 1] - a[i - 1])
            a[i] = max(a[i], chord)
    return a


class Design:
    def __init__(self):
        self.head = Head()
        h = self.head
        self.nape = h.nape_from_tape()
        # Never lower than the chin, or the back edge stops you looking up.
        self.back_edge_z = max(self.nape[1] + P.BACK_EDGE_ABOVE_NAPE,
                               h.landmarks["menton"][2])
        # The screw head and its washer sit inside each side plate, over the ear.
        self.side_x = max(h.ear["x"][1] + P.EAR_CLEARANCE + P.PIVOT_HEAD + P.PIVOT_WASHER + P.BOARD,
                          float(np.abs(h.cranium[:, 0]).max()) + P.CLEARANCE + P.BOARD)
        self.arm_x = self.side_x + P.PIVOT_WASHER + P.BOARD
        self.pivot = self._choose_pivot()
        self._profiles(self.pivot, final=True)
        self._scales()
        self.shell_grid = Grid("shell", self._sym(P.SHELL_SEAMS_X, self.side_x),
                               self._sym_scales(self.a), self.shell_phis, self.shell_r, self.pivot)
        self.face_grid = Grid("face", self._sym(P.FACE_SEAMS_X, self.arm_x),
                              self._sym_scales(self.b), self.face_phis, self.face_r, self.pivot)
        self.shell = self._build_shell()
        self.faceplate = self._build_faceplate()
        self.holes = self._holes()
        self.overlays = self._overlays()

    # --- profiles -------------------------------------------------------------
    def _req(self, pivot):
        """Required (|x|, phi, radius) points: the head inflated by each clearance
        plus the board, so the outer surface must pass outside them."""
        h = self.head
        if not hasattr(self, "_inflated"):
            self._inflated = dict(
                cran=h.inflated("cranium", P.CLEARANCE + P.BOARD),
                face=h.inflated("face", P.FACE_CLEARANCE + P.BOARD),
                ears=h.inflated("ears", P.EAR_CLEARANCE + P.BOARD))
        out = {}
        for k, pts in self._inflated.items():
            if k == "face":
                # Beside the head the flat pivot arms (inner face at arm_x - BOARD)
                # give the clearance sideways; don't make the mask bulge for it.
                pts = pts[np.abs(pts[:, 0]) <= self.side_x - 2.0]
            r, phi = polar(pts, pivot)
            out[k] = (np.abs(pts[:, 0]), phi, r)
        return out

    def _profiles(self, pivot, final=False):
        req = self._req(pivot)
        cx, cphi, cneed = req["cran"]
        front = P.SHELL_FRONT_EDGE
        self.sweep_end = P.FACE_ROWS[0] + P.SWING_ANGLE + 3.0
        band = cx <= P.SHELL_SEAMS_X[0]
        in_sweep = band & (cphi >= front - 3) & (cphi <= self.sweep_end + 3)
        # Vertices sit on the circle; chords between them dip inside it.
        R0 = float(cneed[in_sweep].max()) / math.cos(math.radians(P.SHELL_ROW_STEP / 2))

        # Shell: circle R0 through the swing zone, convex hull of the head beyond.
        back = band & (cphi > self.sweep_end - 10) & (cphi < 268)
        arc = np.linspace(front, self.sweep_end, 40)
        pts = np.vstack([_fu(np.full(arc.size, R0), arc), _fu(cneed[back], cphi[back])])
        fine = np.linspace(front, 266, 400)
        fr = _hull_profile(pts, fine)
        z = pivot[1] + fr * np.sin(np.radians(fine))
        below = np.where((fine > 180) & (z <= self.back_edge_z))[0]
        phi_end = float(fine[below[0]]) if len(below) else 266.0
        n = int(math.ceil((phi_end - front) / P.SHELL_ROW_STEP)) + 1
        phis = np.linspace(front, phi_end, n)
        radii = _hull_profile(pts, phis)
        radii[phis <= self.sweep_end + 1e-9] = R0
        fixed = phis <= self.sweep_end + 1e-9
        radii = _inflate(phis, radii, cphi[band], cneed[band], fixed)
        # The chord between the last circle vertex and the next must also clear.
        shell_phis, shell_r = phis, radii

        # Faceplate: outside the swing circle and clear of the face.
        fphis = np.array(P.FACE_ROWS, float)
        fx, fphi, fneed = req["face"]
        fband = fx <= P.FACE_SEAMS_X[0]
        step = max(abs(a - b) for a, b in zip(fphis, fphis[1:]))
        r_sw = (R0 + P.BOARD + P.SWING_GAP + 0.3) / math.cos(math.radians(step / 2))
        sel = fband & (fphi >= fphis[-1] - 3) & (fphi <= fphis[0] + 3)
        fpts = np.vstack([_fu(fneed[sel], fphi[sel]), _fu(np.full(fphis.size, r_sw), fphis)])
        fr = _hull_profile(fpts, fphis)
        fr = _inflate(fphis, fr, fphi[fband], fneed[fband])
        # Re-hull so every faceplate seam turns the same way (no dents).
        fr = _hull_profile(_fu(fr, fphis), fphis)
        d_min = chord_min_dist(fphis, fr)
        if d_min < R0 + P.BOARD + P.SWING_GAP:
            fr *= (R0 + P.BOARD + P.SWING_GAP + 0.3) / d_min
            d_min = chord_min_dist(fphis, fr)
        if final:
            self.R0 = R0
            self.shell_phis, self.shell_r = shell_phis, shell_r
            self.face_phis, self.face_r = fphis, fr
            self.face_dmin = d_min
            self.req = req
        return R0, float(fr.max())

    def _choose_pivot(self):
        ey, ez = self.head.ear_center
        best = None
        for dy in np.arange(-12, 12.1, 2.0):
            for dz in np.arange(-12, 16.1, 2.0):
                piv = (ey + dy, ez + dz)
                R0, fmax = self._profiles(piv)
                cost = R0 + fmax
                if best is None or cost < best[0]:
                    best = (cost, piv)
        return best[1]

    # --- column scales --------------------------------------------------------
    def _scales(self):
        req = self.req
        sp, sr = self.shell_phis, self.shell_r
        lo, hi = sp[0], sp[-1]
        knots_s = np.array(P.SHELL_SEAMS_X + (self.side_x,))
        xs, need = [], []
        for key in ("cran", "ears"):
            x, phi, nd = req[key]
            m = (phi >= lo) & (phi <= hi)
            xs.append(x[m])
            need.append(nd[m] / profile_radius(sp, sr, phi[m]))
        xs, need = np.concatenate(xs), np.concatenate(need)
        # The side plate carries the pivot and balances the faceplate arm.
        xs = np.append(xs, knots_s[-1])
        need = np.append(need, max((P.SIDE_FRONT_R + 15.0) / sr.min(), P.SIDE_PLATE_R / self.R0))
        a = _fit_scales(knots_s, xs, need)
        self.a = a
        self.knots_s = knots_s

        fp, fr = self.face_phis, self.face_r
        knots_f = np.array(P.FACE_SEAMS_X + (self.arm_x,))
        x, phi, nd = req["face"]
        m = (phi >= min(fp)) & (phi <= max(fp))
        fx = [x[m]]
        fneed = [nd[m] / profile_radius(fp, fr, phi[m])]
        # Swing: the faceplate's inner surface must pass outside the shell.
        xx = np.linspace(0, self.side_x + P.BOARD, 300)
        a_at = np.interp(np.clip(xx, knots_s[0], knots_s[-1]), knots_s, a)
        shell_top = a_at * self.R0
        # Where the shell slopes sideways the perpendicular gap is the radial
        # gap times cos(slope), so ask for proportionally more.
        k = np.clip(np.searchsorted(knots_s, np.clip(xx, knots_s[0], knots_s[-1]), side="right") - 1,
                    0, len(knots_s) - 2)
        slope = np.abs(a[k + 1] - a[k]) * self.R0 / (knots_s[k + 1] - knots_s[k])
        slope = np.where(xx < knots_s[0], 0.0, slope)
        margin = (P.BOARD + P.SWING_GAP + 0.4) * np.sqrt(1 + slope ** 2)
        need = np.where(xx <= self.side_x, shell_top + margin, -np.inf)
        # A steep faceplate panel's inner surface lies up to BOARD further in
        # (towards smaller x) than its outer surface, so clear the shell over
        # that whole reach.
        need = np.array([need[(xx >= v - P.BOARD) & (xx <= v)].max() for v in xx])
        fx.append(xx)
        fneed.append(need / self.face_dmin)
        self.b = _fit_scales(knots_f, np.concatenate(fx), np.concatenate(fneed))
        self.knots_f = knots_f

    @staticmethod
    def _sym(seams, outer):
        right = list(seams) + [outer]
        return [-v for v in reversed(right)] + right

    @staticmethod
    def _sym_scales(s):
        s = list(s)
        return list(reversed(s)) + s

    # --- meshes ---------------------------------------------------------------
    def _radial_out(self, c):
        v = np.array([0.0, c[1] - self.pivot[0], c[2] - self.pivot[1]])
        return v / (np.linalg.norm(v) + 1e-12)

    def _build_shell(self):
        g = self.shell_grid
        m = Mesh("shell")
        ncol = len(g.xs) - 1
        for i in range(ncol):
            for j in range(len(g.phis) - 1):
                m.add_face(g.corners(i, j), f"S{i}.{j}", self._radial_out, cell=(i, j), plate=None)
        for side, i in (("R", 0), ("L", len(g.xs) - 1)):
            x = g.xs[i]
            pts = [g.vertex(i, j) for j in range(len(g.phis))]
            end_r = g.scales[i] * g.radii[-1]
            # Lower jaw lobe: grows smoothly from the back edge to the faceplate's
            # lower arm corner, just behind the jaw line.
            jaw_r = max(end_r, self.b[-1] * self.face_r[-1] + 4.0)
            notch = 360.0 - (abs(P.FACE_ROWS[-1]) + P.SIDE_NOTCH_MARGIN)
            for t in np.linspace(0.0, 1.0, 9)[1:]:
                w = 0.5 - 0.5 * math.cos(math.pi * min(1.0, t * 1.4))
                pts.append(point_at(self.pivot, x, end_r + (jaw_r - end_r) * w,
                                    g.phis[-1] + (notch - g.phis[-1]) * t))
            for phi in np.linspace(notch, 360.0 + g.phis[0], 14):
                pts.append(point_at(self.pivot, x, P.SIDE_FRONT_R, phi))
            sgn = math.copysign(1.0, x)
            m.add_face(pts, f"S-SIDE-{side}", lambda c, s=sgn: np.array([s, 0.0, 0.0]),
                       cell=None, plate=side)
        return m

    def _build_faceplate(self):
        g = self.face_grid
        m = Mesh("faceplate")
        for i in range(len(g.xs) - 1):
            for j in range(len(g.phis) - 1):
                m.add_face(g.corners(i, j), f"F{i}.{j}", self._radial_out, cell=(i, j), plate=None)
        for side, i in (("R", 0), ("L", len(g.xs) - 1)):
            x = g.xs[i]
            pts = [g.vertex(i, j) for j in range(len(g.phis))]
            start, stop = g.phis[-1], g.phis[0] - 360.0
            for phi in np.linspace(start, stop, 40):
                pts.append(point_at(self.pivot, x, P.EAR_DISC_R, phi))
            sgn = math.copysign(1.0, x)
            m.add_face(pts, f"F-ARM-{side}", lambda c, s=sgn: np.array([s, 0.0, 0.0]),
                       cell=None, plate=side)
        return m

    # --- holes and overlays (param space: x in mm, phi in degrees) -----------
    def _holes(self):
        # Each slot sits inside one faceplate panel with board left all round it.
        eye = [(22.0, -5.0), (62.0, -3.0), (66.0, 5.5), (23.0, 3.5)]
        mouth = [(-8.0, -30.5), (8.0, -30.5), (8.0, -27.0), (-8.0, -27.0)]
        holes = {"faceplate": [], "shell": []}
        holes["faceplate"].append(("eye-L", eye))
        holes["faceplate"].append(("eye-R", [(-x, p) for x, p in eye][::-1]))
        holes["faceplate"].append(("mouth", mouth))
        return holes

    def _overlays(self):
        # Overlays are board too, so each one lies flat on a single panel.
        brow = [(18.0, 10.0), (70.0, 10.0), (73.0, 16.0), (18.0, 16.0)]
        cheek = [(44.0, -16.0), (72.0, -16.0), (72.0, -30.0), (62.0, -35.0), (44.0, -28.0)]
        chin = [(-10.0, -58.0), (10.0, -58.0), (10.0, -43.0), (0.0, -40.0), (-10.0, -43.0)]
        mir = lambda poly: [(-x, p) for x, p in poly][::-1]
        out = [
            ("O-BROW-L", "face", brow),
            ("O-BROW-R", "face", mir(brow)),
            ("O-CHEEK-L", "face", cheek),
            ("O-CHEEK-R", "face", mir(cheek)),
            ("O-CHIN", "face", chin),
        ]
        # Crest: one plate per centre panel, behind where the open faceplate reaches.
        ph = self.shell_phis
        for j in range(len(ph) - 1):
            lo, hi = max(ph[j], self.sweep_end + 6.0) + 2.0, ph[j + 1] - 2.0
            if hi - lo >= 8.0:
                k = sum(1 for o in out if o[0].startswith("O-CREST")) + 1
                out.append((f"O-CREST-{k}", "shell", [(-11.0, lo), (11.0, lo), (11.0, hi), (-11.0, hi)]))
        return out

    # --- summary ----------------------------------------------------------------
    def summary(self):
        h = self.head
        return dict(
            head_circumference=round(h.circumference(), 1),
            head_length=round(194.0 * h.sy, 1), head_breadth=round(155.0 * h.sx, 1),
            head_height=round(233.0 * h.sz, 1),
            nape_from_tape=[round(float(v), 1) for v in self.nape],
            back_edge_z=round(float(self.back_edge_z), 1),
            pivot_yz=[round(float(v), 1) for v in self.pivot],
            ear_center_yz=[round(float(v), 1) for v in h.ear_center],
            side_x=round(self.side_x, 1), arm_x=round(self.arm_x, 1),
            R0=round(self.R0, 1),
            shell_rows=[round(float(v), 1) for v in self.shell_phis],
            shell_radii=[round(float(v), 1) for v in self.shell_r],
            face_radii=[round(float(v), 1) for v in self.face_r],
            face_chord_min=round(self.face_dmin, 1),
            shell_scales=[round(float(v), 3) for v in self.a],
            face_scales=[round(float(v), 3) for v in self.b],
        )
