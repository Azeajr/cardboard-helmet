"""Build the Blender scene (head proxy, shell, hinged faceplate, overlays,
pivot hardware) and render preview images."""

import math
import os
import numpy as np
import bpy

from . import params as P
from . import geom2d as g2
from .mesh import point_at

COLORS = {
    "shell": (0.62, 0.08, 0.07, 1.0),
    "face": (0.86, 0.66, 0.24, 1.0),
    "overlay": (0.80, 0.58, 0.18, 1.0),
    "crest": (0.48, 0.05, 0.05, 1.0),
    "slot": (0.02, 0.02, 0.03, 1.0),
    "hardware": (0.55, 0.56, 0.58, 1.0),
    "head": (0.78, 0.74, 0.70, 1.0),
}


def _mat(name, rgba):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.diffuse_color = rgba
    m.roughness = 0.6
    return m


def _obj(name, verts, faces, mat, parent=None, coll=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [list(f) for f in faces])
    me.update()
    ob = bpy.data.objects.new(name, me)
    (coll or bpy.context.scene.collection).objects.link(ob)
    ob.data.materials.append(mat)
    if parent is not None:
        ob.parent = parent
    return ob


def _solidify(ob, t):
    mod = ob.modifiers.new("board", "SOLIDIFY")
    mod.thickness = t
    mod.offset = -1.0          # board lies inside the outer surface
    mod.use_even_offset = True


def _head_mesh(h):
    verts, faces = [], []
    rings = h.rings
    n = len(rings[0])
    for r in rings[:-1]:
        verts += [tuple(p) for p in r]
    apex = len(verts)
    verts.append(tuple(rings[-1][0]))
    for k in range(len(rings) - 2):
        a, b = k * n, (k + 1) * n
        for i in range(n):
            faces.append((a + i, a + (i + 1) % n, b + (i + 1) % n, b + i))
    top = (len(rings) - 2) * n
    for i in range(n):
        faces.append((top + i, top + (i + 1) % n, apex))
    return verts, faces


def _param_patches(d, grid, poly, lift):
    """3D patches of a param-space polygon on a grid surface, lifted outward
    along each panel's normal."""
    verts, faces = [], []
    dense = g2.densify(poly, 1.0)
    for i in range(len(grid.xs) - 1):
        for j in range(len(grid.phis) - 1):
            x0, x1, p0, p1 = grid.cell_rect(i, j)
            clip = g2.clip_rect([np.asarray(p, float) for p in dense], x0, x1, p0, p1)
            if len(clip) < 3 or abs(g2.area(clip)) < 1e-3:
                continue
            pts = [grid.point(p[0], p[1], (i, j)) for p in clip]
            n = np.zeros(3)
            for a, b in zip(pts, pts[1:] + pts[:1]):   # Newell
                n += np.cross(a, b)
            n /= np.linalg.norm(n)
            c = np.mean(pts, axis=0)
            if np.dot(n, [0.0, c[1] - d.pivot[0], c[2] - d.pivot[1]]) < 0:
                n, pts = -n, pts[::-1]     # face outwards, so solidifying inwards lands on the panel
            k = len(verts)
            verts += [tuple(p + n * lift) for p in pts]
            faces.append(list(range(k, k + len(pts))))
    return verts, faces


def _disc(d, x, r, thick, name, mat, parent=None, ring_in=None, n=48):
    verts, faces = [], []
    ang = np.linspace(0, 360, n, endpoint=False)
    for xx in (x, x + thick):
        verts += [tuple(point_at(d.pivot, xx, r, a)) for a in ang]
        if ring_in:
            verts += [tuple(point_at(d.pivot, xx, ring_in, a)) for a in ang]
    if ring_in:
        o0, i0, o1, i1 = 0, n, 2 * n, 3 * n
        for k in range(n):
            k2 = (k + 1) % n
            faces += [(o0 + k, o0 + k2, i0 + k2, i0 + k), (o1 + k, i1 + k, i1 + k2, o1 + k2),
                      (o0 + k, o1 + k, o1 + k2, o0 + k2), (i0 + k, i0 + k2, i1 + k2, i1 + k)]
    else:
        faces += [list(range(n))[::-1], list(range(n, 2 * n))]
        for k in range(n):
            k2 = (k + 1) % n
            faces.append((k, k2, n + k2, n + k))
    return _obj(name, verts, faces, mat, parent)


def build_scene(d, overlay_defs=None):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.scale_length = 0.001
    sc.unit_settings.length_unit = "MILLIMETERS"
    mats = {k: _mat(k, v) for k, v in COLORS.items()}

    hv, hf = _head_mesh(d.head)
    head = _obj("Head proxy", hv, hf, mats["head"])
    head.display_type = "WIRE"

    sm = d.shell
    shell = _obj("Shell", sm.verts, [f["vi"] for f in sm.faces], mats["shell"])
    _solidify(shell, P.BOARD)

    pivot = bpy.data.objects.new("Faceplate pivot (rotate X to open)", None)
    pivot.empty_display_type = "ARROWS"
    pivot.empty_display_size = 30.0
    pivot.location = (0.0, d.pivot[0], d.pivot[1])
    sc.collection.objects.link(pivot)

    fm = d.faceplate
    fverts = [np.asarray(v) - np.array([0.0, d.pivot[0], d.pivot[1]]) for v in fm.verts]
    face = _obj("Faceplate", fverts, [f["vi"] for f in fm.faces], mats["face"], parent=pivot)
    _solidify(face, P.BOARD)

    loc = np.array([0.0, d.pivot[0], d.pivot[1]])
    for name, poly in d.holes["faceplate"]:
        v, f = _param_patches(d, d.face_grid, poly, 0.35)
        _obj(f"Slot {name}", [np.asarray(p) - loc for p in v], f, mats["slot"], parent=pivot)
    for name, gname, poly in d.overlays:
        grid = d.face_grid if gname == "face" else d.shell_grid
        v, f = _param_patches(d, grid, poly, P.BOARD + 0.1)    # on top of the panel
        if gname == "face":
            ob = _obj(name, [np.asarray(p) - loc for p in v], f, mats["overlay"], parent=pivot)
        else:
            ob = _obj(name, v, f, mats["crest"])
        _solidify(ob, P.BOARD)

    # Pivot, inside to out: screw head, washer, side plate, washer, arm, washer, cap.
    w, hd, wr = P.PIVOT_WASHER, P.PIVOT_HEAD, P.PIVOT_WASHER_D / 2
    inside = d.side_x - P.BOARD
    for sgn in (-1.0, 1.0):
        side = "L" if sgn > 0 else "R"
        at = lambda x0, t: x0 if sgn > 0 else -x0 - t   # [x0, x0 + t] on the left, mirrored
        for x0, t, r, what in ((inside - w - hd, hd, 6.0, "Chicago screw head"), (inside - w, w, wr, "Washer in"),
                               (d.side_x, w, wr, "Washer mid"), (d.arm_x, w, wr, "Washer out"),
                               (d.arm_x + w, hd, 6.0, "Chicago screw cap")):
            _disc(d, at(x0, t), r, t, f"{what} {side}", mats["hardware"])
        ob = _disc(d, at(d.arm_x, P.BOARD), P.EAR_DISC_R, P.BOARD, f"E-RING-{side}", mats["face"],
                   ring_in=P.EAR_RING_IN)
        ob.location = (0.0, -d.pivot[0], -d.pivot[1])
        ob.parent = pivot
    return dict(pivot=pivot, head=head)


def set_open(state, ang):
    # Rotating the pivot about -X lifts the faceplate (phi increases).
    state["pivot"].rotation_euler = (math.radians(-ang), 0.0, 0.0)


def render_views(d, state, out_dir, size=900):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "MATERIAL"
    sc.display.shading.show_object_outline = True
    sc.display.shading.object_outline_color = (0.1, 0.1, 0.1)
    sc.display.shading.show_cavity = True
    sc.display.shading.cavity_type = "WORLD"
    sc.render.resolution_x = sc.render.resolution_y = size
    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.quality = 90
    world = bpy.data.worlds.new("World")
    world.color = (1.0, 1.0, 1.0)
    sc.world = world
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"

    target = bpy.data.objects.new("target", None)
    target.location = (0.0, 0.0, -10.0)
    sc.collection.objects.link(target)
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 330.0
    cam.data.clip_end = 5000.0
    sc.collection.objects.link(cam)
    tc = cam.constraints.new("TRACK_TO")
    tc.target = target
    tc.track_axis = "TRACK_NEGATIVE_Z"
    tc.up_axis = "UP_Y"
    sc.camera = cam
    head = state["head"]

    views = {
        "three_quarter": ((-420.0, -620.0, 260.0), 0.0, True),
        "front": ((0.0, -800.0, 0.0), 0.0, False),
        "side_closed": ((800.0, 0.0, 0.0), 0.0, False),
        "side_open": ((800.0, 0.0, 0.0), P.SWING_ANGLE, False),
        "open_three_quarter": ((-420.0, -620.0, 260.0), P.SWING_ANGLE, True),
        "back": ((300.0, 750.0, 200.0), 0.0, False),
    }
    files = {}
    for name, (pos, ang, show_head) in views.items():
        cam.location = pos
        set_open(state, ang)
        head.hide_render = not show_head and ang == 0.0
        path = os.path.join(out_dir, f"{name}.jpg")
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        files[name] = (path, size, size)
    set_open(state, 0.0)
    head.hide_render = False
    return files
