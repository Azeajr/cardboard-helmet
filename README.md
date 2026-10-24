# Cardboard helmet

A flip-up-faceplate cardboard helmet. The 3D model is generated in Blender, checked against a stand-in head built from your measurements, and flattened into full-size stencils with numbered glue tabs.

## Outputs (`out/`)

| File | What it is |
|---|---|
| `cardboard_helmet_stencils.pdf` | Cover, how-to, piece map, then the full-size stencils (US Letter, print at 100%) |
| `helmet.blend` | The model: head proxy, shell, faceplate on its pivot (rotate the *Faceplate pivot* empty about X to open it), overlays, ear discs |
| `report.md` / `report.json` | Every check with its measured numbers, plus the design dimensions |
| `renders/` | The images used in the PDF |

## Rebuild

```sh
./build.sh          # ~1 minute on mains power (several on battery); exits non-zero if any check fails
```

Everything comes from `helmet/params.py`. The fit is driven by:

- `HEAD_CIRCUMFERENCE`: tape around the forehead, just above the brows
- `CHIN_TOP_NAPE`: tape from the chin, over the top, to the nape (used for the back edge)
- `HEAD_LENGTH`, `HEAD_BREADTH`, `HEAD_HEIGHT`: optional straight-line measurements. They are `None` by default, so the head is estimated from the circumference. Measuring them makes the fit more accurate. For length, hold a book flat against your forehead and another against the back of your head and measure between them. For height, put a book on top of your head and measure down to the bottom of your chin.

Clearances, board thickness, swing angle, hardware sizes and facet layout are in the same file.

## What the build checks

1. **Fit**: the shell's inner surface stays `CLEARANCE` from the head proxy everywhere. The head is pushed out along its surface normals, so the gap holds at right angles to the shell, not just straight out from the pivot. The ears and face have their own margins.
2. **Swing**: the faceplate is rotated in 0.5° steps from closed to `SWING_ANGLE`. At each step, both of its board surfaces are tested for intersection with the shell, and the air gap and distance to the head are measured.
3. **Vision**: sight lines from each eye are traced through the closed faceplate and tested against the eye-slot outlines.
4. **No gaps**: 1,728 rays from inside the closed helmet must all hit board.
5. **Putting it on**: a collision-free path exists to lower the helmet onto the head with the faceplate open.
6. **Panels**: every panel is planar, the pieces are flattened exactly (no distortion), every seam is cut on both sides at equal length and has a glue tab, and nothing overlaps.
7. **The PDF itself**: poppler reads the finished PDF back, and every cut line on every stencil page is measured against the model. The 100 mm bars are measured too.

The checks cover the geometry. They don't cover cutting and scoring accuracy, how the board behaves, or how closely the head proxy matches your real head.

## How the geometry guarantees assembly

Each surface is built from columns that are scaled copies of one profile around the pivot axis. Neighbouring columns therefore always form flat trapezoids, so cardboard folds into the exact shape. The forehead region the faceplate slides over is an exact circle about the pivot axis, with the faceplate kept outside it. That makes the swing collision-free at any angle by construction; the build then also checks it numerically.

## Code map

`helmet/head.py` head proxy · `design.py` sizing and meshes · `pieces.py` flattening, tabs, seam numbers · `model.py` how the meshes are split into pieces · `layout.py` page packing · `sheets.py`, `docs.py`, `pdf.py` PDF output · `checks3d.py`, `checks2d.py` verification · `blend.py` Blender scene and renders · `build.py` runs it all.
