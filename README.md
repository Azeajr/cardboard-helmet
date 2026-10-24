# Cardboard helmet

A flip-up-faceplate helmet built from 6 mm double-wall corrugated cardboard. The 3D model is generated in Blender, checked against a stand-in head built from your measurements, and flattened into full-size stencils: one flat panel per facet, with numbered edges that are hot-glued edge to edge.

## Outputs (`out/`)

| File | What it is |
|---|---|
| `cardboard_helmet_stencils.pdf` | Cover, how-to, piece map, then the full-size stencils (US Letter, print at 100%) |
| `helmet.blend` | The model: head proxy, shell, faceplate on its pivot (rotate the *Faceplate pivot* empty about X to open it), overlays, pivot hardware, ear rings |
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

Clearances, board thickness, swing angle, hardware sizes and panel layout are in the same file.

## What the build checks

1. **Fit**: the shell's inner surface stays `CLEARANCE` from the head proxy everywhere. The head is pushed out along its surface normals, so the gap holds at right angles to the shell, not just straight out from the pivot. The ears and face have their own margins.
2. **Swing**: the faceplate is rotated in 0.5° steps from closed to `SWING_ANGLE`. At each step, both of its board surfaces are tested for intersection with the shell, and the air gap and distance to the head are measured. The flat arms ride on washers against the side plates, so their gap is measured separately.
3. **Vision**: sight lines from each eye are traced through the closed faceplate. A sight line counts only if it passes the eye slot in both surfaces of the board, since 6 mm board cuts off oblique views.
4. **No gaps**: 1,728 rays from inside the closed helmet must all hit board.
5. **Putting it on**: a collision-free path exists to lower the helmet onto the head with the faceplate open.
6. **Panels**: every panel is planar, the pieces are flattened exactly (no distortion), every seam is cut on both sides at equal length, and nothing overlaps.
7. **Joints**: every seam turns outwards, so two square-cut edges meet at their outer corners and leave a V on the inside for hot glue. The V must be no wider than `MAX_JOINT_GAP`.
8. **Slots**: the eye and mouth slots leave at least `MIN_BRIDGE` of board to every panel edge.
9. **The PDF itself**: poppler reads the finished PDF back, and every cut line on every stencil page is measured against the model. The 100 mm bars are measured too.

The checks cover the geometry. They don't cover cutting accuracy, how the board behaves, or how closely the head proxy matches your real head.

## How the geometry guarantees assembly

Each surface is built from columns that are scaled copies of one profile around the pivot axis. Neighbouring columns therefore always form flat trapezoids, and each one is cut as its own panel. Every seam turns outwards, so square-cut edges meet exactly at the outer surface, which is the surface the stencils are taken from. In the forehead region the faceplate slides over, the shell's corners lie on a circle about the pivot axis, and the faceplate (including its 6 mm board) stays outside it. That makes the swing collision-free at any angle by construction; the build then also checks it numerically.

## Code map

`helmet/head.py` head proxy · `design.py` sizing and meshes · `pieces.py` flattening, seam numbers, joints · `model.py` how the meshes are split into panels · `layout.py` page packing · `sheets.py`, `docs.py`, `pdf.py` PDF output · `checks3d.py`, `checks2d.py` verification · `blend.py` Blender scene and renders · `build.py` runs it all.

## License

- **Code** (`build.py`, `build.sh`, `helmet/`): [MIT](LICENSE).
- **Design files** (everything in `out/`: the stencil PDF, `helmet.blend`, the renders and the reports): [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-CC-BY-4.0.txt). You may share, adapt and use them, including commercially, as long as you credit Azeajr and link back to this repository.
