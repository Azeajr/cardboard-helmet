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

The shell's columns are scaled copies of one profile around the pivot axis, so neighbouring columns always form flat trapezoids. The forehead region the faceplate slides over is an exact circle about that axis.

Each faceplate column has its own profile, shaped as close to the face as two limits allow: 15 mm from your face, and staying outside the forehead circle while it swings. Where neighbouring columns differ, each panel is split into two triangles along the diagonal that folds outward, so every panel is flat. That freedom is what gives the tapered Iron Man jaw. The faceplate's swing over the shell is then checked numerically at every 0.5°.

The red/gold split follows the Iron Man jawline. The faceplate facets to paint red are marked RED on the stencils.

## Code map

`helmet/head.py` head proxy · `design.py` sizing and meshes · `pieces.py` flattening, tabs, seam numbers · `model.py` how the meshes are split into pieces · `layout.py` page packing · `sheets.py`, `docs.py`, `pdf.py` PDF output · `checks3d.py`, `checks2d.py` verification · `blend.py` Blender scene and renders · `build.py` runs it all.

## License

- **Code** (`build.py`, `build.sh`, `helmet/`): [MIT](LICENSE).
- **Design files** (everything in `out/`: the stencil PDF, `helmet.blend`, the renders and the reports): [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-CC-BY-4.0.txt). You may share, adapt and use them, including commercially, as long as you credit Azeajr and link back to this repository.
