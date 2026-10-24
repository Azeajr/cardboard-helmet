"""Full build, run inside Blender:  ./build.sh  (or blender -b -P build.py)

Generates the 3D model, checks it against the head proxy, flattens it into
stencils, writes out/helmet.blend, out/cardboard_helmet_stencils.pdf and
out/report.md, then re-measures the PDF. Exits non-zero if any check fails."""

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import bpy  # noqa: E402

from helmet import params as P  # noqa: E402
from helmet import model, layout, sheets, docs, checks2d, checks3d, blend  # noqa: E402
from helmet.design import Design  # noqa: E402
from helmet.pdf import PDF  # noqa: E402

OUT = os.path.join(ROOT, "out")
PDF_PATH = os.path.join(OUT, "cardboard_helmet_stencils.pdf")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def summarize(d, r3, r2, pagechk, pdfm):
    c = r3["clearance"]
    sw = r3["swing"]
    v = r3["vision"]
    items = []
    gap = c["head_to_helmet_min"][0]
    items.append(("Head fit", f"at least {gap:.1f} mm between your head and the inside of the helmet "
                  f"(target {P.CLEARANCE:.0f}); {c['crown_to_shell']:.0f} mm above the crown",
                  gap >= P.CLEARANCE - 1.5))
    ear = c["ears_to_shell_min"][0]
    items.append(("Ears", f"{ear:.1f} mm from the side plates", ear >= P.EAR_CLEARANCE - 0.5))
    fc = c["face_to_faceplate_min"][0]
    items.append(("Face", f"faceplate at least {fc:.1f} mm from your face (target {P.FACE_CLEARANCE:.0f}; "
                  f"nose {c['nose_tip_to_faceplate']:.0f}, chin {c['pogonion_to_faceplate']:.0f} mm)",
                  fc >= P.FACE_CLEARANCE - 1.0))
    ok = not sw["intersections"] and sw["min_gap"] >= P.SWING_GAP - 0.3 and sw["min_head"] > 2.0
    items.append(("Faceplate swing", f"tested at {sw['steps']} angles from closed to {P.SWING_ANGLE:.0f} deg: "
                  f"no contact; tightest gap {sw['min_gap']:.1f} mm; never closer than {sw['min_head']:.0f} mm "
                  f"to your head", ok))
    both = v["eye_l"]["straight_ahead"] and v["eye_r"]["straight_ahead"]
    hl = v["eye_l"]["horizontal"]
    vl = v["eye_l"]["vertical"]
    items.append(("Vision", f"both eyes look straight out through the slots; each eye sees "
                  f"{min(hl) if hl else 0} to {max(hl) if hl else 0} deg sideways and "
                  f"{min(vl) if vl else 0} to {max(vl) if vl else 0} deg up/down", both))
    cv = r3["coverage"]
    items.append(("No gaps", f"{cv['rays']} sight lines from inside all hit the helmet"
                  if cv["missed"] == 0 else f"{cv['missed']} of {cv['rays']} sight lines escape", cv["missed"] == 0))
    dn = r3["donning"]
    items.append(("Putting it on", "a collision-free path exists to lower it onto your head (faceplate open)"
                  if dn["ok"] else f"blocked (heights {dn['blocked_heights']})", dn["ok"]))
    mc = r3["mesh"]
    plan = max(m["max_planarity_mm"] for m in mc.values())
    items.append(("Flat panels", f"every panel flat to {plan:.0e} mm; normals consistent",
                  plan < 0.01 and all(m["inconsistent_normals"] == 0 and m["nonmanifold_edges"] == 0
                                      for m in mc.values())))
    items.append(("Pieces match the model", f"every flattened panel within {r2['face_shape_error_mm']:.0e} mm of "
                  f"its 3D shape; {r2['seams']} seams each cut on both sides with matching lengths; "
                  f"{r2['tabs']} glue tabs", r2["face_shape_error_mm"] < 0.01 and not r2["bad_seams"]
                  and not r2["overlaps"] and r2["orientation_errors"] == 0))
    items.append(("Pages", f"{pagechk['pages']} stencil pages; every piece inside the printable area, none touching",
                  not pagechk["out_of_bounds"] and not pagechk["clashes"]))
    if pdfm is not None:
        err = max(p["max_error_mm"] for p in pdfm)
        bars = [p["scale_bar_mm"] for p in pdfm]
        n = sum(p["found"] for p in pdfm)
        ok = err < 0.02 and all(b is not None and abs(b - 100.0) < 0.01 for b in bars)
        items.append(("Printed PDF re-measured", f"{n} cut lines read back from this PDF match the model "
                      f"within {err:.3f} mm; 100 mm bars measure {min(bars):.2f} mm", ok))
    else:
        items.append(("Printed PDF re-measured", "pending", True))
    return items


def write_pdf(d, pieces, pages, renders, items):
    total = 3 + len(pages)
    pdf = PDF(P.PAGE_W, P.PAGE_H, "Cardboard helmet - full-size stencils", "cardboard-helmet")
    docs.cover(pdf, d, renders, items, total, len(pages))
    docs.instructions(pdf, d, pieces, renders, 2, total)
    docs.piece_map(pdf, d, pieces, 3, total)
    sheets.stencil_pages(pdf, pages, 4, total)
    pdf.save(PDF_PATH)
    return 4


def main():
    os.makedirs(os.path.join(OUT, "renders"), exist_ok=True)
    log("design")
    d = Design()
    log("pieces")
    bodies, pieces, seams = model.build(d)
    groups = [("shell", lambda p: p.kind == "shell"), ("faceplate", lambda p: p.kind == "face"),
              ("overlays and ear discs", lambda p: p.kind in ("overlay", "flat"))]
    pages = layout.pack(pieces, groups)
    log(f"{len(pieces)} pieces on {len(pages)} pages; checking flat pieces")
    r2 = checks2d.unfold_checks(bodies, pieces, seams)
    pagechk = checks2d.page_checks(pages)

    log("blender scene")
    state = blend.build_scene(d)
    renders = blend.render_views(d, state, os.path.join(OUT, "renders"))
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "helmet.blend"))

    r3 = checks3d.run_all(d, log)

    log("pdf")
    items = summarize(d, r3, r2, pagechk, None)
    first = write_pdf(d, pieces, pages, renders, items)
    pdfm = checks2d.measure_pdf(PDF_PATH, first, pages)
    items = summarize(d, r3, r2, pagechk, pdfm)
    write_pdf(d, pieces, pages, renders, items)
    pdfm = checks2d.measure_pdf(PDF_PATH, first, pages)      # re-check the final file
    items = summarize(d, r3, r2, pagechk, pdfm)

    report = dict(design=d.summary(), checks=[dict(name=n, detail=t, ok=o) for n, t, o in items],
                  clearance=r3["clearance"], swing={k: v for k, v in r3["swing"].items() if k != "per_angle"},
                  vision=r3["vision"], coverage=r3["coverage"], donning=r3["donning"], mesh=r3["mesh"],
                  unfold={k: v for k, v in r2.items()}, pages=pagechk, pdf=pdfm,
                  pieces=[dict(name=p.name, title=p.title, count=p.count) for p in pieces])
    with open(os.path.join(OUT, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    with open(os.path.join(OUT, "report.md"), "w") as fh:
        fh.write("# Cardboard helmet - build report\n\n")
        for n, t, o in items:
            fh.write(f"- **{'PASS' if o else 'FAIL'}** {n}: {t}\n")
        fh.write("\n## Design\n\n")
        for k, v in d.summary().items():
            fh.write(f"- {k}: {v}\n")
    failed = [n for n, _, o in items if not o]
    for n, t, o in items:
        log(f"{'PASS' if o else 'FAIL'}  {n}: {t}")
    log(f"wrote {PDF_PATH}")
    if failed:
        log("FAILED: " + ", ".join(failed))
        sys.exit(1)


main()
