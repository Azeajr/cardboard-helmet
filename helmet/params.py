"""Every dimension the helmet is built from (millimetres, degrees).

Change values here and re-run ./build.sh: the model, the checks and the
stencil PDF are all regenerated from these numbers.

Axes: X = wearer's left, Y = backwards (the face looks towards -Y), Z = up.
z = 0 is the forehead circumference line (just above the brows).
"""

# --- Your measurements -------------------------------------------------------
HEAD_CIRCUMFERENCE = 550.0   # tape around the forehead, just above the brows
CHIN_TOP_NAPE = 660.0        # tape from the chin, over the top, to the nape
# Optional straight-line measurements; None = estimate from the circumference.
HEAD_LENGTH = None           # between the brows to the back of the head
HEAD_BREADTH = None          # widest side to side, above the ears
HEAD_HEIGHT = None           # top of the head to the bottom of the chin

# --- Fit ---------------------------------------------------------------------
CLEARANCE = 12.0             # head to the inside of the shell
FACE_CLEARANCE = 15.0        # nose, lips and chin to the inside of the faceplate
EAR_CLEARANCE = 5.0          # outer ear to the side plates and the screw heads inside them
BACK_EDGE_ABOVE_NAPE = 40.0  # back edge stops this far above the measured nape

# --- Material ----------------------------------------------------------------
BOARD = 6.0                  # double-wall corrugated cardboard thickness
SWING_GAP = 3.0              # air gap between faceplate and shell while it swings
SWING_ANGLE = 80.0           # how far the faceplate flips up

# --- Pivot hardware ----------------------------------------------------------
# Stack, inside to out: screw head, washer, side plate, washer, arm, washer, cap.
PIVOT_HOLE = 5.0             # Chicago screw (binding post) diameter
PIVOT_WASHER = 1.0           # fender washer thickness (three per pivot)
PIVOT_WASHER_D = 18.0        # fender washer outside diameter
PIVOT_HEAD = 2.0             # Chicago screw head thickness (each end)
SIDE_FRONT_R = 30.0          # side plate front edge: an arc this far around the pivot
SIDE_PLATE_R = 82.0          # side plate radius (at least) where the faceplate passes over it
EAR_DISC_R = 36.0            # rounded end of each faceplate arm around the pivot
EAR_RING_IN = 24.0           # inner radius of the ring glued on each arm around the screw cap

# --- Panels ------------------------------------------------------------------
# Every facet is its own flat panel, joined edge to edge with hot glue.
SHELL_SEAMS_X = (25.0, 65.0)         # seams running front to back over the shell (each side)
FACE_SEAMS_X = (15.0, 76.0, 96.0)    # seams running down the faceplate (each side)
SHELL_FRONT_EDGE = 10.0              # shell front edge, degrees above the pivot
SHELL_ROW_STEP = 30.0                # shell seam spacing around the pivot
FACE_ROWS = (17.0, 9.0, -14.0, -37.0, -60.0)
SIDE_NOTCH_MARGIN = 4.0              # side plate stays this far behind the jaw line
MAX_JOINT_GAP = 8.0                  # widest V allowed inside a joint (filled with hot glue)
MIN_BRIDGE = 6.0                     # narrowest board left between a slot and a panel edge

# --- Paper -------------------------------------------------------------------
PAGE_W, PAGE_H = 215.9, 279.4        # US Letter
MARGIN = 10.0                        # keep everything this far from the paper edge
HEADER_H = 13.0
FOOTER_H = 16.0
PIECE_GAP = 4.0                      # space between pieces on a page
