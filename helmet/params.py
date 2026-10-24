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
EAR_CLEARANCE = 5.0          # outer ear to the inside of the side plates
BACK_EDGE_ABOVE_NAPE = 40.0  # back edge stops this far above the measured nape

# --- Material ----------------------------------------------------------------
BOARD = 1.5                  # chipboard / cereal-box board thickness
SWING_GAP = 1.0              # air gap between faceplate and shell while it swings
SWING_ANGLE = 80.0           # how far the faceplate flips up
FACE_SWING_EXTRA = 0.8       # extra radial margin for the triangulated faceplate's swing

# --- Pivot hardware ----------------------------------------------------------
PIVOT_HOLE = 5.0             # Chicago screw (binding post) diameter
PIVOT_WASHER = 1.0           # washer between the side plate disc and the arm
SIDE_DISC_R = 30.0           # reinforcement disc glued outside each side plate
SIDE_PLATE_R = 82.0          # side plate radius (at least) where the faceplate passes over it
EAR_DISC_R = 36.0            # ear disc / boss on each faceplate arm
EAR_RING_IN = 24.0           # inner radius of the decorative ear rings

# --- Facets ------------------------------------------------------------------
SHELL_FOLDS_X = (15.0, 45.0, 75.0)   # fold lines across the shell (each side)
FACE_FOLDS_X = (15.0, 45.0, 75.0)    # fold lines across the faceplate
SHELL_FRONT_EDGE = 10.0              # shell front edge, degrees above the pivot
SHELL_ROW_STEP = 12.0                # shell fold spacing around the pivot
FACE_ROWS = (17.0, 7.0, -5.0, -15.0, -24.0, -33.0, -42.0, -51.0, -60.0)
EYE_ROW = 1                          # faceplate row band holding the eye slots
SIDE_NOTCH_MARGIN = 1.0              # side plate's jaw lobe stops this far below the faceplate edge

# --- Paper -------------------------------------------------------------------
PAGE_W, PAGE_H = 215.9, 279.4        # US Letter
MARGIN = 10.0                        # keep everything this far from the paper edge
HEADER_H = 13.0
FOOTER_H = 16.0
PIECE_GAP = 4.0                      # space between pieces on a page
TAB_WIDTH = 8.0
