# Robotic Arm Chess Engine

A Franka Emika Panda plays chess against a person. Everything is in one file,
[`chess_arm.py`](chess_arm.py):

| Part | Where in `chess_arm.py` |
|---|---|
| Arm and gripper motion (csc376_franky, through the station miniPC) | `FrankyArm`, `ChessMover` |
| Detecting pieces (RealSense on the gripper, **color** or **depth**) | `BoardVision` |
| Correcting the grasp for off-centre / turned pieces (on/off) | `PickAdjuster` |
| Detecting the other player's move | `CameraWatcher`, `infer_move` |
| Choosing the robot's move | `Stockfish` |

All the values you might need to change (robot IP, piece sizes, colour ranges,
heights, speeds) are at the top of the file under **CONFIGURATION**. Values
marked ESTIMATE are guesses; measure your set.

## Install (robot PC)

```bash
pip install -r requirements.txt
sudo apt install stockfish        # or set STOCKFISH_PATH to the executable
```

The arm is driven only through the course robot server on the station's miniPC
(`csc376_franky`), never libfranka directly. Install `csc376_franky` the way the course
says. Each move is planned on this PC, like the course examples: Ruckig makes a smooth
path, the roboticstoolbox Panda model turns it into joint angles, and the miniPC runs it.
The program asks you to type `yes` once before the robot first moves
(`CONFIRM_ROBOT_MOTION`). Speeds are `SPEED_TRAVEL` / `SPEED_APPROACH` (Ruckig fractions)
and `HOME_SPEED_FACTOR` (at most 0.3).
Any move that would need a joint faster than 0.6 rad/s (`JOINT_VEL_TARGET`) or harder than
6 rad/s^2 (`JOINT_ACC_TARGET`) is slowed to exactly that; the miniPC's limits are 0.6525 and 7.5.

## Setup (once per station)

```bash
python chess_arm.py auto-calibrate     # find the board, confirm the view, then hover over the 4 corner squares
python chess_arm.py view               # live overlay: tune colours / depth until all squares are green
python chess_arm.py pick-test e2 e4    # move one piece to check the grasp
```

**`play` finds the board by itself every time it starts**, so the board can be moved
between games. It looks first where the board was last time. `--no-calibrate` skips
this and reuses the saved position. Calibration is saved to `chess_calibration.json`
next to the script.

**How the board is found:** only the 4 yellow L-shaped corner brackets are used. The
board's own colours are not, because they look too much like the table. The inside corner
of each L is a corner of the board. From home, the bottom-left and bottom-right corners
must be in view; the camera shifts forward (`VIEW_FORWARD_STEP`) until all 4 are in the
image, climbing if the near ones drop out before the far ones appear. Only colour is used:
the board's known 34 cm size and where its 4 corners appear in the image give its distance
and tilt (no depth camera). The squares inside must show a checkerboard pattern
(ranks 3-6 are checked, so calibrate before moving pieces there). Corners are named by
where they are in the camera image (`CORNER_LABELS`): bottom-left h8, bottom-right a8,
top-left h1, top-right a1. The playing area starts
`BOARD_BORDER` (1 cm) in from the corners; squares are 4 cm (`SQUARE_SIZE`), so the board is 34 cm.

**The camera then fits itself to the board** in separate steps, each checked with a new
picture: (1) sideways only, until the board is centred in the image; (2) turn only, about
the camera, until the image is square to the board; (3) up/down only, until the yellow
markers just fit (`VIEW_MARGIN`, 2% in from the edges), re-centring after each step. It
steers by the image, so camera-mount or distance errors only make it take an extra step.
If a move loses the corners it tries half and a quarter of the way. You confirm the view (or search again); then it is saved as the viewing pose, and
the arm hovers 10 cm above a1, h1, h8 and a8 (`AUTO_CAL_HOVER`) using the saved positions only.
At each one you type how far off it is (`forward left [high]` in cm, Enter = on target); the
average miss is taken off the board position and learned as a camera-mount correction
(`mount_fix` in the calibration file, used by every later search), then it checks again.

Colours (H 0-180, S/V 0-255), measured from a photo of the lab board and pieces:

| | Setting | Measured |
|---|---|---|
| Yellow corners | `CORNER_HSV` | H 24-27, S 230-255, V 220-255 |
| Light pieces | `WHITE_PIECE_HSV` | H 18-20, S 75-110, V 200-240 |
| Dark pieces | `BLACK_PIECE_HSV` | H 0-11, S 110-230, V 25-120 |

The RealSense sees colours a bit differently from a phone, so check them with `view`.
If the corners aren't found, open `board_detect_debug.png` (cyan = what was taken as
yellow, red circles = the inside corners found). Brown hue wraps around 180, so a range like `(170, ...)` to `(12, ...)`
means 170-180 and 0-12.

Set before the first run:
* `CAMERA_MOUNT_OFFSET`, `CAMERA_MOUNT_YAW_DEG`: where the camera sits on the gripper.
  Any error here shifts the whole board by the same amount.
* `VIEW_FORWARD`: which way is forward (robot towards board). From home, the bottom-left and
  bottom-right yellow corners must be in view; the camera shifts forward from there.

Hand-guiding isn't available through the course server, so there is no manual calibration.

## Play

```bash
python chess_arm.py play                       # finds the board, robot plays black, colour detection
python chess_arm.py --detect depth play        # use the depth sensor instead of colour
python chess_arm.py --no-adjust play           # turn off the off-centre/rotation correction
python chess_arm.py --robot-color white play
python chess_arm.py --manual-moves play        # type your moves (tests the arm without vision)
python chess_arm.py --dry-run --manual-moves --no-adjust play   # no robot: prints arm commands
```

The game always starts from the standard position. The program waits until the
camera sees the starting setup, then:

1. **Your turn:** the arm holds the camera over the board. When the board has been
   still for `STABLE_FRAMES` readings and matches exactly one legal move, that's
   your move. A hand over the board (depth mode) or a half-made move is ignored.
2. **Robot's turn:** Stockfish picks a move. The arm removes a captured piece to the
   capture area beside the h-file, then moves its piece. It handles castling and en
   passant; after a promotion it asks you to swap the pawn for the new piece.
3. With `VERIFY_ROBOT_MOVES`, the camera checks the robot's move. If a piece
   didn't land where it should, the program lists the wrong squares and waits for
   you to fix them.

## Detection modes

Switch with `DETECTION_MODE` or `--detect`.

* **color** (default): a square holds a white/black piece if enough of its centre
  matches `WHITE_PIECE_HSV` / `BLACK_PIECE_HSV`. With `USE_BACKGROUND_SUBTRACTION`,
  the empty light/dark square colours are learned from ranks 3–6 at setup, so square
  colours similar to a piece colour are ignored. Tune the ranges in `view` (right
  panel: white mask = 255, black mask = 128).
* **depth**: a square is occupied if its median height above the board plane is over
  `DEPTH_PIECE_MIN_HEIGHT`. The plane is fitted on the board border every frame.
  This mode only sees *occupied or not*. That is enough, because the move is
  identified by checking which legal move gives that occupancy.

## Pick adjustment

`ENABLE_PICK_ADJUSTMENT` / `--no-adjust`. It works from the same top-down image the
camera uses to watch the board:

* **Off-centre:** the grasp moves to the piece's actual footprint (limited to
  `ADJUST_MAX_OFFSET` of a square).
* **Turned** (e.g. a knight not facing forward): the gripper rotates so the fingers
  close across the piece's narrow side. It puts the piece down with the gripper
  straight, so the piece ends up squared to the board. Limitation: a knight facing
  backward is still placed facing backward, because its outline looks the same
  either way.
* Tall pieces near the edge of the board appear to lean outward in the image.
  The correction accounts for this, using measured heights in depth mode and
  `PIECE_HEIGHT` in colour mode.

## Things to measure and adjust first

* `PIECE_HEIGHT`, `PIECE_GRIP_WIDTH`, `PIECE_BASE_DIAMETER` (the grasp height is
  `GRASP_HEIGHT_FRACTION` of the piece height)
* `SQUARE_SIZE` (only used to warn if calibration looks wrong)
* `CAPTURE_FIRST_SLOT`: must be reachable and clear of the board
* Colour ranges / depth thresholds, using `view`
