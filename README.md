# Robotic Arm Chess Engine

A Franka Emika Panda plays chess against a person. Everything is in one file,
[`chess_arm.py`](chess_arm.py):

| Part | Where in `chess_arm.py` |
|---|---|
| Arm and gripper motion (panda-py) | `PandaArm`, `ChessMover` |
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

Install the `panda-python` build that matches the robot's libfranka version (see
the panda-py releases page). FCI must be activated in Franka Desk, and the
brakes unlocked, before running anything.

## Setup (once per station)

```bash
python chess_arm.py auto-calibrate     # find the board, then hover over a1/h1/h8 so you can check it
python chess_arm.py view               # live overlay: tune colours / depth until all squares are green
python chess_arm.py pick-test e2 e4    # move one piece to check the grasp
```

**`play` finds the board by itself every time it starts**, so the board can be moved
between games. It looks first where the board was last time. `--no-calibrate` skips
this and reuses the saved position. Calibration is saved to `chess_calibration.json`
next to the script.

**How the board is found:** the board has yellow corner brackets (at least 3 of the 4
must be visible). The program finds them, then takes the wood-coloured region between
them as the board. The grey breadboard is much less saturated than the wood, so it and
its holes are ignored. The board's outline gives its outer corners. On ranks 4 and 5,
which are empty in the starting position, the outermost light squares show where the
thin border ends. Depth gives the board's height and tilt, and which side the robot sits
on (`--robot-color`) decides which corner is a1. The board can be empty or set up. If the
board doesn't fit in the camera image, the arm moves higher and tries again.

Colours are set for the lab board and pieces, measured from a photo (H 0-180, S/V 0-255):

| | Setting | Measured |
|---|---|---|
| Yellow corners | `CORNER_HSV` | H 24-27, S 230-255, V 220-255 |
| Dark squares + border | `BOARD_DARK_HSV` | H 7-8, S 118-140, V 125-160 |
| Light squares | (anything not dark) | H 13-15, S ~60, V ~210 |
| Light pieces | `WHITE_PIECE_HSV` | H 18-20, S 75-110, V 200-240 |
| Dark pieces | `BLACK_PIECE_HSV` | H 0-11, S 110-230, V 25-120 |

The RealSense sees colours a bit differently from a phone, so check them with `view`.
If the board isn't found, open `board_detect_debug.png` (red = board wood, cyan =
yellow corners). Brown hue wraps around 180, so a range like `(170, ...)` to `(12, ...)`
means 170-180 and 0-12.

Set before the first run:
* `CAMERA_MOUNT_OFFSET`, `CAMERA_MOUNT_YAW_DEG`: where the camera sits on the gripper.
  Any error here shifts the whole board by the same amount.
* `BOARD_GUESS_CENTRE`: roughly where the board is, for the very first search.

Manual alternative: `calibrate-board` (hand-guide the fingertips onto a1, h1, h8), then
`calibrate-camera` (click the 4 outer corners of the playing area).

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

* `ROBOT_IP`
* `PIECE_HEIGHT`, `PIECE_GRIP_WIDTH`, `PIECE_BASE_DIAMETER` (the grasp height is
  `GRASP_HEIGHT_FRACTION` of the piece height)
* `SQUARE_SIZE` (only used to warn if calibration looks wrong)
* `CAPTURE_FIRST_SLOT`: must be reachable and clear of the board
* `VIEW_HEIGHT`: the whole playing area must be in the camera image
* Colour ranges / depth thresholds, using `view`
