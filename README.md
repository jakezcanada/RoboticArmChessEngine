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
python chess_arm.py calibrate-board    # hand-guide the closed fingertips onto a1, h1, h8
python chess_arm.py calibrate-camera   # viewing pose + click the 4 outer corners of the playing area
python chess_arm.py view               # live overlay: tune colours / depth until all squares are green
python chess_arm.py pick-test e2 e4    # move one piece to check the grasp
```

Calibration is saved to `chess_calibration.json` next to the script.

## Play

```bash
python chess_arm.py play                       # robot plays black, colour detection, adjustment on
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
