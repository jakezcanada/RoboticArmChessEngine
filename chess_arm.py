#!/usr/bin/env python3
"""Play chess against a Franka Emika Panda.

One file that runs on the station PC. It:
  * moves the arm and gripper through the course robot server on the station's miniPC
    (csc376_franky; never libfranka directly). Moves are planned here with Ruckig and the
    roboticstoolbox Panda model, then sent to the miniPC as joint trajectories.
  * detects the pieces on the board with the gripper-mounted RealSense,
    either by COLOR (default) or by DEPTH
  * detects when the other player has moved
  * corrects the grasp for a piece that is off-centre or turned
    (PickAdjuster; switch it on/off with ENABLE_PICK_ADJUSTMENT or --no-adjust)
  * asks Stockfish for the robot's move

Workflow (run each step once per setup, then play):
    python chess_arm.py auto-calibrate       find the board with the camera (yellow corners) and check it;
                                             `play` also finds the board by itself every time it starts
    python chess_arm.py view                 live detection overlay, for tuning the variables below
    python chess_arm.py pick-test e2 e4      move one piece, to test grasping
    python chess_arm.py play                 play a game

Useful options (put them before the command):
    --detect color|depth   which detection method to use (default: DETECTION_MODE)
    --no-adjust            turn off the off-centre/rotation correction
    --manual-moves         type the human's moves instead of using the camera
    --dry-run              print arm commands instead of moving (no robot needed)
    --robot-color white    robot plays white (default black)
    --no-calibrate         reuse the saved board position instead of finding the board again

Every value marked ESTIMATE is a guess. Measure your set and change it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.engine
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

CALIBRATION_FILE = Path(__file__).with_name("chess_calibration.json")

# --- Stockfish -------------------------------------------------------------
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "stockfish")  # executable name or full path
STOCKFISH_THINK_TIME = 1.0  # seconds per move
STOCKFISH_SKILL_LEVEL = 20  # 0 (weak) .. 20 (full strength)

# --- Game ------------------------------------------------------------------
ROBOT_COLOR = "black"  # "white" or "black"
DETECTION_MODE = "color"  # "color" or "depth"
ENABLE_PICK_ADJUSTMENT = True  # correct grasps for off-centre / rotated pieces
VERIFY_ROBOT_MOVES = True  # after the robot moves, check the board with the camera

# --- Board (measured on the lab board) ------------------------------------
INCH = 0.0254
SQUARE_SIZE = 0.040  # playing square (m); used to warn if the calibration looks wrong
BOARD_BORDER = 0.010  # border between the yellow corners and the playing squares (m)
BOARD_WIDTH = 8 * SQUARE_SIZE + 2 * BOARD_BORDER  # 0.34 m, outer corner to outer corner

# --- Pieces (ESTIMATES for a set that fits 1.5" squares, in metres) --------
P, N, B, R, Q, K = chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING
PIECE_HEIGHT = {P: 0.030, N: 0.042, B: 0.045, R: 0.035, Q: 0.055, K: 0.065}
PIECE_BASE_DIAMETER = {P: 0.022, N: 0.025, B: 0.024, R: 0.024, Q: 0.027, K: 0.028}
PIECE_GRIP_WIDTH = {P: 0.014, N: 0.016, B: 0.014, R: 0.018, Q: 0.016, K: 0.017}  # at grasp height
GRASP_HEIGHT_FRACTION = 0.5  # grasp this far up the piece
MIN_GRASP_HEIGHT = 0.012  # never put the fingertips lower than this above the board
GRIP_FORCE = 15.0  # N (only used if csc376_franky's Gripper has a grasp() call)
GRIP_SPEED = 0.05  # m/s
GRIP_EPSILON = 0.010  # accepted difference between PIECE_GRIP_WIDTH and the real width
GRIP_SQUEEZE = 0.004  # without grasp(): close this much narrower than the piece, so the fingers press on it
GRIPPER_MAX_WIDTH = 0.08  # Franka Hand fully open
OPEN_CLEARANCE = 0.020  # open this much wider than the piece base before descending

# --- Motion ----------------------------------------------------------------
SAFE_HEIGHT = 0.12  # travel height above the board (clears the king)
PLACE_EXTRA_HEIGHT = 0.003  # release this much higher than the grasp height
# Speeds are Ruckig (velocity, acceleration, jerk) fractions of the Panda's limits, as in the
# csc376_franky examples (they use 0.02, 0.01, 0.05). Start slow; raise only after testing.
SPEED_TRAVEL = (0.05, 0.03, 0.08)  # moves at safe height
SPEED_APPROACH = (0.02, 0.01, 0.05)  # going down to / up from a piece
HOME_SPEED_FACTOR = 0.2  # move_home() speed; the miniPC allows at most 0.3
CONFIRM_ROBOT_MOTION = True  # ask "type [yes]" once before the robot first moves

# --- Capture area (in squares, measured from the centre of a1 along the board axes)
#     Default: two columns beside the h-file, outside the board's border.
CAPTURE_FIRST_SLOT = (9.5, 0.0)  # (files, ranks) from a1; 7 = h-file
CAPTURE_SLOT_SPACING = 1.0  # squares between slots
CAPTURE_SLOTS_PER_COLUMN = 8

# --- Camera ----------------------------------------------------------------
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 1280, 720, 30
WARP_PX_PER_SQUARE = 80  # resolution of the top-down board image
WARP_MARGIN = 0.5  # squares of border kept around the playing area (used to find the board plane)
CAMERA_HEIGHT_FALLBACK = 0.50  # camera height above the board if depth is unavailable

# --- Automatic board calibration (auto-calibrate, and the start of every `play`) --
#     The arm looks down and finds the 4 yellow L-shaped corner brackets. The inside corner of
#     each L is a corner of the board. The board's own colours are NOT used (they look too much
#     like the table). The camera then moves straight above the board centre, turned square to
#     it, at whatever height puts the 4 corners at the edges of the image (VIEW_MARGIN in from them).
VIEW_FORWARD = (1.0, 0.0)  # "forward" in the robot base frame (x, y): from the robot towards the board
VIEW_FORWARD_STEP = 0.04  # from home, the camera shifts forward this much at a time until all 4 corners are seen (m)
VIEW_FORWARD_MAX = 0.40  # give up (and ask) after shifting this far forward (m)
CAMERA_MOUNT_OFFSET = (0.05, 0.0, -0.06)  # ESTIMATE: camera optical centre in the gripper (EE) frame (m). MEASURE IT:
#                                           an error here moves the whole board by the same amount.
CAMERA_MOUNT_YAW_DEG = 0.0  # ESTIMATE: rotation of the camera image x-axis from the gripper x-axis, about the gripper z-axis
CORNER_HSV = ((18, 170, 140), (32, 255, 255))  # yellow brackets (HSV; measured H 24-27, S 230-255, V 220-255)
CORNER_MIN_AREA = 0.0002  # a yellow blob must cover this fraction of the image to count as a bracket
BOARD_BORDER_FRACTION = BOARD_BORDER / BOARD_WIDTH  # 1 cm / 34 cm: the playing area starts this far in from the corners
VIEW_MARGIN = 0.02  # the outer board corners end up this fraction of the image width/height in from its edges
VIEW_SEARCH_STEP = 0.07  # if the near corners leave the image before the far ones come in, the board doesn't fit:
#                          climb this much and keep shifting forward
VIEW_FIT_TOLERANCE = 0.005  # stop adjusting the view when the next move would be shorter than this (m)
#  If the corners aren't found, look at board_detect_debug.png (cyan = what was taken as yellow).
AUTO_CAL_HOVER = 0.010  # check height: the fingertips stop this far above a1, h1 and h8

# --- COLOR detection: these are the variables to tune (HSV, OpenCV ranges H 0-180, S/V 0-255)
#     Use `python chess_arm.py view` to see the result live.
#     Measured from a photo of the lab set: light-wood pieces H 18-20, S 75-110, V 200-240;
#     dark-wood pieces H 0-11, S 110-230, V 25-120. Light squares are H 13-15, S ~60;
#     dark squares are brighter than the dark pieces (V 125-160). Hue ranges may wrap around 180.
WHITE_PIECE_HSV = ((16, 70, 150), (24, 150, 255))  # light-wood pieces
BLACK_PIECE_HSV = ((170, 70, 10), (12, 255, 120))  # dark-wood pieces
COLOR_MIN_FRACTION = 0.20  # fraction of a square's sample area that must match a piece color
SAMPLE_RADIUS = 0.35  # sample disk radius, as a fraction of a square
USE_BACKGROUND_SUBTRACTION = True  # ignore pixels that look like the empty square (learned at setup)
BACKGROUND_MIN_DIFF = 25.0  # CIELAB distance from the empty-square colour to count as "piece"

# --- DEPTH detection -------------------------------------------------------
DEPTH_PIECE_MIN_HEIGHT = 0.012  # a square is occupied if its median height exceeds this
HAND_MIN_HEIGHT = 0.10  # anything taller than this over the board is a hand / arm
HAND_MIN_FRACTION = 0.01  # fraction of the board area that must be that tall

# --- Move detection --------------------------------------------------------
STABLE_FRAMES = 8  # identical readings in a row before a board is trusted
FRAME_PERIOD = 0.10  # seconds between readings
MAX_CHANGED_SQUARES = 6  # more changes than this = something is in the way, keep waiting

# --- Pick adjustment -------------------------------------------------------
ADJUST_MAX_OFFSET = 0.45  # largest correction, as a fraction of a square
ADJUST_MIN_ELONGATION = 1.25  # only rotate for pieces whose footprint is this elongated (knights)
ADJUST_MAX_YAW_DEG = 60.0  # largest gripper rotation


# =============================================================================
# GEOMETRY / CALIBRATION
# =============================================================================


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


@dataclass
class Calibration:
    a1: np.ndarray | None = None  # fingertip position touching the centre of a1 (robot base frame)
    h1: np.ndarray | None = None
    h8: np.ndarray | None = None
    R_down: np.ndarray | None = None  # gripper orientation pointing down (from move_to_start)
    q_view: np.ndarray | None = None  # joint angles of the camera viewing pose
    image_corners: np.ndarray | None = None  # pixel corners of the playing area: a1, h1, h8, a8

    @classmethod
    def load(cls, path=CALIBRATION_FILE) -> "Calibration":
        if not Path(path).exists():
            return cls()
        d = json.loads(Path(path).read_text())
        return cls(**{k: (np.array(v) if v is not None else None) for k, v in d.items()})

    def save(self, path=CALIBRATION_FILE):
        d = {k: (v.tolist() if v is not None else None) for k, v in self.__dict__.items()}
        Path(path).write_text(json.dumps(d, indent=2))
        print(f"Saved {path}")

    def require(self, *names):
        missing = [n for n in names if getattr(self, n) is None]
        if missing:
            sys.exit(f"Calibration is missing {missing}. Run auto-calibrate first.")


class BoardFrame:
    """Square and capture-slot positions in the robot base frame, from the taught points.

    Board axes: ex runs a-file -> h-file, ey runs rank 1 -> rank 8, ez points up.
    """

    def __init__(self, cal: Calibration):
        cal.require("a1", "h1", "h8")
        self.origin = cal.a1
        ex = cal.h1 - cal.a1
        self.square = np.linalg.norm(ex) / 7
        ex /= np.linalg.norm(ex)
        ey = cal.h8 - cal.h1
        ey -= ex * ex.dot(ey)
        ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        if ez[2] < 0:
            sys.exit("Board calibration is mirrored: touch a1, then h1, then h8.")
        self.ex, self.ey, self.ez = ex, ey, ez
        if abs(self.square - SQUARE_SIZE) > 0.1 * SQUARE_SIZE:
            print(f"WARNING: measured square size {self.square * 1000:.1f} mm, expected {SQUARE_SIZE * 1000:.1f} mm")

    def point(self, files: float, ranks: float, height: float = 0.0) -> np.ndarray:
        """Base-frame point at (files, ranks) squares from the centre of a1, ``height`` above the board."""
        return self.origin + self.square * (files * self.ex + ranks * self.ey) + height * self.ez

    def board_vector(self, dfiles: float, dranks: float) -> np.ndarray:
        """Base-frame vector for a displacement measured in squares."""
        return self.square * (dfiles * self.ex + dranks * self.ey)

    def location(self, loc, height: float = 0.0) -> np.ndarray:
        """A location is a square index or ("capture", slot)."""
        if isinstance(loc, tuple):
            col, row = divmod(loc[1], CAPTURE_SLOTS_PER_COLUMN)
            f = CAPTURE_FIRST_SLOT[0] + col * CAPTURE_SLOT_SPACING
            r = CAPTURE_FIRST_SLOT[1] + row * CAPTURE_SLOT_SPACING
            return self.point(f, r, height)
        return self.point(chess.square_file(loc), chess.square_rank(loc), height)

    def centre(self, height: float = 0.0) -> np.ndarray:
        return self.point(3.5, 3.5, height)


def loc_name(loc) -> str:
    return f"capture[{loc[1]}]" if isinstance(loc, tuple) else chess.square_name(loc)


# =============================================================================
# ARM (csc376_franky: the course robot server on the station's miniPC)
# =============================================================================


class FrankyArm:
    """The real arm, only through csc376_franky (the miniPC's robot server).

    Each gripper move is planned here like the course examples do: Ruckig makes a smooth
    Cartesian path, the roboticstoolbox Panda model turns it into joint angles (starting from
    the angles read just before, because the miniPC refuses a trajectory if the robot moved
    since), and the miniPC runs it."""

    def __init__(self, R_down: np.ndarray | None):
        import roboticstoolbox as rtb
        from csc376_franky import FrankaJointTrajectoryController, Gripper
        from csc376_franky.motion_generator import RuckigMotionGenerator

        self.robot = FrankaJointTrajectoryController()  # this station's miniPC
        self.gripper = Gripper()
        self.model = rtb.models.Panda()
        self.planner = RuckigMotionGenerator()
        self.R_down = R_down
        self._check_tool_frame()
        if CONFIRM_ROBOT_MOTION:
            print("The robot will now move by itself (home, over the board, picking pieces). Check the way is clear.")
            if input("To allow the real robot to move, type [yes], then press enter\n") != "yes":
                sys.exit("User did not type [yes]; not moving the real robot.")

    def _check_tool_frame(self):
        """Warn if the robot's end-effector frame and the model's differ (fingertip positions would be off)."""
        try:
            O_T_EE = np.array(self.robot.get_current_robot_state().O_T_EE).reshape(4, 4).T
        except Exception as e:  # noqa: BLE001 - only a check
            print(f"  (couldn't read the robot's end-effector pose to compare: {e})")
            return
        gap = np.linalg.norm(O_T_EE[:3, 3] - self.pose[:3, 3])
        if gap > 0.005:
            print(f"WARNING: the robot's end-effector is {gap * 1000:.0f} mm from the roboticstoolbox model's "
                  "fingertip point. Positions will be off by about that much; check the end-effector set in Desk.")

    # state
    @property
    def q(self) -> np.ndarray:
        return np.array(self.robot.get_current_joint_positions(), float)

    @property
    def pose(self) -> np.ndarray:
        """Fingertip pose (4x4, base frame) from the joint angles and the roboticstoolbox Panda model."""
        return np.array(self.model.fkine(self.q).A)

    # motion
    def home(self):
        self.robot.move_home(HOME_SPEED_FACTOR)

    def move_joints(self, q, speed=SPEED_TRAVEL):
        """Go to the gripper pose these joint angles give (there's no plain joint move in the examples)."""
        self._move_pose(np.array(self.model.fkine(np.asarray(q, float)).A), speed)

    def move_to(self, position, yaw: float = 0.0, speed=SPEED_TRAVEL):
        """Move the fingertips to ``position`` pointing down, turned ``yaw`` rad about vertical."""
        if self.R_down is None:
            sys.exit("No gripper-down orientation; run auto-calibrate first.")
        T = np.eye(4)
        T[:3, :3] = rot_z(yaw) @ self.R_down
        T[:3, 3] = position
        self._move_pose(T, speed)

    def _move_pose(self, T, speed):
        from spatialmath import SE3, SO3

        q0 = self.q  # read right before planning
        start = self.model.fkine(q0)
        R = T[:3, :3]
        u, _, vt = np.linalg.svd(R)  # keep the rotation exactly orthonormal for spatialmath
        target = SE3.Rt(SO3(u @ vt, check=False), T[:3, 3])
        if np.linalg.norm(start.t - target.t) < 0.0005 and np.linalg.norm(start.R - target.R) < 1e-3:
            return  # already there
        traj, dt = self.planner.calculate_cartesian_pose_trajectory(start, target, *speed)
        q_traj = self.planner.cartesian_pose_to_joint_trajectory(self.model, q0, traj)
        self.robot.run_joint_trajectory(q_traj, dt)

    def teaching_mode(self, on: bool):
        sys.exit("Hand-guiding isn't available through the course robot server.")

    # gripper
    def open(self, width: float):
        self.gripper.move(min(max(width, 0.0), GRIPPER_MAX_WIDTH), GRIP_SPEED * 2)

    def grasp(self, width: float) -> bool:
        """Close on a piece about ``width`` wide. True if it is held."""
        g = self.gripper
        if hasattr(g, "grasp"):  # not in the examples; used if the course Gripper has it (libfranka-style)
            try:
                return bool(g.grasp(width, GRIP_SPEED, GRIP_FORCE, GRIP_EPSILON, GRIP_EPSILON))
            except TypeError:
                pass
        try:
            g.move(max(width - GRIP_SQUEEZE, 0.0), GRIP_SPEED)
        except Exception:  # noqa: BLE001 - a move that stops on the piece may report failure
            pass
        st = g.state
        return bool(getattr(st, "is_grasped", False)) or st.width > width - GRIP_EPSILON


class DryRunArm:
    """Prints the commands instead of moving. For checking the program without the robot."""

    def __init__(self, R_down=None):
        self.R_down = R_down if R_down is not None else np.diag([1.0, -1.0, -1.0])
        self._pose = np.eye(4)
        self._pose[:3, :3] = self.R_down
        self._q = np.zeros(7)

    q = property(lambda self: self._q.copy())
    pose = property(lambda self: self._pose.copy())

    def home(self):
        print("  [dry] home")

    def move_joints(self, q, speed=SPEED_TRAVEL):
        self._q = np.asarray(q, float)
        print("  [dry] joints", np.round(self._q, 3))

    def move_to(self, position, yaw=0.0, speed=SPEED_TRAVEL):
        self._pose[:3, 3] = position
        print(f"  [dry] move_to {np.round(position, 4)} yaw {math.degrees(yaw):.1f} deg")

    def teaching_mode(self, on):
        print(f"  [dry] teaching mode {'on' if on else 'off'}")

    def open(self, width):
        print(f"  [dry] open {width * 1000:.0f} mm")

    def grasp(self, width):
        print(f"  [dry] grasp {width * 1000:.0f} mm")
        return True


# =============================================================================
# CAMERA (RealSense on the gripper)
# =============================================================================


class RealSenseCamera:
    def __init__(self):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
        cfg.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)
        profile = self.pipe.start(cfg)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)  # depth pixels line up with colour pixels
        i = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = (i.fx, i.fy, i.ppx, i.ppy)
        for _ in range(15):  # let auto-exposure settle
            self.pipe.wait_for_frames()

    def read(self):
        """Return (colour BGR image, depth in metres (0 = no reading), principal point (u, v))."""
        frames = self.align.process(self.pipe.wait_for_frames())
        c, d = frames.get_color_frame(), frames.get_depth_frame()
        intr = c.profile.as_video_stream_profile().intrinsics
        color = np.asanyarray(c.get_data()).copy()
        depth = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale
        return color, depth, (intr.ppx, intr.ppy)

    def close(self):
        self.pipe.stop()


# =============================================================================
# BOARD DETECTION (colour or depth)
# =============================================================================

M_PX = WARP_MARGIN * WARP_PX_PER_SQUARE
WARP_SIZE = int(round((8 + 2 * WARP_MARGIN) * WARP_PX_PER_SQUARE))


def square_centre_px(sq: int) -> np.ndarray:
    """Pixel centre of a square in the top-down image (rank 8 at the top, a-file on the left)."""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return np.array([M_PX + (f + 0.5) * WARP_PX_PER_SQUARE, M_PX + (7 - r + 0.5) * WARP_PX_PER_SQUARE])


def is_light(sq: int) -> bool:
    return (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1


def hsv_mask(hsv, rng) -> np.ndarray:
    """Boolean mask of pixels inside an HSV range. If low hue > high hue the range wraps past 180."""
    (h0, s0, v0), (h1, s1, v1) = rng
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_ok = (h >= h0) & (h <= h1) if h0 <= h1 else (h >= h0) | (h <= h1)
    return hue_ok & (sat >= s0) & (sat <= s1) & (val >= v0) & (val <= v1)


@dataclass
class Observation:
    color: np.ndarray  # top-down colour image
    height: np.ndarray  # top-down height above the board (m, NaN = unknown)
    nadir: np.ndarray  # top-down pixel directly below the camera
    camera_height: float  # metres above the board
    squares: dict = field(default_factory=dict)  # square -> "w" / "b" (colour) or "?" (depth)
    hand: bool = False  # something tall is over the board


class BoardVision:
    """Turns a camera frame into which squares hold which pieces."""

    def __init__(self, camera: RealSenseCamera, cal: Calibration, mode: str):
        import cv2

        cal.require("image_corners")
        self.cv2 = cv2
        self.camera = camera
        self.mode = mode
        P_ = WARP_PX_PER_SQUARE
        dst = np.float32([[M_PX, M_PX + 8 * P_], [M_PX + 8 * P_, M_PX + 8 * P_], [M_PX + 8 * P_, M_PX], [M_PX, M_PX]])
        self.H = cv2.getPerspectiveTransform(np.float32(cal.image_corners), dst)
        self.background = None  # (light Lab mean, dark Lab mean)
        r = SAMPLE_RADIUS * P_
        ys, xs = np.mgrid[-int(r) : int(r) + 1, -int(r) : int(r) + 1]
        self.disk = (xs**2 + ys**2) <= r * r
        self.disk_r = int(r)
        inner = np.zeros((WARP_SIZE, WARP_SIZE), bool)
        a, b = int(M_PX), int(M_PX + 8 * P_)
        inner[a:b, a:b] = True
        self.inner, self.ring = inner, ~inner

    # --- frames -------------------------------------------------------------

    def observe(self) -> Observation:
        cv2 = self.cv2
        color, depth, pp = self.camera.read()
        size = (WARP_SIZE, WARP_SIZE)
        wc = cv2.warpPerspective(color, self.H, size, flags=cv2.INTER_LINEAR)
        wd = cv2.warpPerspective(depth, self.H, size, flags=cv2.INTER_NEAREST)
        nadir = cv2.perspectiveTransform(np.float32([[pp]]), self.H)[0, 0]
        height, cam_h = self._height_map(wd, nadir)
        obs = Observation(wc, height, nadir, cam_h)
        obs.squares, obs.hand = self.classify(obs)
        return obs

    def _height_map(self, wd, nadir):
        """Fit the board plane on the border ring; height = plane depth - measured depth."""
        v, u = np.nonzero(self.ring & (wd > 0))
        if len(u) < 200:
            return np.full(wd.shape, np.nan, np.float32), CAMERA_HEIGHT_FALLBACK
        A = np.column_stack([u, v, np.ones_like(u)]).astype(np.float64)
        z = wd[v, u].astype(np.float64)
        keep = np.ones(len(z), bool)
        for _ in range(3):  # least squares, dropping outliers (pieces in the capture area etc.)
            coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
            keep = np.abs(A @ coef - z) < 0.008
        yy, xx = np.mgrid[0 : wd.shape[0], 0 : wd.shape[1]]
        plane = coef[0] * xx + coef[1] * yy + coef[2]
        height = np.where(wd > 0, plane - wd, np.nan).astype(np.float32)
        cam_h = float(coef[0] * nadir[0] + coef[1] * nadir[1] + coef[2])
        return height, cam_h if 0.1 < cam_h < 1.5 else CAMERA_HEIGHT_FALLBACK

    # --- colour masks -----------------------------------------------------------

    def color_masks(self, obs: Observation):
        """Boolean images: pixels that look like a white piece / a black piece."""
        cv2 = self.cv2
        hsv = cv2.cvtColor(obs.color, cv2.COLOR_BGR2HSV)
        white = hsv_mask(hsv, WHITE_PIECE_HSV)
        black = hsv_mask(hsv, BLACK_PIECE_HSV)
        if USE_BACKGROUND_SUBTRACTION and self.background is not None:
            lab = cv2.cvtColor(obs.color, cv2.COLOR_BGR2LAB).astype(np.float32)
            diff = np.linalg.norm(lab - self._background_image(), axis=2) > BACKGROUND_MIN_DIFF
            white &= diff
            black &= diff
        return white, black

    def _background_image(self):
        if getattr(self, "_bg_img", None) is None:
            img = np.zeros((WARP_SIZE, WARP_SIZE, 3), np.float32)
            img[:] = (self.background[0] + self.background[1]) / 2  # border: average
            half = WARP_PX_PER_SQUARE / 2
            for sq in chess.SQUARES:
                cx, cy = square_centre_px(sq)
                img[int(cy - half) : int(cy + half), int(cx - half) : int(cx + half)] = self.background[0 if is_light(sq) else 1]
            self._bg_img = img
        return self._bg_img

    def learn_background(self, obs: Observation):
        """Learn the colour of empty light and dark squares from ranks 3-6 (empty at the start)."""
        cv2 = self.cv2
        lab = cv2.cvtColor(obs.color, cv2.COLOR_BGR2LAB).astype(np.float32)
        light, dark = [], []
        for sq in chess.SQUARES:
            if 2 <= chess.square_rank(sq) <= 5:
                (light if is_light(sq) else dark).append(self._sample(lab, sq).mean(axis=0))
        self.background = (np.mean(light, axis=0), np.mean(dark, axis=0))
        self._bg_img = None

    def _sample(self, img, sq):
        cx, cy = square_centre_px(sq).astype(int)
        r = self.disk_r
        return img[cy - r : cy + r + 1, cx - r : cx + r + 1][self.disk]

    # --- per-square classification --------------------------------------------

    def classify(self, obs: Observation):
        squares = {}
        hand = False
        if self.mode == "color":
            white, black = self.color_masks(obs)
            for sq in chess.SQUARES:
                fw, fb = self._sample(white, sq).mean(), self._sample(black, sq).mean()
                if max(fw, fb) >= COLOR_MIN_FRACTION:
                    squares[sq] = "w" if fw >= fb else "b"
        else:
            h = obs.height
            tall = np.nan_to_num(h[self.inner], nan=0.0) > HAND_MIN_HEIGHT
            hand = tall.mean() > HAND_MIN_FRACTION
            for sq in chess.SQUARES:
                s = self._sample(h, sq)
                if np.isfinite(s).sum() > 0.3 * s.size and np.nanmedian(s) > DEPTH_PIECE_MIN_HEIGHT:
                    squares[sq] = "?"
        return squares, hand

    def piece_mask(self, obs: Observation, piece: chess.Piece) -> np.ndarray:
        """Pixels belonging to pieces of this colour (colour mode) or to any piece (depth mode)."""
        if self.mode == "color":
            white, black = self.color_masks(obs)
            return white if piece.color == chess.WHITE else black
        return np.nan_to_num(obs.height, nan=0.0) > DEPTH_PIECE_MIN_HEIGHT

    # --- display ----------------------------------------------------------------

    def draw(self, obs: Observation, expected: dict | None = None):
        cv2 = self.cv2
        img = obs.color.copy()
        for sq in chess.SQUARES:
            cx, cy = square_centre_px(sq).astype(int)
            got = obs.squares.get(sq)
            exp = expected.get(sq) if expected is not None else got
            ok = got == exp if self.mode == "color" else (got is None) == (exp is None)
            col = (0, 200, 0) if ok else (0, 0, 255)
            cv2.circle(img, (cx, cy), self.disk_r, col, 1)
            if got:
                cv2.putText(img, got, (cx - 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        if obs.hand:
            cv2.putText(img, "HAND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img


def expected_squares(board: chess.Board, mode: str) -> dict:
    """What the camera should see for this position."""
    if mode == "color":
        return {sq: "w" if p.color == chess.WHITE else "b" for sq, p in board.piece_map().items()}
    return {sq: "?" for sq in board.piece_map()}


def differences(observed: dict, expected: dict, mode: str) -> list[int]:
    if mode == "color":
        return sorted(sq for sq in chess.SQUARES if observed.get(sq) != expected.get(sq))
    return sorted(set(observed) ^ set(expected))


def infer_move(board: chess.Board, observed: dict, mode: str) -> chess.Move | None:
    """The legal move whose result matches what the camera sees, else None.

    Colour mode compares piece colours per square, and falls back to occupancy if
    exactly one move matches that way. Promotions are assumed to be to a queen.
    """
    exact, by_occupancy = [], []
    for move in board.legal_moves:
        board.push(move)
        exp = expected_squares(board, mode)
        board.pop()
        if exp == observed:
            exact.append(move)
        elif set(exp) == set(observed):
            by_occupancy.append(move)
    matches = exact or (by_occupancy if len({(m.from_square, m.to_square) for m in by_occupancy}) == 1 else [])
    if not matches:
        return None
    for m in matches:
        if m.promotion in (None, chess.QUEEN):
            return m
    return matches[0]


# =============================================================================
# PICK ADJUSTMENT (detached; ENABLE_PICK_ADJUSTMENT / --no-adjust)
# =============================================================================


@dataclass
class Correction:
    offset: np.ndarray  # base-frame xyz shift of the grasp point
    yaw: float = 0.0  # gripper rotation about vertical (rad)

    @staticmethod
    def none():
        return Correction(np.zeros(3), 0.0)


class PickAdjuster:
    """Finds where a piece really is inside its square, and how it is turned.

    Uses the top-down image from the viewing pose. The camera looks down from
    above the board centre, so tall pieces near the edge appear to lean outward
    (parallax); each pixel is moved back toward the point below the camera by
    its height (measured in depth mode, half the piece height in colour mode).

    * Position: centroid of the piece's footprint vs the square centre.
    * Rotation: if the footprint is elongated (a knight), turn the gripper so the
      fingers close across its narrow side. The piece is then put down with the
      gripper straight again, so it ends up squared to the board.
    """

    def __init__(self, vision: BoardVision | None, board: BoardFrame, R_down: np.ndarray, enabled: bool):
        self.vision, self.board, self.enabled = vision, board, enabled and vision is not None
        self.closing_axis = R_down[:, 1]  # Panda fingers move along the gripper's y axis

    def correction(self, obs: Observation | None, loc, piece: chess.Piece) -> Correction:
        if not self.enabled or obs is None or isinstance(loc, tuple):
            return Correction.none()
        P_ = WARP_PX_PER_SQUARE
        centre = square_centre_px(loc)
        x0, y0 = np.maximum(centre - 1.3 * P_, 0).astype(int)  # room for a piece's outward lean
        x1, y1 = (centre + 1.3 * P_).astype(int)
        mask = self.vision.piece_mask(obs, piece)[y0:y1, x0:x1].astype(np.uint8)
        n, labels, stats, cents = self.vision.cv2.connectedComponentsWithStats(mask)
        best, best_d = None, 0.7 * P_
        for i in range(1, n):
            if stats[i, 4] < 0.05 * P_ * P_:
                continue
            d = np.linalg.norm(cents[i] + (x0, y0) - centre)
            if d < best_d:
                best, best_d = i, d
        if best is None:
            print(f"  adjust: no piece found on {chess.square_name(loc)}, using square centre")
            return Correction.none()

        blob = (labels == best).astype(np.uint8)
        Hc = obs.camera_height
        if self.vision.mode == "depth":
            # every pixel has a measured height: move it back to where it sits on the board
            vs, us = np.nonzero(blob)
            pts = np.column_stack([us + x0, vs + y0]).astype(np.float64)
            z = np.clip(np.nan_to_num(obs.height[vs + y0, us + x0], nan=0.0), 0, Hc * 0.5)
            pts = obs.nadir + (pts - obs.nadir) * ((Hc - z) / Hc)[:, None]
        else:
            # The silhouette is the footprint smeared outward (away from the camera) by up to
            # L pixels. Keep only pixels that are still inside the silhouette when shifted
            # outward by L: that is the footprint on the board.
            vs, us = np.nonzero(blob)
            c = np.array([us.mean() + x0, vs.mean() + y0])
            radial = c - obs.nadir
            dist = np.linalg.norm(radial)
            h = PIECE_HEIGHT[piece.piece_type]
            L = dist * h / (Hc - h)
            if L >= 1:
                sx, sy = radial / dist * L
                shifted = self.vision.cv2.warpAffine(blob, np.float32([[1, 0, -sx], [0, 1, -sy]]),
                                                     (blob.shape[1], blob.shape[0]), flags=self.vision.cv2.INTER_NEAREST)
                foot = blob & shifted
                if foot.sum() > 0.2 * blob.sum():
                    vs, us = np.nonzero(foot)
            pts = np.column_stack([us + x0, vs + y0]).astype(np.float64)

        d_px = pts.mean(axis=0) - centre
        dfiles, dranks = d_px[0] / P_, -d_px[1] / P_  # image y points toward rank 1
        mag = math.hypot(dfiles, dranks)
        if mag > ADJUST_MAX_OFFSET:
            print(f"  adjust: offset {mag:.2f} squares is large, limiting to {ADJUST_MAX_OFFSET}")
            dfiles, dranks = dfiles * ADJUST_MAX_OFFSET / mag, dranks * ADJUST_MAX_OFFSET / mag
        offset = self.board.board_vector(dfiles, dranks)

        yaw = 0.0
        cov = np.cov((pts - pts.mean(axis=0)).T)
        evals, evecs = np.linalg.eigh(cov)  # ascending
        if evals[0] > 0 and math.sqrt(evals[1] / evals[0]) >= ADJUST_MIN_ELONGATION:
            minor = evecs[:, 0]
            want = self.board.board_vector(minor[0], -minor[1])
            have = self.closing_axis
            yaw = math.atan2(have[0] * want[1] - have[1] * want[0], have[0] * want[0] + have[1] * want[1])
            yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2  # gripper is symmetric
            yaw = float(np.clip(yaw, -math.radians(ADJUST_MAX_YAW_DEG), math.radians(ADJUST_MAX_YAW_DEG)))
        print(
            f"  adjust {chess.square_name(loc)}: shift {np.linalg.norm(offset) * 1000:.1f} mm, "
            f"rotate {math.degrees(yaw):.0f} deg"
        )
        return Correction(offset, yaw)


# =============================================================================
# MOTION: pick and place pieces
# =============================================================================


@dataclass
class Transfer:
    src: object  # square or ("capture", slot)
    dst: object
    piece: chess.Piece
    note: str = ""


def plan_transfers(board: chess.Board, move: chess.Move, capture_slot: int) -> list[Transfer]:
    """Physical piece moves for a chess move (board = position before the move).

    A captured piece is removed before the capturing piece moves in.
    Promotion: the pawn is moved; the swap for a queen is done by a person.
    """
    out = []
    if board.is_en_passant(move):
        victim = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        out.append(Transfer(victim, ("capture", capture_slot), board.piece_at(victim), "en passant capture"))
    elif board.is_capture(move):
        out.append(Transfer(move.to_square, ("capture", capture_slot), board.piece_at(move.to_square), "capture"))
    out.append(Transfer(move.from_square, move.to_square, board.piece_at(move.from_square), board.san(move)))
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        f_from, f_to = (7, 5) if board.is_kingside_castling(move) else (0, 3)
        rook_sq = chess.square(f_from, rank)
        out.append(Transfer(rook_sq, chess.square(f_to, rank), board.piece_at(rook_sq), "castling rook"))
    return out


class ChessMover:
    def __init__(self, arm, board: BoardFrame, adjuster: PickAdjuster, cal: Calibration):
        self.arm, self.board, self.adjuster, self.cal = arm, board, adjuster, cal

    def go_view(self):
        if self.cal.q_view is not None:
            self.arm.move_joints(self.cal.q_view)
        else:
            sys.exit("No viewing pose saved; run auto-calibrate first.")

    @staticmethod
    def grasp_height(piece: chess.Piece) -> float:
        return max(MIN_GRASP_HEIGHT, GRASP_HEIGHT_FRACTION * PIECE_HEIGHT[piece.piece_type])

    def pick(self, loc, piece: chess.Piece, corr: Correction) -> bool:
        target = self.board.location(loc) + corr.offset
        gh = self.grasp_height(piece)
        self.arm.open(PIECE_BASE_DIAMETER[piece.piece_type] + OPEN_CLEARANCE)
        self.arm.move_to(target + self.board.ez * SAFE_HEIGHT, corr.yaw, SPEED_TRAVEL)
        self.arm.move_to(target + self.board.ez * gh, corr.yaw, SPEED_APPROACH)
        ok = self.arm.grasp(PIECE_GRIP_WIDTH[piece.piece_type])
        self.arm.move_to(target + self.board.ez * SAFE_HEIGHT, corr.yaw, SPEED_APPROACH)
        return ok

    def place(self, loc, piece: chess.Piece):
        target = self.board.location(loc)
        h = self.grasp_height(piece) + PLACE_EXTRA_HEIGHT
        self.arm.move_to(target + self.board.ez * SAFE_HEIGHT, 0.0, SPEED_TRAVEL)  # yaw 0: squared up
        self.arm.move_to(target + self.board.ez * h, 0.0, SPEED_APPROACH)
        self.arm.open(PIECE_BASE_DIAMETER[piece.piece_type] + OPEN_CLEARANCE)
        self.arm.move_to(target + self.board.ez * SAFE_HEIGHT, 0.0, SPEED_APPROACH)

    def move_piece(self, t: Transfer, obs: Observation | None):
        print(f"  arm: {loc_name(t.src)} -> {loc_name(t.dst)}  {t.note}")
        corr = self.adjuster.correction(obs, t.src, t.piece)
        for attempt in range(2):
            if self.pick(t.src, t.piece, corr):
                break
            print("  grasp failed" + (", retrying at the square centre" if attempt == 0 else ""))
            corr = Correction.none()
        else:
            self.arm.open(0.08)
            input(f"  Could not pick up the piece on {loc_name(t.src)}. Put it in the gripper's path or "
                  f"move it to {loc_name(t.dst)} by hand, then press Enter.")
            return
        self.place(t.dst, t.piece)


# =============================================================================
# MOVE SOURCES: camera (real) or keyboard (testing)
# =============================================================================


class CameraWatcher:
    """Watches the board from the viewing pose and reports the human's move."""

    def __init__(self, vision: BoardVision):
        self.vision = vision
        self.last_obs: Observation | None = None

    def stable(self):
        """Yield an observation each time the reading has been unchanged for STABLE_FRAMES frames."""
        last, count = None, 0
        while True:
            obs = self.vision.observe()
            self.last_obs = obs
            key = None if obs.hand else tuple(sorted(obs.squares.items()))
            count = count + 1 if key is not None and key == last else 0
            last = key
            if count >= STABLE_FRAMES - 1:
                yield obs
            time.sleep(FRAME_PERIOD)

    def wait_for_board(self, board: chess.Board, message: str, learn_background=False):
        """Block until the camera sees exactly this position; print wrong squares meanwhile."""
        mode = self.vision.mode
        expected = expected_squares(board, mode)
        reported = None
        for obs in self.stable():
            if learn_background and mode == "color":
                self.vision.learn_background(obs)
                obs.squares, obs.hand = self.vision.classify(obs)
            wrong = differences(obs.squares, expected, mode)
            if not wrong:
                return obs
            if wrong != reported:
                print(f"{message}; these squares look wrong: {' '.join(chess.square_name(s) for s in wrong)}")
                reported = wrong

    def wait_for_move(self, board: chess.Board) -> chess.Move:
        mode = self.vision.mode
        before = expected_squares(board, mode)
        reported = None
        for obs in self.stable():
            changed = differences(obs.squares, before, mode)
            if not changed or len(changed) > MAX_CHANGED_SQUARES:
                continue
            move = infer_move(board, obs.squares, mode)
            if move is not None:
                return move
            if changed != reported:  # stable but not a legal move: misread or piece still being adjusted
                print(f"  board doesn't match a legal move (changed: {' '.join(chess.square_name(s) for s in changed)})")
                reported = changed


class KeyboardMoves:
    last_obs = None

    def wait_for_move(self, board: chess.Board) -> chess.Move:
        while True:
            text = input("Your move (e.g. e2e4 or Nf3, q to quit): ").strip()
            if text.lower() == "q":
                raise KeyboardInterrupt
            for parse in (board.parse_san, board.parse_uci):
                try:
                    return parse(text)
                except ValueError:
                    pass
            print("  not a legal move")


# =============================================================================
# ENGINE (Stockfish)
# =============================================================================


class Stockfish:
    def __init__(self, path: str, think_time: float, skill: int):
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)
        except FileNotFoundError:
            sys.exit(f"Stockfish not found at '{path}'. Install it or set STOCKFISH_PATH / --stockfish.")
        if skill < 20:
            self.engine.configure({"Skill Level": skill})
        self.limit = chess.engine.Limit(time=think_time)

    def best_move(self, board: chess.Board) -> chess.Move:
        return self.engine.play(board, self.limit).move

    def close(self):
        self.engine.quit()


# =============================================================================
# AUTOMATIC BOARD CALIBRATION
# =============================================================================


def camera_pose(arm_pose: np.ndarray) -> np.ndarray:
    """Camera pose in the robot base frame, from the gripper pose and the camera mount."""
    T = np.eye(4)
    T[:3, :3] = rot_z(math.radians(CAMERA_MOUNT_YAW_DEG))
    T[:3, 3] = CAMERA_MOUNT_OFFSET
    return arm_pose @ T


def depth_at(depth, u, v, r=3) -> float:
    win = depth[max(int(v) - r, 0) : int(v) + r + 1, max(int(u) - r, 0) : int(u) + r + 1]
    win = win[win > 0]
    return float(np.median(win)) if win.size else float("nan")


def pixel_rays(px, intrinsics) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    px = np.asarray(px, float)
    return np.column_stack([(px[:, 0] - cx) / fx, (px[:, 1] - cy) / fy, np.ones(len(px))])


def fit_plane(depth, mask, intrinsics, samples=3000):
    """Plane (point, normal) in the camera frame through the depth readings under ``mask``.
    Points well off the plane (dark pieces, noise) are dropped."""
    v, u = np.nonzero(mask & (depth > 0))
    if len(u) < 100:
        return None
    rng = np.random.default_rng(0)
    pick = rng.choice(len(u), min(samples, len(u)), replace=False)
    pts = pixel_rays(np.column_stack([u[pick], v[pick]]), intrinsics) * depth[v[pick], u[pick], None]
    # RANSAC first: pieces can cover a third of the board, which drags a plain least-squares fit
    keep, best = None, 0
    for _ in range(200):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((pts - a) @ n) < 0.004
        if inl.sum() > best:
            keep, best = inl, inl.sum()
    if keep is None or best < 50:
        return None
    for _ in range(3):
        c = pts[keep].mean(0)
        n = np.linalg.svd(pts[keep] - c)[2][2]
        keep = np.abs((pts - c) @ n) < 0.005
        if keep.sum() < 50:
            return None
    return c, n


def on_plane(px, plane, intrinsics) -> np.ndarray:
    """Camera-frame points where the pixels' rays meet the plane."""
    c, n = plane
    rays = pixel_rays(px, intrinsics)
    return rays * ((n @ c) / (rays @ n))[:, None]


def project(points_base, cam_T, intrinsics):
    """Robot-base points -> camera pixels."""
    fx, fy, cx, cy = intrinsics
    p = (np.asarray(points_base) - cam_T[:3, 3]) @ cam_T[:3, :3]  # into the camera frame
    return np.column_stack([fx * p[:, 0] / p[:, 2] + cx, fy * p[:, 1] / p[:, 2] + cy])


def corner_mask(color):
    """Pixels the colour of the yellow brackets (small specks removed, screw holes filled)."""
    import cv2

    m = hsv_mask(cv2.cvtColor(color, cv2.COLOR_BGR2HSV), CORNER_HSV).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0


def bracket_corners(mask) -> list[np.ndarray]:
    """Inside corner (pixel) of each L-shaped yellow bracket, biggest bracket first.

    The inside corner of an L is the point of the shape deepest inside its convex hull
    (the deepest "convexity defect"), which is where the board's corner sits."""
    import cv2

    contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    out = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < CORNER_MIN_AREA * mask.size:
            break
        hull = np.sort(cv2.convexHull(c, returnPoints=False), axis=0)
        try:
            defects = cv2.convexityDefects(c, hull)
        except cv2.error:
            continue
        if defects is None or len(defects) == 0:
            continue
        defects = defects.reshape(-1, 4)  # rows: start, end, farthest point, depth * 256
        start, end, far, depth = defects[np.argmax(defects[:, 3])]
        if depth / 256 < 3:  # not an L (no real notch)
            continue
        out.append(c[far, 0].astype(float))
    return out


def locate_board(arm, cam: RealSenseCamera, robot_white: bool):
    """Find the board from where the arm is now, using only the 4 yellow corners.

    Sets ``locate_board.seen`` to how many yellow corners were in the image. Returns a dict (a1, h1, h8, axes = (ex, ey, n, square), outer = board corners a1/h1/h8/a8
    side in the base frame, outer_px / image_corners = pixels of the board / playing-area
    corners) or None."""
    import cv2

    color, depth, _ = cam.read()
    mask = corner_mask(color)
    pts = bracket_corners(mask)
    locate_board.seen = len(pts)
    debug = color.copy()
    debug[mask] = (255, 255, 0)
    for pt in pts:
        cv2.circle(debug, tuple(int(x) for x in pt), 8, (0, 0, 255), 2)
    debug_path = Path(__file__).with_name("board_detect_debug.png")
    cv2.imwrite(str(debug_path), debug)
    if len(pts) < 4:
        print(f"  only {len(pts)} of the 4 yellow corners seen (see {debug_path.name}; cyan = yellow)")
        return None
    pts = np.array(pts[:4])
    h_img, w_img = mask.shape
    if pts[:, 0].min() < 3 or pts[:, 1].min() < 3 or pts[:, 0].max() > w_img - 4 or pts[:, 1].max() > h_img - 4:
        print("  a corner is at the edge of the image")
        return None
    c = pts.mean(0)
    quad = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]  # in order around the board
    inside = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(inside, quad.astype(np.int32), 1)
    plane = fit_plane(depth, (inside > 0) & ~mask, cam.intrinsics)  # RANSAC: pieces don't matter
    if plane is None:
        print("  corners found, but not enough depth readings on the board")
        return None
    cam_T = camera_pose(arm.pose)
    Q = on_plane(quad, plane, cam.intrinsics) @ cam_T[:3, :3].T + cam_T[:3, 3]  # board corners, base frame
    sides = [np.linalg.norm(Q[(i + 1) % 4] - Q[i]) for i in range(4)]
    diags = [np.linalg.norm(Q[2] - Q[0]), np.linalg.norm(Q[3] - Q[1])]
    if max(sides) / min(sides) > 1.15 or max(diags) / min(diags) > 1.1:
        print(f"  the 4 yellow corners don't form a square (sides {np.round(np.array(sides) * 1000)} mm)")
        return None
    centre = Q.mean(0)
    n = np.cross(Q[1] - Q[0], Q[2] - Q[1])
    n /= np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    # ranks run towards the robot when it plays black (it sits behind rank 8)
    towards_robot = np.array([-centre[0], -centre[1], 0.0])
    want = -towards_robot if robot_white else towards_robot
    edges = [Q[(i + 1) % 4] - Q[i] for i in range(4)]
    ey = max((e if e @ want >= 0 else -e for e in edges), key=lambda e: e @ want / np.linalg.norm(e))
    ey = ey - n * (n @ ey)
    ey /= np.linalg.norm(ey)
    ex = np.cross(ey, n)
    rel = Q - centre
    order = [np.argmin(rel @ (ex + ey)), np.argmax(rel @ (ex - ey)), np.argmax(rel @ (ex + ey)), np.argmax(rel @ (ey - ex))]
    if len(set(order)) < 4:
        print("  couldn't tell which corner is which")
        return None
    outer_px, outer = quad[order], Q[order]  # a1, h1, h8, a8 corners of the board
    # playing area: BOARD_BORDER_FRACTION in from the board's edges
    b = BOARD_BORDER_FRACTION
    H = cv2.getPerspectiveTransform(np.float32([[0, 1], [1, 1], [1, 0], [0, 0]]), np.float32(outer_px))
    inner_px = cv2.perspectiveTransform(np.float32([[[b, 1 - b], [1 - b, 1 - b], [1 - b, b], [b, b]]]), H)[0]
    C = on_plane(inner_px, plane, cam.intrinsics) @ cam_T[:3, :3].T + cam_T[:3, 3]
    sq = np.mean([np.linalg.norm(C[(i + 1) % 4] - C[i]) for i in range(4)]) / 8
    a1 = C[0] + sq / 2 * (ex + ey)
    cv2.polylines(debug, [inner_px.astype(np.int32)], True, (0, 255, 0), 2)
    for name, pt in zip(("a1", "h1", "h8", "a8"), inner_px):
        cv2.putText(debug, name, tuple(int(x) for x in pt), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(str(debug_path), debug)
    print(f"  4 corners found: square {sq * 1000:.1f} mm, board centre {np.round(centre, 3)}")
    if abs(np.mean(sides) - BOARD_WIDTH) > 0.05 * BOARD_WIDTH:
        print(f"  WARNING: board measured {np.mean(sides) * 1000:.0f} mm wide, expected {BOARD_WIDTH * 1000:.0f} mm "
              "(check the camera depth / CAMERA_MOUNT_OFFSET)")
    return {"a1": a1, "h1": a1 + 7 * sq * ex, "h8": a1 + 7 * sq * (ex + ey), "axes": (ex, ey, n, sq),
            "outer": outer, "outer_px": outer_px, "image_corners": inner_px.astype(float)}


def view_pose_for(found, intrinsics, R_down):
    """Gripper (position, yaw) that puts the camera straight above the board centre, turned square
    to the board, at the height where the 4 corners fill the image with VIEW_MARGIN to spare."""
    fx, fy, cx, cy = intrinsics
    ex, ey, n, _ = found["axes"]
    Q = found["outer"]
    centre = Q.mean(0)
    R_mount = rot_z(math.radians(CAMERA_MOUNT_YAW_DEG))
    cam_x0 = (R_down @ R_mount)[:, 0]
    # smallest turn that lines the image x-axis up with a board edge
    turn = math.atan2(ex[1], ex[0]) - math.atan2(cam_x0[1], cam_x0[0])
    yaw = (turn + math.pi / 4) % (math.pi / 2) - math.pi / 4
    R_cam = rot_z(yaw) @ R_down @ R_mount
    half_x = np.abs((Q - centre) @ R_cam[:, 0]).max()
    half_y = np.abs((Q - centre) @ R_cam[:, 1]).max()
    room_x = min(cx, CAMERA_WIDTH - cx) - VIEW_MARGIN * CAMERA_WIDTH  # pixels from the image centre to the margin
    room_y = min(cy, CAMERA_HEIGHT - cy) - VIEW_MARGIN * CAMERA_HEIGHT
    h = float(max(fx * half_x / room_x, fy * half_y / room_y))  # no fixed height: whatever makes the board fill the image
    cam_pos = centre + n * h
    R_ee = rot_z(yaw) @ R_down
    return cam_pos - R_ee @ np.asarray(CAMERA_MOUNT_OFFSET), yaw, h


def search_board(arm, cam: RealSenseCamera, robot_white: bool):
    """From home (bottom-left and bottom-right corners in view), shift the camera forward until
    all 4 yellow corners are in the image. Climbs if the board doesn't fit. Returns locate_board's dict or None."""
    start = arm.pose[:3, 3].copy()
    fwd = np.array([*VIEW_FORWARD, 0.0], float)
    fwd /= np.linalg.norm(fwd)
    if (found := locate_board(arm, cam, robot_white)) is not None:
        return found
    if locate_board.seen < 2:
        print("  WARNING: the bottom-left and bottom-right yellow corners should be in view from home.")
    shift, up, seen_two = 0.0, 0.0, locate_board.seen >= 2
    while shift < VIEW_FORWARD_MAX:
        if seen_two and locate_board.seen < 2:  # near corners left the image before the far ones came in
            up += VIEW_SEARCH_STEP
            seen_two = False
            print(f"  the board doesn't fit at this height; going {VIEW_SEARCH_STEP * 100:.0f} cm higher")
        else:
            shift += VIEW_FORWARD_STEP
        print(f"Shifting the camera forward: {shift * 100:.0f} cm from home, {up * 100:.0f} cm up.")
        try:
            arm.move_to(start + fwd * shift + [0, 0, up])
        except Exception as e:  # noqa: BLE001 - out of reach
            print(f"  can't move there ({e})")
            return None
        if (found := locate_board(arm, cam, robot_white)) is not None:
            return found
        seen_two = seen_two or locate_board.seen >= 2
    return None


def auto_calibrate(args, cal: Calibration, arm, cam: RealSenseCamera, check: bool = True):
    """Find the board with the camera and fill in a1, h1, h8, R_down, q_view and image_corners.

    1. From home, shift forward until all 4 yellow corners are in the image.
    2. Zoom: move square above the board at the height where the corners reach the image edges.
    3. The user confirms that view; it is saved.
    4. ``check``: hover over the 4 corner squares (no camera) and ask whether that looks right."""
    robot_white = args.robot_color == "white"
    print("Finding the board by its 4 yellow corners.")
    last = BoardFrame(cal).centre() if cal.a1 is not None else None
    while True:
        arm.home()
        cal.R_down = arm.pose[:3, :3]
        arm.R_down = cal.R_down
        found = search_board(arm, cam, robot_white)
        if found is None:
            if input("Corners not found. Put the board so its bottom-left and bottom-right yellow corners are in view "
                     "from home, then press Enter to try again (q = quit) ").strip() == "q":
                sys.exit("Auto-calibration failed. Check the board position and CORNER_HSV (see board_detect_debug.png).")
            continue

        # zoom: centred, square to the image, corners at the image edges
        for _ in range(4):
            target, yaw, h = view_pose_for(found, cam.intrinsics, cal.R_down)
            if float(np.linalg.norm(target - arm.pose[:3, 3])) < VIEW_FIT_TOLERANCE:
                break
            print(f"Zooming to the board: camera {h:.2f} m above it, turned {math.degrees(yaw):.0f} deg.")
            q_before = arm.q
            try:
                arm.move_to(target, yaw)
            except Exception as e:  # noqa: BLE001 - out of reach: keep the current view
                print(f"  can't move there ({e}); keeping the current view")
                break
            again = None
            for _try in range(3):  # give the camera a moment to settle (exposure, vibration) after the move
                time.sleep(0.5)
                if (again := locate_board(arm, cam, robot_white)) is not None:
                    break
            if again is None:
                print("  lost the corners after moving; going back to the last good view")
                arm.move_joints(q_before)
                break
            found = again

        px = found["outer_px"]
        print(f"The corners span x {px[:, 0].min():.0f}-{px[:, 0].max():.0f}, "
              f"y {px[:, 1].min():.0f}-{px[:, 1].max():.0f} of the {CAMERA_WIDTH}x{CAMERA_HEIGHT} image "
              "(see board_detect_debug.png).")
        answer = input("Are all 4 yellow corners in frame and is this a good view of the board? "
                       "[Y = save / r = search again / q = quit] ").strip().lower()
        if answer == "q":
            sys.exit("Not saved.")
        if answer != "r":
            break

    cal.a1, cal.h1, cal.h8 = found["a1"], found["h1"], found["h8"]
    cal.q_view = arm.q
    cal.image_corners = found["image_corners"]
    cal.save()  # from here on, only plain moves to saved positions; the camera isn't used
    print(f"Saved the view pose and board position to {CALIBRATION_FILE}.")
    board = BoardFrame(cal)

    if check:  # hover just above the 4 corner squares
        arm.open(0.0)
        corners = {"a1": cal.a1, "h1": cal.h1, "h8": cal.h8, "a8": cal.a1 + cal.h8 - cal.h1}
        for name, p in corners.items():
            arm.move_to(p + board.ez * SAFE_HEIGHT)
            arm.move_to(p + board.ez * AUTO_CAL_HOVER, speed=SPEED_APPROACH)
            print(f"  the fingertips should be just above the centre of {name}")
            time.sleep(1.0)
            arm.move_to(p + board.ez * SAFE_HEIGHT, speed=SPEED_APPROACH)
        if input("Were the fingertips over the centres of a1, h1, h8 and a8? [Y/n] ").strip().lower() == "n":
            arm.move_joints(cal.q_view)
            sys.exit("The board position is saved but off. If every point was off the same way, fix CAMERA_MOUNT_OFFSET / "
                     "CAMERA_MOUNT_YAW_DEG; if they were off towards/away from the edges, fix BOARD_BORDER_FRACTION. "
                     "Then run auto-calibrate again.")
    print(f"Square size {board.square * 1000:.1f} mm. Board tilt {math.degrees(math.acos(min(1, board.ez[2]))):.1f} deg.")
    if last is not None:
        print(f"The board moved {np.linalg.norm(board.centre() - last) * 1000:.0f} mm since the last calibration.")
    arm.move_joints(cal.q_view)


def cmd_auto_calibrate(args):
    if args.dry_run:
        sys.exit("auto-calibrate needs the robot and the camera; it can't run with --dry-run.")
    cal = Calibration.load()
    arm = make_arm(args, cal)
    cam = RealSenseCamera()
    try:
        auto_calibrate(args, cal, arm, cam)
    finally:
        cam.close()


# =============================================================================
# COMMANDS
# =============================================================================


def make_arm(args, cal):
    return DryRunArm(cal.R_down) if args.dry_run else FrankyArm(cal.R_down)


def cmd_view(args):
    import cv2

    cal = Calibration.load()
    cal.require("q_view", "image_corners")
    arm = make_arm(args, cal)
    arm.move_joints(cal.q_view)
    cam = RealSenseCamera()
    vision = BoardVision(cam, cal, args.detect)
    start = chess.Board()
    print("Green = matches the starting position. Keys: b = learn empty-square colours "
          "(board set up), q = quit.")
    try:
        while True:
            obs = vision.observe()
            panels = [vision.draw(obs, expected_squares(start, args.detect))]
            if args.detect == "color":
                w, b = vision.color_masks(obs)
                panels.append(cv2.cvtColor((w * 255).astype(np.uint8) | (b * 128).astype(np.uint8), cv2.COLOR_GRAY2BGR))
            else:
                h = np.clip(np.nan_to_num(obs.height) / 0.07 * 255, 0, 255).astype(np.uint8)
                panels.append(cv2.applyColorMap(h, cv2.COLORMAP_JET))
            cv2.imshow("detection (left) / masks: white=255 black=128 or height (right)", np.hstack(panels))
            k = cv2.waitKey(30) & 0xFF
            if k == ord("b"):
                vision.learn_background(obs)
                print("Learned empty-square colours from ranks 3-6.")
            if k == ord("q"):
                break
    finally:
        cam.close()


def build_game_parts(args, need_camera: bool, find_board: bool = False):
    """``find_board``: always locate the board first (play). Otherwise only when there's no calibration."""
    cal = Calibration.load()
    arm = make_arm(args, cal)
    cam = vision = None
    missing = cal.a1 is None or cal.image_corners is None
    if not args.dry_run and (missing or (find_board and not args.no_calibrate)):
        cam = RealSenseCamera()
        auto_calibrate(args, cal, arm, cam, check=False)
    board_frame = BoardFrame(cal)
    if need_camera:
        cal.require("q_view", "image_corners")
        cam = cam or RealSenseCamera()
        vision = BoardVision(cam, cal, args.detect)
    elif cam:
        cam.close()
        cam = None
    R_down = cal.R_down if cal.R_down is not None else arm.R_down
    adjuster = PickAdjuster(vision, board_frame, R_down, args.adjust)
    if args.adjust and vision is None:
        print("Pick adjustment needs the camera; it is off for this run.")
    print(f"Detection: {args.detect if vision else 'none (keyboard)'}; pick adjustment: {'ON' if adjuster.enabled else 'OFF'}")
    return cal, arm, cam, vision, ChessMover(arm, board_frame, adjuster, cal)


def cmd_pick_test(args):
    src, dst = chess.parse_square(args.src), chess.parse_square(args.dst)
    piece = chess.Piece.from_symbol(args.piece)
    cal, arm, cam, vision, mover = build_game_parts(args, need_camera=args.adjust and not args.manual_moves)
    try:
        mover.go_view()
        obs = vision.observe() if vision else None
        mover.move_piece(Transfer(src, dst, piece, "test"), obs)
        mover.go_view()
    finally:
        if cam:
            cam.close()


def cmd_play(args):
    robot_color = chess.WHITE if args.robot_color == "white" else chess.BLACK
    use_camera = not args.manual_moves
    # with --manual-moves the camera is still opened for pick adjustment unless --no-adjust
    cal, arm, cam, vision, mover = build_game_parts(args, need_camera=use_camera or args.adjust, find_board=True)
    watcher = CameraWatcher(vision) if use_camera else KeyboardMoves()
    engine = Stockfish(args.stockfish, args.think_time, args.skill)
    board = chess.Board()
    capture_slot = 0
    try:
        mover.go_view()
        if use_camera:
            print("Set up the pieces in the starting position.")
            watcher.wait_for_board(board, "Waiting for the starting position", learn_background=True)
            print("Starting position seen. Game on.")
        while not board.is_game_over():
            print(f"\n{board}\n")
            if board.turn != robot_color:
                move = watcher.wait_for_move(board)
                print(f"Human: {board.san(move)}")
                board.push(move)
                continue

            move = engine.best_move(board)
            print(f"Robot: {board.san(move)}")
            obs = watcher.last_obs if use_camera else (vision.observe() if vision else None)
            transfers = plan_transfers(board, move, capture_slot)
            capture_slot += sum(isinstance(t.dst, tuple) for t in transfers)
            for t in transfers:
                mover.move_piece(t, obs)
            board.push(move)
            mover.go_view()
            if move.promotion:
                input(f"Please swap the pawn on {chess.square_name(move.to_square)} for a "
                      f"{chess.piece_name(move.promotion)}, then press Enter.")
            if use_camera and VERIFY_ROBOT_MOVES:
                watcher.wait_for_board(board, "The board doesn't match the robot's move; please fix it")
            elif use_camera:
                watcher.last_obs = vision.observe()
        print(f"\n{board}\nGame over: {board.result()}")
    except KeyboardInterrupt:
        print("\nGame stopped.")
    finally:
        engine.close()
        if cam:
            cam.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="print arm commands instead of moving the robot")
    p.add_argument("--detect", choices=["color", "depth"], default=DETECTION_MODE)
    p.add_argument("--adjust", dest="adjust", action="store_true", default=ENABLE_PICK_ADJUSTMENT)
    p.add_argument("--no-adjust", dest="adjust", action="store_false", help="turn off pick adjustment")
    p.add_argument("--manual-moves", action="store_true", help="type the human's moves instead of using the camera")
    p.add_argument("--robot-color", choices=["white", "black"], default=ROBOT_COLOR)
    p.add_argument("--stockfish", default=STOCKFISH_PATH)
    p.add_argument("--think-time", type=float, default=STOCKFISH_THINK_TIME)
    p.add_argument("--skill", type=int, default=STOCKFISH_SKILL_LEVEL)
    p.add_argument("--no-calibrate", action="store_true", help="play: reuse the saved board position instead of finding it again")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("auto-calibrate", help="find the board by its yellow corners, then hover over a1/h1/h8 to check")
    sub.add_parser("view", help="live detection overlay for tuning")
    t = sub.add_parser("pick-test", help="move one piece, e.g. pick-test e2 e4 --piece P")
    t.add_argument("src")
    t.add_argument("dst")
    t.add_argument("--piece", default="P", help="piece symbol (P N B R Q K = white, lowercase = black)")
    sub.add_parser("play", help="play a game")
    args = p.parse_args()
    {
        "auto-calibrate": cmd_auto_calibrate,
        "view": cmd_view,
        "pick-test": cmd_pick_test,
        "play": cmd_play,
    }[args.command](args)


if __name__ == "__main__":
    main()
