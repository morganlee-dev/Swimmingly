"""
Swimming Stroke Analyser
========================
Underwater side-on camera, moving or fixed.

Pipeline:
  1. Lane rope hue-transition detection  → pixels_per_metre calibration
  2. Transition drift tracking           → distance per stroke
  3. MediaPipe Tasks PoseLandmarker      → 33 joint landmarks per frame
  4. Underwater stroke phase detection   → catch / pull / push / recovery
  5. Joint angle calculation             → angle time-series per joint
  6. Stroke segmentation                 → one array of angles per stroke
  7. Spatial normalisation               → hip-centred, torso-scaled
  8. Dynamic Time Warping                → temporal alignment of two strokes
  9. Pearson correlation                 → pattern similarity per joint
 10. Mean Absolute Error on angles       → positional accuracy per joint

Dependencies:
    pip install opencv-python mediapipe numpy scipy dtaidistance

Model file (download before running):
    wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

Three model sizes available — trade accuracy for speed:
    pose_landmarker_lite.task    — fastest,  least accurate
    pose_landmarker_full.task    — balanced
    pose_landmarker_heavy.task   — slowest,  most accurate (recommended underwater)
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import math
import os


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_heavy.task")
MARKER_SPACING_M = 1.0        # measure this in your pool with a tape measure
RESEED_INTERVAL  = 15         # re-detect rope transitions every N frames
MIN_JOINT_VIS    = 0.4        # minimum visibility score to trust a joint

JOINT_PAIRS = {               # (vertex, pointA, pointB) for angle calculation
    "r_elbow":    (14, 12, 16),
    "l_elbow":    (13, 11, 15),
    "r_shoulder": (12, 14, 24),
    "l_shoulder": (11, 13, 23),
    "r_hip":      (24, 12, 26),
    "l_hip":      (23, 11, 25),
    "r_knee":     (26, 24, 28),
    "l_knee":     (25, 23, 27),
}

JOINT_NAMES = list(JOINT_PAIRS.keys())


# ---------------------------------------------------------------------------
# 3. Joint angle calculation
# ---------------------------------------------------------------------------

def calculate_angle(a, b, c):
    """
    Calculate the angle at vertex b formed by points a, b, c.

    Full equation:
        BA = (ax-bx, ay-by)
        BC = (cx-bx, cy-by)
        angle = arccos( (BA·BC) / (|BA| × |BC|) )

    Returns angle in degrees.
    """
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba      = a - b
    bc      = c - b
    denom   = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 0.0
    cosine  = np.dot(ba, bc) / denom
    return math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))


def extract_angles(landmarks, frame_w, frame_h):
    """
    Extract all joint angles from a mediapipe.tasks landmark list.
    landmarks: list of NormalizedLandmark objects (result.pose_landmarks[0])
    Returns dict of {joint_name: angle_degrees} or None if key joints
    have low visibility.
    """
    h, w = frame_h, frame_w

    def pt(idx):
        return (landmarks[idx].x * w, landmarks[idx].y * h)

    def vis(idx):
        return landmarks[idx].visibility

    key_joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
    if any(vis(i) < MIN_JOINT_VIS for i in key_joints):
        return None

    angles = {}
    for name, (vertex, a, b) in JOINT_PAIRS.items():
        angles[name] = calculate_angle(pt(a), pt(vertex), pt(b))

    return angles


# ---------------------------------------------------------------------------
# 4. Spatial normalisation
# ---------------------------------------------------------------------------

def normalise_landmarks(landmarks, frame_w, frame_h):
    """
    Convert raw landmark coordinates to hip-centred, torso-scaled coordinates.
    Removes position and size differences between swimmers.

    Steps:
        1. Subtract hip midpoint from all joints  (removes position)
        2. Divide by torso length                 (removes size/scale)

    landmarks: list of NormalizedLandmark objects (result.pose_landmarks[0])
    Returns flat numpy array of shape (66,) — 33 joints × (x, y).
    Returns None if torso length is near zero.
    """
    h, w = frame_h, frame_w

    hip_mid = np.array([
        ((landmarks[23].x + landmarks[24].x) / 2) * w,
        ((landmarks[23].y + landmarks[24].y) / 2) * h,
    ])
    shoulder_mid = np.array([
        ((landmarks[11].x + landmarks[12].x) / 2) * w,
        ((landmarks[11].y + landmarks[12].y) / 2) * h,
    ])

    torso_length = np.linalg.norm(shoulder_mid - hip_mid)
    if torso_length < 1e-6:
        return None

    joints = []
    for i in range(33):
        x = (landmarks[i].x * w - hip_mid[0]) / torso_length
        y = (landmarks[i].y * h - hip_mid[1]) / torso_length
        joints.extend([x, y])

    return np.array(joints, dtype=np.float32)


# ---------------------------------------------------------------------------
# 5. Underwater stroke phase detection
# ---------------------------------------------------------------------------

class StrokePhase:
    ENTRY    = "entry"
    CATCH    = "catch"
    PULL     = "pull"
    PUSH     = "push"
    RECOVERY = "recovery"


def get_stroke_phase(landmarks, frame_w, frame_h, prev_wrist_x):
    """
    Determine the current stroke phase from landmark positions.
    Uses left arm (indices 11, 15, 23, 24).
    landmarks: list of NormalizedLandmark objects
    """
    h, w = frame_h, frame_w

    wrist_x    = landmarks[15].x * w
    wrist_y    = landmarks[15].y * h
    shoulder_x = landmarks[11].x * w
    shoulder_y = landmarks[11].y * h
    hip_x      = ((landmarks[23].x + landmarks[24].x) / 2) * w

    wrist_ahead    = wrist_x < shoulder_x
    wrist_below    = wrist_y > shoulder_y
    wrist_past_hip = wrist_x >= hip_x
    wrist_past_shoulder = wrist_x >= shoulder_x

    if wrist_ahead and not wrist_below:
        return StrokePhase.ENTRY
    elif wrist_ahead and wrist_below:
        return StrokePhase.CATCH
    # elif not wrist_ahead and not wrist_past_hip:
    elif not wrist_ahead:
        return StrokePhase.PULL
    # elif wrist_past_hip:
    #     return StrokePhase.PUSH
    else:
        return StrokePhase.RECOVERY


# ---------------------------------------------------------------------------
# 6. Stroke recorder
# ---------------------------------------------------------------------------

class StrokeRecorder:
    """
    Detects stroke boundaries and records angle sequences and normalised
    landmark sequences for each completed stroke.
    """

    def __init__(self):
        self.prev_phase       = None
        self.prev_wrist_x     = None

        self.current_angles   = []
        self.current_norm     = []
        self.current_wrist    = []

        self.completed_angles = []
        self.completed_norm   = []
        self.completed_wrists = []
        self.stroke_count     = 0

    def update(self, landmarks, frame_w, frame_h):
        """
        landmarks: list of NormalizedLandmark objects (result.pose_landmarks[0])
        """
        h, w = frame_h, frame_w

        wrist_x = landmarks[15].x * w
        if self.prev_wrist_x is None:
            self.prev_wrist_x = wrist_x

        phase  = get_stroke_phase(landmarks, w, h, self.prev_wrist_x)
        angles = extract_angles(landmarks, w, h)
        norm   = normalise_landmarks(landmarks, w, h)

        if angles is not None:
            self.current_angles.append([angles[n] for n in JOINT_NAMES])
        if norm is not None:
            self.current_norm.append(norm)

        self.current_wrist.append((int(wrist_x), int(landmarks[15].y * h)))

        # Stroke boundary — recovery/push → entry
        if (self.prev_phase in (StrokePhase.RECOVERY, StrokePhase.PULL)
                and phase in (StrokePhase.ENTRY, StrokePhase.CATCH)):
        # if (self.prev_phase != StrokePhase.ENTRY
        #         and phase == StrokePhase.ENTRY):
            if len(self.current_angles) > 5:
                self.completed_angles.append(
                    np.array(self.current_angles, dtype=np.float32)
                )
                self.completed_norm.append(
                    np.array(self.current_norm, dtype=np.float32)
                )
                self.completed_wrists.append(self.current_wrist.copy())
                self.stroke_count += 1

            # dist_tracker.reset_stroke_distance()
            self.current_angles = []
            self.current_norm   = []
            self.current_wrist  = []

        self.prev_phase   = phase
        self.prev_wrist_x = wrist_x

        return phase


# ---------------------------------------------------------------------------
# 7. Drawing utilities
# ---------------------------------------------------------------------------

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 23), (12, 24), (23, 24),                        # torso
    (23, 25), (24, 26), (25, 27), (26, 28),              # legs
    (27, 29), (28, 30), (29, 31), (30, 32),              # feet
    (15, 17), (15, 19), (15, 21),                        # left hand
    (16, 18), (16, 20), (16, 22),                        # right hand
]

def draw_landmarks_on_frame(frame, landmarks, frame_w, frame_h):
    """
    Draw pose skeleton directly with OpenCV.
    No proto conversion needed — works with the tasks API landmark list directly.
    """
    h, w = frame_h, frame_w

    # Draw connections first (behind the dots)
    for start, end in POSE_CONNECTIONS:
        x1 = int(landmarks[start].x * w)
        y1 = int(landmarks[start].y * h)
        x2 = int(landmarks[end].x * w)
        y2 = int(landmarks[end].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw joint dots on top
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def draw_overlay(frame, phase, recorder):
    h, w = frame.shape[:2]

    # if calibrated:
    #     cv2.putText(frame,
    #                 f"Scale: {dist_tracker.pixels_per_metre:.1f} px/m",
    #                 (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # else:
    #     cv2.putText(frame,
    #                 "Calibrating — need 2+ rope transitions visible",
    #                 (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

    lines = [
        f"Phase:      {phase}",
        f"Strokes:    {recorder.stroke_count}",
        # f"Dist (lap): {dist_tracker.total_distance_m:.2f} m",
        # f"Stroke:     {dist_tracker.current_stroke_dist:.2f} m",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 58 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


# ---------------------------------------------------------------------------
# 9. Main loop
# ---------------------------------------------------------------------------

def run(video_source=0, compare_after_n_strokes=2):
    """
    Main capture and analysis loop.

    video_source: 0 for webcam, or a file path string for recorded video.
    compare_after_n_strokes: when this many strokes have been completed,
        compare stroke 1 vs stroke 2 and print the report.
    """

    # --- Build the PoseLandmarker using the new tasks API ---
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options      = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        running_mode=mp_vision.RunningMode.VIDEO,  # VIDEO mode requires timestamps
    )

    vid = os.path.join(os.path.dirname(__file__), video_source)

    print("Opening video source...")
    cap = cv2.VideoCapture(vid)
    # cap       = cv2.VideoCapture(video_source)
    fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
    recorder  = StrokeRecorder()
    rope_band = None
    frame_idx = 0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            h, w = frame.shape[:2]

            # --- Contrast enhancement (helps MediaPipe underwater) ---
            lab      = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b  = cv2.split(lab)
            clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l        = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


            # --- Pose estimation (mediapipe.tasks API) ---
            # Wrap the numpy frame in mp.Image — required by the tasks API
            rgb      = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # VIDEO running mode requires a monotonically increasing timestamp (ms)
            timestamp_ms = int((frame_idx / fps) * 1000)
            result       = landmarker.detect_for_video(mp_image, timestamp_ms)

            phase = StrokePhase.RECOVERY

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                draw_landmarks_on_frame(frame, landmarks, w, h)  # pass w, h
                phase = recorder.update(landmarks, w, h)

            # --- Overlay ---
            draw_overlay(frame, phase,recorder)

            cv2.imshow("Swim Stroke Analyser", frame)

            # --- Auto-compare after N strokes ---
            if recorder.stroke_count >= compare_after_n_strokes:
                if len(recorder.completed_angles) >= 2:
                    result_cmp = compare_strokes(
                        recorder.completed_angles[0],
                        recorder.completed_angles[1],
                    )
                    print_comparison_report(result_cmp)
                break  # remove this line to keep recording indefinitely

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Pass a video file path to analyse recorded footage:
    #   run(video_source="swim_footage.mp4")
    # Pass 0 to use a live webcam:
    #   run(video_source=0)
    run(video_source="swim_footage_2.mp4", compare_after_n_strokes=10)
