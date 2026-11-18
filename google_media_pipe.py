
import cv2
import mediapipe as mp
import numpy as np
import os
import requests

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# 1. CONFIGURATION
# ==========================================

VIDEO_PATH = "test_swing.mp4"

MODEL_PATH = "pose_landmarker_full.task"

OUTPUT_DIR = "outputs"

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "test_swing_pose_only.mp4"
)

MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/"
    "pose_landmarker_full.task"
)

FRAME_WIDTH = 1024
FRAME_HEIGHT = 576


# ==========================================
# 2. DOWNLOAD MODEL AUTOMATICALLY
# ==========================================

def download_model():

    if os.path.exists(MODEL_PATH):
        print("[OK] MediaPipe model already exists.")
        return

    print("[DOWNLOAD] Downloading MediaPipe model...")

    response = requests.get(
        MODEL_URL,
        timeout=120
    )

    response.raise_for_status()

    with open(MODEL_PATH, "wb") as file:
        file.write(response.content)

    print("[OK] Model downloaded.")


download_model()

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 3. INITIALIZE MEDIAPIPE
# ==========================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.PoseLandmarkerOptions(

    base_options=base_options,

    running_mode=vision.RunningMode.VIDEO,

    num_poses=1,

    min_pose_detection_confidence=0.5,

    min_pose_presence_confidence=0.5,

    min_tracking_confidence=0.5,

    output_segmentation_masks=False
)

detector = vision.PoseLandmarker.create_from_options(
    options
)


# ==========================================
# 4. POSE CONNECTIONS
# ==========================================

POSE_CONNECTIONS = (
    vision.PoseLandmarksConnections.POSE_LANDMARKS
)


# ==========================================
# 5. COORDINATE CONVERSION
# ==========================================

def to_pixel_coordinates(point, width, height):

    return (
        int(point[0] * width),
        int(point[1] * height)
    )


# ==========================================
# 6. DRAW FULL BODY POSE

# ==========================================
# 6. DRAW FULL BODY POSE (NO FACE)
# ==========================================

FACE_LANDMARKS = set(range(0, 11))   # nose, eyes, ears, mouth = indices 0-10

def draw_pose(frame, landmarks):

    height, width, _ = frame.shape

    # Draw skeleton connections (skip anything touching the face)

    for connection in POSE_CONNECTIONS:

        if connection.start in FACE_LANDMARKS or connection.end in FACE_LANDMARKS:
            continue

        start = landmarks[connection.start]
        end = landmarks[connection.end]

        p1 = to_pixel_coordinates(
            [start.x, start.y],
            width,
            height
        )

        p2 = to_pixel_coordinates(
            [end.x, end.y],
            width,
            height
        )

        cv2.line(
            frame,
            p1,
            p2,
            (255, 255, 255),
            2
        )

    # Draw body landmarks (skip the face)

    for index, landmark in enumerate(landmarks):

        if index in FACE_LANDMARKS:
            continue

        point = to_pixel_coordinates(
            [landmark.x, landmark.y],
            width,
            height
        )

        cv2.circle(
            frame,
            point,
            4,
            (0, 255, 255),
            -1
        )

# ==========================================
# 7. VIDEO PROCESSING ENGINE
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    detector.close()

    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


fps = cap.get(cv2.CAP_PROP_FPS)

if not fps or fps <= 0:
    fps = 30.0


total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (FRAME_WIDTH, FRAME_HEIGHT)
)

if not writer.isOpened():

    cap.release()
    detector.close()

    raise RuntimeError(
        f"Could not create output video: {OUTPUT_PATH}"
    )


frame_number = 0

print("\n================================")
print("VISIONCOACH POSE ENGINE STARTED")
print("================================")

print(f"Input: {VIDEO_PATH}")
print(f"Output: {OUTPUT_PATH}")
print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")


# ==========================================
# 8. PROCESS EVERY FRAME
# ==========================================

try:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        height, width, _ = frame.shape

        # Convert BGR to RGB

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Generate timestamp

        timestamp_ms = int(
            (frame_number - 1) * 1000 / fps
        )

        # MediaPipe pose detection

        detection_result = detector.detect_for_video(
            image,
            timestamp_ms
        )

        # Draw pose only

        if detection_result.pose_landmarks:

            landmarks = detection_result.pose_landmarks[0]

            draw_pose(
                frame,
                landmarks
            )

        # Write analyzed frame

        writer.write(frame)

        # Progress tracking

        if frame_number % 30 == 0:

            progress = (
                frame_number / total_frames * 100
                if total_frames > 0
                else 0
            )

            print(
                f"\rProcessing: {frame_number}/{total_frames} "
                f"({progress:.1f}%)",
                end=""
            )


finally:

    cap.release()

    writer.release()

    detector.close()

    print("\n\n================================")
    print("VISIONCOACH PROCESSING COMPLETED")
    print("================================")

    print(f"Saved video: {OUTPUT_PATH}")

    print("================================")
