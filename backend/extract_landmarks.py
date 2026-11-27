
import csv
import cv2
import sys

from pathlib import Path
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_FILE = "yolo11n-pose.pt"

# Scene-change thresholds.
# These are initial prototype thresholds and need testing.
CUT_THRESHOLD = 0.65
PERSISTENCE_THRESHOLD = 0.30

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
]


# ==========================================
# SCENE CHANGE DETECTION
# ==========================================

def get_histogram(frame):

    # Resize frame to reduce processing time
    frame = cv2.resize(frame, (320, 180))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256]
    )

    cv2.normalize(
        histogram,
        histogram,
        0,
        1,
        cv2.NORM_MINMAX
    )

    return histogram


def histogram_difference(hist1, hist2):

    return cv2.compareHist(
        hist1,
        hist2,
        cv2.HISTCMP_BHATTACHARYYA
    )


def detect_scene_cuts(video_path):

    print("\nChecking video continuity...")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open video for validation.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    success, previous_frame = cap.read()

    if not success:
        cap.release()
        raise RuntimeError("Video is empty or unreadable.")

    previous_hist = get_histogram(previous_frame)

    frame_index = 1
    candidate = None
    detected_cuts = []

    while True:

        success, current_frame = cap.read()

        if not success:
            break

        current_hist = get_histogram(current_frame)

        # Confirm a possible cut using the following frame.
        if candidate is not None:

            candidate_index, old_hist, new_hist = candidate

            persistence_difference = histogram_difference(
                new_hist,
                current_hist
            )

            old_scene_difference = histogram_difference(
                old_hist,
                current_hist
            )

            if (
                persistence_difference < PERSISTENCE_THRESHOLD
                and old_scene_difference > CUT_THRESHOLD
            ):

                cut_time = candidate_index / fps

                detected_cuts.append({
                    "frame": candidate_index,
                    "time_seconds": round(cut_time, 2)
                })

            candidate = None

        # Check for a major visual change.
        difference = histogram_difference(
            previous_hist,
            current_hist
        )

        if difference > CUT_THRESHOLD:

            candidate = (
                frame_index,
                previous_hist.copy(),
                current_hist.copy()
            )

        previous_hist = current_hist
        frame_index += 1

    cap.release()

    return detected_cuts


# ==========================================
# FIND LATEST UPLOADED VIDEO
# ==========================================

extensions = {".mp4", ".mov", ".webm"}

videos = [
    file for file in UPLOAD_DIR.iterdir()
    if file.suffix.lower() in extensions
] if UPLOAD_DIR.exists() else []

if not videos:

    print("No video found in uploads folder.")
    sys.exit(1)

video_path = max(
    videos,
    key=lambda file: file.stat().st_mtime
)

print("Processing:", video_path.name)


# ==========================================
# VALIDATE VIDEO BEFORE YOLO
# ==========================================

try:

    scene_cuts = detect_scene_cuts(video_path)

except Exception as error:

    print("\nVIDEO VALIDATION FAILED")
    print("Reason:", str(error))
    sys.exit(1)


if scene_cuts:

    print("\nVIDEO REJECTED")
    print("------------------------------")
    print("Reason: Significant scene cut or camera transition detected.")

    for cut in scene_cuts:
        print(
            f"Possible cut at {cut['time_seconds']} seconds "
            f"(frame {cut['frame']})"
        )

    print("\nPlease upload a new video.")

    print(
        "REJECTION_REASON: "
        "Your video contains a scene cut or camera transition. "
        "Please upload one continuous recording of your batting shot."
    )

    sys.exit(2)


print("\nVideo continuity check passed.")
print("Starting YOLO pose extraction...")


# ==========================================
# LOAD YOLO MODEL
# ==========================================

model = YOLO(MODEL_FILE)

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():

    print("Could not open video.")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30


# ==========================================
# LANDMARK EXTRACTION
# ==========================================

csv_path = OUTPUT_DIR / "landmarks.csv"

frame_index = 0
detected_frames = 0

with open(csv_path, "w", newline="") as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "video",
        "frame",
        "timestamp_seconds",
        "person_index",
        "keypoint",
        "x_pixels",
        "y_pixels",
        "x_normalized",
        "y_normalized",
        "confidence"
    ])

    print("\nExtracting body landmarks...")

    while True:

        success, frame = cap.read()

        if not success:
            break

        height, width = frame.shape[:2]

        results = model.predict(
            frame,
            conf=0.35,
            verbose=False
        )

        keypoints = results[0].keypoints

        if keypoints is not None and keypoints.xy is not None:

            pixel_coordinates = keypoints.xy.cpu().numpy()

            normalized_coordinates = keypoints.xyn.cpu().numpy()

            if keypoints.conf is not None:
                confidence_values = keypoints.conf.cpu().numpy()
            else:
                confidence_values = None

            if len(pixel_coordinates) > 0:
                detected_frames += 1

            for person_index in range(len(pixel_coordinates)):

                for point_index, point_name in enumerate(KEYPOINT_NAMES):

                    x_pixel, y_pixel = pixel_coordinates[
                        person_index, point_index
                    ]

                    x_normalized, y_normalized = normalized_coordinates[
                        person_index, point_index
                    ]

                    if confidence_values is not None:

                        confidence = confidence_values[
                            person_index, point_index
                        ]

                    else:
                        confidence = 0

                    writer.writerow([
                        video_path.name,
                        frame_index,
                        round(frame_index / fps, 4),
                        person_index,
                        point_name,
                        round(float(x_pixel), 2),
                        round(float(y_pixel), 2),
                        round(float(x_normalized), 6),
                        round(float(y_normalized), 6),
                        round(float(confidence), 4)
                    ])

        frame_index += 1

        if frame_index % 50 == 0:
            print("Processed frames:", frame_index)


cap.release()


# ==========================================
# COMPLETION
# ==========================================

print("\nLANDMARK EXTRACTION COMPLETED!")
print("--------------------------------")
print("Total frames:", frame_index)
print("Frames with detected people:", detected_frames)
print("CSV saved at:", csv_path.resolve())
