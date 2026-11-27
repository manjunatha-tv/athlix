import csv
import json
import cv2
import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
import shutil
import re
from ultralytics import YOLO

from phase_detection import detect_movement_phases


# ==========================================
# APP CONFIGURATION
# ==========================================

app = FastAPI(title="ATHIX API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
REFERENCE_DIR = Path("references")

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
REFERENCE_DIR.mkdir(exist_ok=True)

# Serve generated per-frame pose data to the frontend.
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

MAX_FILE_SIZE = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm"}

MODEL_FILE = "yolo11n-pose.pt"

CUT_THRESHOLD = 0.65
PERSISTENCE_THRESHOLD = 0.30
CONFIDENCE_THRESHOLD = 0.5

# Trainer references are stored by technique.
VALID_TECHNIQUES = {"straight-drive", "cover-drive", "pull-shot", "cut-shot"}
VALID_HANDS = {"right", "left"}
VALID_CAMERA_VIEWS = {"side", "front", "three-quarter"}

# Compare eight joint-angle patterns.
ANGLE_COLUMNS = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "left_hip_angle",
    "right_hip_angle",
]

# The first prototype uses equal feature weights.
# This is an engineering prototype, not a coach-validated scoring model.
ANGLE_WEIGHTS = {name: 1.0 for name in ANGLE_COLUMNS}

# Resample both movements to the same normalized timeline.
COMPARISON_SAMPLES = 100

MODEL = None


# ==========================================
# KEYPOINT NAMES
# ==========================================

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
    "right_ankle",
]


# ==========================================
# LOAD YOLO MODEL
# ==========================================

def get_model():
    global MODEL

    if MODEL is None:
        print("Loading YOLO pose model...")
        MODEL = YOLO(MODEL_FILE)

    return MODEL


# ==========================================
# SCENE CUT DETECTION
# ==========================================

def get_histogram(frame):
    frame = cv2.resize(frame, (320, 180))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256],
    )

    cv2.normalize(
        histogram,
        histogram,
        0,
        1,
        cv2.NORM_MINMAX,
    )

    return histogram


def histogram_difference(hist1, hist2):
    return cv2.compareHist(
        hist1,
        hist2,
        cv2.HISTCMP_BHATTACHARYYA,
    )


def detect_scene_cuts(video_path):
    print("\nChecking video continuity...")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("ATHIX could not read the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    success, previous_frame = cap.read()

    if not success:
        cap.release()
        raise ValueError("The uploaded video is empty or unreadable.")

    previous_hist = get_histogram(previous_frame)

    frame_index = 1
    candidate = None
    detected_cuts = []

    while True:
        success, current_frame = cap.read()

        if not success:
            break

        current_hist = get_histogram(current_frame)

        if candidate is not None:
            candidate_index, old_hist, new_hist = candidate

            persistence_difference = histogram_difference(
                new_hist,
                current_hist,
            )

            old_scene_difference = histogram_difference(
                old_hist,
                current_hist,
            )

            if (
                persistence_difference < PERSISTENCE_THRESHOLD
                and old_scene_difference > CUT_THRESHOLD
            ):
                detected_cuts.append(round(candidate_index / fps, 2))

            candidate = None

        difference = histogram_difference(
            previous_hist,
            current_hist,
        )

        if difference > CUT_THRESHOLD:
            candidate = (
                frame_index,
                previous_hist.copy(),
                current_hist.copy(),
            )

        previous_hist = current_hist
        frame_index += 1

    cap.release()
    return detected_cuts


# ==========================================
# YOLO LANDMARK EXTRACTION
# ==========================================

def extract_landmarks(video_path, csv_path):
    print("\nStarting YOLO pose extraction...")

    model = get_model()
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Could not open video for pose extraction.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    frame_index = 0
    detected_frames = 0

    try:
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
                "confidence",
            ])

            while True:
                success, frame = cap.read()

                if not success:
                    break

                results = model.predict(
                    frame,
                    conf=0.35,
                    verbose=False,
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
                                person_index,
                                point_index,
                            ]

                            x_normalized, y_normalized = normalized_coordinates[
                                person_index,
                                point_index,
                            ]

                            if confidence_values is not None:
                                confidence = confidence_values[
                                    person_index,
                                    point_index,
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
                                round(float(confidence), 4),
                            ])

                frame_index += 1

                if frame_index % 50 == 0:
                    print("Processed frames:", frame_index)

    finally:
        cap.release()

    if frame_index == 0:
        raise ValueError("No video frames could be extracted.")

    if detected_frames == 0:
        raise ValueError("No person was detected in the uploaded video.")

    print("\nPOSE EXTRACTION COMPLETED")
    print("Total frames:", frame_index)
    print("Frames with detected people:", detected_frames)
    print("Landmarks saved:", csv_path)

    return {
        "total_frames": frame_index,
        "detected_frames": detected_frames,
    }


# ==========================================
# SELECT PRIMARY PERSON
# ==========================================

def select_primary_person(landmarks_df):
    """
    If multiple people are detected, choose one person consistently
    per frame: the person with the highest mean keypoint confidence.
    """
    required = {
        "frame",
        "person_index",
        "keypoint",
        "confidence",
    }

    if not required.issubset(landmarks_df.columns):
        raise ValueError("Landmark CSV is missing person-selection columns.")

    confidence = (
        landmarks_df.groupby(["frame", "person_index"])["confidence"]
        .mean()
        .reset_index(name="mean_confidence")
    )

    selected = (
        confidence.sort_values(
            ["frame", "mean_confidence"],
            ascending=[True, False],
        )
        .drop_duplicates(subset=["frame"], keep="first")
        [["frame", "person_index"]]
    )

    primary = landmarks_df.merge(
        selected,
        on=["frame", "person_index"],
        how="inner",
    )

    return primary


# ==========================================
# JOINT ANGLE CALCULATION
# ==========================================

def calculate_angle(a, b, c):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return np.nan

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def get_point(frame, landmark, coordinate_columns=("x_pixels", "y_pixels")):
    point = frame[frame["keypoint"] == landmark]

    if point.empty:
        return None

    row = point.iloc[0]

    if float(row["confidence"]) < CONFIDENCE_THRESHOLD:
        return None

    return [
        float(row[coordinate_columns[0]]),
        float(row[coordinate_columns[1]]),
    ]


def get_joint_angle(frame, first, middle, last):
    a = get_point(frame, first)
    b = get_point(frame, middle)
    c = get_point(frame, last)

    if a is None or b is None or c is None:
        return np.nan

    return calculate_angle(a, b, c)


def calculate_joint_angles(landmarks_path, angles_path):
    print("\nCalculating user joint angles...")

    df = pd.read_csv(landmarks_path)

    required_columns = {
        "frame",
        "timestamp_seconds",
        "person_index",
        "keypoint",
        "x_pixels",
        "y_pixels",
        "confidence",
    }

    if not required_columns.issubset(df.columns):
        raise ValueError("Landmark CSV is missing required columns.")

    df = select_primary_person(df)

    results = []

    for frame_number, frame in df.groupby("frame", sort=True):
        frame = frame.reset_index(drop=True)
        timestamp = float(frame["timestamp_seconds"].iloc[0])
        person_index = int(frame["person_index"].iloc[0])

        result = {
            "frame": int(frame_number),
            "timestamp_seconds": timestamp,
            "person_index": person_index,

            "left_elbow_angle": get_joint_angle(
                frame, "left_shoulder", "left_elbow", "left_wrist"
            ),
            "right_elbow_angle": get_joint_angle(
                frame, "right_shoulder", "right_elbow", "right_wrist"
            ),
            "left_knee_angle": get_joint_angle(
                frame, "left_hip", "left_knee", "left_ankle"
            ),
            "right_knee_angle": get_joint_angle(
                frame, "right_hip", "right_knee", "right_ankle"
            ),
            "left_shoulder_angle": get_joint_angle(
                frame, "left_elbow", "left_shoulder", "left_hip"
            ),
            "right_shoulder_angle": get_joint_angle(
                frame, "right_elbow", "right_shoulder", "right_hip"
            ),
            "left_hip_angle": get_joint_angle(
                frame, "left_shoulder", "left_hip", "left_knee"
            ),
            "right_hip_angle": get_joint_angle(
                frame, "right_shoulder", "right_hip", "right_knee"
            ),
        }

        results.append(result)

    if not results:
        raise ValueError("No joint-angle measurements could be calculated.")

    angles_df = pd.DataFrame(results)
    angles_df.to_csv(angles_path, index=False)

    print("User joint angles calculated:", len(angles_df))
    print("Joint angle file:", angles_path)

    return angles_df


# ==========================================
# ANGLE SMOOTHING
# ==========================================

def smooth_joint_angles(angles_df, smoothed_path):
    print("\nSmoothing joint angles...")

    group = angles_df.sort_values("frame").copy()

    for column in ANGLE_COLUMNS:
        values = group[column].interpolate(
            method="linear",
            limit=5,
            limit_area="inside",
        )

        group[column] = values.rolling(
            window=5,
            center=True,
            min_periods=1,
        ).median()

    group.to_csv(smoothed_path, index=False)

    print("Angle smoothing completed.")
    print("Smoothed file:", smoothed_path)

    return group


# ==========================================
# REFERENCE ANGLE EXTRACTION
# ==========================================

def calculate_reference_angle(frame_landmarks, first, middle, last):
    """
    Reference positions are 3D world coordinates.
    This calculates a 3D joint angle from the three mapped bone positions.
    """
    try:
        a = frame_landmarks[first]
        b = frame_landmarks[middle]
        c = frame_landmarks[last]

        return calculate_angle(
            [a["x"], a["y"], a["z"]],
            [b["x"], b["y"], b["z"]],
            [c["x"], c["y"], c["z"]],
        )
    except (KeyError, TypeError):
        return np.nan


def get_reference_folder(technique):
    return REFERENCE_DIR / technique


def load_reference_angles(technique, batting_hand="right", camera_view="side"):
    """Load the saved trainer's smoothed 2D joint-angle sequence."""
    folder = get_reference_folder(technique)
    metadata_path = folder / "metadata.json"
    angles_path = folder / "smoothed_joint_angles.csv"

    if not metadata_path.exists() or not angles_path.exists():
        return None, (
            f"No trainer reference is saved for {technique}. "
            "Save a completed trainer analysis first."
        )

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read trainer reference metadata: {error}")

    if metadata.get("status") != "ready":
        return None, "The trainer reference is not marked as ready."
    if metadata.get("batting_hand") != batting_hand:
        return None, "The player's batting hand does not match the trainer reference."
    if metadata.get("camera_view") != camera_view:
        return None, "The player's camera view does not match the trainer reference."

    try:
        reference_df = pd.read_csv(angles_path)
    except Exception as error:
        raise ValueError(f"Could not read trainer joint angles: {error}")

    required = {"timestamp_seconds", *ANGLE_COLUMNS}
    if not required.issubset(reference_df.columns) or len(reference_df) < 2:
        raise ValueError("Trainer reference joint-angle file is incomplete.")

    return reference_df, None


# TIME NORMALIZATION AND COMPARISON
# ==========================================

def resample_sequence(df, columns, sample_count=COMPARISON_SAMPLES):
    """
    Convert a movement sequence to a normalized 0..1 timeline.
    Each angle is interpolated independently.
    """
    clean = df.copy()

    if "timestamp_seconds" in clean.columns:
        clean = clean.sort_values("timestamp_seconds")

    clean = clean.reset_index(drop=True)

    if len(clean) < 2:
        raise ValueError("At least two frames are required for comparison.")

    source_time = np.linspace(0.0, 1.0, len(clean))
    target_time = np.linspace(0.0, 1.0, sample_count)

    sampled = {}

    for column in columns:
        values = pd.to_numeric(clean[column], errors="coerce").to_numpy(
            dtype=float
        )

        valid = np.isfinite(values)

        if valid.sum() < 2:
            sampled[column] = np.full(sample_count, np.nan)
            continue

        sampled[column] = np.interp(
            target_time,
            source_time[valid],
            values[valid],
        )

    return pd.DataFrame(sampled)


def compare_movement(user_df, reference_df):
    """
    Prototype score:
    - resample both sequences to a common normalized timeline
    - compare the eight angle trajectories
    - convert mean absolute angular error into a 0..100 similarity score

    This is not a validated cricket coaching score.
    """
    user_sequence = resample_sequence(user_df, ANGLE_COLUMNS)
    reference_sequence = resample_sequence(reference_df, ANGLE_COLUMNS)

    feature_results = {}
    weighted_scores = []
    weighted_errors = []

    for column in ANGLE_COLUMNS:
        user_values = user_sequence[column].to_numpy(dtype=float)
        reference_values = reference_sequence[column].to_numpy(dtype=float)

        valid = np.isfinite(user_values) & np.isfinite(reference_values)

        if valid.sum() < 10:
            feature_results[column] = {
                "similarity_percent": None,
                "mean_absolute_error_degrees": None,
                "valid_samples": int(valid.sum()),
                "status": "insufficient_data",
            }
            continue

        errors = np.abs(user_values[valid] - reference_values[valid])
        mae = float(np.mean(errors))

        # 90 degrees of mean error maps to zero.
        # This simple mapping is a prototype choice, not a validated threshold.
        similarity = float(np.clip(100.0 * (1.0 - mae / 90.0), 0.0, 100.0))

        feature_results[column] = {
            "similarity_percent": round(similarity, 2),
            "mean_absolute_error_degrees": round(mae, 2),
            "valid_samples": int(valid.sum()),
            "status": "compared",
        }

        weight = ANGLE_WEIGHTS[column]
        weighted_scores.append(similarity * weight)
        weighted_errors.append(weight)

    if not weighted_scores:
        return {
            "status": "unavailable",
            "message": (
                "Not enough reliable joint-angle data was detected "
                "to calculate a similarity score."
            ),
            "overall_similarity_percent": None,
            "feature_scores": feature_results,
        }

    overall = sum(weighted_scores) / sum(weighted_errors)

    return {
        "status": "experimental",
        "label": "Experimental joint-angle similarity",
        "overall_similarity_percent": round(float(overall), 2),
        "features_compared": len(weighted_scores),
        "comparison_samples": COMPARISON_SAMPLES,
        "feature_scores": feature_results,
        "limitations": [
            "The score measures similarity to the saved trainer recording, not objectively correct batting technique.",
            "Camera angle, perspective, occlusion and pose-detection errors can affect joint angles.",
            "The scoring formula has not been validated by cricket coaches or biomechanics experts.",
            "The current comparison uses eight body joint-angle patterns; it does not measure bat or ball movement.",
        ],
    }




# ==========================================
# FRAME-BY-FRAME VIDEO OVERLAY DATA
# ==========================================

def build_frame_overlay_data(landmarks_path, user_angles_df, reference_df):
    """
    Build compact, normalized YOLO keypoints and a frame-by-frame
    experimental similarity score for synchronised frontend canvas rendering.
    """
    landmarks_df = pd.read_csv(landmarks_path)
    primary_df = select_primary_person(landmarks_df)

    user_angles = user_angles_df.sort_values("frame").reset_index(drop=True)
    reference = (
        resample_sequence(reference_df, ANGLE_COLUMNS, len(user_angles))
        if reference_df is not None else None
    )

    angle_rows = {
        int(row["frame"]): row
        for _, row in user_angles.iterrows()
    }

    frames = []
    grouped = primary_df.groupby("frame", sort=True)

    for frame_number, frame_df in grouped:
        frame_number = int(frame_number)
        first_row = frame_df.iloc[0]
        timestamp = float(first_row["timestamp_seconds"])

        keypoints = {}
        for _, point in frame_df.iterrows():
            name = str(point["keypoint"])
            confidence = float(point["confidence"])
            if confidence < 0.20:
                continue
            keypoints[name] = {
                "x": float(point["x_normalized"]),
                "y": float(point["y_normalized"]),
                "confidence": confidence,
            }

        score = None
        user_row = angle_rows.get(frame_number)
        if user_row is not None and reference is not None:
            errors = []
            reference_index = (
                0 if len(user_angles) < 2 else
                int(round(
                    (user_angles.index[user_angles["frame"] == frame_number][0]
                     / (len(user_angles) - 1)) * (len(reference) - 1)
                ))
            )
            reference_row = reference.iloc[reference_index]

            for column in ANGLE_COLUMNS:
                try:
                    user_value = float(user_row[column])
                    ref_value = float(reference_row[column])
                except (TypeError, ValueError, KeyError):
                    continue
                if np.isfinite(user_value) and np.isfinite(ref_value):
                    errors.append(abs(user_value - ref_value))

            if len(errors) >= 2:
                score = round(float(np.clip(
                    100.0 * (1.0 - np.mean(errors) / 90.0), 0.0, 100.0
                )), 1)

        frames.append({
            "frame": frame_number,
            "timestamp_seconds": timestamp,
            "similarity_percent": score,
            "keypoints": keypoints,
        })

    return {
        "status": "success",
        "scoring_label": "Experimental frame-by-frame joint-angle similarity",
        "frames": frames,
        "skeleton_edges": [
            ["left_shoulder", "right_shoulder"],
            ["left_shoulder", "left_elbow"],
            ["left_elbow", "left_wrist"],
            ["right_shoulder", "right_elbow"],
            ["right_elbow", "right_wrist"],
            ["left_shoulder", "left_hip"],
            ["right_shoulder", "right_hip"],
            ["left_hip", "right_hip"],
            ["left_hip", "left_knee"],
            ["left_knee", "left_ankle"],
            ["right_hip", "right_knee"],
            ["right_knee", "right_ankle"],
        ],
    }


# ==========================================
# RENDER AN ACTUAL YOLO-ANNOTATED VIDEO
# ==========================================

def render_annotated_video(source_path, output_path, overlay_data):
    """Burn the detected YOLO skeleton and frame score into a separate video."""
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise ValueError("Could not reopen the uploaded video to render the YOLO overlay.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Could not read video dimensions for YOLO overlay rendering.")

    # WebM/VP8 is widely supported by Chromium browsers. Fall back to MP4V
    # on OpenCV builds that do not include a VP8 encoder.
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"VP80"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        output_path.unlink(missing_ok=True)
        output_path = output_path.with_suffix(".mp4")
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            "OpenCV could not create the annotated video. Check the FFmpeg codecs included with your OpenCV installation."
        )

    frames = overlay_data.get("frames", [])
    edges = overlay_data.get("skeleton_edges", [])
    frame_lookup = {int(item["frame"]): item for item in frames}
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            pose_frame = frame_lookup.get(frame_index)
            if pose_frame:
                points = pose_frame.get("keypoints", {})
                pixel_points = {}
                for name, point in points.items():
                    if float(point.get("confidence", 0)) < 0.20:
                        continue
                    x = int(np.clip(float(point["x"]), 0, 1) * (width - 1))
                    y = int(np.clip(float(point["y"]), 0, 1) * (height - 1))
                    pixel_points[name] = (x, y)

                # Draw the bones first, then joints.
                for start_name, end_name in edges:
                    start = pixel_points.get(start_name)
                    end = pixel_points.get(end_name)
                    if start and end:
                        cv2.line(frame, start, end, (99, 255, 217), max(3, width // 420), cv2.LINE_AA)

                for x, y in pixel_points.values():
                    radius = max(4, width // 300)
                    cv2.circle(frame, (x, y), radius + 2, (99, 255, 217), -1, cv2.LINE_AA)
                    cv2.circle(frame, (x, y), radius, (255, 255, 255), -1, cv2.LINE_AA)

                score = pose_frame.get("similarity_percent")
                score_text = f"{float(score):.1f}%" if score is not None else "N/A"
                # Compact dark HUD with the score burned into the video.
                panel_w = min(310, max(210, width // 3))
                panel_h = max(72, height // 7)
                overlay = frame.copy()
                cv2.rectangle(overlay, (18, 18), (18 + panel_w, 18 + panel_h), (10, 15, 8), -1)
                cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
                cv2.putText(frame, "ATHIX  |  YOLO POSE", (32, 18 + panel_h // 3), cv2.FONT_HERSHEY_SIMPLEX, max(0.45, width / 1800), (190, 205, 170), 1, cv2.LINE_AA)
                cv2.putText(frame, f"SIMILARITY  {score_text}", (32, 18 + (panel_h * 2) // 3), cv2.FONT_HERSHEY_SIMPLEX, max(0.55, width / 1450), (99, 255, 217), 2, cv2.LINE_AA)

            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    if frame_index == 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("The annotated video could not be generated.")

    return {"frames_rendered": frame_index, "fps": round(float(fps), 2), "width": width, "height": height, "filename": output_path.name}


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "message": "ATHIX backend is running",
    }


# ==========================================
# SAVE AN EXISTING ANALYSIS AS TRAINER REFERENCE
# ==========================================

@app.post("/api/references/save-latest")
async def save_latest_analysis_as_reference(
    upload_id: str = Form(...),
    technique: str = Form(...),
    batting_hand: str = Form("right"),
    camera_view: str = Form("side"),
    trainer_name: str = Form(""),
    replace_existing: bool = Form(False),
):
    """Save the four CSV outputs from a previous /api/analyze request."""
    if not re.fullmatch(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        upload_id,
    ):
        raise HTTPException(status_code=400, detail="Invalid analysis ID.")
    if technique not in VALID_TECHNIQUES:
        raise HTTPException(status_code=400, detail="Invalid batting technique.")
    if batting_hand not in VALID_HANDS:
        raise HTTPException(status_code=400, detail="batting_hand must be right or left.")
    if camera_view not in VALID_CAMERA_VIEWS:
        raise HTTPException(status_code=400, detail="Invalid camera view.")

    source_files = {
        "landmarks.csv": OUTPUT_DIR / f"{upload_id}_landmarks.csv",
        "joint_angles.csv": OUTPUT_DIR / f"{upload_id}_joint_angles.csv",
        "smoothed_joint_angles.csv": OUTPUT_DIR / f"{upload_id}_smoothed_joint_angles.csv",
        "movement_phases.csv": OUTPUT_DIR / f"{upload_id}_movement_phases.csv",
    }
    missing = [name for name, path in source_files.items() if not path.is_file()]
    if missing:
        raise HTTPException(
            status_code=404,
            detail="Analysis files missing: " + ", ".join(missing)
            + ". This analysis may have been created before the updated backend.",
        )

    angle_df = pd.read_csv(source_files["smoothed_joint_angles.csv"])
    required = {"timestamp_seconds", *ANGLE_COLUMNS}
    if not required.issubset(angle_df.columns) or len(angle_df) < 2:
        raise HTTPException(status_code=422, detail="Insufficient joint-angle data.")
    valid_angles = int(angle_df[ANGLE_COLUMNS].notna().sum().sum())
    if valid_angles == 0:
        raise HTTPException(status_code=422, detail="No valid joint angles were extracted.")

    phase_df = pd.read_csv(source_files["movement_phases.csv"])
    phase_counts = {}
    if "phase" in phase_df.columns:
        phase_counts = {
            str(key): int(value)
            for key, value in phase_df["phase"].value_counts(dropna=False).items()
        }

    reference_folder = get_reference_folder(technique)
    if reference_folder.exists() and not replace_existing:
        raise HTTPException(
            status_code=409,
            detail="A reference already exists for this technique. Confirm replacement first.",
        )

    staging = REFERENCE_DIR / f".staging_{uuid4()}"
    backup = REFERENCE_DIR / f".backup_{uuid4()}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for target_name, source_path in source_files.items():
            shutil.copy2(source_path, staging / target_name)

        metadata = {
            "reference_id": f"{technique}_{uuid4()}",
            "technique": technique,
            "batting_hand": batting_hand,
            "camera_view": camera_view,
            "trainer_name": trainer_name.strip() or "Not provided",
            "source_upload_id": upload_id,
            "total_frames": int(len(angle_df)),
            "valid_joint_angle_values": valid_angles,
            "phase_distribution": phase_counts,
            "files": list(source_files.keys()),
            "status": "ready",
            "verified_by_coach": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scoring_note": "Similarity is relative to this trainer video; it is not a validated measure of correct technique.",
        }
        with (staging / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        if reference_folder.exists():
            reference_folder.rename(backup)
        try:
            staging.rename(reference_folder)
        except Exception:
            if backup.exists():
                backup.rename(reference_folder)
            raise
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
    except Exception as error:
        shutil.rmtree(staging, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Could not save reference: {error}")

    return {
        "status": "success",
        "message": f"Trainer reference for {technique} saved.",
        "reference": metadata,
    }


@app.get("/api/references/{technique}")
def get_reference_status(technique: str):
    if technique not in VALID_TECHNIQUES:
        raise HTTPException(status_code=400, detail="Invalid batting technique.")
    metadata_path = get_reference_folder(technique) / "metadata.json"
    if not metadata_path.exists():
        return {"status": "missing", "technique": technique}
    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=500, detail="Reference metadata is unreadable.")
    return {"status": metadata.get("status", "unknown"), "reference": metadata}


# ==========================================
# MAIN VIDEO ANALYSIS ENDPOINT
# ==========================================

@app.post("/api/analyze")
async def analyze_video(
    video: UploadFile = File(...),
    technique: str = Form(...),
    batting_hand: str = Form("right"),
    camera_view: str = Form("side"),
):
    extension = Path(video.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only MP4, MOV and WebM videos are supported.",
        )

    if technique not in {
        "straight-drive",
        "cover-drive",
        "pull-shot",
        "cut-shot",
    }:
        raise HTTPException(
            status_code=400,
            detail="Invalid batting technique.",
        )

    if batting_hand not in VALID_HANDS:
        raise HTTPException(status_code=400, detail="batting_hand must be right or left.")
    if camera_view not in VALID_CAMERA_VIEWS:
        raise HTTPException(status_code=400, detail="Invalid camera view.")

    upload_id = str(uuid4())
    file_name = f"{upload_id}{extension}"
    file_path = UPLOAD_DIR / file_name

    landmarks_path = OUTPUT_DIR / f"{upload_id}_landmarks.csv"
    angles_path = OUTPUT_DIR / f"{upload_id}_joint_angles.csv"
    smoothed_path = OUTPUT_DIR / f"{upload_id}_smoothed_joint_angles.csv"
    phases_path = OUTPUT_DIR / f"{upload_id}_movement_phases.csv"

    total_size = 0

    # Save uploaded video in chunks and enforce size limit.
    try:
        with file_path.open("wb") as output:
            while True:
                chunk = await video.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the 20 MB limit.",
                    )

                output.write(chunk)

    except Exception:
        file_path.unlink(missing_ok=True)
        raise

    finally:
        await video.close()

    output_files = [
        landmarks_path,
        angles_path,
        smoothed_path,
        phases_path,
        OUTPUT_DIR / f"{upload_id}_overlay.json",
        OUTPUT_DIR / f"{upload_id}_annotated.webm",
        OUTPUT_DIR / f"{upload_id}_annotated.mp4",
    ]

    try:
        # VIDEO CONTINUITY VALIDATION
        scene_cuts = await run_in_threadpool(
            detect_scene_cuts,
            file_path,
        )

        if scene_cuts:
            cut_times = ", ".join(
                f"{time:.2f}s" for time in scene_cuts
            )

            raise HTTPException(
                status_code=422,
                detail=(
                    "Your video contains a scene cut or camera transition. "
                    f"Possible cut detected at: {cut_times}. "
                    "ATHIX requires one continuous recording of your batting shot. "
                    "Please upload a new video without cuts or transitions."
                ),
            )

        # YOLO LANDMARK EXTRACTION
        extraction_result = await run_in_threadpool(
            extract_landmarks,
            file_path,
            landmarks_path,
        )

        # JOINT ANGLE CALCULATION
        angles_df = await run_in_threadpool(
            calculate_joint_angles,
            landmarks_path,
            angles_path,
        )

        # ANGLE SMOOTHING
        smoothed_df = await run_in_threadpool(
            smooth_joint_angles,
            angles_df,
            smoothed_path,
        )

        # MOVEMENT PHASE DETECTION
        phase_result = await run_in_threadpool(
            detect_movement_phases,
            smoothed_path,
            phases_path,
        )

        # FRAME-BY-FRAME POSE OVERLAY DATA
        reference_df, reference_message = await run_in_threadpool(
            load_reference_angles,
            technique,
            batting_hand,
            camera_view,
        )

        overlay_path = OUTPUT_DIR / f"{upload_id}_overlay.json"
        overlay_data = await run_in_threadpool(
            build_frame_overlay_data,
            landmarks_path,
            smoothed_df,
            reference_df,
        )

        if overlay_data is not None:
            with overlay_path.open("w", encoding="utf-8") as overlay_file:
                json.dump(overlay_data, overlay_file, separators=(",", ":"))

        annotated_video_path = OUTPUT_DIR / f"{upload_id}_annotated.webm"
        annotated_video_result = await run_in_threadpool(
            render_annotated_video,
            file_path,
            annotated_video_path,
            overlay_data,
        ) if overlay_data is not None else None

        # AVERAGE JOINT ANGLES
        average_angles = {}

        for column in ANGLE_COLUMNS:
            average = smoothed_df[column].mean()

            average_angles[column] = (
                round(float(average), 2)
                if pd.notna(average)
                else None
            )

        # REFERENCE COMPARISON
        if reference_df is not None:
            similarity_result = await run_in_threadpool(
                compare_movement,
                smoothed_df,
                reference_df,
            )
        else:
            similarity_result = {
                "status": "unavailable",
                "label": "Similarity score unavailable",
                "overall_similarity_percent": None,
                "message": reference_message,
                "feature_scores": {},
            }

        return {
            "status": "success",
            "message": (
                "Video processed successfully. Pose landmarks, joint angles, "
                "smoothed measurements and movement phases generated."
            ),
            "upload_id": upload_id,
            "filename": file_name,
            "technique": technique,
            "batting_hand": batting_hand,
            "camera_view": camera_view,
            "size_mb": round(total_size / (1024 * 1024), 2),
            "video_validation": "passed",
            "analysis_status": "movement_segmentation_completed",
            "total_frames": extraction_result["total_frames"],
            "detected_frames": extraction_result["detected_frames"],
            "landmarks_file": landmarks_path.name,
            "joint_angles_file": angles_path.name,
            "smoothed_angles_file": smoothed_path.name,
            "movement_phases_file": phases_path.name,
            "phase_frames_analysed": phase_result["frames_analysed"],
            "phase_distribution": phase_result["phase_distribution"],
            "average_joint_angles": average_angles,
            "similarity": similarity_result,
            "pose_overlay": {
                "available": overlay_data is not None,
                "url": f"/outputs/{overlay_path.name}" if overlay_data is not None else None,
                "frame_count": len(overlay_data["frames"]) if overlay_data is not None else 0,
            },
            "annotated_video": {
                "available": annotated_video_result is not None,
                "url": f"/outputs/{annotated_video_result['filename']}" if annotated_video_result is not None else None,
                "frames_rendered": annotated_video_result["frames_rendered"] if annotated_video_result else 0,
                "fps": annotated_video_result["fps"] if annotated_video_result else None,
            },
        }

    except HTTPException:
        raise

    except Exception as error:
        print("\nATHIX ANALYSIS ERROR:", error)

        for output_file in output_files:
            output_file.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=(
                "ATHIX could not complete the analysis. "
                "Please check the backend terminal and try another video."
            ),
        )

    finally:
        file_path.unlink(missing_ok=True)
    