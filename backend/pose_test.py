
from pathlib import Path
import cv2
from ultralytics import YOLO

# Folders
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Find the most recently uploaded cricket video
video_extensions = {".mp4", ".mov", ".webm"}

videos = [
    file for file in UPLOAD_DIR.iterdir()
    if file.suffix.lower() in video_extensions
] if UPLOAD_DIR.exists() else []

if not videos:
    print("No videos found in the uploads folder.")
    print("Upload a video through ATHIX first.")
    raise SystemExit

video_path = max(videos, key=lambda file: file.stat().st_mtime)

print("Input video:", video_path)

# Load YOLO11 Nano Pose model
print("Loading YOLO11 Pose model...")
model = YOLO("yolo11n-pose.pt")

# Open video
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Error: Could not open the video.")
    raise SystemExit

# Read video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 30

# Create output video
output_path = OUTPUT_DIR / "pose_output.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    print("Error: Could not create output video.")
    raise SystemExit

print("Starting pose detection...")

frame_count = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    # Detect human body keypoints
    results = model.predict(
        frame,
        conf=0.35,
        verbose=False
    )

    # Draw skeleton and keypoints
    annotated_frame = results[0].plot()

    # Save processed frame
    writer.write(annotated_frame)

    frame_count += 1

    if frame_count % 50 == 0:
        print("Processed frames:", frame_count)

# Release resources
cap.release()
writer.release()

print("\nPose detection completed!")
print("Total frames processed:", frame_count)
print("Output saved at:", output_path.resolve())
