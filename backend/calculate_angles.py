
import pandas as pd
import numpy as np

# Input and output files
INPUT_FILE = "landmarks.csv"
OUTPUT_FILE = "joint_angles.csv"

# Minimum confidence required for a landmark
CONFIDENCE_THRESHOLD = 0.5

# Load extracted pose landmarks
df = pd.read_csv(INPUT_FILE)


def calculate_angle(a, b, c):
    """
    Calculate the angle at point B
    using three 2D coordinates.
    """

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


def get_point(frame, landmark):
    """Return coordinates only if confidence is sufficient."""

    point = frame[frame["keypoint"] == landmark]

    if point.empty:
        return None

    row = point.iloc[0]

    if row["confidence"] < CONFIDENCE_THRESHOLD:
        return None

    return [row["x_pixels"], row["y_pixels"]]


def get_joint_angle(frame, first, middle, last):
    """Calculate a joint angle using three landmarks."""

    a = get_point(frame, first)
    b = get_point(frame, middle)
    c = get_point(frame, last)

    if a is None or b is None or c is None:
        return np.nan

    return calculate_angle(a, b, c)


results = []

# Process each video frame
for frame_number, frame in df.groupby("frame"):

    frame = frame.reset_index(drop=True)

    timestamp = frame["timestamp_seconds"].iloc[0]

    result = {
        "frame": frame_number,
        "timestamp_seconds": timestamp,

        # Elbow angles
        "left_elbow_angle": get_joint_angle(
            frame, "left_shoulder", "left_elbow", "left_wrist"
        ),

        "right_elbow_angle": get_joint_angle(
            frame, "right_shoulder", "right_elbow", "right_wrist"
        ),

        # Knee angles
        "left_knee_angle": get_joint_angle(
            frame, "left_hip", "left_knee", "left_ankle"
        ),

        "right_knee_angle": get_joint_angle(
            frame, "right_hip", "right_knee", "right_ankle"
        ),

        # Shoulder angles
        "left_shoulder_angle": get_joint_angle(
            frame, "left_elbow", "left_shoulder", "left_hip"
        ),

        "right_shoulder_angle": get_joint_angle(
            frame, "right_elbow", "right_shoulder", "right_hip"
        ),

        # Hip angles
        "left_hip_angle": get_joint_angle(
            frame, "left_shoulder", "left_hip", "left_knee"
        ),

        "right_hip_angle": get_joint_angle(
            frame, "right_shoulder", "right_hip", "right_knee"
        )
    }

    results.append(result)


# Convert results into a DataFrame
angles_df = pd.DataFrame(results)

# Save results
angles_df.to_csv(OUTPUT_FILE, index=False)

print("\nJOINT ANGLE CALCULATION COMPLETE")
print("--------------------------------")
print("Frames processed:", len(angles_df))
print("Output file:", OUTPUT_FILE)

print("\nAverage joint angles:")
print(
    angles_df.drop(
        columns=["frame", "timestamp_seconds"]
    ).mean().round(2)
)

print("\nMissing measurements:")
print(
    angles_df.isna().sum()
)

print("\nSaved successfully!")
