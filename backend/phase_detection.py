
import pandas as pd
import numpy as np
from pathlib import Path


MINIMUM_ACTIVITY_THRESHOLD = 8.0

ANGLE_COLUMNS = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_knee_angle",
    "right_knee_angle"
]


def detect_movement_phases(input_path, output_path):

    print("\nDetecting batting movement phases...")

    df = pd.read_csv(input_path)

    required_columns = [
        "frame",
        "timestamp_seconds",
        "person_index"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column missing: {column}"
            )

    available_angles = [
        column for column in ANGLE_COLUMNS
        if column in df.columns
    ]

    if not available_angles:
        raise ValueError("No joint-angle columns found.")

    results = []

    for person_index, group in df.groupby(
        "person_index",
        sort=True
    ):

        group = group.sort_values(
            "timestamp_seconds"
        ).copy()

        group = group.reset_index(drop=True)

        if len(group) < 10:
            print(
                f"Skipping person {person_index}: "
                "not enough frames."
            )
            continue

        time = group[
            "timestamp_seconds"
        ].to_numpy(dtype=float)

        velocities = []

        for column in available_angles:

            angles = group[column].to_numpy(dtype=float)

            velocity = np.full(
                len(angles),
                np.nan
            )

            for i in range(1, len(angles)):

                dt = time[i] - time[i - 1]

                if (
                    dt <= 0
                    or not np.isfinite(angles[i])
                    or not np.isfinite(angles[i - 1])
                ):
                    continue

                velocity[i] = abs(
                    angles[i] - angles[i - 1]
                ) / dt

            velocities.append(velocity)

        velocity_matrix = np.array(
            velocities
        ).T

        movement_score = np.full(
            len(group),
            np.nan
        )

        for i in range(len(group)):

            valid = velocity_matrix[i][
                np.isfinite(velocity_matrix[i])
            ]

            if len(valid) > 0:
                movement_score[i] = np.median(valid)

        group["movement_score"] = movement_score

        group["movement_score"] = (
            group["movement_score"]
            .rolling(
                window=5,
                center=True,
                min_periods=1
            )
            .median()
        )

        valid_scores = group[
            "movement_score"
        ].dropna()

        if len(valid_scores) < 5:
            continue

        median_score = valid_scores.median()

        mad = np.median(
            np.abs(valid_scores - median_score)
        )

        adaptive_threshold = (
            median_score + 2.5 * mad
        )

        threshold = max(
            MINIMUM_ACTIVITY_THRESHOLD,
            adaptive_threshold
        )

        print(
            f"Person {person_index} threshold:",
            round(threshold, 2),
            "degrees/second"
        )

        group["is_active"] = (
            group["movement_score"] > threshold
        )

        active = group[
            "is_active"
        ].to_numpy(dtype=bool)

        # Fill one-frame gaps.
        for i in range(1, len(active) - 1):

            if (
                active[i - 1]
                and not active[i]
                and active[i + 1]
            ):
                active[i] = True

        group["is_active"] = active

        active_indices = np.where(active)[0]

        phases = ["transition"] * len(group)

        if len(active_indices) == 0:

            phases = ["low_activity"] * len(group)

        else:

            first_active = active_indices[0]
            last_active = active_indices[-1]

            for i in range(len(group)):

                if i < first_active:
                    phases[i] = "preparation"

                elif active[i]:
                    phases[i] = "active_movement"

                elif i > last_active:
                    phases[i] = "follow_through"

                else:
                    phases[i] = "transition"

        group["is_active"] = active
        group["phase"] = phases

        results.append(group)

        print(
            "Active movement frames:",
            int(group["is_active"].sum())
        )

        print(
            "Phase counts:",
            group["phase"].value_counts().to_dict()
        )

    if not results:
        raise ValueError(
            "Could not identify usable movement data."
        )

    final_df = pd.concat(
        results,
        ignore_index=True
    )

    final_df.to_csv(
        output_path,
        index=False
    )

    phase_distribution = (
        final_df["phase"]
        .value_counts()
        .to_dict()
    )

    print("\nMOVEMENT SEGMENTATION COMPLETED")
    print("Frames analysed:", len(final_df))
    print("Output:", output_path)
    print("Phase distribution:", phase_distribution)

    return {
        "frames_analysed": len(final_df),
        "phase_distribution": {
            str(key): int(value)
            for key, value in phase_distribution.items()
        }
    }


# ==========================================
# MANUAL TESTING
# ==========================================

if __name__ == "__main__":

    output_dir = Path("outputs")

    files = list(
        output_dir.glob("*_smoothed_joint_angles.csv")
    )

    if not files:
        raise FileNotFoundError(
            "No smoothed joint-angle CSV found."
        )

    latest_file = max(
        files,
        key=lambda file: file.stat().st_mtime
    )

    output_file = output_dir / (
        latest_file.stem.replace(
            "_smoothed_joint_angles",
            "_movement_phases"
        ) + ".csv"
    )

    detect_movement_phases(
        latest_file,
        output_file
    )
