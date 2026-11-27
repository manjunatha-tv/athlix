
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("outputs")

# Find the latest analysis files
angle_files = list(OUTPUT_DIR.glob("*_smoothed_joint_angles.csv"))
phase_files = list(OUTPUT_DIR.glob("*_movement_phases.csv"))

if not angle_files or not phase_files:
    print("Required analysis files were not found.")
    raise SystemExit

angle_file = max(angle_files, key=lambda f: f.stat().st_mtime)
phase_file = max(phase_files, key=lambda f: f.stat().st_mtime)

print("Loading:", angle_file.name)
print("Loading:", phase_file.name)

angles = pd.read_csv(angle_file)
phases = pd.read_csv(phase_file)

# Combine joint angles and movement phases
df = pd.merge(
    angles,
    phases[["frame", "phase"]],
    on="frame",
    how="inner"
)

if df.empty:
    print("No matching frames found between the files.")
    raise SystemExit

# Use timestamps for the horizontal axis
time = df["timestamp_seconds"]

# Joint angle groups
angle_groups = {
    "Elbow Angles": [
        "left_elbow_angle",
        "right_elbow_angle"
    ],
    "Knee Angles": [
        "left_knee_angle",
        "right_knee_angle"
    ]
}

# Phase colours
phase_colors = {
    "preparation": "green",
    "active_movement": "orange",
    "transition": "purple",
    "follow_through": "blue"
}

fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

for ax, (title, columns) in zip(axes, angle_groups.items()):

    # Draw joint angle lines
    for column in columns:
        if column in df.columns:
            ax.plot(
                time,
                df[column],
                marker="o",
                linewidth=2,
                label=column.replace("_", " ").title()
            )

    # Highlight movement phases
    current_phase = None
    start_time = None

    for i in range(len(df)):
        phase = df["phase"].iloc[i]

        if phase != current_phase:
            if current_phase is not None:
                ax.axvspan(
                    start_time,
                    time.iloc[i],
                    color=phase_colors.get(current_phase, "grey"),
                    alpha=0.12
                )

            current_phase = phase
            start_time = time.iloc[i]

    # Finish the final phase
    if current_phase is not None:
        ax.axvspan(
            start_time,
            time.iloc[-1],
            color=phase_colors.get(current_phase, "grey"),
            alpha=0.12
        )

    ax.set_title(title)
    ax.set_ylabel("Angle (degrees)")
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[-1].set_xlabel("Time (seconds)")

fig.suptitle("ATHIX - Straight Drive Joint Angle Analysis")
plt.tight_layout()

# Save graph
graph_path = OUTPUT_DIR / "straight_drive_angle_graph.png"
plt.savefig(graph_path, dpi=150)

print("\nGRAPH GENERATED SUCCESSFULLY")
print("Saved at:", graph_path.resolve())

plt.show()
