
import pandas as pd
from pathlib import Path

from batting_config import get_batting_configuration

OUTPUT_DIR = Path("outputs")

config = get_batting_configuration()

batting_side = config["batting_hand"]
opposite_side = "left" if batting_side == "right" else "right"

angle_files = list(OUTPUT_DIR.glob("*_smoothed_joint_angles.csv"))
phase_files = list(OUTPUT_DIR.glob("*_movement_phases.csv"))

if not angle_files or not phase_files:
    print("Required analysis files not found.")
    raise SystemExit

angle_file = max(angle_files, key=lambda f: f.stat().st_mtime)
phase_file = max(phase_files, key=lambda f: f.stat().st_mtime)

angles = pd.read_csv(angle_file)
phases = pd.read_csv(phase_file)

# Match frames and people where possible
merge_keys = ["frame"]

if "person_index" in angles.columns and "person_index" in phases.columns:
    merge_keys.append("person_index")

if "phase" not in phases.columns:
    print("Movement phase column not found.")
    raise SystemExit

data = pd.merge(
    angles,
    phases[merge_keys + ["phase"]],
    on=merge_keys,
    how="inner"
)

if data.empty:
    print("No matching frames found.")
    raise SystemExit

# Build metrics using the selected batting hand
metrics = {
    "Batting-side elbow": f"{batting_side}_elbow_angle",
    "Opposite-side elbow": f"{opposite_side}_elbow_angle",
    "Batting-side shoulder": f"{batting_side}_shoulder_angle",
    "Opposite-side shoulder": f"{opposite_side}_shoulder_angle",
    "Batting-side hip": f"{batting_side}_hip_angle",
    "Opposite-side hip": f"{opposite_side}_hip_angle",
    "Batting-side knee": f"{batting_side}_knee_angle",
    "Opposite-side knee": f"{opposite_side}_knee_angle"
}

available_metrics = {
    label: column
    for label, column in metrics.items()
    if column in data.columns
}

if not available_metrics:
    print("No required joint measurements found.")
    raise SystemExit

# Calculate statistics for each movement phase
rows = []

for phase_name, group in data.groupby("phase"):

    for label, column in available_metrics.items():

        values = group[column].dropna()

        rows.append({
            "phase": phase_name,
            "measurement": label,
            "mean_angle": round(values.mean(), 2) if not values.empty else None,
            "minimum_angle": round(values.min(), 2) if not values.empty else None,
            "maximum_angle": round(values.max(), 2) if not values.empty else None,
            "valid_frames": len(values)
        })

report = pd.DataFrame(rows)

output_file = OUTPUT_DIR / "straight_drive_batting_report.csv"

report.to_csv(output_file, index=False)

print("\nATHIX STRAIGHT DRIVE REPORT")
print("---------------------------")
print("Batting hand:", batting_side)
print("Frames analysed:", len(data))
print("\nPhase-wise measurements:")
print(report.to_string(index=False))
print("\nSaved:", output_file.resolve())
