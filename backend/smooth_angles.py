
import pandas as pd

# Load calculated joint angles
df = pd.read_csv("joint_angles.csv")

angle_columns = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "left_hip_angle",
    "right_hip_angle"
]

# Create a separate dataframe
smoothed = df.copy()

for column in angle_columns:

    # Fill only very short gaps (maximum 5 consecutive frames)
    values = df[column].interpolate(
        method="linear",
        limit=5,
        limit_area="inside"
    )

    # Apply a rolling median to reduce sudden fluctuations
    smoothed[column] = values.rolling(
        window=5,
        center=True,
        min_periods=1
    ).median()

# Save smoothed measurements
smoothed.to_csv("smoothed_joint_angles.csv", index=False)

print("\nANGLE SMOOTHING COMPLETE")
print("------------------------")
print("Frames processed:", len(smoothed))
print("Output file: smoothed_joint_angles.csv")

print("\nRemaining missing measurements:")
print(smoothed[angle_columns].isna().sum())

print("\nSaved successfully!")
