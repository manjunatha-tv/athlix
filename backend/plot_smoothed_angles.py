
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("smoothed_joint_angles.csv")

joint_groups = {
    "Elbow Movement": [
        "left_elbow_angle",
        "right_elbow_angle"
    ],
    "Shoulder Movement": [
        "left_shoulder_angle",
        "right_shoulder_angle"
    ],
    "Hip Movement": [
        "left_hip_angle",
        "right_hip_angle"
    ],
    "Knee Movement": [
        "left_knee_angle",
        "right_knee_angle"
    ]
}

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

for ax, (title, columns) in zip(axes, joint_groups.items()):

    for column in columns:
        ax.plot(
            df["timestamp_seconds"],
            df[column],
            label=column.replace("_angle", "").replace("_", " ")
        )

    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Angle (degrees)")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig("smoothed_angles_graph.png", dpi=300)
plt.show()

print("Graphs generated successfully!")
print("Saved as smoothed_angles_graph.png")
