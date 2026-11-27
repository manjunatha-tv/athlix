
# ==========================================
# ATHIX BATTING CONFIGURATION
# ==========================================

TECHNIQUE = "straight-drive"

# Current reference batter
BATTING_HAND = "right"

# YOLO anatomical keypoint names
BODY_SIDES = {
    "left": {
        "shoulder": "left_shoulder",
        "elbow": "left_elbow",
        "wrist": "left_wrist",
        "hip": "left_hip",
        "knee": "left_knee",
        "ankle": "left_ankle"
    },
    "right": {
        "shoulder": "right_shoulder",
        "elbow": "right_elbow",
        "wrist": "right_wrist",
        "hip": "right_hip",
        "knee": "right_knee",
        "ankle": "right_ankle"
    }
}


def get_batting_configuration(hand=BATTING_HAND):

    if hand not in ("left", "right"):
        raise ValueError(
            "Batting hand must be 'left' or 'right'."
        )

    opposite_hand = (
        "left" if hand == "right" else "right"
    )

    return {
        "technique": TECHNIQUE,
        "batting_hand": hand,
        "dominant_side": BODY_SIDES[hand],
        "opposite_side": BODY_SIDES[opposite_hand]
    }


if __name__ == "__main__":

    config = get_batting_configuration()

    print("\nATHIX BATTING CONFIGURATION")
    print("---------------------------")

    print("Technique:", config["technique"])
    print("Batting hand:", config["batting_hand"])

    print("\nDominant side:")

    for joint, landmark in config["dominant_side"].items():
        print(f"{joint}: {landmark}")

    print("\nOpposite side:")

    for joint, landmark in config["opposite_side"].items():
        print(f"{joint}: {landmark}")
