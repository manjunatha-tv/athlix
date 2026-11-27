
# ==========================================
# ATHIX 3D REFERENCE SKELETON MAPPING
# ==========================================

# Actual bone names obtained from cricket-batter.glb

REFERENCE_BONES = {
    "nose": "mixamorig:Head_06",

    "left_shoulder": "mixamorig:LeftArm_09",
    "right_shoulder": "mixamorig:RightArm_020",

    "left_elbow": "mixamorig:LeftForeArm_010",
    "right_elbow": "mixamorig:RightForeArm_021",

    "left_wrist": "mixamorig:LeftHand_00",
    "right_wrist": "mixamorig:RightHand_022",

    "left_hip": "mixamorig:LeftUpLeg_031",
    "right_hip": "mixamorig:RightUpLeg_036",

    "left_knee": "mixamorig:LeftLeg_032",
    "right_knee": "mixamorig:RightLeg_037",

    "left_ankle": "mixamorig:LeftFoot_033",
    "right_ankle": "mixamorig:RightFoot_038"
}


# Bones we want to analyse
IMPORTANT_BONES = [
    "mixamorig:Hips_01",
    "mixamorig:Spine_02",
    "mixamorig:Spine1_03",
    "mixamorig:Spine2_04",

    "mixamorig:LeftArm_09",
    "mixamorig:LeftForeArm_010",
    "mixamorig:LeftHand_00",

    "mixamorig:RightArm_020",
    "mixamorig:RightForeArm_021",
    "mixamorig:RightHand_022",

    "mixamorig:LeftUpLeg_031",
    "mixamorig:LeftLeg_032",
    "mixamorig:LeftFoot_033",

    "mixamorig:RightUpLeg_036",
    "mixamorig:RightLeg_037",
    "mixamorig:RightFoot_038"
]


ANIMATION_NAME = "Virat Playing Cricket"


def print_reference_mapping():

    print("\nATHIX REFERENCE SKELETON")
    print("========================")

    print("Reference animation:", ANIMATION_NAME)

    print("\nMapped body landmarks:")

    for landmark, bone in REFERENCE_BONES.items():
        print(f"{landmark:20} -> {bone}")

    print("\nImportant movement bones:")

    for bone in IMPORTANT_BONES:
        print("-", bone)

    print("\nTotal mapped landmarks:", len(REFERENCE_BONES))
    print("Total important bones:", len(IMPORTANT_BONES))


if __name__ == "__main__":
    print_reference_mapping()
