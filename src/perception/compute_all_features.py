"""
Batch runner to compute features for clips with valid pose and bed region JSONs.
"""

import os
import glob
import subprocess

POSE_DIR = "outputs/pose"
BED_REGIONS_DIR = "data/bed_regions"


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    pose_files = sorted(glob.glob(os.path.join(POSE_DIR, "*_pose.json")))

    for pose_path in pose_files:
        name = os.path.basename(pose_path).replace("_pose.json", "")
        bed_region_path = os.path.join(BED_REGIONS_DIR, f"{name}.json")

        if not os.path.exists(bed_region_path):
            print(f"WARNING: no bed region for {name}, skipping.")
            continue

        out_path = os.path.join("outputs", "features", f"{name}_features.json")
        run(["python", "src/perception/compute_features.py",
             "--pose_json", pose_path, "--bed_region", bed_region_path, "--out", out_path])

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()