"""
Phase 3 - Batch runner: classify_frames.py + smooth_states.py for every clip
that has a features JSON.

Usage:
    python run_phase3_all.py
"""

import os
import glob
import subprocess

FEATURES_DIR = "outputs/features"


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    feature_files = sorted(glob.glob(os.path.join(FEATURES_DIR, "*_features.json")))

    for feature_path in feature_files:
        base = os.path.basename(feature_path).replace("_features.json", "")
        print(f"\n========== Processing {base} ==========")

        candidates_path = os.path.join("outputs", "states", f"{base}_candidates.json")
        timeline_path = os.path.join("outputs", "timeline", f"{base}_timeline.json")

        run(["python", "src/temporal/classify_frames.py",
             "--features", feature_path, "--out", candidates_path])
        bed_region_path = os.path.join("data", "bed_regions", f"{base}.json")
        run(["python", "src/temporal/smooth_states.py",
             "--candidates", candidates_path, "--bed_region", bed_region_path, "--out", timeline_path])

    print("\nAll clips processed through Phase 3.")


if __name__ == "__main__":
    main()