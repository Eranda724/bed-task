"""
Batch runner to detect bed exit/return events for every clip timeline.
"""

import os
import glob
import subprocess

TIMELINE_DIR = "outputs/timeline"


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    timeline_files = sorted(glob.glob(os.path.join(TIMELINE_DIR, "*_timeline.json")))

    for timeline_path in timeline_files:
        base = os.path.basename(timeline_path).replace("_timeline.json", "")
        out_path = os.path.join("outputs", "events", f"{base}_events.json")
        
        features_path = os.path.join("outputs", "features", f"{base}_features.json")
        bed_region_path = os.path.join("data", "bed_regions", f"{base}.json")
        run(["python", "src/events/detect_bed_events.py",
             "--timeline", timeline_path, "--features", features_path,
             "--bed_region", bed_region_path, "--out", out_path])

    print("\nEvent detection complete.")


if __name__ == "__main__":
    main()