"""
Phase 4 - Batch runner: duration_summary.py for every clip timeline.

Usage:
    python run_phase4_all.py
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
        out_path = os.path.join("outputs", "summary", f"{base}_summary.json")
        run(["python", "src/outputs/duration_summary.py",
             "--timeline", timeline_path, "--out", out_path])

    print("\nAll clips processed through Phase 4.")


if __name__ == "__main__":
    main()