"""
Batch runner to generate duration summaries for every clip timeline.
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
        events_path = os.path.join("outputs", "events", f"{base}_events.json")
        run(["python", "src/outputs/duration_summary.py",
             "--timeline", timeline_path, "--events", events_path, "--out", out_path])

    print("\nSummary generation complete.")


if __name__ == "__main__":
    main()