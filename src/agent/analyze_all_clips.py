"""
Batch runner to perform agentic analysis for every clip.
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
        events_path = os.path.join("outputs", "events", f"{base}_events.json")
        features_path = os.path.join("outputs", "features", f"{base}_features.json")
        out_path = os.path.join("outputs", "agent", f"{base}_reasoning.json")
        run(["python", "src/agent/agent_analysis.py",
             "--timeline", timeline_path, "--events", events_path, "--features", features_path, "--out", out_path])

    print("\nAgent analysis complete.")


if __name__ == "__main__":
    main()