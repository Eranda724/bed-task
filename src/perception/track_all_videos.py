"""
Batch process for video perception (frames, pose tracking, visualization).
Expects bed region JSONs to be manually created via mark_bed_region.py.
"""

import os
import glob
import subprocess
import sys

VIDEOS_DIR = "data/videos"
BED_REGIONS_DIR = "data/bed_regions"


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    video_paths = sorted(glob.glob(os.path.join(VIDEOS_DIR, "*.mp4")))

    for video_path in video_paths:
        name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\nProcessing {name}...")

        bed_region_path = os.path.join(BED_REGIONS_DIR, f"{name}.json")
        if not os.path.exists(bed_region_path):
            print(f"WARNING: no bed region for {name}, run mark_bed_region.py first. Skipping.")
            continue

        frames_dir = os.path.join("outputs", "frames", name)
        pose_json = os.path.join("outputs", "pose", f"{name}_pose.json")
        viz_dir = os.path.join("outputs", "viz", name)

        run([sys.executable, "src/perception/extract_frames.py", "--video", video_path])
        run([sys.executable, "src/perception/detect_pose_track.py",
             "--frames_dir", frames_dir, "--video_name", name])
        run([sys.executable, "src/perception/visualize_pose.py",
             "--pose_json", pose_json, "--frames_dir", frames_dir,
             "--bed_region", bed_region_path, "--out_dir", viz_dir])

    print("\nAll clips processed.")


if __name__ == "__main__":
    main()