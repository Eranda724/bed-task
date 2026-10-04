import glob
import os
import subprocess
import sys

def main():
    videos = sorted(glob.glob("data/videos/*.mp4"))
    if not videos:
        print("No videos found in data/videos/")
        return

    print(f"Found {len(videos)} videos.")
    print("For each video, a window will appear.")
    print("Click 4 corners of the bed, then press 's' to save and move to the next one.")
    print("Press 'q' to skip a video.")
    print("Press Ctrl+C in the console to abort the entire process.\n")

    for idx, video in enumerate(videos, start=1):
        print(f"[{idx}/{len(videos)}] Marking bed for: {video}")
        cmd = [sys.executable, "src/perception/mark_bed_region.py", "--video", video]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"Error marking {video}. Skipping...")
        except KeyboardInterrupt:
            print("\nProcess aborted by user.")
            break

    print("\nAll done!")

if __name__ == "__main__":
    main()
