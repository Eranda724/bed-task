"""
Frame extraction tool.
Samples frames from a video at a fixed time interval (default every 0.5s).
"""

import argparse
import json
import os
import cv2


def extract_frames(video_path: str, interval_sec: float, out_dir: str):
    """
    Extract frames from video_path every interval_sec seconds.

    Returns a list of dicts: {frame_index, timestamp_sec, image_path}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError(f"Could not read FPS for: {video_path}")

    frame_interval = max(1, round(fps * interval_sec))

    os.makedirs(out_dir, exist_ok=True)

    records = []
    frame_index = 0
    saved_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            timestamp_sec = frame_index / fps
            out_path = os.path.join(out_dir, f"frame_{saved_index:04d}.jpg")
            cv2.imwrite(out_path, frame)
            records.append({
                "frame_index": saved_index,
                "timestamp_sec": round(timestamp_sec, 3),
                "image_path": out_path,
            })
            saved_index += 1

        frame_index += 1

    cap.release()

    # Save timestamps for later stages
    index_path = os.path.join(out_dir, "frames_index.json")
    with open(index_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Extracted {len(records)} frames from {video_path} "
          f"(video fps={fps:.2f}, sampling every {interval_sec}s)")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract sampled frames from a video.")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--interval", type=float, default=0.5,
                         help="Sampling interval in seconds (default: 0.5)")
    parser.add_argument("--out_dir", default=None,
                         help="Output directory for extracted frames "
                              "(default: outputs/frames/<video_name>/)")
    args = parser.parse_args()

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = args.out_dir or os.path.join("outputs", "frames", video_name)

    extract_frames(args.video, args.interval, out_dir)