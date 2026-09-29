"""
Merge per-clip ground truth CSVs into one timeline, offsetting each
clip's timestamps by the cumulative duration of preceding clips.

Usage:
    python merge_ground_truth.py --order case01_turning_in_bed case05_leaving_bed
"""

import argparse
import csv
import os
import cv2


def get_duration_sec(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frame_count / fps


def parse_time(t: str) -> float:
    parts = [float(p) for p in t.split(":")]
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    h, m, s = parts
    return h * 3600 + m * 60 + s


def format_time(t: float) -> str:
    m = int(t // 60)
    s = t % 60
    return f"{m:02d}:{s:05.2f}"


def merge(order, videos_dir, gt_dir, out_states_path, out_events_path):
    offset = 0.0
    merged_states = []
    merged_events = []

    for clip_name in order:
        video_path = os.path.join(videos_dir, f"{clip_name}.mp4")
        states_path = os.path.join(gt_dir, f"{clip_name}_states.csv")
        events_path = os.path.join(gt_dir, f"{clip_name}_events.csv")

        with open(states_path) as f:
            for row in csv.DictReader(f):
                merged_states.append({
                    "start_time": format_time(parse_time(row["start_time"]) + offset),
                    "end_time": format_time(parse_time(row["end_time"]) + offset),
                    "state": row["state"],
                })

        if os.path.exists(events_path):
            with open(events_path) as f:
                for row in csv.DictReader(f):
                    if row.get("event"):
                        merged_events.append({
                            "event": row["event"],
                            "time": format_time(parse_time(row["time"]) + offset),
                        })

        offset += get_duration_sec(video_path)

    with open(out_states_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["start_time", "end_time", "state"])
        writer.writeheader()
        writer.writerows(merged_states)

    with open(out_events_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["event", "time"])
        writer.writeheader()
        writer.writerows(merged_events)

    print(f"Merged {len(order)} clips, total duration {offset:.2f}s")
    print(f"Wrote {out_states_path} and {out_events_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", nargs="+", required=True,
                         help="Clip base names in merge order, e.g. case01_turning_in_bed case05_leaving_bed")
    parser.add_argument("--videos_dir", default="data/videos")
    parser.add_argument("--gt_dir", default="data/ground_truth")
    parser.add_argument("--out_states", default="data/ground_truth/merged_full_states.csv")
    parser.add_argument("--out_events", default="data/ground_truth/merged_full_events.csv")
    args = parser.parse_args()

    merge(args.order, args.videos_dir, args.gt_dir, args.out_states, args.out_events)