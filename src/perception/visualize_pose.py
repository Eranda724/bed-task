"""
Phase 1 - Step 5: Sanity-check visualization

Draws detected keypoints, skeleton connections, bounding box, track ID,
and the bed polygon onto sampled frames, and saves them as images.

This step exists because trusting numeric output (keypoint coordinates,
confidences) without visually confirming it first is how silent bugs slip
into a pipeline. Cheap to run, catches most integration mistakes early.

Usage:
    python visualize_pose.py --pose_json outputs/pose/case05_leaving_bed_pose.json \
                              --frames_dir outputs/frames/case05_leaving_bed \
                              --bed_region data/bed_regions/case05_leaving_bed.json \
                              --out_dir outputs/viz/case05_leaving_bed
"""

import argparse
import json
import os
import cv2
import numpy as np

# COCO skeleton connections (pairs of keypoint indices), matching the
# KEYPOINT_NAMES order used in detect_pose_track.py
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),          # arms + shoulders
    (5, 11), (6, 12), (11, 12),                       # torso
    (11, 13), (13, 15), (12, 14), (14, 16),           # legs
    (0, 1), (0, 2), (1, 3), (2, 4),                   # head
]

CONF_THRESHOLD = 0.3  # below this, skip drawing that keypoint


def draw_frame(image, frame_record, bed_polygon=None):
    img = image.copy()

    if bed_polygon:
        pts = np.array(bed_polygon, dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.putText(img, "BED", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2)

    for person in frame_record["persons"]:
        x1, y1, x2, y2 = map(int, person["box_xyxy"])
        track_id = person["track_id"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID:{track_id}" if track_id is not None else "ID:?"
        cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        kpts = person["keypoints"]
        coords = {}
        for k in kpts:
            if k["confidence"] is not None and k["confidence"] >= CONF_THRESHOLD:
                coords[k["name"]] = (int(k["x"]), int(k["y"]))
                cv2.circle(img, coords[k["name"]], 4, (0, 0, 255), -1)

        name_list = [k["name"] for k in kpts]
        for i, j in SKELETON:
            n1, n2 = name_list[i], name_list[j]
            if n1 in coords and n2 in coords:
                cv2.line(img, coords[n1], coords[n2], (0, 200, 255), 2)

    timestamp = frame_record.get("timestamp_sec")
    if timestamp is not None:
        cv2.putText(img, f"t={timestamp:.2f}s", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img


def visualize(pose_json_path: str, frames_dir: str, bed_region_path: str, out_dir: str):
    with open(pose_json_path) as f:
        pose_data = json.load(f)

    bed_polygon = None
    if bed_region_path and os.path.exists(bed_region_path):
        with open(bed_region_path) as f:
            bed_polygon = json.load(f)["bed_polygon"]

    os.makedirs(out_dir, exist_ok=True)

    for frame_record in pose_data:
        frame_path = os.path.join(frames_dir, frame_record["frame_file"])
        image = cv2.imread(frame_path)
        if image is None:
            print(f"Warning: could not read {frame_path}, skipping.")
            continue

        annotated = draw_frame(image, frame_record, bed_polygon)
        out_path = os.path.join(out_dir, frame_record["frame_file"])
        cv2.imwrite(out_path, annotated)

    print(f"Saved {len(pose_data)} annotated frames to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize pose/tracking/bed-region for sanity check.")
    parser.add_argument("--pose_json", required=True)
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--bed_region", default=None)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    visualize(args.pose_json, args.frames_dir, args.bed_region, args.out_dir)