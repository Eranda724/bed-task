"""
Person detection, pose estimation, and tracking.
Runs YOLOv8-pose on sampled frames to extract bounding boxes, 
17 body keypoints, and track IDs. Includes custom deduplication 
to remove overlapping false-positive bounding boxes.

"""

import argparse
import json
import os
from ultralytics import YOLO


KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def compute_iou(box1, box2):
    """IoU (intersection over union) between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter_area

    return inter_area / union if union > 0 else 0.0


def deduplicate_persons(persons, iou_threshold=0.5):
    """
    Removes duplicate detections of the same person. When two detections
    overlap above iou_threshold, keep only the one with higher detection
    confidence.
    """
    kept = []
    sorted_persons = sorted(persons, key=lambda p: p["detection_confidence"], reverse=True)

    for person in sorted_persons:
        is_duplicate = False
        for kept_person in kept:
            if compute_iou(person["box_xyxy"], kept_person["box_xyxy"]) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(person)

    return kept


def load_frame_index(frames_dir: str):
    """
    Loads timestamps saved by extract_frames.py (frames_index.json).
    """
    index_path = os.path.join(frames_dir, "frames_index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)
    return None


def run_pose_tracking(frames_dir: str, out_path: str, model_name: str = "yolov8n-pose.pt"):
    model = YOLO(model_name)

    frame_files = sorted(
        f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".jpg")
    )
    frame_index_data = load_frame_index(frames_dir)

    results_out = []

    for i, fname in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, fname)

        results = model.track(source=frame_path, persist=True, verbose=False, conf=0.5)
        result = results[0]

        timestamp_sec = None
        if frame_index_data and i < len(frame_index_data):
            timestamp_sec = frame_index_data[i]["timestamp_sec"]

        frame_record = {
            "frame_index": i,
            "frame_file": fname,
            "timestamp_sec": timestamp_sec,
            "persons": [],
        }

        if result.keypoints is not None and result.boxes is not None:
            boxes = result.boxes
            keypoints = result.keypoints

            for person_idx in range(len(boxes)):
                track_id = int(boxes.id[person_idx].item()) if boxes.id is not None else None
                box = boxes.xyxy[person_idx].tolist()
                conf = float(boxes.conf[person_idx].item())

                kpts_xy = keypoints.xy[person_idx].tolist()
                kpts_conf = keypoints.conf[person_idx].tolist() if keypoints.conf is not None else None

                keypoint_records = []
                for k_idx, name in enumerate(KEYPOINT_NAMES):
                    x, y = kpts_xy[k_idx]
                    kconf = kpts_conf[k_idx] if kpts_conf else None
                    keypoint_records.append({
                        "name": name, "x": x, "y": y, "confidence": kconf
                    })

                frame_record["persons"].append({
                    "track_id": track_id,
                    "box_xyxy": box,
                    "detection_confidence": conf,
                    "keypoints": keypoint_records,
                })

        frame_record["persons"] = deduplicate_persons(frame_record["persons"])
        results_out.append(frame_record)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results_out, f, indent=2)

    print(f"Processed {len(frame_files)} frames -> {out_path}")
    return results_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pose detection + tracking on extracted frames.")
    parser.add_argument("--frames_dir", required=True, help="Directory of extracted frames")
    parser.add_argument("--video_name", required=True, help="Name used for output file")
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    out_path = args.out or os.path.join("outputs", "pose", f"{args.video_name}_pose.json")
    run_pose_tracking(args.frames_dir, out_path)