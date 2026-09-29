"""
Phase 1 - Step 4: Person detection + pose estimation + tracking

Runs YOLOv8-pose on each sampled frame to get:
    - person bounding box
    - 17 body keypoints (COCO format) with per-keypoint confidence
    - a tracking ID (via Ultralytics' built-in ByteTrack), so the same
      person keeps the same ID across frames - this matters for the PDF's
      "caregiver entering the scene" difficult case, where two people may
      appear and we need to know which one is the patient being monitored.

Model choice: YOLOv8n-pose (the "nano" / smallest variant).
    - Free, open-source (Ultralytics).
    - Combines person detection + pose in a single forward pass, instead of
      needing a separate detector and a separate pose model.
    - "n" (nano) size chosen deliberately for CPU-only inference (AMD
      Radeon 850M has no CUDA support) - trades some accuracy for speed,
      acceptable since we only process ~30 sampled frames per 15s clip.

COCO keypoint order (used by YOLOv8-pose), 17 keypoints:
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

Usage:
    python detect_pose_track.py --frames_dir outputs/frames/case05_leaving_bed \
                                 --video_name case05_leaving_bed
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


def load_frame_index(frames_dir: str):
    """
    Loads timestamps saved by extract_frames.py (frames_index.json),
    so each frame's real video time can be attached to its detections.
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

        # persist=True keeps tracker state across calls so IDs stay
        # consistent across frames within this video
        results = model.track(source=frame_path, persist=True, verbose=False)
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