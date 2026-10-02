"""
Phase 2 - Per-frame evidence extraction

Converts raw pose keypoints (from Phase 1) into meaningful numeric signals
per frame, per tracked person:
    - torso_angle_deg: angle of shoulder-to-hip line vs vertical
      (0 = upright, 90 = horizontal/lying)
    - height_width_ratio: bounding box height / width
      (low = lying/spread, high = standing/sitting upright)
    - bed_overlap_ratio: fraction of person's box area overlapping bed polygon
    - motion_speed: pixel distance moved by body center since previous frame
    - visibility_score: average confidence across all keypoints
      (low score -> frame is unreliable -> should become UNKNOWN later)

These signals are what Phase 3's state machine will threshold against,
instead of trying to classify raw pixels directly.

Usage:
    python compute_features.py --pose_json outputs/pose/case05_leaving_bed_pose.json \
                                --bed_region data/bed_regions/case05_leaving_bed.json \
                                --out outputs/features/case05_leaving_bed_features.json
"""

import argparse
import json
import math
import os


def get_keypoint(person, name):
    for kp in person["keypoints"]:
        if kp["name"] == name:
            return kp
    return None


def midpoint(kp1, kp2):
    if kp1 is None or kp2 is None:
        return None
    if kp1["confidence"] is None or kp2["confidence"] is None:
        return None
    if kp1["confidence"] < 0.3 or kp2["confidence"] < 0.3:
        return None
    return ((kp1["x"] + kp2["x"]) / 2, (kp1["y"] + kp2["y"]) / 2)


def compute_torso_angle(person):
    """
    Angle of the shoulder-midpoint -> hip-midpoint line, measured from
    vertical (0 = perfectly upright/vertical, 90 = perfectly horizontal).
    Returns None if shoulders/hips aren't confidently detected.
    """
    l_sh = get_keypoint(person, "left_shoulder")
    r_sh = get_keypoint(person, "right_shoulder")
    l_hip = get_keypoint(person, "left_hip")
    r_hip = get_keypoint(person, "right_hip")

    shoulder_mid = midpoint(l_sh, r_sh)
    hip_mid = midpoint(l_hip, r_hip)

    if shoulder_mid is None or hip_mid is None:
        return None

    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]

    if dx == 0 and dy == 0:
        return None

    # angle from vertical: 0 = straight up/down line, 90 = horizontal line
    angle_from_vertical = math.degrees(math.atan2(abs(dx), abs(dy)))
    return angle_from_vertical


def compute_height_width_ratio(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    return height / width


def compute_bed_overlap_ratio(box_xyxy, bed_polygon):
    """
    Approximates overlap using the bed polygon's bounding rectangle
    intersected with the person's box. Simple and explainable rather
    than exact polygon clipping - sufficient given our bed polygon is
    roughly rectangular (4 corners).
    """
    if not bed_polygon:
        return None

    xs = [p[0] for p in bed_polygon]
    ys = [p[1] for p in bed_polygon]
    bed_x1, bed_x2 = min(xs), max(xs)
    bed_y1, bed_y2 = min(ys), max(ys)

    px1, py1, px2, py2 = box_xyxy

    inter_x1 = max(px1, bed_x1)
    inter_y1 = max(py1, bed_y1)
    inter_x2 = min(px2, bed_x2)
    inter_y2 = min(py2, bed_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    person_area = max((px2 - px1) * (py2 - py1), 1e-6)
    return inter_area / person_area


def compute_visibility_score(person):
    confidences = [kp["confidence"] for kp in person["keypoints"] if kp["confidence"] is not None]
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)


def get_box_center(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def compute_features(pose_data, bed_polygon):
    features_out = []
    prev_centers = {}  # track_id -> (x, y) from previous frame

    for frame_record in pose_data:
        frame_features = {
            "frame_index": frame_record["frame_index"],
            "timestamp_sec": frame_record["timestamp_sec"],
            "persons": [],
        }

        for person in frame_record["persons"]:
            track_id = person["track_id"]
            box = person["box_xyxy"]

            torso_angle = compute_torso_angle(person)
            height_width_ratio = compute_height_width_ratio(box)
            bed_overlap = compute_bed_overlap_ratio(box, bed_polygon)
            visibility = compute_visibility_score(person)

            center = get_box_center(box)
            motion_speed = None

            if track_id is not None and track_id in prev_centers:
                # Reliable case: tracker gave us a stable ID
                prev_x, prev_y = prev_centers[track_id]
                motion_speed = math.hypot(center[0] - prev_x, center[1] - prev_y)
                prev_centers[track_id] = center
            elif track_id is None and "untracked_last" in prev_centers:

                prev_x, prev_y = prev_centers["untracked_last"]
                dist = math.hypot(center[0] - prev_x, center[1] - prev_y)
                motion_speed = dist if dist < 150 else None
                prev_centers["untracked_last"] = center
            else:
                prev_centers["untracked_last"] = center

            if track_id is not None:
                prev_centers[track_id] = center

            frame_features["persons"].append({
                "track_id": track_id,
                "torso_angle_deg": torso_angle,
                "height_width_ratio": round(height_width_ratio, 3),
                "bed_overlap_ratio": round(bed_overlap, 3) if bed_overlap is not None else None,
                "motion_speed": round(motion_speed, 2) if motion_speed is not None else None,
                "visibility_score": round(visibility, 3),
            })

        features_out.append(frame_features)

    return features_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute per-frame evidence features.")
    parser.add_argument("--pose_json", required=True)
    parser.add_argument("--bed_region", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.pose_json) as f:
        pose_data = json.load(f)
    with open(args.bed_region) as f:
        bed_polygon = json.load(f)["bed_polygon"]

    features = compute_features(pose_data, bed_polygon)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(features, f, indent=2)

    print(f"Computed features for {len(features)} frames -> {args.out}")