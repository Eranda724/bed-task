"""
Per-frame candidate state classification.
Converts per-frame features into a candidate state label per
frame, using threshold rules derived from observed data.
"""


import argparse
import json
import os


# Thresholds - derived from real observed data:
#   lying:    torso_angle ~86-88 deg, height/width ~0.25-0.30
#   standing/walking: torso_angle ~1-4 deg, height/width ~3.0+
TORSO_ANGLE_LYING_MIN = 60.0
TORSO_ANGLE_UPRIGHT_MAX = 30.0
HEIGHT_WIDTH_LYING_MAX = 0.6   # tightened: real sitting data ranges 0.57-1.82,
                                # overlapping the old 0.8 cutoff; lying data is 0.25-0.47
HEIGHT_WIDTH_UPRIGHT_MIN = 2.0
MOTION_WALKING_MIN = 10.0
BED_OVERLAP_ON_BED_MIN = 0.3
VISIBILITY_MIN = 0.35

def classify_single_person(person_features):
    torso_angle = person_features.get("torso_angle_deg")
    height_width = person_features.get("height_width_ratio")
    bed_overlap = person_features.get("bed_overlap_ratio")
    on_bed_point = person_features.get("on_bed_point")
    motion = person_features.get("motion_speed")
    visibility = person_features.get("visibility_score", 0.0)

    if visibility is None or visibility < VISIBILITY_MIN:
        return "UNKNOWN", "low_visibility"
    if torso_angle is None or height_width is None:
        return "UNKNOWN", "missing_pose_signal"

    # Use OR to combine hip-point and bed-overlap signals for stability.
    on_bed = bool(on_bed_point) or (bed_overlap is not None and bed_overlap >= BED_OVERLAP_ON_BED_MIN)

    # Lying requires BOTH a horizontal torso AND a low, wide box.
    if torso_angle >= TORSO_ANGLE_LYING_MIN and height_width <= HEIGHT_WIDTH_LYING_MAX:
        if on_bed:
            return "LYING_IN_BED", "horizontal_on_bed"
        else:
            return "UNKNOWN", "horizontal_off_bed_unusual"

    if torso_angle <= TORSO_ANGLE_UPRIGHT_MAX and height_width >= HEIGHT_WIDTH_UPRIGHT_MIN:
        if motion is not None and motion >= MOTION_WALKING_MIN:
            return "WALKING", "upright_moving"
        else:
            return "STANDING", "upright_still"

    if on_bed:
        return "SITTING_ON_BED", "bent_posture_on_bed"
    else:
        return "SITTING_OUTSIDE_BED", "bent_posture_off_bed"

def classify_frames(features_data):
    output = []
    for frame in features_data:
        frame_result = {
            "frame_index": frame["frame_index"],
            "timestamp_sec": frame["timestamp_sec"],
            "persons": [],
        }

        if not frame["persons"]:
            # No detection at all this frame
            frame_result["persons"].append({
                "track_id": None,
                "candidate_state": "UNKNOWN",
                "reason": "no_detection",
            })
        else:
            for person in frame["persons"]:
                state, reason = classify_single_person(person)
                frame_result["persons"].append({
                    "track_id": person["track_id"],
                    "candidate_state": state,
                    "reason": reason,
                    "box_center": person.get("box_center"),
                })

        output.append(frame_result)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify per-frame candidate states.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.features) as f:
        features_data = json.load(f)

    results = classify_frames(features_data)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Classified {len(results)} frames -> {args.out}")