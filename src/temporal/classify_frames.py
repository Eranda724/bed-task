"""
Phase 3 - Stage A: Per-frame candidate state classification

Converts per-frame features (Phase 2) into a candidate state label per
frame, using threshold rules derived from observed data:
    - LYING_IN_BED:        torso_angle > 60 deg, on bed
    - SITTING_ON_BED:      torso_angle 30-60 deg (or upright+low motion), on bed
    - SITTING_OUTSIDE_BED: same posture, NOT on bed
    - STANDING:            torso_angle < 30 deg, low motion, upright ratio
    - WALKING:             torso_angle < 30 deg, high motion
    - OUT_OF_BED:          no confident posture read, but recently off-bed
    - UNKNOWN:             low visibility / no detection / insufficient evidence

This is Stage A only - a per-frame guess. Stage B (temporal smoothing)
will clean up flicker and enforce valid transitions.

Usage:
    python classify_frames.py --features outputs/features/case05_leaving_bed_features.json \
                               --out outputs/states/case05_leaving_bed_candidates.json
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

    # On-bed decided by EITHER signal confirming it - the hip-point test
    # is accurate but flickers right at the bed polygon's edge (jittery
    # pose near a boundary); bed_overlap_ratio is more stable but less
    # precise when a person sits upright with much of their box above the
    # bed's flat footprint. Combining with OR uses whichever signal is
    # more reliable in each situation.
    on_bed = bool(on_bed_point) or (bed_overlap is not None and bed_overlap >= BED_OVERLAP_ON_BED_MIN)

    # Lying requires BOTH a horizontal torso AND a low, wide box - using
    # AND (not OR) prevents a single borderline signal from overriding a
    # clearly contradicting one (e.g. height/width dipping to 0.79 during
    # a sitting posture while torso_angle correctly shows upright).
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