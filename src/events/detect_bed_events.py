"""
Bed exit and return event detection.
Scans timeline segments for the specific sequence patterns for bed exits and returns.
"""

import argparse
import json
import os
import math


IN_BED_STATES = {"LYING_IN_BED", "SITTING_ON_BED"}
OUT_OF_BED_ACTIVE_STATES = {"WALKING", "STANDING", "SITTING_OUTSIDE_BED", "OUT_OF_BED"}

NEAR_BED_DISTANCE_PX = 250


def confidence_for_segment(segment):
    flag = segment.get("transition_flag", "expected")
    if flag == "expected" or flag == "start_of_clip":
        return 0.92
    elif flag == "abrupt":
        return 0.65
    else:  # involves_unknown
        return 0.55


def detect_bed_exits(segments):
    """
    Looks for: IN_BED_STATE -> STANDING -> WALKING
    Confirmed at the moment WALKING begins.
    """
    events = []
    for i in range(len(segments) - 2):
        seg_bed = segments[i]
        seg_standing = segments[i + 1]
        seg_walking = segments[i + 2]

        if (seg_bed["state"] in IN_BED_STATES
                and seg_standing["state"] == "STANDING"
                and seg_walking["state"] == "WALKING"):
            events.append({
                "event": "bed_exit",
                "start_time": seg_standing["start_time"],
                "confirmed_time": seg_walking["start_time"],
                "previous_state": seg_bed["state"],
                "current_state": seg_walking["state"],
                "confidence": confidence_for_segment(seg_walking),
                "decision": "MONITOR"
            })
    return events


MIN_IN_BED_SEC_FOR_DIRECT_EXIT = 1.5  # the IN_BED segment must have lasted at
                                       # least this long to rule out flicker
                                       # misclassifications at the bed edge
                                       # (case04's SITTING_ON_BED was only 0.5s)


def detect_bed_exits_direct(segments):
    """
    Fallback pattern: IN_BED_STATE -> WALKING (skipping STANDING).
    Only valid if IN_BED segment lasted >= MIN_IN_BED_SEC_FOR_DIRECT_EXIT.
    Confidence is capped at 0.70 due to the missing intermediate state.
    """
    events = []
    for i in range(len(segments) - 1):
        seg_bed = segments[i]
        seg_walking = segments[i + 1]

        in_bed_duration = seg_bed["end_time_sec"] - seg_bed["start_time_sec"]
        if (seg_bed["state"] in IN_BED_STATES
                and seg_walking["state"] == "WALKING"
                and in_bed_duration >= MIN_IN_BED_SEC_FOR_DIRECT_EXIT):
            raw_conf = confidence_for_segment(seg_walking)
            capped_conf = min(raw_conf, 0.70)
            events.append({
                "event": "bed_exit",
                "start_time": seg_bed["start_time"],
                "confirmed_time": seg_walking["start_time"],
                "previous_state": seg_bed["state"],
                "current_state": seg_walking["state"],
                "confidence": capped_conf,
                "decision": "MONITOR",
            })
    return events


def detect_bed_exits_from_clip_start(segments, first_segment_near_bed):
    """
    Fallback pattern: Initial STANDING -> WALKING.
    Checks if person started near the bed when recording began, otherwise
    ignores to prevent flagging someone simply walking into the room.
    """
    events = []
    if len(segments) >= 2 and first_segment_near_bed:
        first, second = segments[0], segments[1]
        if first["state"] == "STANDING" and second["state"] == "WALKING":
            events.append({
                "event": "bed_exit",
                "start_time": first["start_time"],
                "confirmed_time": second["start_time"],
                "previous_state": "UNKNOWN (not visible before clip start)",
                "current_state": second["state"],
                "confidence": 0.5,
                "decision": "MONITOR",
            })
    return events

def detect_bed_returns(segments):
    """
    Looks for: OUT_OF_BED_ACTIVE_STATE -> SITTING_ON_BED -> LYING_IN_BED
    Confirmed at the moment LYING_IN_BED begins.
    """
    events = []
    for i in range(len(segments) - 2):
        seg_out = segments[i]
        seg_sitting = segments[i + 1]
        seg_lying = segments[i + 2]

        if (seg_out["state"] in OUT_OF_BED_ACTIVE_STATES
                and seg_sitting["state"] == "SITTING_ON_BED"
                and seg_lying["state"] == "LYING_IN_BED"):
            events.append({
                "event": "return_to_bed",
                "start_time": seg_sitting["start_time"],
                "confirmed_time": seg_lying["start_time"],
                "previous_state": seg_out["state"],
                "current_state": seg_lying["state"],
                "confidence": confidence_for_segment(seg_lying),
                "decision": "NORMAL"
            })
    return events

def compute_bed_centroid(bed_polygon):
    if not bed_polygon:
        return None
    xs = [p[0] for p in bed_polygon]
    ys = [p[1] for p in bed_polygon]
    return [sum(xs) / len(xs), sum(ys) / len(ys)]


def is_first_segment_near_bed(features_data, first_segment_end_time, bed_centroid):
    """
    Checks whether the person's average position during the clip's first
    segment (0 to first_segment_end_time) is close to the bed centroid.
    """
    if bed_centroid is None:
        return False  # no bed info - can't confirm proximity, so don't guess

    positions = []
    for frame in features_data:
        if frame["timestamp_sec"] > first_segment_end_time:
            break
        for person in frame["persons"]:
            center = person.get("box_center")
            if center:
                positions.append(center)

    if not positions:
        return False

    avg_x = sum(p[0] for p in positions) / len(positions)
    avg_y = sum(p[1] for p in positions) / len(positions)
    dist = math.hypot(avg_x - bed_centroid[0], avg_y - bed_centroid[1])
    return dist <= NEAR_BED_DISTANCE_PX

def detect_events(segments, features_data=None, bed_centroid=None):
    events = detect_bed_exits(segments) + detect_bed_returns(segments)

    if not any(e["event"] == "bed_exit" for e in events):
        # IN_BED -> WALKING (no STANDING intermediate)
        # Only fires if the IN_BED segment was long enough to be real (not flicker).
        events += detect_bed_exits_direct(segments)

    if not any(e["event"] == "bed_exit" for e in events):
        # clip starts already mid-exit (STANDING/WALKING at t=0)
        first_segment_near_bed = False
        if features_data is not None and bed_centroid is not None and segments:
            first_segment_near_bed = is_first_segment_near_bed(
                features_data, segments[0]["end_time_sec"], bed_centroid
            )
        events += detect_bed_exits_from_clip_start(segments, first_segment_near_bed)

    events.sort(key=lambda e: e["start_time"])
    return events

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect bed exit/return events from a timeline.")
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--features", default=None,
                         help="Optional path to features JSON, used to confirm bed "
                              "proximity for the clip-start exit fallback pattern.")
    parser.add_argument("--bed_region", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.timeline) as f:
        segments = json.load(f)

    features_data = None
    if args.features and os.path.exists(args.features):
        with open(args.features) as f:
            features_data = json.load(f)

    bed_centroid = None
    if args.bed_region and os.path.exists(args.bed_region):
        with open(args.bed_region) as f:
            bed_polygon = json.load(f)["bed_polygon"]
        bed_centroid = compute_bed_centroid(bed_polygon)

    events = detect_events(segments, features_data, bed_centroid)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(events, f, indent=2)

    print(f"Found {len(events)} event(s) -> {args.out}")
    for e in events:
        print(f"  {e['event']}: {e['start_time']} -> {e['confirmed_time']} "
              f"({e['previous_state']} -> {e['current_state']}, conf={e['confidence']})")