"""
Temporal smoothing and segment building.
1. Removes short-lived flicker states (noise).
2. Merges identical consecutive states into timeline segments.
3. Flags transitions as 'expected' or 'abrupt' based on adjacency rules.
"""

import argparse
import json
import os
import math


MIN_FRAMES_TO_CONFIRM = 2
MAX_PLAUSIBLE_JUMP_PX = 150
MAX_DISTANCE_FROM_BED_PX = 220

# Physically adjacent states, used to flag abrupt transitions for review.
EXPECTED_NEIGHBORS = {
    "LYING_IN_BED": {"SITTING_ON_BED", "UNKNOWN"},
    "SITTING_ON_BED": {"LYING_IN_BED", "STANDING", "SITTING_OUTSIDE_BED", "UNKNOWN"},
    "SITTING_OUTSIDE_BED": {"STANDING", "SITTING_ON_BED", "UNKNOWN"},
    "STANDING": {"SITTING_ON_BED", "SITTING_OUTSIDE_BED", "WALKING", "OUT_OF_BED", "UNKNOWN"},
    "WALKING": {"STANDING", "OUT_OF_BED", "UNKNOWN"},
    "OUT_OF_BED": {"WALKING", "STANDING", "UNKNOWN"},
    "UNKNOWN": set(),  # UNKNOWN can follow/precede anything, never flagged itself
}

def _distance(p1, p2):
    if p1 is None or p2 is None:
        return float("inf")
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def compute_bed_centroid(bed_polygon):
    """
    Returns the [x, y] centroid of the bed polygon.
    """
    if not bed_polygon:
        return None
    xs = [p[0] for p in bed_polygon]
    ys = [p[1] for p in bed_polygon]
    return [sum(xs) / len(xs), sum(ys) / len(ys)]

def extract_primary_sequence(candidates_data, bed_centroid=None):
    """
    Extracts state sequence for the primary person.
    Uses bed-anchored disambiguation for multi-person frames.
    """
    has_multiple_people = any(len(f["persons"]) >= 2 for f in candidates_data)

    sequence = []

    if not has_multiple_people:
        # Simple case: trust the single detection each frame, no restriction.
        for frame in candidates_data:
            persons = frame["persons"]
            state = persons[0]["candidate_state"] if persons else "UNKNOWN"
            sequence.append({
                "frame_index": frame["frame_index"],
                "timestamp_sec": frame["timestamp_sec"],
                "state": state,
            })
        return sequence

    # Multi-person case: bed-anchored disambiguation
    for frame in candidates_data:
        persons = frame["persons"]
        state = "UNKNOWN"

        if persons:
            if bed_centroid is not None:
                chosen = min(persons, key=lambda p: _distance(p.get("box_center"), bed_centroid))
                dist = _distance(chosen.get("box_center"), bed_centroid)
                if dist <= MAX_DISTANCE_FROM_BED_PX:
                    state = chosen["candidate_state"]
            else:
                # No bed info available - fall back to first detection
                state = persons[0]["candidate_state"]

        sequence.append({
            "frame_index": frame["frame_index"],
            "timestamp_sec": frame["timestamp_sec"],
            "state": state,
        })

    return sequence

def remove_flicker(sequence, min_frames=MIN_FRAMES_TO_CONFIRM):
    """
    Collapses short runs of states into neighboring states to remove flicker.
    """
    if len(sequence) < 2:
        return sequence

    states = [s["state"] for s in sequence]
    cleaned = states.copy()

    i = 0
    while i < len(states):
        run_start = i
        run_state = states[i]
        run_end = i
        while run_end + 1 < len(states) and states[run_end + 1] == run_state:
            run_end += 1
        run_length = run_end - run_start + 1

        if run_length < min_frames:
            before_state = cleaned[run_start - 1] if run_start > 0 else None
            after_state = states[run_end + 1] if run_end + 1 < len(states) else None

            if before_state is not None and after_state is not None and before_state == after_state:
                for j in range(run_start, run_end + 1):
                    cleaned[j] = before_state
            elif before_state is not None and after_state is None:
                # flicker at the very end of the clip - extend the
                # preceding state forward instead of leaving a stray blip
                for j in range(run_start, run_end + 1):
                    cleaned[j] = before_state
            elif before_state is None and after_state is not None:
                # flicker at the very start of the clip - extend the
                # following state backward
                for j in range(run_start, run_end + 1):
                    cleaned[j] = after_state

        i = run_end + 1

    for idx, state in enumerate(cleaned):
        sequence[idx]["state"] = state

    return sequence

def build_segments(sequence, clip_end_sec=None):
    """
    Merges consecutive identical states into segments with start/end times.
    Segments tile the observed period without gaps.
    """
    if not sequence:
        return []

    segments = []
    current_state = sequence[0]["state"]
    current_start = sequence[0]["timestamp_sec"]

    for entry in sequence[1:]:
        if entry["state"] != current_state:
            segments.append({
                "state": current_state,
                "start_time_sec": current_start,
                "end_time_sec": entry["timestamp_sec"],  # end = next segment's start; no gap
            })
            current_state = entry["state"]
            current_start = entry["timestamp_sec"]

    # Last segment: extend to the true clip end
    last_detected = sequence[-1]["timestamp_sec"]
    true_end = clip_end_sec if clip_end_sec is not None else last_detected
    segments.append({
        "state": current_state,
        "start_time_sec": current_start,
        "end_time_sec": true_end,
    })

    # If detection ended before the clip end, append an explicit UNKNOWN tail
    if clip_end_sec is not None and last_detected < clip_end_sec - 1e-6:
        # Split the last segment if we need to append an UNKNOWN tail.
        if current_state != "UNKNOWN":
            segments[-1]["end_time_sec"] = last_detected  # shrink back to last detection
            segments.append({
                "state": "UNKNOWN",
                "start_time_sec": last_detected,
                "end_time_sec": true_end,
            })
        # If already UNKNOWN, no need to append another.

    return segments


def flag_transitions(segments):
    """
    Annotates segments with 'expected' or 'abrupt' transition flags based on EXPECTED_NEIGHBORS.
    """
    for i in range(1, len(segments)):
        prev_state = segments[i - 1]["state"]
        curr_state = segments[i]["state"]

        if prev_state == "UNKNOWN" or curr_state == "UNKNOWN":
            segments[i]["transition_flag"] = "involves_unknown"
        elif curr_state in EXPECTED_NEIGHBORS.get(prev_state, set()):
            segments[i]["transition_flag"] = "expected"
        else:
            segments[i]["transition_flag"] = "abrupt"

    if segments:
        segments[0]["transition_flag"] = "start_of_clip"

    return segments


def format_timestamp(seconds):
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"


def _compute_bed_centroid(candidates_data):
    """
    Estimates the bed centroid as the median box_center of bed-related states.
    """
    xs, ys = [], []
    for frame in candidates_data:
        for person in frame["persons"]:
            if person.get("candidate_state") in ("LYING_IN_BED", "SITTING_ON_BED"):
                center = person.get("box_center")
                if center:
                    xs.append(center[0])
                    ys.append(center[1])
    if not xs:
        return None
    # Use median to be robust to outlier detections
    xs.sort()
    ys.sort()
    return [xs[len(xs) // 2], ys[len(ys) // 2]]


def process(candidates_data, bed_centroid=None):
    # Determine the true clip end to extend the final segment properly.
    clip_end_sec = None
    if candidates_data:
        clip_end_sec = max(f["timestamp_sec"] for f in candidates_data)

    sequence = extract_primary_sequence(candidates_data, bed_centroid)
    sequence = remove_flicker(sequence)
    segments = build_segments(sequence, clip_end_sec=clip_end_sec)
    
    OUT_OF_BED_TRIGGERS = {"WALKING", "STANDING", "SITTING_OUTSIDE_BED"}
    if len(segments) > 1:
        last_idx = len(segments) - 1
        if segments[last_idx]["state"] == "UNKNOWN" and segments[last_idx-1]["state"] in OUT_OF_BED_TRIGGERS:
            segments[last_idx]["state"] = "OUT_OF_BED"

    segments = flag_transitions(segments)
    for seg in segments:
        seg["start_time"] = format_timestamp(seg["start_time_sec"])
        seg["end_time"] = format_timestamp(seg["end_time_sec"])

    return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smooth candidate states into a clean timeline.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--bed_region", default=None,
                         help="Optional path to bed region JSON, used to anchor "
                              "primary-person selection near the bed when multiple "
                              "people are detected in the same frame.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.candidates) as f:
        candidates_data = json.load(f)

    bed_centroid = None
    if args.bed_region and os.path.exists(args.bed_region):
        with open(args.bed_region) as f:
            bed_polygon = json.load(f)["bed_polygon"]
        bed_centroid = compute_bed_centroid(bed_polygon)

    segments = process(candidates_data, bed_centroid)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(segments, f, indent=2)

    print(f"Built {len(segments)} segments -> {args.out}")
    for seg in segments:
        flag = f" [{seg['transition_flag']}]" if seg['transition_flag'] != 'start_of_clip' else ""
        print(f"  {seg['start_time']} - {seg['end_time']}  {seg['state']}{flag}")