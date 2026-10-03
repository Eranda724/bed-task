"""
Phase 3 - Stage B: Temporal smoothing and segment building

Takes Stage A's per-frame candidate states and:
    1. Removes short-lived "flicker" states (a state lasting fewer than
       MIN_FRAMES_TO_CONFIRM consecutive frames is noise, not a real
       transition - e.g. one frame misread as STANDING during continuous
       WALKING due to a brief dip in motion_speed).
    2. Merges consecutive identical states into clean segments, which
       directly form the timeline (PDF section 4).
    3. Flags transitions between segments as "expected" or "abrupt" based
       on a simple adjacency rule (e.g. LYING_IN_BED -> WALKING with no
       STANDING in between is physically unusual and worth flagging for
       the agent to double check in Phase 6).

This only tracks ONE person (the primary tracked person) per clip for now,
since all current state logic assumes a single patient per frame. Phase 6
(agentic analysis) will handle multi-person disambiguation for the
caregiver case.

Usage:
    python smooth_states.py --candidates outputs/states/case05_leaving_bed_candidates.json \
                             --out outputs/timeline/case05_leaving_bed_timeline.json
"""

import argparse
import json
import os
import math


MIN_FRAMES_TO_CONFIRM = 2  # a state must persist this many consecutive
                            # frames before we accept it as real, not noise
MAX_PLAUSIBLE_JUMP_PX = 150
MAX_DISTANCE_FROM_BED_PX = 220

# States that are physically adjacent / expected to transition directly
# into each other. Used only to flag abrupt transitions for review, not
# to block them outright (the agent decides what to do with the flag).
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
    Returns the [x, y] centroid (average of corner points) of the bed
    polygon, used as a spatial anchor: the patient is expected to stay
    closer to the bed than a visiting caregiver on average.
    """
    if not bed_polygon:
        return None
    xs = [p[0] for p in bed_polygon]
    ys = [p[1] for p in bed_polygon]
    return [sum(xs) / len(xs), sum(ys) / len(ys)]

def extract_primary_sequence(candidates_data, bed_centroid=None):
    """
    Pulls out one state value per frame for the PRIMARY (patient) person.

    Uses two different strategies depending on the clip:

    1. SINGLE-PERSON CLIPS (no frame ever has 2+ simultaneous detections):
       The vast majority of clips. There's never anyone to disambiguate
       between, so we simply trust whichever single detection is present
       each frame. No bed-distance restriction is applied here, since
       legitimate behavior (walking away from the bed, leaving the room)
       moves far from the bed by definition - restricting by bed distance
       would wrongly reject exactly the behavior these clips are meant to
       test.

    2. MULTI-PERSON CLIPS (at least one frame has 2+ people, e.g. a
       caregiver scenario): Disambiguation is genuinely needed. Here we
       anchor to the bed centroid, since the patient is expected to stay
       closer to the bed, on average, than a visiting caregiver who moves
       around the room. Frames where even the closest candidate is too far
       from the bed are marked UNKNOWN rather than guessing.
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
    Collapses any run of identical states shorter than min_frames into
    a neighboring state. Runs touching the start or end of the clip are
    now also handled (previously only interior runs were smoothed,
    missing boundary flicker like a single stray frame at the very end).
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
    This is the core timeline format (PDF section 4).

    Each segment's end_time_sec is set to the next segment's start_time_sec so
    that segments tile the observed period without gaps (previously the end was
    the last *detected frame* inside the segment, leaving a 0.5s hole at every
    boundary equal to the sampling interval).

    The final segment extends to clip_end_sec (the timestamp of the last sampled
    frame). If the last detection is earlier than clip_end_sec, an UNKNOWN
    segment is appended to cover the remaining tail.
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
        # The last segment already ends at true_end — but only if the last state
        # wasn't already covering all the way to true_end above. We need to
        # split: the last detected segment ends at last_detected, then UNKNOWN
        # covers last_detected → true_end.
        if current_state != "UNKNOWN":
            segments[-1]["end_time_sec"] = last_detected  # shrink back to last detection
            segments.append({
                "state": "UNKNOWN",
                "start_time_sec": last_detected,
                "end_time_sec": true_end,
            })
        # If last state was already UNKNOWN, it already extends to true_end; fine.

    return segments


def flag_transitions(segments):
    """
    Adds a 'transition_flag' to each segment noting whether the move FROM
    the previous segment INTO this one is expected or abrupt, based on
    EXPECTED_NEIGHBORS. This doesn't change the data, just annotates it
    for Phase 6 (agentic analysis) to use when deciding whether a moment
    needs more context.
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
    Estimates the bed centroid as the median box_center of all persons whose
    candidate_state is a bed-related state (LYING_IN_BED, SITTING_ON_BED).
    Falls back to None if no such detections exist in this clip.
    This gives a coarse but usually reliable anchor for where the bed is,
    without needing to pass the full bed polygon into this module.
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
    # Determine the true clip end from the last sampled frame in the candidates.
    # This is used to extend the final timeline segment all the way to the end
    # of the video, and to fill any post-detection gap with UNKNOWN.
    clip_end_sec = None
    if candidates_data:
        clip_end_sec = max(f["timestamp_sec"] for f in candidates_data)

    sequence = extract_primary_sequence(candidates_data, bed_centroid)
    sequence = remove_flicker(sequence)
    segments = build_segments(sequence, clip_end_sec=clip_end_sec)
    
    OUT_OF_BED_TRIGGERS = {"WALKING", "STANDING", "SITTING_OUTSIDE_BED"}
    for i in range(1, len(segments)):
        if segments[i]["state"] == "UNKNOWN" and segments[i-1]["state"] in OUT_OF_BED_TRIGGERS:
            segments[i]["state"] = "OUT_OF_BED"

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