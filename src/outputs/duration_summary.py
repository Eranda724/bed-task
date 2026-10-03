"""
Phase 4 - Duration summaries

Reads a clip's timeline (Phase 3 segments) and produces the activity
duration summary in the exact format shown in PDF section 3/7:
    - total observation time
    - per-state duration (human readable + seconds)
    - bed_summary: time in bed vs out of bed, bed exit count (placeholder,
      real counting happens in Phase 5)

"IN_BED" states: LYING_IN_BED, SITTING_ON_BED
"OUT_OF_BED" states: SITTING_OUTSIDE_BED, STANDING, WALKING, OUT_OF_BED
UNKNOWN is reported separately, not folded into either bucket, since we
genuinely don't know the person's bed status during those segments -
folding it into either side would overstate our confidence.

Usage:
    python duration_summary.py --timeline outputs/timeline/case05_leaving_bed_timeline.json \
                                --out outputs/summary/case05_leaving_bed_summary.json
"""

import argparse
import json
import os


IN_BED_STATES = {"LYING_IN_BED", "SITTING_ON_BED"}
OUT_OF_BED_STATES = {"SITTING_OUTSIDE_BED", "STANDING", "WALKING", "OUT_OF_BED"}

ALL_STATES = [
    "lying_in_bed", "sitting_on_bed", "sitting_outside_bed",
    "standing", "walking", "out_of_bed", "unknown",
]


def format_duration(seconds):
    m = int(seconds // 60)
    s = int(round(seconds % 60))
    return f"{m}m {s:02d}s"


def summarize(segments, events=None):
    durations_sec = {state: 0.0 for state in ALL_STATES}
    total_time = 0.0
    bed_exit_count = 0
    bed_return_count = 0
    longest_out_of_bed_period_sec = 0.0
    current_out = 0.0

    if events:
        for e in events:
            if e.get("event") == "bed_exit":
                bed_exit_count += 1
            elif e.get("event") == "return_to_bed":
                bed_return_count += 1
    if segments:
        total_time = segments[-1]["end_time_sec"] - segments[0]["start_time_sec"]

    for seg in segments:
        duration = max(0.0, seg["end_time_sec"] - seg["start_time_sec"])
        key = seg["state"].lower()
        if key in durations_sec:
            durations_sec[key] += duration

        if seg["state"] in OUT_OF_BED_STATES:
            current_out += duration
            longest_out_of_bed_period_sec = max(longest_out_of_bed_period_sec, current_out)
        else:
            current_out = 0.0

    in_bed_sec = durations_sec["lying_in_bed"] + durations_sec["sitting_on_bed"]
    out_of_bed_sec = (durations_sec["sitting_outside_bed"] + durations_sec["standing"]
                       + durations_sec["walking"] + durations_sec["out_of_bed"])

    activity_summary = {
        state: format_duration(durations_sec[state]) for state in ALL_STATES
    }
    activity_duration_sec = {
        state: round(durations_sec[state], 2) for state in ALL_STATES
    }

    result = {
        "total_observation_time": format_duration(total_time),
        "observation_duration_sec": round(total_time, 2),
        "activity_summary": activity_summary,
        "activity_duration_sec": activity_duration_sec,
        "bed_summary": {
            "time_in_bed": format_duration(in_bed_sec),
            "time_out_of_bed": format_duration(out_of_bed_sec),
            "time_unknown": format_duration(durations_sec["unknown"]),
            "bed_exit_count": bed_exit_count,
        },
        "total_in_bed_sec": round(in_bed_sec, 2),
        "total_out_of_bed_sec": round(out_of_bed_sec, 2),
        "total_unknown_sec": round(durations_sec["unknown"], 2),
        "longest_out_of_bed_period_sec": round(longest_out_of_bed_period_sec, 2),
        "bed_exit_count": bed_exit_count,
        "bed_return_count": bed_return_count,
        "final_state": segments[-1]["state"] if segments else "UNKNOWN",
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute duration summary from a timeline.")
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--events", default=None, help="Path to events JSON (from Phase 5)")
    args = parser.parse_args()

    with open(args.timeline) as f:
        segments = json.load(f)

    events = None
    if args.events and os.path.exists(args.events):
        with open(args.events) as f:
            events = json.load(f)

    result = summarize(segments, events)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote summary -> {args.out}")
    print(json.dumps(result, indent=2))