"""
Phase 6 - Batch runner: agent_analysis.py for every clip.

Usage:
    python run_phase6_all.py
"""

import argparse
import json
import os


# ── thresholds ──────────────────────────────────────────────────────────────
MIN_WALKING_SEC_FOR_EXIT = 3.0   # exits with less walking than this are likely
                                  # repositioning, not a true exit
HIGH_CONFIDENCE = 0.85
LOW_CONFIDENCE  = 0.60
PROLONGED_ABSENCE_SEC = 600.0


# ── helpers ──────────────────────────────────────────────────────────────────

def total_state_time(segments, state):
    """Return the total seconds the timeline spends in `state`."""
    return sum(
        seg["end_time_sec"] - seg["start_time_sec"]
        for seg in segments
        if seg["state"] == state
    )


def states_after_event(segments, confirmed_time_str):
    """
    Return the list of segment states that START at or after the event's
    confirmed_time. Used to detect returns-to-bed that immediately follow
    a detected exit.
    """
    after = []
    for seg in segments:
        if seg["start_time"] >= confirmed_time_str:
            after.append(seg["state"])
    return after


def format_seconds(sec):
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:05.2f}"


def time_to_seconds(t_str):
    """Convert 'MM:SS.ss' formatted time string to seconds."""
    parts = t_str.split(":")
    return int(parts[0]) * 60 + float(parts[1])


def out_of_bed_duration(event, segments, clip_end_sec):
    """
    Conservative estimate of how long the person has been out of bed:
    time from the confirmed exit to the end of the clip.
    This is a lower bound - the true absence may extend beyond the clip.
    """
    exit_time = time_to_seconds(event["confirmed_time"])
    return max(0.0, clip_end_sec - exit_time)


# ── per-event reasoning ───────────────────────────────────────────────────────

def reason_bed_exit(event, segments):
    conf = event["confidence"]
    confirmed_time = event["confirmed_time"]
    clip_end_sec = segments[-1]["end_time_sec"] if segments else 0.0

    # How much walking is in this clip?
    walk_total = total_state_time(segments, "WALKING")

    # Did they come back to bed after the exit?
    states_after = states_after_event(segments, confirmed_time)
    returned = "LYING_IN_BED" in states_after or "SITTING_ON_BED" in states_after

    # How long have they been out of bed (lower bound: exit -> clip end)?
    absence_duration = out_of_bed_duration(event, segments, clip_end_sec)

    # ── prolonged absence: highest-priority check (overrides all others) ────
    # PDF section 7 explicitly names "unexpected prolonged absence from bed"
    # as an alert trigger. This rule fires regardless of detection confidence
    # because the duration itself IS the clinical risk signal.
    # NOTE: none of our 15s test clips are long enough to trigger this
    # (correctly - a 15s clip cannot show a 10-minute absence). It exists
    # to make the logic inspectable and to handle real-world longer videos.
    if absence_duration >= PROLONGED_ABSENCE_SEC:
        action = "ALERT"
        rationale = (
            f"Bed exit at {event['start_time']} has not resolved with a return to bed "
            f"within {PROLONGED_ABSENCE_SEC:.0f}s (observed absence so far: "
            f"{absence_duration:.1f}s). Per policy, a prolonged unexpected absence "
            f"from bed triggers an alert regardless of exit-detection confidence "
            f"({conf})."
        )
    # ── downgrade conditions ────────────────────────────────────────────────
    elif returned:
        action = "NORMAL"
        rationale = (
            f"Bed exit detected at {event['start_time']} (confirmed {confirmed_time}), "
            f"but the patient returned to bed within the same clip. "
            f"Sequence: {event['previous_state']} -> STANDING -> WALKING -> back to bed. "
            f"No alert required."
        )
    elif walk_total < MIN_WALKING_SEC_FOR_EXIT:
        action = "MONITOR"
        rationale = (
            f"Bed exit pattern detected at {event['start_time']} (confirmed {confirmed_time}), "
            f"but total walking time is only {walk_total:.1f}s "
            f"(threshold: {MIN_WALKING_SEC_FOR_EXIT}s). "
            f"This may be a brief repositioning at the bed edge rather than a true exit. "
            f"Confidence: {conf}. Flagging for review."
        )
    elif conf >= HIGH_CONFIDENCE:
        action = "ALERT"
        rationale = (
            f"High-confidence bed exit detected at {event['start_time']} "
            f"(confirmed walking away at {confirmed_time}). "
            f"Previous state: {event['previous_state']}. "
            f"Total walking time: {walk_total:.1f}s. "
            f"Confidence: {conf}. Caregiver attention recommended."
        )
    elif conf >= LOW_CONFIDENCE:
        action = "MONITOR"
        rationale = (
            f"Bed exit detected at {event['start_time']} (confirmed {confirmed_time}) "
            f"but with moderate confidence ({conf}) - the transition sequence was abrupt "
            f"or had an unexpected jump. Total walking time: {walk_total:.1f}s. "
            f"Flagging for caregiver review."
        )
    else:
        action = "MONITOR"
        rationale = (
            f"Low-confidence bed exit at {event['start_time']} (confirmed {confirmed_time}). "
            f"The person was already standing/walking when the clip began "
            f"({event['previous_state']}); we cannot confirm they were in bed beforehand. "
            f"Confidence: {conf}. Monitoring recommended."
        )

    return {**event, "action": action, "rationale": rationale}


def reason_return_to_bed(event, segments):
    conf = event["confidence"]

    if conf >= HIGH_CONFIDENCE:
        action = "NORMAL"
        rationale = (
            f"Patient returned to bed at {event['start_time']} "
            f"(confirmed lying down at {event['confirmed_time']}). "
            f"Sequence: {event['previous_state']} -> SITTING_ON_BED -> LYING_IN_BED. "
            f"Confidence: {conf}. Normal return to bed."
        )
    else:
        action = "MONITOR"
        rationale = (
            f"Return-to-bed pattern detected at {event['start_time']} "
            f"(confirmed {event['confirmed_time']}) with confidence {conf}. "
            f"The transition sequence was abrupt or involved an unexpected state jump. "
            f"Verifying patient is settled."
        )

    return {**event, "action": action, "rationale": rationale}


def reason_event(event, segments):
    if event["event"] == "bed_exit":
        return reason_bed_exit(event, segments)
    elif event["event"] == "return_to_bed":
        return reason_return_to_bed(event, segments)
    else:
        return {**event, "action": "MONITOR",
                "rationale": f"Unknown event type '{event['event']}' - flagging for manual review."}


# ── clip-level summary ────────────────────────────────────────────────────────

def clip_summary(segments, reasoned_events):
    lying_sec   = total_state_time(segments, "LYING_IN_BED")
    sitting_sec = total_state_time(segments, "SITTING_ON_BED")
    walking_sec = total_state_time(segments, "WALKING")
    unknown_sec = total_state_time(segments, "UNKNOWN")
    clip_duration = segments[-1]["end_time_sec"] if segments else 0.0

    has_alert   = any(e["action"] == "ALERT"   for e in reasoned_events)
    has_monitor = any(e["action"] == "MONITOR" for e in reasoned_events)

    if not reasoned_events:
        if lying_sec > 0 and walking_sec == 0:
            classification = "in_bed_static"
        elif unknown_sec > clip_duration * 0.5:
            classification = "ambiguous"
        else:
            classification = "in_bed_active"
    elif has_alert:
        classification = "alert"
    elif has_monitor:
        classification = "monitor"
    else:
        classification = "normal"

    return {
        "clip_classification": classification,
        "clip_duration_sec": round(clip_duration, 2),
        "time_lying_sec":    round(lying_sec,   2),
        "time_sitting_sec":  round(sitting_sec, 2),
        "time_walking_sec":  round(walking_sec, 2),
        "time_unknown_sec":  round(unknown_sec, 2),
        "event_count":       len(reasoned_events),
        "alert_count":       sum(1 for e in reasoned_events if e["action"] == "ALERT"),
        "monitor_count":     sum(1 for e in reasoned_events if e["action"] == "MONITOR"),
        "normal_count":      sum(1 for e in reasoned_events if e["action"] == "NORMAL"),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def analyze(segments, events):
    reasoned_events = [reason_event(e, segments) for e in events]
    summary = clip_summary(segments, reasoned_events)
    return {
        "summary": summary,
        "events":  reasoned_events,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6: Agentic analysis of timeline + events.")
    parser.add_argument("--timeline", required=True, help="Path to timeline JSON (Phase 3 output).")
    parser.add_argument("--events",   required=True, help="Path to events JSON (Phase 5 output).")
    parser.add_argument("--out",      required=True, help="Output path for reasoning JSON.")
    args = parser.parse_args()

    with open(args.timeline) as f:
        segments = json.load(f)

    events = []
    if os.path.exists(args.events):
        with open(args.events) as f:
            events = json.load(f)

    result = analyze(segments, events)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    s = result["summary"]
    print(f"Clip classification: {s['clip_classification']}")
    print(f"  Duration: {s['clip_duration_sec']:.1f}s | "
          f"Lying: {s['time_lying_sec']:.1f}s | "
          f"Walking: {s['time_walking_sec']:.1f}s | "
          f"Unknown: {s['time_unknown_sec']:.1f}s")
    print(f"  Events: {s['event_count']} "
          f"(ALERT: {s['alert_count']}, MONITOR: {s['monitor_count']}, NORMAL: {s['normal_count']})")
    for e in result["events"]:
        print(f"  [{e['action']}] {e['event']} @ {e['start_time']} - {e['rationale'][:80]}...")
    print(f"-> {args.out}")