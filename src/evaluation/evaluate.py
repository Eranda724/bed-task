"""
Evaluates system output against ground truth across all clips.
Calculates state classification accuracy, bed-exit metrics, and duration errors.
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict


EVENT_TIME_TOLERANCE_SEC = 2.0  # a predicted event within this many seconds

SAMPLE_INTERVAL_SEC = 0.5  # must match extract_frames.py's --interval
EQUIVALENT_STATES = {
    "OUT_OF_BED": "UNKNOWN",
}

def parse_time(t: str) -> float:
    parts = [float(p) for p in t.strip().split(":")]
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    h, m, s = parts
    return h * 3600 + m * 60 + s


def load_ground_truth_states(path):
    """Returns a list of (start_sec, end_sec, state) tuples."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append((parse_time(row["start_time"]), parse_time(row["end_time"]), row["state"]))
    return rows


def load_ground_truth_events(path):
    """Returns a list of (event_type, time_sec) tuples."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("event"):
                rows.append((row["event"], parse_time(row["time"])))
    return rows


def state_at_time(gt_rows, t):
    for start, end, state in gt_rows:
        if start <= t <= end:
            return state
    return None  # no ground truth coverage at this timestamp


def predicted_state_at_time(timeline_segments, t):
    for seg in timeline_segments:
        if seg["start_time_sec"] <= t <= seg["end_time_sec"]:
            return seg["state"]
    return "UNKNOWN"


def sample_timestamps(clip_duration, interval=SAMPLE_INTERVAL_SEC):
    ts = []
    t = 0.0
    while t <= clip_duration:
        ts.append(round(t, 2))
        t += interval
    return ts


def evaluate_states(clip_name, gt_states_path, timeline_path):
    gt_rows = load_ground_truth_states(gt_states_path)
    if not gt_rows or not os.path.exists(timeline_path):
        return None

    with open(timeline_path) as f:
        timeline = json.load(f)

    clip_duration = max(end for _, end, _ in gt_rows)
    timestamps = sample_timestamps(clip_duration)

    results = []
    for t in timestamps:
        gt_state = state_at_time(gt_rows, t)
        if gt_state is None:
            continue
        pred_state = predicted_state_at_time(timeline, t)
        gt_state_normalized = EQUIVALENT_STATES.get(gt_state, gt_state)
        pred_state_normalized = EQUIVALENT_STATES.get(pred_state, pred_state)
        results.append((t, gt_state_normalized, pred_state_normalized))

    return results


def evaluate_events(clip_name, gt_events_path, pred_events_path, event_type=None):
    gt_events = load_ground_truth_events(gt_events_path)
    pred_events = []
    if os.path.exists(pred_events_path):
        with open(pred_events_path) as f:
            pred_events = json.load(f)

    if event_type:
        gt_events = [e for e in gt_events if e[0] == event_type]
        pred_events = [e for e in pred_events if e["event"] == event_type]

    matched_gt = set()
    matched_pred = set()

    for pi, pe in enumerate(pred_events):
        pred_confirmed = parse_time(pe["confirmed_time"])
        pred_start     = parse_time(pe["start_time"])
        for gi, (gtype, gtime) in enumerate(gt_events):
            if gi in matched_gt:
                continue
            best_dist = min(abs(pred_confirmed - gtime), abs(pred_start - gtime))
            if pe["event"] == gtype and best_dist <= EVENT_TIME_TOLERANCE_SEC:
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    true_positives = len(matched_pred)
    false_positives = len(pred_events) - len(matched_pred)
    false_negatives = len(gt_events) - len(matched_gt)

    return {
        "clip": clip_name,
        "ground_truth_events": len(gt_events),
        "predicted_events": len(pred_events),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def evaluate_durations(clip_name, gt_states_path, summary_path):
    gt_rows = load_ground_truth_states(gt_states_path)
    if not gt_rows or not os.path.exists(summary_path):
        return None

    gt_duration_sec = defaultdict(float)
    for start, end, state in gt_rows:
        normalized_state = EQUIVALENT_STATES.get(state, state)
        gt_duration_sec[normalized_state.lower()] += max(0.0, end - start)

    with open(summary_path) as f:
        summary = json.load(f)
    pred_duration_sec = summary["activity_duration_sec"]

    errors = {}
    for state in set(list(gt_duration_sec.keys()) + list(pred_duration_sec.keys())):
        gt_val = gt_duration_sec.get(state, 0.0)
        pred_val = pred_duration_sec.get(state, 0.0)
        errors[state] = {
            "ground_truth_sec": round(gt_val, 2),
            "predicted_sec": round(pred_val, 2),
            "error_sec": round(abs(gt_val - pred_val), 2),
        }

    return errors


def build_confusion_matrix(all_state_results):
    matrix = defaultdict(lambda: defaultdict(int))
    correct = 0
    total = 0
    for _, gt_state, pred_state in all_state_results:
        matrix[gt_state][pred_state] += 1
        total += 1
        if gt_state == pred_state:
            correct += 1
    accuracy = correct / total if total > 0 else 0.0
    return matrix, accuracy, correct, total


def main(clips_dir, out_path):
    clip_names = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(clips_dir, "*.mp4"))
    )

    all_state_results = []
    per_clip_state_results = {}
    event_exit_results = []
    event_return_results = []
    duration_eval_results = {}
    failure_examples = []

    for clip in clip_names:
        gt_states_path = os.path.join("data", "ground_truth", f"{clip}_states.csv")
        gt_events_path = os.path.join("data", "ground_truth", f"{clip}_events.csv")
        timeline_path = os.path.join("outputs", "timeline", f"{clip}_timeline.json")
        events_path = os.path.join("outputs", "events", f"{clip}_events.json")
        summary_path = os.path.join("outputs", "summary", f"{clip}_summary.json")

        state_results = evaluate_states(clip, gt_states_path, timeline_path)
        if state_results:
            all_state_results.extend(state_results)
            per_clip_state_results[clip] = state_results
            clip_correct = sum(1 for _, g, p in state_results if g == p)
            clip_total = len(state_results)
            for t, gt_state, pred_state in state_results:
                if gt_state != pred_state:
                    failure_examples.append({
                        "clip": clip, "timestamp_sec": t,
                        "ground_truth": gt_state, "predicted": pred_state,
                    })

        event_exit_result = evaluate_events(clip, gt_events_path, events_path, "bed_exit")
        if event_exit_result:
            event_exit_results.append(event_exit_result)
            
        event_return_result = evaluate_events(clip, gt_events_path, events_path, "return_to_bed")
        if event_return_result:
            event_return_results.append(event_return_result)

        duration_result = evaluate_durations(clip, gt_states_path, summary_path)
        if duration_result:
            duration_eval_results[clip] = duration_result

    matrix, accuracy, correct, total = build_confusion_matrix(all_state_results)
    matrix_readable = {gt: dict(preds) for gt, preds in matrix.items()}

    exit_tp = sum(e["true_positives"] for e in event_exit_results)
    exit_fp = sum(e["false_positives"] for e in event_exit_results)
    exit_fn = sum(e["false_negatives"] for e in event_exit_results)
    exit_precision = exit_tp / (exit_tp + exit_fp) if (exit_tp + exit_fp) > 0 else None
    exit_recall = exit_tp / (exit_tp + exit_fn) if (exit_tp + exit_fn) > 0 else None

    return_tp = sum(e["true_positives"] for e in event_return_results)
    return_fp = sum(e["false_positives"] for e in event_return_results)
    return_fn = sum(e["false_negatives"] for e in event_return_results)
    return_precision = return_tp / (return_tp + return_fp) if (return_tp + return_fp) > 0 else None
    return_recall = return_tp / (return_tp + return_fn) if (return_tp + return_fn) > 0 else None

    overall_duration_errors = defaultdict(list)
    for clip, errors in duration_eval_results.items():
        for state, vals in errors.items():
            overall_duration_errors[state].append(vals["error_sec"])
    avg_duration_errors = {
        state: round(sum(errs) / len(errs), 2)
        for state, errs in overall_duration_errors.items()
    }

    failure_examples_sorted = sorted(
        failure_examples,
        key=lambda f: 0 if f["ground_truth"] != "UNKNOWN" and f["predicted"] != "UNKNOWN" else 1
    )[:10]

    report = {
        "state_classification": {
            "overall_accuracy": round(accuracy, 4),
            "correct_frames": correct,
            "total_frames": total,
            "confusion_matrix": matrix_readable,
        },
        "bed_exit_metrics": {
            "per_clip": event_exit_results,
            "true_positives": exit_tp,
            "false_positives": exit_fp,
            "false_negatives": exit_fn,
            "precision": round(exit_precision, 4) if exit_precision is not None else None,
            "recall": round(exit_recall, 4) if exit_recall is not None else None,
        },
        "return_to_bed_metrics": {
            "per_clip": event_return_results,
            "true_positives": return_tp,
            "false_positives": return_fp,
            "false_negatives": return_fn,
            "precision": round(return_precision, 4) if return_precision is not None else None,
            "recall": round(return_recall, 4) if return_recall is not None else None,
        },
        "duration_estimation": {
            "per_clip": duration_eval_results,
            "average_error_sec_by_state": avg_duration_errors,
        },
        "failure_examples": failure_examples_sorted,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"State classification accuracy: {accuracy:.1%} ({correct}/{total} frames)")
    print(f"\nConfusion matrix (ground_truth -> predicted counts):")
    for gt_state, preds in matrix_readable.items():
        print(f"  {gt_state}: {preds}")
    print(f"\nBed Exits: TP={exit_tp} FP={exit_fp} FN={exit_fn}")
    if exit_precision is not None:
        print(f"  Precision: {exit_precision:.1%}  Recall: {exit_recall:.1%}")
        
    print(f"\nReturn to Bed: TP={return_tp} FP={return_fp} FN={return_fn}")
    if return_precision is not None:
        print(f"  Precision: {return_precision:.1%}  Recall: {return_recall:.1%}")
    print(f"\nAverage duration error by state (seconds):")
    for state, err in avg_duration_errors.items():
        print(f"  {state}: {err}s")
    print(f"\nFull report -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full evaluation against ground truth.")
    parser.add_argument("--clips_dir", default="data/videos")
    parser.add_argument("--out", default="outputs/evaluation_report.json")
    args = parser.parse_args()

    main(args.clips_dir, args.out)