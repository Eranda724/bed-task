import glob
import json
import os

import cv2

EVENT_KEYS = {"event", "start_time", "confirmed_time", "previous_state",
              "current_state", "confidence", "decision"}
SUMMARY_KEYS = {"observation_duration_sec", "activity_duration_sec",
                "bed_exit_count", "bed_return_count", "total_in_bed_sec",
                "total_out_of_bed_sec", "longest_out_of_bed_period_sec",
                "final_state"}
DECISIONS = {"NORMAL", "MONITOR", "ALERT"}
END_KEYS = ("end_sec", "end_time_sec", "end")
START_KEYS = ("start_sec", "start_time_sec", "start")


def load_segments(path):
    data = json.load(open(path))
    if isinstance(data, list):
        return data
    for k in ("segments", "timeline"):
        if k in data:
            return data[k]
    return []


def pick(seg, keys):
    for k in keys:
        if k in seg:
            return seg[k]
    return None


def video_length(path):
    c = cv2.VideoCapture(path)
    fps = c.get(cv2.CAP_PROP_FPS)
    return c.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else None


def check_files():
    clips = sorted(glob.glob("data/videos/*.mp4"))
    print(f"clips found: {len(clips)} (PDF section 8 lists 13 cases)")
    for c in clips:
        n = os.path.splitext(os.path.basename(c))[0]
        for p in [f"data/ground_truth/{n}_states.csv",
                  f"data/ground_truth/{n}_events.csv",
                  f"data/bed_regions/{n}.json",
                  f"outputs/timeline/{n}_timeline.json",
                  f"outputs/summary/{n}_summary.json",
                  f"outputs/events/{n}_events.json",
                  f"outputs/agent/{n}_reasoning.json"]:
            if not os.path.exists(p):
                print(f"[MISSING] {p}")
    return clips


def check_timelines(clips):
    for c in clips:
        n = os.path.splitext(os.path.basename(c))[0]
        p = f"outputs/timeline/{n}_timeline.json"
        if not os.path.exists(p):
            continue
        segs = load_segments(p)
        if not segs:
            print(f"[TIMELINE] {n}: no segments found")
            continue
        ends = [pick(s, END_KEYS) for s in segs]
        starts = [pick(s, START_KEYS) for s in segs]
        if None in ends or None in starts:
            print(f"[TIMELINE] {n}: unknown key names, segment keys = {list(segs[0].keys())}")
            continue
        for i in range(1, len(segs)):
            if abs(starts[i] - ends[i - 1]) > 0.01:
                print(f"[TIMELINE] {n}: gap/overlap between segment {i-1} and {i} "
                      f"({ends[i-1]} -> {starts[i]})")
        length = video_length(c)
        if length is not None and abs(ends[-1] - length) > 0.6:
            print(f"[TIMELINE] {n}: ends at {ends[-1]}s but video is {length:.2f}s")


def check_summaries():
    for f in sorted(glob.glob("outputs/summary/*_summary.json")):
        s = json.load(open(f))
        name = os.path.basename(f)
        missing = SUMMARY_KEYS - set(s)
        if missing:
            print(f"[SUMMARY] {name} missing: {sorted(missing)}")
        if "bed_exit_count" not in s.get("bed_summary", {}):
            print(f"[SUMMARY] {name} bed_summary has no bed_exit_count")
        dur = s.get("activity_duration_sec", {})
        total = s.get("observation_duration_sec")
        if dur and total:
            diff = abs(sum(dur.values()) - total)
            if diff > 1.0:
                print(f"[SUMMARY] {name} durations sum {sum(dur.values()):.1f}s vs video {total}s")


def check_events():
    for f in sorted(glob.glob("outputs/events/*_events.json")):
        data = json.load(open(f))
        events = data if isinstance(data, list) else data.get("events", [])
        name = os.path.basename(f)
        for e in events:
            missing = EVENT_KEYS - set(e)
            if missing:
                print(f"[EVENT] {name} missing: {sorted(missing)}")
            if str(e.get("decision", "")).upper() not in DECISIONS:
                print(f"[EVENT] {name} bad decision: {e.get('decision')}")


clips = check_files()
check_timelines(clips)
check_summaries()
check_events()
print("done")