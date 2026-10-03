"""
Phase 6 - Agentic Analysis
"""

import argparse
import json
import os
import statistics

# thresholds
MIN_WALKING_SEC_FOR_EXIT = 3.0
HIGH_CONFIDENCE = 0.85

def total_state_time(segments, state):
    return sum(seg["end_time_sec"] - seg["start_time_sec"] for seg in segments if seg["state"] == state)

def time_to_seconds(t_str):
    parts = t_str.split(":")
    return int(parts[0]) * 60 + float(parts[1])

def get_segment_idx_at_time(segments, time_str):
    time_sec = time_to_seconds(time_str)
    for i, seg in enumerate(segments):
        # Allow small epsilon
        if seg["start_time_sec"] - 1e-3 <= time_sec <= seg["end_time_sec"] + 1e-3:
            return i
    return -1

def analyze_previous_segment(segments, current_idx):
    if current_idx > 0:
        prev = segments[current_idx - 1]
        curr = segments[current_idx]
        gap = curr["start_time_sec"] - prev["end_time_sec"]
        return prev, f"Previous segment was {prev['state']} from {prev['start_time']} to {prev['end_time']} (ended {gap:.1f}s earlier)."
    return None, "No previous segment exists."

def analyze_next_segment(segments, current_idx):
    if current_idx < len(segments) - 1:
        nxt = segments[current_idx + 1]
        return nxt, f"Following segment is {nxt['state']} from {nxt['start_time']} to {nxt['end_time']}."
    return None, "No following segment exists."

def check_on_bed_or_floor(segment, features):
    start = segment["start_time_sec"]
    end = segment["end_time_sec"]
    if segment["state"] == "UNKNOWN":
        return None, "Person not detected in UNKNOWN segment; cannot determine bed overlap."
    
    overlaps = []
    for f in features:
        if start - 1e-3 <= f["timestamp_sec"] <= end + 1e-3:
            if f["persons"]:
                overlaps.append(f["persons"][0].get("bed_overlap_ratio", 0.0))
    if not overlaps:
        return None, "No overlap data available."
    
    med = statistics.median(overlaps)
    return med, f"Median bed_overlap_ratio over this segment is {med:.2f}."

def reason_bed_exit(event, segments, features):
    conf = event["confidence"]
    trace = []
    
    trace.append({
        "observation": f"Detected {event['event']} at {event['start_time']} with confidence {conf:.2f}.",
        "action": "Determine if evidence is already enough.",
        "finding": None,
        "conclusion": None
    })
    
    idx = get_segment_idx_at_time(segments, event["start_time"])
    
    if conf >= HIGH_CONFIDENCE:
        trace[-1]["finding"] = "High confidence exit. Evidence is sufficient."
        trace[-1]["conclusion"] = "BED_EXIT confirmed."
        return {**event, "action": "ALERT", "reasoning_trace": trace}
        
    trace[-1]["finding"] = "Confidence is low/moderate. More context needed."
    
    trace.append({
        "observation": "Need to check context around the event.",
        "action": "analyze_previous_segment",
        "finding": None,
        "conclusion": None
    })
    prev_seg, prev_desc = analyze_previous_segment(segments, idx)
    trace[-1]["finding"] = prev_desc
    
    trace.append({
        "observation": "Need to check subsequent behavior.",
        "action": "analyze_next_segment",
        "finding": None,
        "conclusion": None
    })
    next_seg, next_desc = analyze_next_segment(segments, idx)
    trace[-1]["finding"] = next_desc
    
    walk_total = total_state_time(segments, "WALKING")
    if walk_total < MIN_WALKING_SEC_FOR_EXIT:
        trace.append({
            "observation": f"Total walking time is {walk_total:.1f}s, less than {MIN_WALKING_SEC_FOR_EXIT}s threshold.",
            "action": "Assess clinical risk of brief walk.",
            "finding": "Activity is likely repositioning at the bed edge rather than a true exit.",
            "conclusion": "BED_EXIT not confirmed."
        })
        return {**event, "action": "MONITOR", "reasoning_trace": trace}
        
    trace.append({
        "observation": "All context gathered.",
        "action": "Determine alert level.",
        "finding": "Exit sequence verified despite initial low confidence.",
        "conclusion": "BED_EXIT confirmed."
    })
    return {**event, "action": "MONITOR", "reasoning_trace": trace}

def reason_return_to_bed(event, segments, features):
    trace = [{
        "observation": f"Detected {event['event']} at {event['start_time']} with confidence {event['confidence']:.2f}.",
        "action": "Determine if evidence is already enough.",
        "finding": "Return to bed is clear.",
        "conclusion": "RETURN_TO_BED confirmed."
    }]
    return {**event, "action": "NORMAL", "reasoning_trace": trace}

def reason_segment(seg, features):
    trace = []
    trace.append({
        "observation": f"Person is LYING_IN_BED from {seg['start_time']} to {seg['end_time']}.",
        "action": "Determine if evidence is already enough.",
        "finding": "Must verify if person is actually in bed or on the floor.",
        "conclusion": None
    })
    trace.append({
        "observation": "Need spatial context.",
        "action": "check_on_bed_or_floor",
        "finding": None,
        "conclusion": None
    })
    med_overlap, overlap_desc = check_on_bed_or_floor(seg, features)
    trace[-1]["finding"] = overlap_desc
    
    if med_overlap is not None and med_overlap < 0.3:
        trace.append({
            "observation": f"Overlap {med_overlap:.2f} is below on-bed threshold.",
            "action": "Determine clinical risk.",
            "finding": "Person is horizontal but not confidently on the bed. May be on the floor.",
            "conclusion": "Activity cannot be confidently determined."
        })
        return {"event": "lying_check", "segment": seg, "action": "MONITOR", "reasoning_trace": trace}
        
    trace.append({
        "observation": "Overlap confirms person is in bed.",
        "action": "Determine alert level.",
        "finding": "Normal resting state.",
        "conclusion": "Patient resting in bed."
    })
    return {"event": "lying_check", "segment": seg, "action": "NORMAL", "reasoning_trace": trace}

def analyze(segments, events, features):
    reasoned_items = []
    
    for e in events:
        if e["event"] == "bed_exit":
            reasoned_items.append(reason_bed_exit(e, segments, features))
        elif e["event"] == "return_to_bed":
            reasoned_items.append(reason_return_to_bed(e, segments, features))
            
    for seg in segments:
        if seg["state"] == "LYING_IN_BED":
            reasoned_items.append(reason_segment(seg, features))
            
    lying_sec   = total_state_time(segments, "LYING_IN_BED")
    walking_sec = total_state_time(segments, "WALKING")
    unknown_sec = total_state_time(segments, "UNKNOWN")
    clip_duration = segments[-1]["end_time_sec"] if segments else 0.0

    has_alert   = any(e["action"] == "ALERT"   for e in reasoned_items)
    has_monitor = any(e["action"] == "MONITOR" for e in reasoned_items)

    if not events: 
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

    summary = {
        "clip_classification": classification,
        "clip_duration_sec": round(clip_duration, 2),
        "event_count":       len(events),
        "alert_count":       sum(1 for e in reasoned_items if e["action"] == "ALERT"),
        "monitor_count":     sum(1 for e in reasoned_items if e["action"] == "MONITOR"),
        "normal_count":      sum(1 for e in reasoned_items if e["action"] == "NORMAL"),
    }
    
    return {
        "summary": summary,
        "agent_analysis": reasoned_items,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--events",   required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out",      required=True)
    args = parser.parse_args()

    with open(args.timeline) as f:
        segments = json.load(f)

    events = []
    if os.path.exists(args.events):
        with open(args.events) as f:
            events = json.load(f)
            
    with open(args.features) as f:
        features = json.load(f)

    result = analyze(segments, events, features)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Clip classification: {result['summary']['clip_classification']}")
    print(f"-> {args.out}")