"""
Phase 1 - Step 3: Bed region annotation

Opens the first frame of a video and lets you click 4 points to define the
bed's polygon. This is a manual, one-time-per-video step rather than a
trained bed-detector model. The PDF allows this level of technical freedom
("bed-region detection" is listed as an option, not a requirement to train
a model), and a fixed camera + manual polygon is simple, reliable, and easy
to justify/explain in an interview - no bed-detector training needed.

Controls:
    - Left click: add a point (click 4 corners of the bed area, in order)
    - 'r': reset points for this frame
    - 's': save polygon to JSON and quit
    - 'q': quit without saving

Usage:
    python mark_bed_region.py --video data/videos/case05_leaving_bed.mp4
"""

import argparse
import json
import os
import numpy as np
import cv2

points = []


def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"Point added: ({x}, {y})")


def mark_bed_region(video_path: str, out_path: str):
    global points
    points = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read first frame of: {video_path}")

    window_name = "Click 4 corners of the bed, then press 's' to save"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)

    while True:
        display_frame = frame.copy()
        for i, pt in enumerate(points):
            cv2.circle(display_frame, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(display_frame, str(i + 1), (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(points) >= 2:
            cv2.polylines(display_frame, [np.array(points)],
                          isClosed=(len(points) >= 4), color=(0, 255, 0), thickness=2)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            points = []
            print("Points reset.")
        elif key == ord('s'):
            if len(points) < 3:
                print("Need at least 3 points before saving. Keep clicking.")
                continue
            break
        elif key == ord('q'):
            print("Quit without saving.")
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"bed_polygon": points}, f, indent=2)

    print(f"Saved bed polygon ({len(points)} points) to {out_path}")
    return points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mark bed region on first frame of a video.")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--out", default=None,
                         help="Output JSON path (default: data/bed_regions/<video_name>.json)")
    args = parser.parse_args()

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    out_path = args.out or os.path.join("data", "bed_regions", f"{video_name}.json")

    mark_bed_region(args.video, out_path)