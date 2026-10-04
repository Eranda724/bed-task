import subprocess
import sys

def run_step(cmd, description):
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"{'='*50}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Pipeline failed at step: {description}")
        sys.exit(e.returncode)

def main():
    steps = [
        ([sys.executable, "src/perception/track_all_videos.py"], "Perception (Frames, Pose, Tracking, Viz)"),
        ([sys.executable, "src/perception/compute_all_features.py"], "Feature Extraction"),
        ([sys.executable, "src/temporal/generate_all_timelines.py"], "Temporal Smoothing and Timelines"),
        ([sys.executable, "src/events/detect_all_events.py"], "Event Detection"),
        ([sys.executable, "src/outputs/generate_all_summaries.py"], "Duration Summaries"),
        ([sys.executable, "src/agent/analyze_all_clips.py"], "Agentic Reasoning and Alerts"),
        ([sys.executable, "src/evaluation/evaluate.py"], "Evaluation Report Generation")
    ]
    
    for cmd, desc in steps:
        run_step(cmd, desc)
        
    print(f"\n{'='*50}")
    print("Pipeline completed successfully! All outputs generated.")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
