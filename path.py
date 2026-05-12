import argparse
import sys
import subprocess
from pathlib import Path
import main

def main_path():
    """
    Main entry point for HawkSight Path Annotation.
    Runs the tracking pipeline and then launches the dashboard.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--input",       required=True, help="Path to input video file")
    p.add_argument("--robot_weights", required=True, help="Path to YOLO robot weights (.pt)")
    p.add_argument("--fuel_weights", required=True, help="Path to YOLO fuel weights (.pt)")
    p.add_argument("--output",      default="out_persistent.mp4", help="Path to output video file")
    p.add_argument("--max-robots",  type=int,   default=6, help="Max unique robots to track")
    p.add_argument("--device", default="cpu", help="device to run the model on")
    
    # Allow unknown args to be passed to main.process_video if needed
    args, unknown = p.parse_known_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Run the tracking pipeline
    print(f"\n[HawkSight] Starting tracking pipeline...")
    print(f"            Input:   {args.input}")
    print(f"            Robot Weights: {args.robot_weights}")
    print(f"            Fuel Weights: {args.fuel_weights}")
    
    try:
        registry = main.process_video(
            input_path   = Path(args.input),
            output_path  = output_path,
            robot_weights_path= Path(args.robot_weights),
            fuel_weights_path= Path(args.fuel_weights),
            max_robots   = args.max_robots,
            device= args.device
        )
    except Exception as e:
        print(f"\n[HawkSight] Error during tracking: {e}")
        sys.exit(1)

    # 2. Save robot registry and paths to JSON
    json_path = Path("robot_paths.json")
    print(f"\n[HawkSight] Saving paths to {json_path}...")
    registry.save_json(json_path)

    # 3. Launch the dashboard to display results
    print("\n[HawkSight] Launching dashboard...")
    try:
        # We use subprocess to launch dashboard.py so it runs independently
        subprocess.Popen([sys.executable, "dashboard.py"])
        print("[HawkSight] Dashboard launched successfully.")
    except Exception as e:
        print(f"[HawkSight] Failed to launch dashboard: {e}")

if __name__ == "__main__":
    main_path()
