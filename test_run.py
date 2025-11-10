"""
Bat Swing Analysis - CLI Entrypoint

This script runs the full analysis pipeline on a single video file.

Example Usage:
python run_analysis.py \
    --video data/sample_videos/swing1.mp4 \
    --yolo-model yolov8n.pt \
    --scale 0.004 \
    --output_dir results/swing1
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from detectors.yolov8_detector import YOLOv8Detector
from pose.mediapipe_pose import MediaPipePose
from metrics.swing_metrics import (
    PixelScaler,
    compute_derivatives,
    detect_swings,
    analyze_swing,
)
from utils.geometry import interpolate_points, estimate_bat_tip, smooth_trajectory
from viz.overlay import create_swing_gif
from viz.plots import plot_swing_analytics

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define the keypoints we need from MediaPipe
REQUIRED_LANDMARKS = [
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "RIGHT_WRIST",
    "RIGHT_HIP",
]


def get_video_fps(cap: cv2.VideoCapture) -> float:
    """Gets the FPS from the video capture object."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        logging.warning("Could not get valid FPS from video, defaulting to 30.0")
        return 30.0
    return fps


def process_frame(
    frame: np.ndarray,
    frame_idx: int,
    detector: YOLOv8Detector,
    pose_estimator: MediaPipePose,
) -> Optional[Dict[str, Any]]:
    """
    Processes a single video frame to detect objects and estimate pose.
    
    This simplified version assumes a single player of interest.
    """
    # 1. Run Detection
    detections = detector.detect(frame)

    # Filter for 'person' and 'bat'
    person_dets = [d for d in detections if d["class_name"] == "person"]
    bat_dets = [d for d in detections if d["class_name"] == "bat"]
    
    # 2. Find Player of Interest (assume largest 'person' bbox)
    if not person_dets:
        logging.warning(f"No 'person' detected in frame {frame_idx}")
        return None
    
    # Sort by bbox area (w*h) descending
    person_dets.sort(key=lambda d: (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]), reverse=True)
    player_bbox = person_dets[0]["bbox"]

    # 3. Run Pose Estimation
    # We run pose on the full frame for simplicity, as MediaPipe is fast
    # and handles finding the person itself.
    pose_results = pose_estimator.process_frame(frame)
    if not pose_results:
        logging.warning(f"No pose detected in frame {frame_idx}")
        return None

    # 4. Extract Key Keypoints
    landmarks = pose_results.get("landmarks_2d_pixels", {})
    frame_keypoints = {}
    missing_keypoint = False
    for key in REQUIRED_LANDMARKS:
        if key not in landmarks:
            logging.warning(f"Missing keypoint '{key}' in frame {frame_idx}")
            missing_keypoint = True
        frame_keypoints[key] = landmarks.get(key)
    
    if missing_keypoint:
        # We can't proceed without all keypoints for this frame
        return {"keypoints": {key: (np.nan, np.nan) for key in REQUIRED_LANDMARKS}, "bat_tip": (np.nan, np.nan)}

    # 5. Find Bat and Bat-Tip
    # Find the bat detection closest to the player's right wrist
    right_wrist_pt = frame_keypoints["RIGHT_WRIST"]
    bat_tip = (np.nan, np.nan)

    if bat_dets and right_wrist_pt:
        # Find bat closest to wrist
        bat_dets.sort(key=lambda d: np.linalg.norm(
            np.array([(d['bbox'][0] + d['bbox'][2]) / 2, (d['bbox'][1] + d['bbox'][3]) / 2]) - np.array(right_wrist_pt)
        ))
        bat_bbox = bat_dets[0]["bbox"]
        
        # Estimate bat tip
        bat_tip = estimate_bat_tip(bat_bbox, right_wrist_pt)
    else:
        logging.warning(f"No 'bat' detected or wrist missing in frame {frame_idx}")

    return {"keypoints": frame_keypoints, "bat_tip": bat_tip}


def run_pipeline(
    video_path: str,
    yolo_model_path: str,
    scale_mps: float,
    output_dir: Path,
    fps_override: Optional[float] = None,
):
    """
    Main analysis pipeline function.
    """
    logging.info(f"Starting analysis for: {video_path}")
    logging.info(f"Using scale: {scale_mps:.5f} meters/pixel")
    
    # 1. Initialization
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = YOLOv8Detector(yolo_model_path)
    pose_estimator = MediaPipePose()
    scaler = PixelScaler(pixel_ref=1.0, meter_ref=scale_mps) # 1 pixel = scale_mps meters

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file {video_path}")
        sys.exit(1)

    fps = fps_override if fps_override else get_video_fps(cap)
    logging.info(f"Video FPS: {fps}")

    all_frames = []
    all_keypoints = []
    all_bat_tips = []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=frame_count, desc="Processing frames")

    # 2. Per-Frame Processing Loop
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Store BGR frame for GIF creation later
        all_frames.append(frame)

        frame_data = process_frame(frame, frame_idx, detector, pose_estimator)

        if frame_data:
            all_keypoints.append(frame_data["keypoints"])
            all_bat_tips.append(frame_data["bat_tip"])
        else:
            # Append NaNs if processing failed
            all_keypoints.append({key: (np.nan, np.nan) for key in REQUIRED_LANDMARKS})
            all_bat_tips.append((np.nan, np.nan))
        
        pbar.update(1)
        frame_idx += 1

    cap.release()
    pbar.close()
    logging.info(f"Processed {len(all_frames)} frames.")

    if not all_frames:
        logging.error("No frames processed. Exiting.")
        return

    # 3. Trajectory Generation and Post-Processing
    logging.info("Generating and smoothing trajectories...")
    
    # Convert lists of tuples/dicts into structured numpy arrays
    tip_trajectory = np.array(all_bat_tips, dtype=float)
    
    keypoint_trajectories = {}
    for key in REQUIRED_LANDMARKS:
        keypoint_trajectories[key] = np.array([kp[key] for kp in all_keypoints], dtype=float)

    # Interpolate and smooth all trajectories
    # A larger window (e.g., 11) is good for smoothing high-speed motion
    smooth_window = 11
    poly_order = 2
    
    tip_trajectory_smooth = smooth_trajectory(
        interpolate_points(tip_trajectory), window=smooth_window, poly=poly_order
    )
    
    kp_trajectories_smooth = {}
    for key in REQUIRED_LANDMARKS:
        kp_trajectories_smooth[key] = smooth_trajectory(
            interpolate_points(keypoint_trajectories[key]), window=smooth_window, poly=poly_order
        )

    # 4. Compute Derivatives and Detect Swings
    logging.info("Computing derivatives and detecting swings...")
    
    _, speed_px_per_frame, accel_px_per_frame_sq = compute_derivatives(
        tip_trajectory_smooth, fps=fps
    )
    speed_mps = scaler.to_mps(speed_px_per_frame, fps)

    # Detect swings: speed > 2.0 m/s for at least 5 frames
    swings = detect_swings(
        speed_mps, speed_thresh_mps=2.0, min_swing_frames=int(fps * 0.1) # min 100ms
    )
    logging.info(f"Detected {len(swings)} swings.")

    # 5. Analyze Each Swing and Generate Outputs
    all_swing_metrics = []
    
    # Consolidate all data for analysis functions
    full_trajectory_data = {
        "bat_tip": tip_trajectory_smooth,
        "speed_mps": speed_mps,
        "accel_px_per_frame_sq": accel_px_per_frame_sq,
        **kp_trajectories_smooth # Adds "RIGHT_SHOULDER", "RIGHT_WRIST" etc.
    }

    for i, (start_frame, end_frame) in enumerate(swings):
        logging.info(f"Analyzing swing {i} (frames {start_frame}-{end_frame})...")
        
        swing_metrics = analyze_swing(
            full_data=full_trajectory_data,
            swing_frames=(start_frame, end_frame),
            fps=fps,
            scaler=scaler
        )
        
        swing_metrics["video"] = Path(video_path).name
        swing_metrics["swing_id"] = i
        all_swing_metrics.append(swing_metrics)

        # Generate visualizations
        try:
            # A. Plot
            plot_path = output_dir / f"swing_{i}_plots.png"
            logging.info(f"Generating plot: {plot_path}")
            plot_swing_analytics(
                swing_metrics["time_series"],
                output_path=plot_path,
                title=f"Swing {i} Analysis"
            )

            # B. GIF
            gif_path = output_dir / f"swing_{i}_overlay.gif"
            logging.info(f"Generating GIF: {gif_path}")
            
            # Prepare data for overlay
            overlay_data = {
                "bat_tip": tip_trajectory, # Show raw trajectory
                "bat_tip_smooth": tip_trajectory_smooth,
                "keypoints": all_keypoints, # Show raw keypoints
            }
            
            create_swing_gif(
                frames=all_frames,
                overlay_data=overlay_data,
                swing_frames=(start_frame, end_frame),
                output_path=str(gif_path),
                fps=fps
            )
        except Exception as e:
            logging.error(f"Failed to generate visualization for swing {i}: {e}")

    # 6. Save Final Metrics CSV
    if all_swing_metrics:
        df = pd.DataFrame(all_swing_metrics)
        
        # Reorder columns for readability
        csv_cols = [
            "video", "swing_id", "start_frame", "end_frame", "duration_s",
            "peak_speed_mps", "peak_angular_velocity_rps",
            "mean_angular_velocity_rps", "swing_angle_deg", "smoothness_score"
        ]
        # Add any extra columns (like time_series) at the end
        extra_cols = [col for col in df.columns if col not in csv_cols]
        df = df[csv_cols + extra_cols]
        
        csv_path = output_dir / "metrics.csv"
        df.to_csv(csv_path, index=False, float_format="%.4f")
        logging.info(f"Metrics saved to {csv_path}")
    else:
        logging.warning("No swings were analyzed. No metrics.csv file will be saved.")

    logging.info("Analysis complete.")


def main():
    parser = argparse.ArgumentParser(description="Bat Swing Analysis CLI")
    parser.add_argument(
        "--video", type=str, required=True, help="Path to the input video file."
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n.pt",
        help="Path to the YOLOv8 model file (.pt).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        required=True,
        help="Pixel-to-meter scale (meters per pixel). E.g., 0.004",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/default",
        help="Directory to save analysis results (CSV, plots, GIFs).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override video FPS. If not set, it's read from the video file.",
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    
    run_pipeline(
        video_path=args.video,
        yolo_model_path=args.yolo_model,
        scale_mps=args.scale,
        output_dir=output_path,
        fps_override=args.fps,
    )


if __name__ == "__main__":
    print("TESTING: Main block is running") # <-- ADD THIS LINE
    main()