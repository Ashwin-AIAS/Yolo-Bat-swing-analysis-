"""
Functions for drawing overlays and creating animated GIFs.
"""
import numpy as np
import cv2
import imageio
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from pathlib import Path
import logging

# Define some colors (BGR format)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_ORANGE = (0, 165, 255)


def draw_frame_overlay(
    frame: np.ndarray,
    frame_idx: int,
    overlay_data: Dict[str, Any],
    max_trajectory_points: int = 100
) -> np.ndarray:
    """
    Draws all annotations on a single frame.
    
    Args:
        frame (np.ndarray): The video frame (BGR).
        frame_idx (int): The current frame index.
        overlay_data (Dict[str, Any]): Dictionary containing trajectories
            like 'bat_tip', 'keypoints', etc.
    
    Returns:
        np.ndarray: The frame with overlays drawn.
    """
    vis_frame = frame.copy()
    h, w, _ = vis_frame.shape

    # 1. Draw Keypoints
    keypoints = overlay_data.get("keypoints", [])
    if frame_idx < len(keypoints):
        frame_kps = keypoints[frame_idx]
        for name, (x, y) in frame_kps.items():
            if not np.isnan(x):
                cv2.circle(vis_frame, (int(x), int(y)), 5, COLOR_GREEN, -1, cv2.LINE_AA)
                
        # Draw skeleton lines
        lines = [("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
                 ("RIGHT_SHOULDER", "RIGHT_HIP")]
        for p1_name, p2_name in lines:
            if p1_name in frame_kps and p2_name in frame_kps:
                p1 = frame_kps[p1_name]
                p2 = frame_kps[p2_name]
                if not np.isnan(p1).any() and not np.isnan(p2).any():
                    cv2.line(vis_frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                             COLOR_WHITE, 2, cv2.LINE_AA)

    # 2. Draw Bat-Tip and Trajectory
    bat_tip_traj = overlay_data.get("bat_tip_smooth", []) # Use smooth for overlay
    if frame_idx < len(bat_tip_traj):
        
        # Draw trajectory history
        start_idx = max(0, frame_idx - max_trajectory_points)
        trajectory_pts = bat_tip_traj[start_idx:frame_idx+1]
        
        # Filter out NaNs
        trajectory_pts = trajectory_pts[~np.isnan(trajectory_pts).any(axis=1)].astype(int)

        if len(trajectory_pts) > 1:
            cv2.polylines(vis_frame, [trajectory_pts], isClosed=False, 
                          color=COLOR_ORANGE, thickness=2, lineType=cv2.LINE_AA)
        
        # Draw current bat-tip
        current_tip = bat_tip_traj[frame_idx]
        if not np.isnan(current_tip).any():
            cv2.circle(vis_frame, (int(current_tip[0]), int(current_tip[1])), 
                       8, COLOR_RED, -1, cv2.LINE_AA)

    # 3. Draw Frame Index
    cv2.putText(vis_frame, f"Frame: {frame_idx}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_WHITE, 2, cv2.LINE_AA)
                
    return vis_frame


# --- THIS ENTIRE FUNCTION IS REPLACED ---
def create_swing_gif(
    video_path: str,
    overlay_data: Dict[str, Any],
    swing_frames: Tuple[int, int],
    output_path: str,
    fps: float,
    padding_frames: int = 15
):
    """
    Creates an animated GIF for a specific swing event.
    Reads frames directly from the video file to save memory.
    """
    start_f, end_f = swing_frames
    
    # Add padding
    start_f = max(0, start_f - padding_frames)
    end_f = end_f + padding_frames # We'll check the upper bound inside the loop
    
    gif_frames = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"GIF Error: Could not open video {video_path}")
        return

    # Set the video capture to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    
    current_frame_idx = start_f
    
    desc = f"Creating GIF {Path(output_path).name}"
    for i in tqdm(range(start_f, end_f + 1), desc=desc, leave=False):
        ret, frame = cap.read()
        if not ret:
            break # Reached end of video

        # Draw overlay
        overlay_frame = draw_frame_overlay(
            frame=frame,
            frame_idx=current_frame_idx,
            overlay_data=overlay_data
        )
        
        # Convert BGR (OpenCV) to RGB (imageio)
        rgb_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
        gif_frames.append(rgb_frame)
        current_frame_idx += 1
        
    cap.release()
    
    if not gif_frames:
        logging.warning("No frames were extracted for the GIF.")
        return

    # Save the GIF
    imageio.mimsave(output_path, gif_frames, fps=min(fps, 30)) # Cap FPS at 30 for GIFs
# --- END OF REPLACEMENT ---
