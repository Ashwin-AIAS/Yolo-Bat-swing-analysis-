"""
Core logic for swing detection and metric calculation.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, welch
from typing import Tuple, Dict, Any, Optional

from utils.geometry import calculate_angle_between_vectors, interpolate_points, smooth_trajectory

class PixelScaler:
    """Converts pixel measurements to metric (meter) measurements."""
    
    def __init__(self, pixel_ref: float, meter_ref: float):
        """
        Initializes the scaler.
        
        Args:
            pixel_ref (float): The number of pixels (e.g., 250).
            meter_ref (float): The corresponding number of meters (e.g., 1.0).
        """
        if pixel_ref <= 0:
            raise ValueError("pixel_ref must be positive")
        self.scale = meter_ref / pixel_ref  # meters per pixel

    def to_meters(self, pixels: float) -> float:
        """Converts pixels to meters."""
        return pixels * self.scale

    def to_mps(self, pixels_per_frame: float, fps: float) -> float:
        """Converts (pixels per frame) to (meters per second)."""
        return (pixels_per_frame * self.scale) * fps


def compute_derivatives(
    trajectory: np.ndarray, fps: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes position, velocity (speed), and acceleration from a trajectory.
    Handles NaNs by interpolation and applies smoothing.
    
    Args:
        trajectory (np.ndarray): Array of (x, y) points, shape (N, 2).
        fps (float): Frames per second of the video.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - pos_smooth (np.ndarray): Smoothed (x, y) trajectory.
            - speed_px_per_frame (np.ndarray): Speed (scalar) in pixels/frame. Shape (N,).
            - accel_px_per_frame_sq (np.ndarray): Acceleration (scalar) in pixels/frame^2. Shape (N,).
    """
    # 1. Interpolate NaNs
    pos_interp = interpolate_points(trajectory)
    
    # 2. Smooth
    # Use a relatively wide window for acceleration calculation
    pos_smooth = smooth_trajectory(pos_interp, window=7, poly=2)

    # 3. Compute velocity (pixels/frame)
    # np.gradient computes central difference, dx is (1 / (2*dt))
    # Here dt = 1 frame, so we get (pixels / 2 frames).
    # Using edge_order=2 gives more accurate boundary estimates.
    velocity_px_per_frame = np.gradient(pos_smooth, axis=0, edge_order=2) # Shape (N, 2)
    
    # 4. Compute speed (scalar magnitude of velocity)
    speed_px_per_frame = np.linalg.norm(velocity_px_per_frame, axis=1) # Shape (N,)

    # 5. Compute acceleration (pixels/frame^2)
    acceleration_px_per_frame_sq = np.gradient(velocity_px_per_frame, axis=0, edge_order=2) # Shape (N, 2)
    
    # 6. Compute acceleration magnitude
    accel_magnitude_px_per_frame_sq = np.linalg.norm(acceleration_px_per_frame_sq, axis=1) # Shape (N,)
    
    return pos_smooth, speed_px_per_frame, accel_magnitude_px_per_frame_sq


def compute_angular_velocity(
    shoulder_pts: np.ndarray, 
    tip_pts: np.ndarray, 
    fps: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the angular velocity of the shoulder-to-tip vector.
    
    Args:
        shoulder_pts (np.ndarray): Trajectory of the shoulder (N, 2).
        tip_pts (np.ndarray): Trajectory of the bat-tip (N, 2).
        fps (float): Frames per second.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - angles_rad (np.ndarray): Angle of the vector in radians (N,).
            - ang_vel_rad_per_sec (np.ndarray): Angular velocity in rad/s (N,).
    """
    # Create the shoulder-to-tip vectors
    vectors = tip_pts - shoulder_pts # Shape (N, 2)
    
    # Compute the angle of each vector relative to the positive x-axis
    # arctan2(y, x)
    angles_rad = np.arctan2(vectors[:, 1], vectors[:, 0])
    
    # 'Unwrap' the angle to handle transitions from +pi to -pi
    angles_unwrapped = np.unwrap(angles_rad)
    
    # Compute angular velocity (rad/frame) using gradient
    ang_vel_rad_per_frame = np.gradient(angles_unwrapped, edge_order=2)
    
    # Convert to rad/s
    ang_vel_rad_per_sec = ang_vel_rad_per_frame * fps
    
    return angles_rad, ang_vel_rad_per_sec


def compute_smoothness_score(acceleration_signal: np.ndarray) -> float:
    """
    Computes a smoothness score (0-1) based on spectral entropy.
    A smooth signal (few frequencies) has low entropy -> high score.
    A jerky signal (many frequencies) has high entropy -> low score.
    
    Args:
        acceleration_signal (np.ndarray): 1D signal (e.g., magnitude of acceleration).
    
    Returns:
        float: Smoothness score [0, 1].
    """
    if np.all(acceleration_signal == 0):
        return 1.0  # Perfectly smooth

    # 1. Compute Power Spectral Density (PSD)
    # nperseg=None uses the whole signal length
    f, Pxx = welch(acceleration_signal, nperseg=min(256, len(acceleration_signal)))

    if np.sum(Pxx) == 0:
        return 1.0

    # 2. Normalize PSD to create a probability distribution
    psd_norm = Pxx / np.sum(Pxx)
    
    # 3. Compute Shannon Entropy
    # Use a small epsilon to avoid log(0)
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    
    # 4. Normalize entropy
    # Max entropy for a signal of length M is log2(M)
    if len(psd_norm) <= 1:
        return 1.0 # Not enough data to compute entropy
        
    max_entropy = np.log2(len(psd_norm))
    normalized_entropy = entropy / max_entropy
    
    # 5. Smoothness = 1 - Normalized Entropy
    # High entropy (jerky) -> low smoothness
    # Low entropy (smooth) -> high smoothness
    return max(0.0, 1.0 - normalized_entropy)


def detect_swings(
    speed_mps: np.ndarray, 
    speed_thresh_mps: float = 2.0, 
    min_swing_frames: int = 5
) -> List[Tuple[int, int]]:
    """
    Detects swing events based on speed thresholding.
    
    Args:
        speed_mps (np.ndarray): 1D array of speed in m/s.
        speed_thresh_mps (float): The speed threshold to define a swing.
        min_swing_frames (int): The minimum number of contiguous frames
                                above threshold to be considered a swing.

    Returns:
        List[Tuple[int, int]]: A list of (start_frame, end_frame) tuples.
    """
    try:
        from scipy.ndimage import label
    except ImportError:
        raise ImportError("scipy is required for swing detection. `pip install scipy`")
        
    above_threshold = speed_mps > speed_thresh_mps
    
    # Find contiguous blocks of 'True'
    labeled_array, num_features = label(above_threshold)
    
    swings = []
    for i in range(1, num_features + 1):
        indices = np.where(labeled_array == i)[0]
        if len(indices) >= min_swing_frames:
            start_frame = indices[0]
            end_frame = indices[-1]
            
            # TODO: Add padding? For now, just return the core motion.
            swings.append((start_frame, end_frame))
            
    return swings


def analyze_swing(
    full_data: Dict[str, np.ndarray],
    swing_frames: Tuple[int, int],
    fps: float,
    scaler: PixelScaler
) -> Dict[str, Any]:
    """
    Calculates all metrics for a single detected swing.

    Args:
        full_data (Dict[str, np.ndarray]): A dict containing all trajectories
            (e.g., 'bat_tip', 'RIGHT_SHOULDER', 'speed_mps', 'accel_px_per_frame_sq').
        swing_frames (Tuple[int, int]): (start_frame, end_frame) of the swing.
        fps (float): Video FPS.
        scaler (PixelScaler): The initialized pixel-to-meter scaler.

    Returns:
        Dict[str, Any]: A dictionary of scalar metrics for this swing.
    """
    start, end = swing_frames
    
    # 1. Slice data for the swing event
    swing_duration_s = (end - start) / fps
    swing_speed_mps = full_data["speed_mps"][start:end]
    swing_accel_px = full_data["accel_px_per_frame_sq"][start:end]
    swing_shoulder_pts = full_data["RIGHT_SHOULDER"][start:end]
    swing_tip_pts = full_data["bat_tip"][start:end]
    
    # 2. Linear Speed Metrics
    peak_speed_mps = np.max(swing_speed_mps)
    
    # 3. Angular Velocity Metrics
    swing_angles_rad, swing_ang_vel_rps = compute_angular_velocity(
        swing_shoulder_pts, swing_tip_pts, fps
    )
    
    # Use absolute velocity for peak/mean
    abs_ang_vel = np.abs(swing_ang_vel_rps)
    peak_angular_velocity_rps = np.max(abs_ang_vel)
    mean_angular_velocity_rps = np.mean(abs_ang_vel)
    
    # 4. Swing Angle
    # Total angular displacement
    start_angle = swing_angles_rad[0]
    end_angle = swing_angles_rad[-1]
    # We use np.unwrap on the [start, end] pair to get the correct path
    swing_angle_rad = np.abs(np.unwrap([start_angle, end_angle])[1] - start_angle)
    swing_angle_deg = np.rad2deg(swing_angle_rad)
    
    # 5. Smoothness
    smoothness_score = compute_smoothness_score(swing_accel_px)

    # 6. Time series data (for plotting)
    time_s = np.arange(len(swing_speed_mps)) / fps
    time_series_data = {
        "time_s": time_s,
        "speed_mps": swing_speed_mps,
        "angle_deg": np.rad2deg(swing_angles_rad),
        "angular_velocity_rps": swing_ang_vel_rps,
    }

    return {
        "start_frame": start,
        "end_frame": end,
        "duration_s": swing_duration_s,
        "peak_speed_mps": peak_speed_mps,
        "peak_angular_velocity_rps": peak_angular_velocity_rps,
        "mean_angular_velocity_rps": mean_angular_velocity_rps,
        "swing_angle_deg": swing_angle_deg,
        "smoothness_score": smoothness_score,
        "time_series": time_series_data, # Store this for plotting
    }