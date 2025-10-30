"""
Geometry and signal processing utility functions.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Tuple, List, Optional

def interpolate_points(points: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Interpolates NaN values in a 2D trajectory.

    Args:
        points (np.ndarray): An (N, 2) array of (x, y) coordinates
                             that may contain np.nan.
        method (str): Interpolation method (e.g., 'linear', 'slinear', 'quadratic').

    Returns:
        np.ndarray: An (N, 2) array with NaNs filled.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Input array must have shape (N, 2)")

    # Use pandas for simple and robust interpolation
    df = pd.DataFrame(points, columns=['x', 'y'])
    df_interp = df.interpolate(method=method, axis=0, limit_direction='both')
    
    # Fill any remaining NaNs (e.g., at the very start/end)
    df_interp = df_interp.fillna(method='bfill').fillna(method='ffill')
    
    return df_interp.to_numpy()


def smooth_trajectory(
    points: np.ndarray, window: int = 5, poly: int = 2
) -> np.ndarray:
    """
    Applies a Savitzky-Golay filter to smooth a 2D trajectory.

    Args:
        points (np.ndarray): An (N, 2) array of (x, y) coordinates.
        window (int): The length of the filter window (must be odd).
        poly (int): The order of the polynomial used to fit the samples.

    Returns:
        np.ndarray: The smoothed (N, 2) array.
    """
    if points.shape[0] < window:
        # Not enough points to filter, return original
        return points
        
    # Ensure window is odd
    if window % 2 == 0:
        window += 1
        
    # Apply filter to x and y coordinates separately
    x_smooth = savgol_filter(points[:, 0], window, poly)
    y_smooth = savgol_filter(points[:, 1], window, poly)
    
    return np.stack([x_smooth, y_smooth], axis=1)


def calculate_angle(
    p1: Tuple[float, float], 
    p2: Tuple[float, float], 
    p3: Tuple[float, float]
) -> float:
    """
    Calculates the angle (in radians) at p2, formed by p1-p2-p3.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    v1 = p1 - p2
    v2 = p3 - p2
    
    dot_prod = np.dot(v1, v2)
    norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    if norm_prod == 0:
        return 0.0  # Avoid division by zero
        
    cos_theta = np.clip(dot_prod / norm_prod, -1.0, 1.0)
    return np.arccos(cos_theta)


def calculate_angle_between_vectors(
    v1: np.ndarray, v2: np.ndarray
) -> float:
    """Calculates the angle (in radians) between two 2D vectors."""
    dot_prod = np.dot(v1, v2)
    norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    if norm_prod == 0:
        return 0.0
        
    cos_theta = np.clip(dot_prod / norm_prod, -1.0, 1.0)
    return np.arccos(cos_theta)
    

def get_bbox_ends(bbox: List[int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Estimates the two 'ends' of a bounding box by finding the two
    farthest corners from the center. This approximates the major axis.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    if w > h:
        # Horizontal-ish box
        return (x1, int(cy)), (x2, int(cy))
    else:
        # Vertical-ish box
        return (int(cx), y1), (int(cx), y2)

def estimate_bat_tip(
    bat_bbox: List[int], 
    wrist_point: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Estimates the bat-tip position.
    
    Heuristic: The tip is the 'end' of the bat's bounding box
    that is FARTHEST from the player's wrist.

    Args:
        bat_bbox (List[int]): [x1, y1, x2, y2] for the bat.
        wrist_point (Tuple[int, int]): (x, y) for the player's wrist.

    Returns:
        Tuple[int, int]: The estimated (x, y) of the bat tip.
    """
    if not wrist_point or any(np.isnan(wrist_point)):
        # No wrist, just return the center of the bat box
        return (int((bat_bbox[0] + bat_bbox[2]) / 2), 
                int((bat_bbox[1] + bat_bbox[3]) / 2))

    end1, end2 = get_bbox_ends(bat_bbox)
    
    dist1 = np.linalg.norm(np.array(end1) - np.array(wrist_point))
    dist2 = np.linalg.norm(np.array(end2) - np.array(wrist_point))
    
    if dist1 > dist2:
        return end1
    else:
        return end2