"""
Wrapper for the SORT (Simple Online and Realtime Tracking) algorithm.

Note: This is provided as requested by the prompt, but the main
pipeline (run_analysis.py) was simplified to a single-player focus
and does not currently use this tracker. This module is ready to be
integrated into a multi-player tracking pipeline.
"""
import numpy as np
from typing import List, Optional
try:
    from sort_track.sort import Sort
except ImportError:
    print("Warning: 'sort-track' package not found. SORTTracker will not be available.")
    print("Install with: pip install sort-track")
    Sort = None

class SORTTracker:
    """
    A simple wrapper for the SORT tracking algorithm.
    """
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initializes the SORT tracker.

        Args:
            max_age (int): Maximum number of frames to keep a track alive without a detection.
            min_hits (int): Minimum number of detections to start a track.
            iou_threshold (float): IOU threshold for matching detections to tracks.
        """
        if Sort is None:
            raise ImportError("SORTTracker requires the 'sort-track' package.")
        
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    def update(self, detections: List[List[float]]) -> np.ndarray:
        """
        Updates the tracker with new detections for the current frame.

        Args:
            detections (List[List[float]]): A list of detections, where each
                detection is [x1, y1, x2, y2, score].

        Returns:
            np.ndarray: An array of active tracks, where each row is
                [x1, y1, x2, y2, track_id].
        """
        if not detections:
            # SORT expects an empty array of shape (0, 5) if no detections
            detections_np = np.empty((0, 5))
        else:
            detections_np = np.array(detections)
            
        # Update the tracker
        tracked_objects = self.tracker.update(detections_np)
        
        return tracked_objects