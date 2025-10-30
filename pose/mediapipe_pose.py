"""
MediaPipe Pose Estimation Wrapper
"""
import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, Any, Optional

class MediaPipePose:
    """
    Wrapper class for the MediaPipe Pose solution.
    """
    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5):
        """
        Initializes the MediaPipe Pose model.

        Args:
            model_complexity (int): Complexity of the pose model (0, 1, or 2).
            min_detection_confidence (float): Minimum confidence for detection.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )
        
        # Get the names of all landmarks
        self.landmark_names = [lm.name for lm in self.mp_pose.PoseLandmark]

    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Processes a single frame to find pose landmarks.

        Args:
            frame (np.ndarray): The input frame (BGR format).

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing:
                - 'landmarks_2d_pixels': Dict[str, Tuple(int, int)]
                - 'landmarks_3d_world': Dict[str, Tuple(float, float, float)]
            Returns None if no pose is detected.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False # Improve performance
        
        results = self.pose.process(frame_rgb)
        
        frame_rgb.flags.writeable = True # Not needed, but good practice

        if not results.pose_landmarks:
            return None

        h, w, _ = frame.shape
        
        landmarks_2d = {}
        landmarks_3d = {}

        # Process 2D pixel landmarks
        for i, lm in enumerate(results.pose_landmarks.landmark):
            lm_name = self.landmark_names[i]
            # Convert normalized (0.0-1.0) coords to pixel coords
            px, py = int(lm.x * w), int(lm.y * h)
            landmarks_2d[lm_name] = (px, py)

        # Process 3D world landmarks (relative to hip center)
        if results.pose_world_landmarks:
             for i, lm in enumerate(results.pose_world_landmarks.landmark):
                lm_name = self.landmark_names[i]
                landmarks_3d[lm_name] = (lm.x, lm.y, lm.z)

        return {
            "landmarks_2d_pixels": landmarks_2d,
            "landmarks_3d_world": landmarks_3d,
            "raw_results": results # For debugging or advanced use
        }

    def close(self):
        """Releases the pose model resources."""
        self.pose.close()