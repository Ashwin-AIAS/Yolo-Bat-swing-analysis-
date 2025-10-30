"""
YOLOv8 Detector Wrapper
"""
import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO

class YOLOv8Detector:
    """
    Wrapper class for the Ultralytics YOLOv8 object detector.
    """
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initializes the YOLOv8 model.

        Args:
            model_path (str): Path to the YOLOv8 model file (.pt).
        """
        try:
            self.model = YOLO(model_path)
            self.model_names = self.model.names
        except Exception as e:
            raise ImportError(f"Could not load YOLOv8 model from {model_path}. "
                              f"Ensure 'ultralytics' is installed and the path is correct. Error: {e}")

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Performs object detection on a single frame.

        Args:
            frame (np.ndarray): The input frame (BGR format).
            conf_threshold (float): Confidence threshold for detections.

        Returns:
            List[Dict[str, Any]]: A list of detection dictionaries.
                Each dict has: {'bbox': [x1, y1, x2, y2], 'class_id': int, 'class_name': str, 'conf': float}
        """
        # Note: YOLO processes BGR frames correctly.
        results = self.model.predict(frame, conf=conf_threshold, verbose=False)
        
        detections = []
        if not results:
            return detections

        res = results[0]  # Get results for the first (only) image
        boxes = res.boxes.cpu().numpy() # Get boxes in numpy format

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": class_id,
                "class_name": self.model_names.get(class_id, "unknown"),
                "conf": conf,
            })
            
        return detections