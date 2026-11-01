"""
MediaPipe Pose detection wrapper supporting video frames, preprocessing,
and normalized landmark extraction.
"""
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Dict, List, Optional, Tuple, Any

class PoseDetector:
    """Wrapper around MediaPipe PoseLandmarker for real-time human pose estimation."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: str = "full"  # "lite" or "full"
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            filename = "pose_landmarker_full.task" if model_complexity == "full" else "pose_landmarker_lite.task"
            model_path = os.path.join(base_dir, "models", filename)

        if not os.path.exists(model_path):
            # Fallback to alternate model file if exists
            fallback = os.path.join(base_dir, "models", "pose_landmarker_lite.task")
            if os.path.exists(fallback):
                model_path = fallback
            else:
                raise FileNotFoundError(f"PoseLandmarker model not found at: {model_path}")

        self.model_path = model_path
        self.base_options = python.BaseOptions(model_asset_path=self.model_path)
        self.options = vision.PoseLandmarkerOptions(
            base_options=self.base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.IMAGE
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(self.options)

    def preprocess_frame(
        self,
        frame: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        apply_denoise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies Image & Video Processing (IMV) steps:
        - Resizing (optional)
        - Color-space conversion (BGR to RGB)
        - Optional Gaussian blur noise reduction
        Returns:
            (preprocessed_rgb_frame, original_bgr_frame)
        """
        processed_bgr = frame.copy()
        if target_size is not None:
            processed_bgr = cv2.resize(processed_bgr, target_size, interpolation=cv2.INTER_AREA)

        if apply_denoise:
            # Gaussian blur for noise reduction
            processed_bgr = cv2.GaussianBlur(processed_bgr, (3, 3), 0)

        # Color-space conversion: OpenCV uses BGR, MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        return rgb_frame, processed_bgr

    def detect(self, frame_bgr: np.ndarray) -> Optional[List[Dict[str, float]]]:
        """
        Detects body landmarks in a single BGR video frame.
        Returns:
            List of 33 normalized landmark dicts with keys ('x', 'y', 'z', 'visibility', 'presence')
            or None if no person is detected.
        """
        rgb_frame, _ = self.preprocess_frame(frame_bgr)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.landmarker.detect(mp_image)

        if not detection_result or not detection_result.pose_landmarks:
            return None

        # Return the first detected person's landmarks
        first_pose = detection_result.pose_landmarks[0]
        landmarks_list = []
        for lm in first_pose:
            landmarks_list.append({
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "visibility": float(getattr(lm, "visibility", 1.0)),
                "presence": float(getattr(lm, "presence", 1.0)),
            })

        return landmarks_list

    def close(self):
        """Releases the landmarker resources."""
        if hasattr(self, "landmarker") and self.landmarker:
            self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
