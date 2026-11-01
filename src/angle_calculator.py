"""
Joint angle calculation and geometric vector analysis for yoga posture analysis.
Supports 2D and 3D angle calculations, segment inclinations, and relative distances.
"""
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from .config import (
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL,
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    NOSE
)

class AngleCalculator:
    """Calculates geometric joint angles and body alignments from landmark coordinates."""

    @staticmethod
    def calculate_angle_2d(
        a: Union[Tuple[float, float], np.ndarray, List[float]],
        b: Union[Tuple[float, float], np.ndarray, List[float]],
        c: Union[Tuple[float, float], np.ndarray, List[float]]
    ) -> float:
        """
        Calculates the interior angle at joint 'b' formed by segments ba and bc in degrees [0, 180].
        Args:
            a: Coordinates (x, y) of first endpoint
            b: Coordinates (x, y) of vertex joint
            c: Coordinates (x, y) of second endpoint
        Returns:
            Angle in degrees [0, 180]
        """
        a_arr = np.array([a[0], a[1]], dtype=np.float64)
        b_arr = np.array([b[0], b[1]], dtype=np.float64)
        c_arr = np.array([c[0], c[1]], dtype=np.float64)

        ba = a_arr - b_arr
        bc = c_arr - b_arr

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0.0

        dot = np.dot(ba, bc)
        cosine = dot / (norm_ba * norm_bc)
        cosine = np.clip(cosine, -1.0, 1.0)

        angle_rad = np.arccos(cosine)
        return float(np.degrees(angle_rad))

    @staticmethod
    def calculate_angle_3d(
        a: Union[Tuple[float, float, float], np.ndarray, List[float]],
        b: Union[Tuple[float, float, float], np.ndarray, List[float]],
        c: Union[Tuple[float, float, float], np.ndarray, List[float]]
    ) -> float:
        """Calculates 3D angle at vertex 'b' formed by ba and bc."""
        a_arr = np.array([a[0], a[1], a[2]], dtype=np.float64)
        b_arr = np.array([b[0], b[1], b[2]], dtype=np.float64)
        c_arr = np.array([c[0], c[1], c[2]], dtype=np.float64)

        ba = a_arr - b_arr
        bc = c_arr - b_arr

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0.0

        dot = np.dot(ba, bc)
        cosine = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    @staticmethod
    def calculate_inclination_angle(p1: Union[Tuple[float, float], np.ndarray],
                                    p2: Union[Tuple[float, float], np.ndarray]) -> float:
        """
        Calculates angle of line segment (p1->p2) relative to the horizontal plane in degrees.
        0 degrees = perfectly horizontal. 90 degrees = perfectly vertical.
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle_rad = math.atan2(abs(dy), abs(dx))
        return float(math.degrees(angle_rad))

    @staticmethod
    def calculate_distance(p1: Union[Tuple[float, float], np.ndarray],
                           p2: Union[Tuple[float, float], np.ndarray]) -> float:
        """Calculates 2D Euclidean distance between two points."""
        return float(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))

    def compute_all_joint_angles(self, landmarks: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Computes key anatomical angles given an array of 33 normalized landmarks.
        Returns a dictionary of named angles in degrees.
        """
        if not landmarks or len(landmarks) < 33:
            return {}

        def pt(idx: int) -> Tuple[float, float]:
            lm = landmarks[idx]
            return (lm["x"], lm["y"])

        angles = {}

        # Elbows
        angles["left_elbow"] = self.calculate_angle_2d(pt(LEFT_SHOULDER), pt(LEFT_ELBOW), pt(LEFT_WRIST))
        angles["right_elbow"] = self.calculate_angle_2d(pt(RIGHT_SHOULDER), pt(RIGHT_ELBOW), pt(RIGHT_WRIST))

        # Shoulders (Arm to torso angle)
        angles["left_shoulder"] = self.calculate_angle_2d(pt(LEFT_HIP), pt(LEFT_SHOULDER), pt(LEFT_ELBOW))
        angles["right_shoulder"] = self.calculate_angle_2d(pt(RIGHT_HIP), pt(RIGHT_SHOULDER), pt(RIGHT_ELBOW))

        # Hips (Torso to thigh angle)
        angles["left_hip"] = self.calculate_angle_2d(pt(LEFT_SHOULDER), pt(LEFT_HIP), pt(LEFT_KNEE))
        angles["right_hip"] = self.calculate_angle_2d(pt(RIGHT_SHOULDER), pt(RIGHT_HIP), pt(RIGHT_KNEE))

        # Knees
        angles["left_knee"] = self.calculate_angle_2d(pt(LEFT_HIP), pt(LEFT_KNEE), pt(LEFT_ANKLE))
        angles["right_knee"] = self.calculate_angle_2d(pt(RIGHT_HIP), pt(RIGHT_KNEE), pt(RIGHT_ANKLE))

        # Ankles
        angles["left_ankle"] = self.calculate_angle_2d(pt(LEFT_KNEE), pt(LEFT_ANKLE), pt(LEFT_FOOT_INDEX))
        angles["right_ankle"] = self.calculate_angle_2d(pt(RIGHT_KNEE), pt(RIGHT_ANKLE), pt(RIGHT_FOOT_INDEX))

        # Arm horizontal inclination relative to floor (horizontal plane)
        angles["left_arm_incline"] = self.calculate_inclination_angle(pt(LEFT_SHOULDER), pt(LEFT_WRIST))
        angles["right_arm_incline"] = self.calculate_inclination_angle(pt(RIGHT_SHOULDER), pt(RIGHT_WRIST))

        # Spine / Torso inclination relative to vertical
        mid_shoulder = ((pt(LEFT_SHOULDER)[0] + pt(RIGHT_SHOULDER)[0]) / 2.0,
                        (pt(LEFT_SHOULDER)[1] + pt(RIGHT_SHOULDER)[1]) / 2.0)
        mid_hip = ((pt(LEFT_HIP)[0] + pt(RIGHT_HIP)[0]) / 2.0,
                   (pt(LEFT_HIP)[1] + pt(RIGHT_HIP)[1]) / 2.0)
        torso_horiz = self.calculate_inclination_angle(mid_hip, mid_shoulder)
        angles["torso_vertical_incline"] = abs(90.0 - torso_horiz)

        # Inverted V / Downward Dog hip hinge angle
        mid_knee = ((pt(LEFT_KNEE)[0] + pt(RIGHT_KNEE)[0]) / 2.0,
                    (pt(LEFT_KNEE)[1] + pt(RIGHT_KNEE)[1]) / 2.0)
        angles["hip_hinge"] = self.calculate_angle_2d(mid_shoulder, mid_hip, mid_knee)

        # Arms straight span (Wrist - Shoulder - Wrist)
        angles["arms_span"] = self.calculate_angle_2d(pt(LEFT_WRIST), pt(LEFT_SHOULDER), pt(RIGHT_WRIST))

        return angles
