"""
Sample Yoga Posture Generator.
Generates synthetic geometric landmark sets and rendered human silhouette frames
for testing and demonstrating all 7 yoga poses with correct and incorrect variations.
"""
from typing import Dict, List, Tuple
import cv2
import numpy as np
from .config import (
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    NOSE, POSE_CONNECTIONS
)

class SamplePoseGenerator:
    """Creates realistic normalized landmark sets for all target yoga poses."""

    @staticmethod
    def get_blank_landmarks() -> List[Dict[str, float]]:
        return [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.99, "presence": 0.99} for _ in range(33)]

    @classmethod
    def generate_tadasana(cls, correct: bool = True) -> List[Dict[str, float]]:
        """Generates landmarks for Tadasana (Mountain Pose)."""
        lms = cls.get_blank_landmarks()
        lms[NOSE] = {"x": 0.50, "y": 0.15, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[LEFT_SHOULDER] = {"x": 0.45, "y": 0.25, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_SHOULDER] = {"x": 0.55, "y": 0.25, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        lms[LEFT_ELBOW] = {"x": 0.44, "y": 0.40, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_ELBOW] = {"x": 0.56, "y": 0.40, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        lms[LEFT_WRIST] = {"x": 0.44, "y": 0.55, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_WRIST] = {"x": 0.56, "y": 0.55, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        lms[LEFT_HIP] = {"x": 0.47, "y": 0.52, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_HIP] = {"x": 0.53, "y": 0.52, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        if correct:
            # Straight knees
            lms[LEFT_KNEE] = {"x": 0.47, "y": 0.72, "z": 0.0, "visibility": 1.0, "presence": 1.0}
            lms[RIGHT_KNEE] = {"x": 0.53, "y": 0.72, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        else:
            # Sagging/bent knees
            lms[LEFT_KNEE] = {"x": 0.42, "y": 0.70, "z": 0.0, "visibility": 1.0, "presence": 1.0}
            lms[RIGHT_KNEE] = {"x": 0.58, "y": 0.70, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        lms[LEFT_ANKLE] = {"x": 0.47, "y": 0.92, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_ANKLE] = {"x": 0.53, "y": 0.92, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        return lms

    @classmethod
    def generate_warrior_ii(cls, correct: bool = True) -> List[Dict[str, float]]:
        """Generates landmarks for Warrior II (Virabhadrasana II)."""
        lms = cls.get_blank_landmarks()
        lms[NOSE] = {"x": 0.50, "y": 0.20, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[LEFT_SHOULDER] = {"x": 0.45, "y": 0.30, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_SHOULDER] = {"x": 0.55, "y": 0.30, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        # Horizontal arms
        if correct:
            lms[LEFT_ELBOW] = {"x": 0.32, "y": 0.30, "z": 0.0, "visibility": 1.0, "presence": 1.0}
            lms[RIGHT_ELBOW] = {"x": 0.68, "y": 0.30, "z": 0.0, "visibility": 1.0, "presence": 1.0}
            lms[LEFT_WRIST] = {"x": 0.18, "y": 0.30, "z": 0.0, "visibility": 1.0, "presence": 1.0}
            lms[RIGHT_WRIST] = {"x": 0.82, "y": 0.30, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        else:
            # Drooping arms
            lms[LEFT_ELBOW] = {"x": 0.34, "y": 0.42, "z": 0.0, "visibility": 1.0, "presence": 1.0}
            lms[RIGHT_ELBOW] = {"x": 0.66, "y": 0.42, "z": 0.0, "visibility": 1.0, "presence": 1.0}
            lms[LEFT_WRIST] = {"x": 0.22, "y": 0.55, "z": 0.0, "visibility": 1.0, "presence": 1.0}
            lms[RIGHT_WRIST] = {"x": 0.78, "y": 0.55, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        lms[LEFT_HIP] = {"x": 0.46, "y": 0.52, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_HIP] = {"x": 0.54, "y": 0.52, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        # Left leg front bent ~90°, Right leg back straight
        if correct:
            lms[LEFT_KNEE] = {"x": 0.32, "y": 0.52, "z": 0.0, "visibility": 1.0, "presence": 1.0}
            lms[LEFT_ANKLE] = {"x": 0.32, "y": 0.88, "z": 0.0, "visibility": 1.0, "presence": 1.0}
            lms[RIGHT_KNEE] = {"x": 0.66, "y": 0.70, "z": 0.0, "visibility": 1.0, "presence": 1.0}
            lms[RIGHT_ANKLE] = {"x": 0.78, "y": 0.88, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        else:
            # Front knee under-bent (~135°)
            lms[LEFT_KNEE] = {"x": 0.32, "y": 0.66, "z": 0.0, "visibility": 1.0, "presence": 1.0}
            lms[LEFT_ANKLE] = {"x": 0.32, "y": 0.88, "z": 0.0, "visibility": 1.0, "presence": 1.0}
            lms[RIGHT_KNEE] = {"x": 0.66, "y": 0.70, "z": 0.0, "visibility": 1.0, "presence": 1.0}
            lms[RIGHT_ANKLE] = {"x": 0.78, "y": 0.88, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        return lms

    @classmethod
    def generate_tree_pose(cls, correct: bool = True) -> List[Dict[str, float]]:
        """Generates landmarks for Tree Pose (Vrikshasana)."""
        lms = cls.get_blank_landmarks()
        lms[NOSE] = {"x": 0.50, "y": 0.16, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[LEFT_SHOULDER] = {"x": 0.45, "y": 0.26, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_SHOULDER] = {"x": 0.55, "y": 0.26, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        # Hands in prayer in front of chest
        lms[LEFT_ELBOW] = {"x": 0.42, "y": 0.35, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_ELBOW] = {"x": 0.58, "y": 0.35, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[LEFT_WRIST] = {"x": 0.49, "y": 0.34, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_WRIST] = {"x": 0.51, "y": 0.34, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        lms[LEFT_HIP] = {"x": 0.47, "y": 0.52, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_HIP] = {"x": 0.53, "y": 0.52, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        # Standing leg: Left leg straight
        lms[LEFT_KNEE] = {"x": 0.47, "y": 0.72, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[LEFT_ANKLE] = {"x": 0.47, "y": 0.92, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        if correct:
            # Right knee flared outwards, foot at inner thigh
            lms[RIGHT_KNEE] = {"x": 0.68, "y": 0.65, "z": 0.0, "visibility": 1.0, "presence": 1.0}
            lms[RIGHT_ANKLE] = {"x": 0.50, "y": 0.68, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        else:
            # Low foot / unbent leg
            lms[RIGHT_KNEE] = {"x": 0.56, "y": 0.73, "z": 0.0, "visibility": 1.0, "presence": 1.0}
            lms[RIGHT_ANKLE] = {"x": 0.54, "y": 0.88, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        return lms

    @classmethod
    def generate_triangle_pose(cls, correct: bool = True) -> List[Dict[str, float]]:
        """Generates landmarks for Triangle Pose (Trikonasana)."""
        lms = cls.get_blank_landmarks()
        lms[NOSE] = {"x": 0.44, "y": 0.38, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        # Tilted shoulders
        lms[LEFT_SHOULDER] = {"x": 0.38, "y": 0.48, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_SHOULDER] = {"x": 0.48, "y": 0.34, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        # Left arm down toward foot, Right arm straight up to sky
        lms[LEFT_ELBOW] = {"x": 0.35, "y": 0.62, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[LEFT_WRIST] = {"x": 0.33, "y": 0.78, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_ELBOW] = {"x": 0.53, "y": 0.22, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_WRIST] = {"x": 0.58, "y": 0.10, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        lms[LEFT_HIP] = {"x": 0.45, "y": 0.58, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_HIP] = {"x": 0.55, "y": 0.54, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        # Wide straight legs
        lms[LEFT_KNEE] = {"x": 0.38, "y": 0.74, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[LEFT_ANKLE] = {"x": 0.32, "y": 0.90, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_KNEE] = {"x": 0.62, "y": 0.72, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_ANKLE] = {"x": 0.72, "y": 0.90, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        return lms

    @classmethod
    def generate_downward_dog(cls, correct: bool = True) -> List[Dict[str, float]]:
        """Generates landmarks for Downward Dog (Adho Mukha Svanasana)."""
        lms = cls.get_blank_landmarks()
        # Inverted V: Hips at highest y apex (smallest y value)
        lms[LEFT_HIP] = {"x": 0.50, "y": 0.35, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_HIP] = {"x": 0.52, "y": 0.35, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        lms[LEFT_SHOULDER] = {"x": 0.35, "y": 0.55, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_SHOULDER] = {"x": 0.37, "y": 0.55, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[NOSE] = {"x": 0.33, "y": 0.60, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        lms[LEFT_ELBOW] = {"x": 0.28, "y": 0.68, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_ELBOW] = {"x": 0.30, "y": 0.68, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[LEFT_WRIST] = {"x": 0.20, "y": 0.82, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_WRIST] = {"x": 0.22, "y": 0.82, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        if correct:
            # Straight legs
            lms[LEFT_KNEE] = {"x": 0.62, "y": 0.58, "z": 0.0, "visibility": 1.0, "presence": 1.0}
            lms[RIGHT_KNEE] = {"x": 0.64, "y": 0.58, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        else:
            # Sagging/bent knees
            lms[LEFT_KNEE] = {"x": 0.58, "y": 0.65, "z": 0.0, "visibility": 1.0, "presence": 1.0}
            lms[RIGHT_KNEE] = {"x": 0.60, "y": 0.65, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        lms[LEFT_ANKLE] = {"x": 0.74, "y": 0.82, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_ANKLE] = {"x": 0.76, "y": 0.82, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        return lms

    @classmethod
    def generate_cobra_pose(cls, correct: bool = True) -> List[Dict[str, float]]:
        """Generates landmarks for Cobra Pose (Bhujangasana)."""
        lms = cls.get_blank_landmarks()
        # Head & chest lifted up
        lms[NOSE] = {"x": 0.24, "y": 0.36, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[LEFT_SHOULDER] = {"x": 0.30, "y": 0.45, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_SHOULDER] = {"x": 0.32, "y": 0.45, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        # Arms supporting lift
        lms[LEFT_ELBOW] = {"x": 0.30, "y": 0.60, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_ELBOW] = {"x": 0.32, "y": 0.60, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[LEFT_WRIST] = {"x": 0.30, "y": 0.75, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_WRIST] = {"x": 0.32, "y": 0.75, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        # Hips on ground
        lms[LEFT_HIP] = {"x": 0.50, "y": 0.75, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_HIP] = {"x": 0.52, "y": 0.75, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        # Legs extended back horizontally
        lms[LEFT_KNEE] = {"x": 0.66, "y": 0.77, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_KNEE] = {"x": 0.68, "y": 0.77, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[LEFT_ANKLE] = {"x": 0.82, "y": 0.78, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_ANKLE] = {"x": 0.84, "y": 0.78, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        return lms

    @classmethod
    def generate_chair_pose(cls, correct: bool = True) -> List[Dict[str, float]]:
        """Generates landmarks for Chair Pose (Utkatasana)."""
        lms = cls.get_blank_landmarks()
        # Arms raised diagonally up/forward
        lms[LEFT_WRIST] = {"x": 0.42, "y": 0.15, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_WRIST] = {"x": 0.58, "y": 0.15, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[LEFT_ELBOW] = {"x": 0.44, "y": 0.25, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_ELBOW] = {"x": 0.56, "y": 0.25, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        lms[NOSE] = {"x": 0.50, "y": 0.30, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[LEFT_SHOULDER] = {"x": 0.46, "y": 0.36, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_SHOULDER] = {"x": 0.54, "y": 0.36, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        # Hips shifted back
        lms[LEFT_HIP] = {"x": 0.36, "y": 0.54, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_HIP] = {"x": 0.40, "y": 0.54, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        if correct:
            # Knees bent ~100°-115°
            lms[LEFT_KNEE] = {"x": 0.50, "y": 0.68, "z": 0.0, "visibility": 1.0, "presence": 1.0}
            lms[RIGHT_KNEE] = {"x": 0.54, "y": 0.68, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        else:
            # Barely bent knees (> 145°)
            lms[LEFT_KNEE] = {"x": 0.38, "y": 0.72, "z": 0.0, "visibility": 1.0, "presence": 1.0}
            lms[RIGHT_KNEE] = {"x": 0.42, "y": 0.72, "z": 0.0, "visibility": 1.0, "presence": 1.0}

        lms[LEFT_ANKLE] = {"x": 0.42, "y": 0.88, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        lms[RIGHT_ANKLE] = {"x": 0.46, "y": 0.88, "z": 0.0, "visibility": 1.0, "presence": 1.0}
        return lms

    @classmethod
    def render_pose_frame(
        cls,
        landmarks: List[Dict[str, float]],
        width: int = 640,
        height: int = 480,
        bg_color: Tuple[int, int, int] = (30, 30, 36)
    ) -> np.ndarray:
        """Renders synthetic human pose silhouette on a canvas."""
        img = np.full((height, width, 3), bg_color, dtype=np.uint8)

        def to_px(idx: int):
            return (int(landmarks[idx]["x"] * width), int(landmarks[idx]["y"] * height))

        # Draw bones
        for p1_idx, p2_idx in POSE_CONNECTIONS:
            pt1 = to_px(p1_idx)
            pt2 = to_px(p2_idx)
            cv2.line(img, pt1, pt2, (200, 200, 200), 5, cv2.LINE_AA)

        # Draw joints
        for idx in range(33):
            pt = to_px(idx)
            cv2.circle(img, pt, 7, (255, 215, 0), -1, cv2.LINE_AA)

        # Head circle
        nose_pt = to_px(NOSE)
        cv2.circle(img, (nose_pt[0], nose_pt[1] - 8), 24, (220, 220, 220), -1, cv2.LINE_AA)
        return img
