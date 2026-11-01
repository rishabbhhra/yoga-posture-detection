"""
Rule-based Yoga Pose Classification Engine.
Classifies live landmarks into 1 of 7 target poses:
- Tadasana (Mountain Pose)
- Tree Pose (Vrikshasana)
- Warrior II (Virabhadrasana II)
- Triangle Pose (Trikonasana)
- Downward Dog (Adho Mukha Svanasana)
- Cobra Pose (Bhujangasana)
- Chair Pose (Utkatasana)
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from .config import (
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    NOSE
)
from .angle_calculator import AngleCalculator

class YogaPoseClassifier:
    """Classifies detected pose landmarks using geometric spatial relations and joint angles."""

    def __init__(self):
        self.calc = AngleCalculator()

    def classify(self, landmarks: List[Dict[str, float]]) -> Tuple[str, float, Dict[str, float]]:
        """
        Classifies the human pose from 33 landmarks.
        Returns:
            (best_pose_name, confidence_score, candidate_scores_dict)
        """
        if not landmarks or len(landmarks) < 33:
            return "No Person Detected", 0.0, {}

        # Visibility check
        key_landmarks = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE]
        avg_vis = np.mean([landmarks[i].get("visibility", 1.0) for i in key_landmarks])
        if avg_vis < 0.4:
            return "Landmarks Obscured", 0.0, {}

        angles = self.calc.compute_all_joint_angles(landmarks)
        scores: Dict[str, float] = {
            "Tadasana": 0.0,
            "Tree Pose": 0.0,
            "Warrior II": 0.0,
            "Triangle Pose": 0.0,
            "Downward Dog": 0.0,
            "Cobra Pose": 0.0,
            "Chair Pose": 0.0,
        }

        def p(idx: int):
            return (landmarks[idx]["x"], landmarks[idx]["y"])

        # Spatial helpers
        l_sh, r_sh = p(LEFT_SHOULDER), p(RIGHT_SHOULDER)
        l_hip, r_hip = p(LEFT_HIP), p(RIGHT_HIP)
        l_knee, r_knee = p(LEFT_KNEE), p(RIGHT_KNEE)
        l_ank, r_ank = p(LEFT_ANKLE), p(RIGHT_ANKLE)
        l_wri, r_wri = p(LEFT_WRIST), p(RIGHT_WRIST)

        mid_shoulder_y = (l_sh[1] + r_sh[1]) / 2.0
        mid_hip_y = (l_hip[1] + r_hip[1]) / 2.0
        mid_wrist_y = (l_wri[1] + r_wri[1]) / 2.0
        mid_ankle_y = (l_ank[1] + r_ank[1]) / 2.0

        ankle_dist = self.calc.calculate_distance(l_ank, r_ank)
        shoulder_dist = max(self.calc.calculate_distance(l_sh, r_sh), 0.05)
        stance_ratio = ankle_dist / shoulder_dist

        lk_angle = angles.get("left_knee", 180.0)
        rk_angle = angles.get("right_knee", 180.0)
        lh_angle = angles.get("left_hip", 180.0)
        rh_angle = angles.get("right_hip", 180.0)
        ls_angle = angles.get("left_shoulder", 0.0)
        rs_angle = angles.get("right_shoulder", 0.0)

        # ---------------------------------------------------------------------
        # 1. Downward Dog Check (Inverted V, hips highest, head towards floor)
        # ---------------------------------------------------------------------
        # In Downward Dog: Hips are significantly higher in the frame (smaller y) than shoulders
        # Wrists & ankles are both near the bottom / ground (large y)
        if (mid_hip_y < mid_shoulder_y - 0.04) and (mid_wrist_y > mid_shoulder_y) and (lk_angle > 140.0 and rk_angle > 140.0):
            dog_score = 0.5
            hip_hinge = angles.get("hip_hinge", 90.0)
            if 60.0 <= hip_hinge <= 110.0:
                dog_score += 0.35
            if lk_angle > 155.0 and rk_angle > 155.0:
                dog_score += 0.15
            scores["Downward Dog"] = min(1.0, dog_score)

        # ---------------------------------------------------------------------
        # 2. Cobra Pose Check (Prone backbend, hips grounded, chest lifted)
        # ---------------------------------------------------------------------
        # Prone: Hips are low in the frame (mid_hip_y > 0.60) and legs extended horizontally behind
        # Shoulders are lifted higher than hips (mid_shoulder_y < mid_hip_y - 0.12)
        if (mid_hip_y > 0.55) and (mid_shoulder_y < mid_hip_y - 0.10) and (mid_ankle_y > 0.55):
            cobra_score = 0.4
            spine_arch = angles.get("left_hip", 180.0)
            if 105.0 <= spine_arch <= 160.0:
                cobra_score += 0.35
            if (lk_angle > 150.0 and rk_angle > 150.0):
                cobra_score += 0.25
            scores["Cobra Pose"] = min(1.0, cobra_score)

        # ---------------------------------------------------------------------
        # 3. Tree Pose Check (Standing on one leg, other knee acutely bent)
        # ---------------------------------------------------------------------
        # Standing pose (head above hips, hips above knees)
        is_standing = mid_shoulder_y < mid_hip_y
        if is_standing:
            left_standing = (lk_angle > 160.0 and rk_angle < 100.0)
            right_standing = (rk_angle > 160.0 and lk_angle < 100.0)
            if left_standing or right_standing:
                tree_score = 0.6
                # Check lifted foot height: ankle of bent leg is higher than standing knee
                if left_standing and (r_ank[1] < l_knee[1] + 0.10):
                    tree_score += 0.25
                elif right_standing and (l_ank[1] < r_knee[1] + 0.10):
                    tree_score += 0.25
                # Hands prayer or raised
                if ls_angle > 40.0 and rs_angle > 40.0:
                    tree_score += 0.15
                scores["Tree Pose"] = min(1.0, tree_score)

        # ---------------------------------------------------------------------
        # 4. Warrior II Check (Wide stance, one knee ~90°, one leg straight, arms wide horizontal)
        # ---------------------------------------------------------------------
        one_knee_bent_90 = (70.0 <= lk_angle <= 120.0 and rk_angle > 150.0) or \
                           (70.0 <= rk_angle <= 120.0 and lk_angle > 150.0)
        arms_outstretched = (ls_angle >= 65.0 and rs_angle >= 65.0) and (stance_ratio > 1.2)
        if one_knee_bent_90 and is_standing:
            warrior_score = 0.5
            if arms_outstretched:
                warrior_score += 0.35
            if stance_ratio > 1.5:
                warrior_score += 0.15
            scores["Warrior II"] = min(1.0, warrior_score)

        # ---------------------------------------------------------------------
        # 5. Triangle Pose Check (Wide stance, both legs straight, lateral torso bend, arms vertical span)
        # ---------------------------------------------------------------------
        both_legs_straight = (lk_angle > 155.0 and rk_angle > 155.0)
        arms_span = angles.get("arms_span", 0.0)
        shoulder_tilt = self.calc.calculate_inclination_angle(l_sh, r_sh)
        if both_legs_straight and (stance_ratio > 1.3) and is_standing:
            # In triangle pose, the torso/shoulders are strongly tilted to one side
            if shoulder_tilt > 25.0 or abs(l_wri[1] - r_wri[1]) > 0.3:
                triangle_score = 0.55
                if arms_span > 140.0:
                    triangle_score += 0.30
                if shoulder_tilt > 35.0:
                    triangle_score += 0.15
                scores["Triangle Pose"] = min(1.0, triangle_score)

        # ---------------------------------------------------------------------
        # 6. Chair Pose Check (Both knees bent 75°-130°, hips flexed, arms reached overhead)
        # ---------------------------------------------------------------------
        both_knees_bent = (70.0 <= lk_angle <= 135.0) and (70.0 <= rk_angle <= 135.0)
        both_hips_flexed = (65.0 <= lh_angle <= 135.0) and (65.0 <= rh_angle <= 135.0)
        if both_knees_bent and both_hips_flexed and is_standing and (stance_ratio < 1.4):
            chair_score = 0.6
            # Arms reaching up
            if ls_angle > 110.0 and rs_angle > 110.0:
                chair_score += 0.30
            if abs(lk_angle - rk_angle) < 25.0:
                chair_score += 0.10
            scores["Chair Pose"] = min(1.0, chair_score)

        # ---------------------------------------------------------------------
        # 7. Tadasana Check (Both legs straight, torso erect, standing tall)
        # ---------------------------------------------------------------------
        torso_erect = (lh_angle > 155.0 and rh_angle > 155.0)
        feet_together = (stance_ratio < 1.3)
        if both_legs_straight and torso_erect and is_standing and feet_together:
            tadasana_score = 0.65
            if lk_angle > 165.0 and rk_angle > 165.0:
                tadasana_score += 0.20
            if lh_angle > 165.0 and rh_angle > 165.0:
                tadasana_score += 0.15
            scores["Tadasana"] = min(1.0, tadasana_score)

        # Find best candidate
        best_pose = max(scores, key=scores.get)
        best_score = scores[best_pose]

        if best_score < 0.40:
            return "Transition / Unknown Pose", best_score, scores

        return best_pose, float(best_score), scores
