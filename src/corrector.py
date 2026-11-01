"""
Posture Correctness Evaluation and Corrective Feedback Engine.
Compares measured joint angles against biomechanical yoga standards,
flags erroneous joints, computes posture accuracy scores, and generates actionable cues.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from .config import (
    YOGA_POSES, AngleRule,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX
)
from .angle_calculator import AngleCalculator

@dataclass
class JointFeedback:
    joint_name: str
    measured_angle: float
    target_min: float
    target_max: float
    is_correct: bool
    status: str  # "CORRECT", "SLIGHT_DEVIATION", "INCORRECT"
    feedback_message: str
    affected_landmarks: Tuple[int, int, int]
    score: float  # 0.0 to 1.0

@dataclass
class PostureEvaluationResult:
    pose_name: str
    overall_status: str  # "EXCELLENT ✓", "GOOD ✓", "NEEDS IMPROVEMENT ⚠️", "INCORRECT ❌"
    overall_score: float  # 0 to 100
    is_posture_correct: bool
    joint_evaluations: List[JointFeedback]
    feedback_summary: List[str]
    incorrect_landmarks: List[int]
    correct_landmarks: List[int]
    measured_angles: Dict[str, float]

class PostureCorrector:
    """Evaluates posture correctness and formulates specific biomechanical feedback."""

    def __init__(self):
        self.calc = AngleCalculator()

    def evaluate_posture(
        self,
        pose_name: str,
        landmarks: List[Dict[str, float]]
    ) -> PostureEvaluationResult:
        """
        Evaluates a detected pose against its posture criteria.
        """
        if pose_name not in YOGA_POSES or not landmarks or len(landmarks) < 33:
            return PostureEvaluationResult(
                pose_name=pose_name,
                overall_status="UNKNOWN",
                overall_score=0.0,
                is_posture_correct=False,
                joint_evaluations=[],
                feedback_summary=["Ensure full body is visible within the camera frame."],
                incorrect_landmarks=[],
                correct_landmarks=[],
                measured_angles={}
            )

        pose_info = YOGA_POSES[pose_name]
        raw_rules: List[AngleRule] = pose_info["rules"]

        # Adapt dynamic rules for bilateral poses (Warrior II, Tree Pose, Triangle Pose)
        rules = self._resolve_dynamic_rules(pose_name, raw_rules, landmarks)

        joint_evaluations: List[JointFeedback] = []
        feedback_summary: List[str] = []
        incorrect_landmarks: set = set()
        correct_landmarks: set = set()
        total_weighted_score = 0.0
        total_weight = 0.0
        measured_angles: Dict[str, float] = {}

        def pt(idx: int) -> Tuple[float, float]:
            return (landmarks[idx]["x"], landmarks[idx]["y"])

        for rule in rules:
            pA, pB, pC = rule.points
            if pA == 0 and pB == 0 and pC == 0:
                continue

            angle = self.calc.calculate_angle_2d(pt(pA), pt(pB), pt(pC))
            measured_angles[rule.joint_name] = round(angle, 1)

            # Tolerance margins: 10 degrees buffer
            buffer = 8.0
            is_correct = (rule.min_angle - 2.0) <= angle <= (rule.max_angle + 2.0)

            if is_correct:
                status = "CORRECT"
                score = 1.0
                feedback_msg = f"{rule.joint_name}: {angle:.1f}° - Good alignment ✓"
                correct_landmarks.add(pB)
            else:
                score = max(0.0, 1.0 - (min(abs(angle - rule.min_angle), abs(angle - rule.max_angle)) / 45.0))
                incorrect_landmarks.add(pB)
                if angle < rule.min_angle:
                    status = "NEEDS_IMPROVEMENT" if angle >= (rule.min_angle - buffer) else "INCORRECT"
                    feedback_msg = f"{rule.joint_name} ({angle:.1f}°): {rule.feedback_too_low}"
                else:
                    status = "NEEDS_IMPROVEMENT" if angle <= (rule.max_angle + buffer) else "INCORRECT"
                    feedback_msg = f"{rule.joint_name} ({angle:.1f}°): {rule.feedback_too_high}"

                feedback_summary.append(feedback_msg)

            total_weighted_score += score * rule.importance
            total_weight += rule.importance

            joint_evaluations.append(JointFeedback(
                joint_name=rule.joint_name,
                measured_angle=round(angle, 1),
                target_min=rule.min_angle,
                target_max=rule.max_angle,
                is_correct=is_correct,
                status=status,
                feedback_message=feedback_msg,
                affected_landmarks=(pA, pB, pC),
                score=score
            ))

        overall_score = round((total_weighted_score / max(total_weight, 1.0)) * 100.0, 1)

        if overall_score >= 88.0:
            overall_status = "EXCELLENT ✓"
            is_posture_correct = True
            if not feedback_summary:
                feedback_summary.append("Outstanding posture! Maintain this position and breathe steadily.")
        elif overall_score >= 70.0:
            overall_status = "GOOD ✓"
            is_posture_correct = True
            if not feedback_summary:
                feedback_summary.append("Good form! Make minor adjustments to perfect the alignment.")
        elif overall_score >= 48.0:
            overall_status = "NEEDS IMPROVEMENT ⚠️"
            is_posture_correct = False
        else:
            overall_status = "INCORRECT ❌"
            is_posture_correct = False

        return PostureEvaluationResult(
            pose_name=pose_info["name"],
            overall_status=overall_status,
            overall_score=overall_score,
            is_posture_correct=is_posture_correct,
            joint_evaluations=joint_evaluations,
            feedback_summary=feedback_summary,
            incorrect_landmarks=list(incorrect_landmarks),
            correct_landmarks=list(correct_landmarks - incorrect_landmarks),
            measured_angles=measured_angles
        )

    def _resolve_dynamic_rules(
        self,
        pose_name: str,
        rules: List[AngleRule],
        landmarks: List[Dict[str, float]]
    ) -> List[AngleRule]:
        """Resolves dynamic side assignments (left vs right) for asymmetrical poses."""
        def pt(idx: int) -> Tuple[float, float]:
            return (landmarks[idx]["x"], landmarks[idx]["y"])

        lk_angle = self.calc.calculate_angle_2d(pt(LEFT_HIP), pt(LEFT_KNEE), pt(LEFT_ANKLE))
        rk_angle = self.calc.calculate_angle_2d(pt(RIGHT_HIP), pt(RIGHT_KNEE), pt(RIGHT_ANKLE))

        resolved = []

        if pose_name == "Warrior II":
            # Determine which leg is front (bent ~90°) vs back (straight)
            left_is_front = abs(lk_angle - 90.0) < abs(rk_angle - 90.0)
            if left_is_front:
                resolved = [
                    AngleRule("Front Knee (Left)", (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 82.0, 105.0,
                              "Ease up front knee slightly past 90°", "Bend left knee deeper towards 90° over ankle", 1.5),
                    AngleRule("Back Leg (Right)", (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 165.0, 180.0,
                              "Straighten right leg firmly", "Keep right leg extended", 1.3),
                    AngleRule("Front Arm (Left)", (LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW), 80.0, 105.0,
                              "Raise left arm to shoulder height", "Lower left arm level with shoulder", 1.0),
                    AngleRule("Back Arm (Right)", (RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW), 80.0, 105.0,
                              "Raise right arm to shoulder height", "Lower right arm level with shoulder", 1.0),
                    AngleRule("Left Elbow", (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), 160.0, 180.0,
                              "Extend left arm straight", "Keep left arm straight", 0.8),
                    AngleRule("Right Elbow", (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST), 160.0, 180.0,
                              "Extend right arm straight", "Keep right arm straight", 0.8),
                ]
            else:
                resolved = [
                    AngleRule("Front Knee (Right)", (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 82.0, 105.0,
                              "Ease up front knee slightly past 90°", "Bend right knee deeper towards 90° over ankle", 1.5),
                    AngleRule("Back Leg (Left)", (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 165.0, 180.0,
                              "Straighten left leg firmly", "Keep left leg extended", 1.3),
                    AngleRule("Front Arm (Right)", (RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW), 80.0, 105.0,
                              "Raise right arm to shoulder height", "Lower right arm level with shoulder", 1.0),
                    AngleRule("Back Arm (Left)", (LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW), 80.0, 105.0,
                              "Raise left arm to shoulder height", "Lower left arm level with shoulder", 1.0),
                    AngleRule("Right Elbow", (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST), 160.0, 180.0,
                              "Extend right arm straight", "Keep right arm straight", 0.8),
                    AngleRule("Left Elbow", (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), 160.0, 180.0,
                              "Extend left arm straight", "Keep left arm straight", 0.8),
                ]

        elif pose_name == "Tree Pose":
            # Standing leg is straighter (higher knee angle), bent leg has lower knee angle
            standing_is_left = lk_angle > rk_angle
            if standing_is_left:
                resolved = [
                    AngleRule("Standing Leg (Left)", (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 165.0, 180.0,
                              "Keep standing left leg firmly straight", "Keep standing leg strong", 1.3),
                    AngleRule("Bent Leg (Right)", (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 25.0, 85.0,
                              "Open right knee outwards to open the hip", "Place right foot against inner thigh", 1.3),
                    AngleRule("Torso Vertical", (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE), 160.0, 180.0,
                              "Engage core to stand upright", "Maintain tall spine", 1.1),
                ]
            else:
                resolved = [
                    AngleRule("Standing Leg (Right)", (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 165.0, 180.0,
                              "Keep standing right leg firmly straight", "Keep standing leg strong", 1.3),
                    AngleRule("Bent Leg (Left)", (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 25.0, 85.0,
                              "Open left knee outwards to open the hip", "Place left foot against inner thigh", 1.3),
                    AngleRule("Torso Vertical", (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE), 160.0, 180.0,
                              "Engage core to stand upright", "Maintain tall spine", 1.1),
                ]

        elif pose_name == "Triangle Pose":
            resolved = [
                AngleRule("Left Knee", (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 160.0, 180.0,
                          "Straighten left leg; avoid bending", "Keep left knee straight", 1.2),
                AngleRule("Right Knee", (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 160.0, 180.0,
                          "Straighten right leg; avoid bending", "Keep right knee straight", 1.2),
                AngleRule("Arms Span", (LEFT_WRIST, LEFT_SHOULDER, RIGHT_WRIST), 150.0, 180.0,
                          "Align both arms vertically through shoulders", "Keep arms wide open", 1.3),
                AngleRule("Torso Lateral Hinge", (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE), 75.0, 125.0,
                          "Hinge cleanly from hip crease", "Lengthen spine through crown", 1.1),
            ]
        else:
            resolved = rules

        return resolved
