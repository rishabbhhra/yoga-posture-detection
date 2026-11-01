"""
Visualizer module for rendering skeleton landmarks, joint angles, color-coded feedback,
and HUD overlays onto video frames.
"""
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
from .config import (
    POSE_CONNECTIONS,
    COLOR_CORRECT,
    COLOR_WARNING,
    COLOR_INCORRECT,
    COLOR_NEUTRAL,
    COLOR_SKELETON,
    COLOR_LANDMARK,
    COLOR_BG_CARD,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE
)
from .corrector import PostureEvaluationResult

class Visualizer:
    """Renders pose skeletons, real-time angle annotations, and HUD feedback onto frames."""

    @staticmethod
    def draw_skeleton(
        frame: np.ndarray,
        landmarks: List[Dict[str, float]],
        eval_result: Optional[PostureEvaluationResult] = None,
        draw_angles: bool = True
    ) -> np.ndarray:
        """
        Draws skeleton connections and joints color-coded by posture correctness.
        """
        out = frame.copy()
        h, w, _ = out.shape

        if not landmarks or len(landmarks) < 33:
            return out

        incorrect_set = set(eval_result.incorrect_landmarks) if eval_result else set()
        correct_set = set(eval_result.correct_landmarks) if eval_result else set()

        def to_px(idx: int) -> Tuple[int, int]:
            lm = landmarks[idx]
            return (int(lm["x"] * w), int(lm["y"] * h))

        # 1. Draw connections
        for p1_idx, p2_idx in POSE_CONNECTIONS:
            pt1 = to_px(p1_idx)
            pt2 = to_px(p2_idx)

            # Determine connection color
            if p1_idx in incorrect_set or p2_idx in incorrect_set:
                color = COLOR_INCORRECT
                thickness = 4
            elif p1_idx in correct_set and p2_idx in correct_set:
                color = COLOR_CORRECT
                thickness = 3
            else:
                color = COLOR_SKELETON
                thickness = 2

            cv2.line(out, pt1, pt2, color, thickness, cv2.LINE_AA)

        # 2. Draw landmarks
        for idx, lm in enumerate(landmarks):
            px, py = to_px(idx)
            # Skip if out of frame bounds
            if px < 0 or px >= w or py < 0 or py >= h:
                continue

            if idx in incorrect_set:
                # Warning ring
                cv2.circle(out, (px, py), 9, COLOR_INCORRECT, -1, cv2.LINE_AA)
                cv2.circle(out, (px, py), 13, (0, 0, 255), 2, cv2.LINE_AA)
            elif idx in correct_set:
                cv2.circle(out, (px, py), 7, COLOR_CORRECT, -1, cv2.LINE_AA)
                cv2.circle(out, (px, py), 10, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(out, (px, py), 4, COLOR_LANDMARK, -1, cv2.LINE_AA)

        # 3. Draw key angle callouts next to joints
        if draw_angles and eval_result and eval_result.measured_angles:
            joint_map = {
                "Left Knee": LEFT_KNEE,
                "Right Knee": RIGHT_KNEE,
                "Front Knee (Left)": LEFT_KNEE,
                "Front Knee (Right)": RIGHT_KNEE,
                "Left Elbow": LEFT_ELBOW,
                "Right Elbow": RIGHT_ELBOW,
                "Left Hip": LEFT_HIP,
                "Right Hip": RIGHT_HIP,
            }

            for name, angle_val in eval_result.measured_angles.items():
                for k, idx in joint_map.items():
                    if k.lower() in name.lower():
                        pt = to_px(idx)
                        text = f"{int(round(angle_val))}°"
                        # Background tag
                        text_sz, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        badge_x = pt[0] + 12
                        badge_y = pt[1] - 8
                        cv2.rectangle(
                            out,
                            (badge_x - 3, badge_y - text_sz[1] - 3),
                            (badge_x + text_sz[0] + 3, badge_y + 3),
                            (20, 20, 20),
                            -1
                        )
                        tag_color = COLOR_INCORRECT if idx in incorrect_set else COLOR_CORRECT
                        cv2.putText(out, text, (badge_x, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tag_color, 1, cv2.LINE_AA)
                        break

        return out

    @staticmethod
    def draw_hud(
        frame: np.ndarray,
        pose_name: str,
        eval_result: Optional[PostureEvaluationResult] = None,
        fps: Optional[float] = None
    ) -> np.ndarray:
        """
        Renders HUD cards displaying Pose, Status, Score, and dynamic feedback messages.
        """
        out = frame.copy()
        h, w, _ = out.shape

        # Top Overlay Card
        header_h = 75
        overlay = out.copy()
        cv2.rectangle(overlay, (15, 15), (w - 15, 15 + header_h), COLOR_BG_CARD, -1)
        alpha = 0.75
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

        # Border for top card
        border_color = COLOR_CORRECT if (eval_result and eval_result.is_posture_correct) else (
            COLOR_WARNING if (eval_result and eval_result.overall_score > 45) else (100, 100, 100)
        )
        cv2.rectangle(out, (15, 15), (w - 15, 15 + header_h), border_color, 2)

        # Pose Name
        cv2.putText(
            out,
            f"Pose: {pose_name}",
            (30, 48),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Posture Status & Score
        if eval_result:
            status_text = f"Status: {eval_result.overall_status}"
            score_text = f"Score: {eval_result.overall_score}%"

            status_color = COLOR_CORRECT if "EXCELLENT" in eval_result.overall_status or "GOOD" in eval_result.overall_status else (
                COLOR_WARNING if "NEEDS" in eval_result.overall_status else COLOR_INCORRECT
            )
            cv2.putText(out, status_text, (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)
            cv2.putText(out, score_text, (w - 180, 48), cv2.FONT_HERSHEY_DUPLEX, 0.75, status_color, 2, cv2.LINE_AA)

        # FPS indicator
        if fps is not None:
            cv2.putText(out, f"{fps:.1f} FPS", (w - 130, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        # Bottom Feedback Card (if feedback items exist)
        if eval_result and eval_result.feedback_summary:
            max_feedback = min(2, len(eval_result.feedback_summary))
            card_h = 35 + max_feedback * 26
            b_top = h - card_h - 15
            b_bot = h - 15

            fb_overlay = out.copy()
            cv2.rectangle(fb_overlay, (15, b_top), (w - 15, b_bot), COLOR_BG_CARD, -1)
            cv2.addWeighted(fb_overlay, 0.8, out, 0.2, 0, out)
            cv2.rectangle(out, (15, b_top), (w - 15, b_bot), (80, 80, 80), 1)

            cv2.putText(out, "CORRECTIVE GUIDANCE:", (30, b_top + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 215, 0), 1, cv2.LINE_AA)

            for i in range(max_feedback):
                fb_msg = eval_result.feedback_summary[i]
                y_pos = b_top + 46 + i * 24
                # Icon prefix
                prefix = "• "
                cv2.putText(out, prefix + fb_msg, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

        return out
