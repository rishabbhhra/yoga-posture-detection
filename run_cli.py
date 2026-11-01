"""
Command-Line Interface (CLI) runner for Real-Time Yoga Posture Detection & Correction.
Supports:
  1. Live Webcam feed (using OpenCV)
  2. Video file evaluation
  3. Image file evaluation
  4. Interactive synthetic pose demo
"""
import argparse
import time
from typing import Any, Optional
import cv2
import numpy as np

from src.pose_detector import PoseDetector
from src.angle_calculator import AngleCalculator
from src.classifier import YogaPoseClassifier
from src.corrector import PostureCorrector
from src.visualizer import Visualizer
from src.sample_generator import SamplePoseGenerator

def process_stream(source: Any = 0, target_pose: Optional[str] = None):
    print("=" * 60)
    print("🧘 Real-Time Yoga Posture Detection & Correction System")
    print(f"📹 Source: {source} | Target Mode: {target_pose if target_pose else 'Auto-Detect'}")
    print("⌨️  Press 'q' or ESC in OpenCV window to quit.")
    print("=" * 60)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source {source}")
        return

    detector = PoseDetector(model_complexity="full")
    classifier = YogaPoseClassifier()
    corrector = PostureCorrector()
    prev_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            # 1. Detect Landmarks
            landmarks = detector.detect(frame)

            eval_res = None
            detected_pose = "Searching for person..."

            if landmarks:
                # 2. Classify Pose
                best_pose, conf, _ = classifier.classify(landmarks)
                active_pose = target_pose if target_pose else best_pose
                detected_pose = f"{active_pose} ({int(conf * 100)}%)" if not target_pose else active_pose

                # 3. Posture Evaluation & Correction
                if active_pose in corrector.calc.compute_all_joint_angles(landmarks) or True:
                    eval_res = corrector.evaluate_posture(active_pose, landmarks)

                # 4. Render Skeleton & Annotations
                frame = Visualizer.draw_skeleton(frame, landmarks, eval_res, draw_angles=True)

            # 5. Render HUD Overlay
            frame = Visualizer.draw_hud(frame, detected_pose, eval_res, fps=fps)

            # Show window
            cv2.imshow("Yoga Posture Detection & Correction", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("\nSession completed successfully.")

def run_demo(pose_name: str, correct: bool = True):
    print(f"\n🧘 Running Synthetic Demo for: {pose_name} (Correct: {correct})")
    generators = {
        "Tadasana": SamplePoseGenerator.generate_tadasana,
        "Tree Pose": SamplePoseGenerator.generate_tree_pose,
        "Warrior II": SamplePoseGenerator.generate_warrior_ii,
        "Triangle Pose": SamplePoseGenerator.generate_triangle_pose,
        "Downward Dog": SamplePoseGenerator.generate_downward_dog,
        "Cobra Pose": SamplePoseGenerator.generate_cobra_pose,
        "Chair Pose": SamplePoseGenerator.generate_chair_pose,
    }

    if pose_name not in generators:
        print(f"Unknown pose '{pose_name}'. Choose from: {list(generators.keys())}")
        return

    lms = generators[pose_name](correct=correct)
    classifier = YogaPoseClassifier()
    corrector = PostureCorrector()

    detected_pose, conf, scores = classifier.classify(lms)
    eval_res = corrector.evaluate_posture(pose_name, lms)

    # Render frame
    frame = SamplePoseGenerator.render_pose_frame(lms, width=800, height=600)
    frame = Visualizer.draw_skeleton(frame, lms, eval_res, draw_angles=True)
    frame = Visualizer.draw_hud(frame, f"{detected_pose} (Demo)", eval_res, fps=60.0)

    print("\n" + "=" * 50)
    print(f"Detected Pose: {detected_pose}")
    print(f"Confidence:    {conf * 100:.1f}%")
    print(f"Status:        {eval_res.overall_status}")
    print(f"Score:         {eval_res.overall_score}%")
    print("-" * 50)
    print("Joint Angles:")
    for j_name, angle in eval_res.measured_angles.items():
        print(f"  • {j_name:30s}: {angle:.1f}°")
    print("-" * 50)
    print("Feedback:")
    for fb in eval_res.feedback_summary:
        print(f"  • {fb}")
    print("=" * 50)

    out_filename = f"demo_{pose_name.lower().replace(' ', '_')}.jpg"
    cv2.imwrite(out_filename, frame)
    print(f"📸 Saved annotated demonstration frame to: {out_filename}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Yoga Posture Detection & Correction")
    parser.add_argument("--source", type=str, default="demo", choices=["webcam", "video", "image", "demo"],
                        help="Input source: webcam, video, image, or synthetic demo")
    parser.add_argument("--file", type=str, default=None, help="Path to video or image file")
    parser.add_argument("--pose", type=str, default="Warrior II",
                        choices=["Tadasana", "Tree Pose", "Warrior II", "Triangle Pose", "Downward Dog", "Cobra Pose", "Chair Pose"],
                        help="Target yoga pose for guidance/demo")
    parser.add_argument("--incorrect", action="store_true", help="Generate incorrect pose variant in demo mode")
    args = parser.parse_args()

    if args.source == "demo":
        run_demo(args.pose, correct=not args.incorrect)
    elif args.source == "webcam":
        process_stream(0, target_pose=args.pose)
    elif args.source in ["video", "image"]:
        if not args.file:
            print("Please provide --file <path> when using video or image source.")
        else:
            process_stream(args.file, target_pose=args.pose)
