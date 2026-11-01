"""
Real-Time Yoga Posture Detection & Correction System
Streamlit Web Application (Step 8 of IMV Lab Project Report)
"""
import os
import sys
import time
import tempfile
from typing import Any, Optional, Dict, List, Tuple
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pose_detector import PoseDetector
from src.angle_calculator import AngleCalculator
from src.classifier import YogaPoseClassifier
from src.corrector import PostureCorrector
from src.visualizer import Visualizer
from src.sample_generator import SamplePoseGenerator
from src.config import YOGA_POSES, COLOR_CORRECT, COLOR_WARNING, COLOR_INCORRECT

# Streamlit Page Config
st.set_page_config(
    page_title="Yoga Posture AI | Real-Time Feedback",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sleek Fitness Dashboard UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #10b981, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.1rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-val {
        font-size: 1.7rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .feedback-box-good {
        background: rgba(16, 185, 129, 0.15);
        border-left: 5px solid #10b981;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .feedback-box-warn {
        background: rgba(245, 158, 11, 0.15);
        border-left: 5px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .feedback-box-err {
        background: rgba(239, 68, 68, 0.15);
        border-left: 5px solid #ef4444;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .status-badge-good {
        color: #10b981;
        font-weight: 700;
        font-size: 1.2rem;
    }
    .status-badge-warn {
        color: #f59e0b;
        font-weight: 700;
        font-size: 1.2rem;
    }
    .status-badge-bad {
        color: #ef4444;
        font-weight: 700;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache detector model to avoid re-instantiation across reruns
@st.cache_resource
def get_detector(complexity: str):
    return PoseDetector(model_complexity=complexity)

@st.cache_resource
def get_classifier():
    return YogaPoseClassifier()

@st.cache_resource
def get_corrector():
    return PostureCorrector()

def main():
    st.markdown('<div class="main-title">🧘 Real-Time Yoga Posture Detection & Correction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Image & Video Processing (IMV) Project • MediaPipe Pose • Geometric Joint Angle Analysis • Real-Time AI Coaching</div>', unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.header("⚙️ Configuration & Modes")

    mode = st.sidebar.radio(
        "Choose Mode:",
        ["🧪 Interactive Pose Playground", "📷 Live Webcam Analysis", "📁 Upload Video / Image", "📖 Yoga Poses Reference Guide"],
        index=0
    )

    pose_selection = st.sidebar.selectbox(
        "Target Pose Guidance:",
        ["Auto-Detect Pose"] + list(YOGA_POSES.keys()),
        index=0
    )
    target_pose_name = None if pose_selection == "Auto-Detect Pose" else pose_selection

    model_complexity = st.sidebar.selectbox(
        "MediaPipe Model Complexity:",
        ["full", "lite"],
        index=0,
        help="'full' has higher keypoint accuracy; 'lite' is optimized for ultra-fast processing."
    )

    draw_angles = st.sidebar.checkbox("Show Joint Angle Badges", value=True)
    show_hud = st.sidebar.checkbox("Show In-Video HUD Cards", value=True)

    detector = get_detector(model_complexity)
    classifier = get_classifier()
    corrector = get_corrector()

    # =========================================================================
    # MODE 1: Interactive Pose Playground (Instant demo & testing of all 7 poses)
    # =========================================================================
    if mode == "🧪 Interactive Pose Playground":
        st.subheader("🧪 Synthetic Pose Playground & Angle Analyzer")
        st.info("Test and inspect all 7 yoga poses with both correct form and common beginner mistakes without needing a camera!")

        col_cfg1, col_cfg2 = st.columns([2, 1])
        with col_cfg1:
            demo_pose = st.selectbox("Select Pose to Test:", list(YOGA_POSES.keys()), index=2)
        with col_cfg2:
            pose_condition = st.radio("Posture Condition:", ["Correct Form ✓", "Common Mistakes / Deviations ⚠️"], horizontal=True)

        is_correct = (pose_condition == "Correct Form ✓")

        # Pose generators map
        gen_map = {
            "Tadasana": SamplePoseGenerator.generate_tadasana,
            "Tree Pose": SamplePoseGenerator.generate_tree_pose,
            "Warrior II": SamplePoseGenerator.generate_warrior_ii,
            "Triangle Pose": SamplePoseGenerator.generate_triangle_pose,
            "Downward Dog": SamplePoseGenerator.generate_downward_dog,
            "Cobra Pose": SamplePoseGenerator.generate_cobra_pose,
            "Chair Pose": SamplePoseGenerator.generate_chair_pose,
        }

        lms = gen_map[demo_pose](correct=is_correct)
        detected_pose, conf, scores = classifier.classify(lms)
        active_pose = target_pose_name if target_pose_name else detected_pose
        eval_res = corrector.evaluate_posture(active_pose, lms)

        # Render Frame
        canvas = SamplePoseGenerator.render_pose_frame(lms, width=720, height=540)
        canvas = Visualizer.draw_skeleton(canvas, lms, eval_res, draw_angles=draw_angles)
        if show_hud:
            canvas = Visualizer.draw_hud(canvas, f"{detected_pose} ({int(conf * 100)}%)", eval_res, fps=60.0)

        col_view, col_stats = st.columns([3, 2])

        with col_view:
            st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption=f"Visualization: {demo_pose} ({pose_condition})", use_container_width=True)

        with col_stats:
            render_dashboard_metrics(detected_pose, conf, eval_res)
            render_joint_angle_table(eval_res)

        st.markdown("---")
        render_feedback_cards(eval_res)

    # =========================================================================
    # MODE 2: Live Webcam Analysis
    # =========================================================================
    elif mode == "📷 Live Webcam Analysis":
        st.subheader("📷 Live Webcam Posture Analysis")
        st.write("Perform yoga poses in front of your webcam to receive real-time posture scoring and corrective guidance.")

        tab_cam1, tab_cam2 = st.tabs(["Snapshot / Periodic Capture", "Continuous OpenCV Stream"])

        with tab_cam1:
            st.caption("Capture photos directly using your browser camera for high-resolution instant posture check.")
            camera_img = st.camera_input("Take a snapshot of your yoga pose")

            if camera_img is not None:
                pil_image = Image.open(camera_img)
                frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                with st.spinner("Analyzing posture with MediaPipe..."):
                    lms = detector.detect(frame)

                if lms:
                    detected_pose, conf, scores = classifier.classify(lms)
                    active_pose = target_pose_name if target_pose_name else detected_pose
                    eval_res = corrector.evaluate_posture(active_pose, lms)

                    annotated = Visualizer.draw_skeleton(frame, lms, eval_res, draw_angles=draw_angles)
                    if show_hud:
                        annotated = Visualizer.draw_hud(annotated, f"{detected_pose}", eval_res)

                    col_img, col_metrics = st.columns([3, 2])
                    with col_img:
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Evaluated Pose Snapshot", use_container_width=True)
                    with col_metrics:
                        render_dashboard_metrics(detected_pose, conf, eval_res)
                        render_joint_angle_table(eval_res)

                    render_feedback_cards(eval_res)
                else:
                    st.warning("⚠️ No person detected in the camera frame. Make sure your whole body is well-lit and visible.")

        with tab_cam2:
            st.info("Run continuous local stream using OpenCV. Press Start to initiate webcam feed.")
            run_live = st.checkbox("Start Live Video Stream", value=False)
            frame_slot = st.empty()
            metric_slot = st.empty()

            if run_live:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("❌ Unable to access local webcam (device 0). Please verify camera permissions.")
                else:
                    prev_t = time.time()
                    while run_live:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Failed to grab camera frame.")
                            break

                        curr_t = time.time()
                        fps = 1.0 / max(curr_t - prev_t, 1e-4)
                        prev_t = curr_t

                        lms = detector.detect(frame)
                        eval_res = None
                        detected_pose = "Scanning..."
                        conf = 0.0

                        if lms:
                            detected_pose, conf, _ = classifier.classify(lms)
                            active_pose = target_pose_name if target_pose_name else detected_pose
                            eval_res = corrector.evaluate_posture(active_pose, lms)
                            frame = Visualizer.draw_skeleton(frame, lms, eval_res, draw_angles=draw_angles)

                        if show_hud:
                            frame = Visualizer.draw_hud(frame, detected_pose, eval_res, fps=fps)

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_slot.image(frame_rgb, use_container_width=True)

                        if eval_res:
                            with metric_slot.container():
                                col_a, col_b, col_c = st.columns(3)
                                col_a.metric("Pose", active_pose)
                                col_b.metric("Score", f"{eval_res.overall_score}%")
                                col_c.metric("Status", eval_res.overall_status)

                    cap.release()

    # =========================================================================
    # MODE 3: Upload Video / Image
    # =========================================================================
    elif mode == "📁 Upload Video / Image":
        st.subheader("📁 Upload Media for Posture Analysis")
        uploaded_file = st.file_uploader("Upload an image (JPG, PNG) or video (MP4, MOV, AVI):", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

        if uploaded_file is not None:
            file_ext = uploaded_file.name.split(".")[-1].lower()

            if file_ext in ["jpg", "jpeg", "png"]:
                # Process single image
                pil_image = Image.open(uploaded_file)
                frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                with st.spinner("Processing image..."):
                    lms = detector.detect(frame)

                if lms:
                    detected_pose, conf, scores = classifier.classify(lms)
                    active_pose = target_pose_name if target_pose_name else detected_pose
                    eval_res = corrector.evaluate_posture(active_pose, lms)

                    annotated = Visualizer.draw_skeleton(frame, lms, eval_res, draw_angles=draw_angles)
                    if show_hud:
                        annotated = Visualizer.draw_hud(annotated, detected_pose, eval_res)

                    col1, col2 = st.columns([3, 2])
                    with col1:
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Analyzed {uploaded_file.name}", use_container_width=True)
                    with col2:
                        render_dashboard_metrics(detected_pose, conf, eval_res)
                        render_joint_angle_table(eval_res)

                    render_feedback_cards(eval_res)
                else:
                    st.warning("No human pose detected in uploaded image.")

            else:
                # Video file
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
                tfile.write(uploaded_file.read())
                video_path = tfile.name

                st.video(uploaded_file)
                if st.button("🚀 Analyze Video Frames"):
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    st_frame_slot = st.empty()
                    prog_bar = st.progress(0.0)

                    frame_idx = 0
                    scores_history = []

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_idx += 1
                        # Process 1 of every 3 frames for smooth playback & efficiency
                        if frame_idx % 3 != 0:
                            continue

                        lms = detector.detect(frame)
                        eval_res = None
                        detected_pose = "Analyzing..."
                        if lms:
                            detected_pose, conf, _ = classifier.classify(lms)
                            active_pose = target_pose_name if target_pose_name else detected_pose
                            eval_res = corrector.evaluate_posture(active_pose, lms)
                            scores_history.append(eval_res.overall_score)
                            frame = Visualizer.draw_skeleton(frame, lms, eval_res, draw_angles=draw_angles)

                        if show_hud:
                            frame = Visualizer.draw_hud(frame, detected_pose, eval_res)

                        st_frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        prog_bar.progress(min(1.0, frame_idx / max(total_frames, 1)))

                    cap.release()
                    if scores_history:
                        st.success(f"Video analysis complete! Average Posture Score: {np.mean(scores_history):.1f}%")

    # =========================================================================
    # MODE 4: Yoga Poses Reference Guide
    # =========================================================================
    elif mode == "📖 Yoga Poses Reference Guide":
        st.subheader("📖 7 Target Yoga Poses - Biomechanical Reference")
        st.write("Detailed guide on the 7 evaluated yoga poses, anatomical joint angle criteria, and physiological benefits.")

        for p_name, p_data in YOGA_POSES.items():
            with st.expander(f"🧘 {p_data['name']} ({p_data['difficulty']})", expanded=False):
                st.markdown(f"**Description:** {p_data['description']}")
                st.markdown(f"**Benefits:** {p_data['benefits']}")
                st.markdown("**Biomechanical Angle Rules:**")
                rules_table = []
                for r in p_data["rules"]:
                    rules_table.append({
                        "Joint": r.joint_name,
                        "Target Range": f"{r.min_angle:.0f}° - {r.max_angle:.0f}°",
                        "Low Feedback": r.feedback_too_low,
                        "High Feedback": r.feedback_too_high,
                    })
                st.dataframe(rules_table, use_container_width=True)

def render_dashboard_metrics(detected_pose: str, conf: float, eval_res: Any):
    """Renders key metrics cards."""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94a3b8; font-size: 0.9rem;">DETECTED POSE</div>
            <div class="metric-val" style="color: #38bdf8;">{detected_pose}</div>
            <div style="color: #64748b; font-size: 0.8rem;">Confidence: {conf*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        score_val = eval_res.overall_score if eval_res else 0.0
        score_color = "#10b981" if score_val >= 80 else ("#f59e0b" if score_val >= 50 else "#ef4444")
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94a3b8; font-size: 0.9rem;">POSTURE SCORE</div>
            <div class="metric-val" style="color: {score_color};">{score_val}%</div>
            <div style="color: {score_color}; font-size: 0.85rem; font-weight: 600;">{eval_res.overall_status if eval_res else 'N/A'}</div>
        </div>
        """, unsafe_allow_html=True)

def render_joint_angle_table(eval_res: Any):
    """Renders tabular comparison of measured vs target joint angles."""
    if not eval_res or not eval_res.joint_evaluations:
        return

    st.markdown("#### 📐 Joint Angle Telemetry")
    table_rows = []
    for j in eval_res.joint_evaluations:
        status_icon = "🟢" if j.is_correct else ("🟡" if j.status == "NEEDS_IMPROVEMENT" else "🔴")
        table_rows.append({
            "Joint": j.joint_name,
            "Measured": f"{j.measured_angle:.1f}°",
            "Target Range": f"{j.target_min:.0f}° - {j.target_max:.0f}°",
            "Status": f"{status_icon} {j.status}",
        })
    st.dataframe(table_rows, use_container_width=True)

def render_feedback_cards(eval_res: Any):
    """Renders formatted corrective action items."""
    st.markdown("#### 💡 Posture Correction & Biomechanical Guidance")
    if not eval_res or not eval_res.feedback_summary:
        st.info("No feedback available.")
        return

    for fb in eval_res.feedback_summary:
        if "✓" in fb or "Outstanding" in fb or "Good" in fb:
            st.markdown(f'<div class="feedback-box-good">✅ <strong>Alignment:</strong> {fb}</div>', unsafe_allow_html=True)
        elif "⚠️" in fb or "slightly" in fb or "NEEDS" in fb:
            st.markdown(f'<div class="feedback-box-warn">⚠️ <strong>Correction:</strong> {fb}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="feedback-box-err">❌ <strong>Correction:</strong> {fb}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
