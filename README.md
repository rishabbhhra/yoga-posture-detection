# Real-Time Yoga Posture Detection and Correction Using Computer Vision

A computer-vision system that captures video frames via webcam, extracts 33 human skeletal landmarks using **MediaPipe Pose**, computes 2D/3D joint angles using geometric vector mathematics, classifies yoga poses, evaluates posture correctness against biomechanical standards, and delivers real-time visual and text-based corrective guidance.

---

## 📋 Features

- **Live & Video Input Processing:** Captures live webcam feed frame-by-frame using OpenCV or processes uploaded videos / static images.
- **Image Preprocessing:** Resizing, color-space conversions (BGR ↔ RGB), Gaussian blur denoising, and normalization.
- **MediaPipe Pose Estimation:** 33 anatomical landmarks (shoulders, elbows, wrists, hips, knees, ankles, feet) with both `full` and `lite` pre-downloaded offline models.
- **Geometric Vector Joint Angle Calculation:**
  $$\theta = \arccos\left(\frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}\right) \times \frac{180}{\pi}$$
  Calculates knee, hip, shoulder, elbow, spine inclination, and inverted-V hinge angles.
- **Classification of 7 Yoga Poses:**
  1. **Tadasana (Mountain Pose)**
  2. **Tree Pose (Vrikshasana)**
  3. **Warrior II (Virabhadrasana II)**
  4. **Triangle Pose (Trikonasana)**
  5. **Downward Dog (Adho Mukha Svanasana)**
  6. **Cobra Pose (Bhujangasana)**
  7. **Chair Pose (Utkatasana)**
- **Intelligent Posture Correction & Biomechanical Feedback:**
  - Compares joint angles with expected physiological ranges.
  - Generates specific joint corrective tips (e.g., *"Bend left knee deeper towards 90° over ankle"*).
  - Dynamically detects lead leg vs. trailing leg for asymmetrical lunges and balancing poses.
  - Overall Posture Accuracy Score (0–100%) and Status (`EXCELLENT ✓`, `GOOD ✓`, `NEEDS IMPROVEMENT ⚠️`, `INCORRECT ❌`).
- **Real-Time Visual Overlays:**
  - Color-coded skeleton connections: **Green** for correct limbs, **Red/Orange** for misaligned joints.
  - Real-time joint angle badges directly over joints on the video feed.
  - Heads-Up Display (HUD) showing pose name, score, and prioritized corrective tips.
- **Modern Streamlit Web App:**
  - Snapshot camera mode & continuous OpenCV streaming.
  - Video and image file uploader with frame-by-frame analysis.
  - Synthetic Pose Playground for testing without a webcam.
  - Biomechanical reference library.

---

## 🏗️ System Architecture

```
Webcam / Video File
        ↓
Frame Extraction (OpenCV)
        ↓
Image Preprocessing (Resize, BGR → RGB, Denoising)
        ↓
Human Pose Estimation (MediaPipe PoseLandmarker)
        ↓
33 Landmark Coordinate Extraction
        ↓
Feature & Joint Angle Calculation (Vectors / Geometry)
        ↓
Yoga Pose Classification Engine
        ↓
Posture Correctness Evaluation & Deviation Analysis
        ↓
Real-Time Feedback UI (Streamlit & OpenCV Overlay HUD)
```

---

## 📂 Directory Structure

```
yoga_posture_detection/
├── models/
│   ├── pose_landmarker_lite.task     # MediaPipe high-speed model
│   └── pose_landmarker_full.task     # MediaPipe high-accuracy model
├── src/
│   ├── __init__.py
│   ├── config.py                     # Landmark indices, connections, angle rules for 7 poses
│   ├── pose_detector.py              # MediaPipe wrapper & preprocessing pipeline
│   ├── angle_calculator.py           # 2D/3D angle, inclination & distance formulas
│   ├── classifier.py                 # 7-pose rule-based classification engine
│   ├── corrector.py                  # Deviation scoring & actionable feedback generator
│   ├── visualizer.py                 # Skeleton rendering, color coding & HUD overlays
│   └── sample_generator.py           # Synthetic pose generator for testing & demo
├── tests/
│   ├── __init__.py
│   └── test_angles_and_poses.py      # Unit test suite verifying all 7 poses
├── app.py                            # Streamlit Web Application
├── run_cli.py                        # Command-line & OpenCV runner
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Requirements

Ensure you are in the `yoga_posture_detection` directory:

```bash
cd yoga_posture_detection
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run via Command-Line / OpenCV

#### A. Synthetic Pose Demonstration (No camera needed)
```bash
# Demo Warrior II
python run_cli.py --source demo --pose "Warrior II"

# Demo Chair Pose with intentional mistakes
python run_cli.py --source demo --pose "Chair Pose" --incorrect
```

#### B. Live Webcam Processing
```bash
# Auto-detect pose from webcam
python run_cli.py --source webcam

# Target coaching mode for a specific pose
python run_cli.py --source webcam --pose "Tree Pose"
```

#### C. Video or Image File Processing
```bash
python run_cli.py --source video --file path/to/yoga_video.mp4
python run_cli.py --source image --file path/to/yoga_photo.jpg
```

---

## 🧪 Running Unit Tests

Run the complete test suite verifying vector angle computations, pose classification, and corrective scoring:

```bash
python -m unittest tests/test_angles_and_poses.py
```

---

## 📊 Summary of Evaluated Yoga Poses & Angle Criteria

| Pose | Key Angles / Alignments | Target Range | Corrective Focus |
| :--- | :--- | :--- | :--- |
| **Tadasana (Mountain)** | Left & Right Knees, Hips, Spine | 165° – 180° | Straight legs, tall vertical posture |
| **Tree Pose** | Standing Knee, Bent Knee, Torso | Standing: 165°–180°, Bent: 25°–85° | One-leg stability, hip opening |
| **Warrior II** | Front Knee, Back Knee, Arms Level | Front: 82°–105°, Back: 165°–180°, Arms: 80°–105° | Front knee at 90°, arms horizontal |
| **Triangle Pose** | Both Knees, Arm Span, Hip Crease | Knees: 160°–180°, Arm Span: 155°–180° | Straight legs, vertical arm extension |
| **Downward Dog** | Hip Apex, Knees, Arms, Shoulders | Hip Apex: 60°–95°, Knees: 155°–180° | Inverted V shape, hips lifted high |
| **Cobra Pose** | Spine Arch, Elbows, Legs Grounded | Spine: 110°–155°, Elbows: 115°–170° | Chest lifted, shoulders back, hips grounded |
| **Chair Pose** | Knees Bent, Hips Flexed, Arms Raised | Knees: 75°–125°, Hips: 70°–115°, Arms: 135°–180° | Hips back, knees bent, arms reaching up |

---

## 📜 Technology Stack

- **Language:** Python 3.12+
- **Computer Vision:** OpenCV (`cv2`)
- **Pose Estimation:** Google MediaPipe Pose Tasks (`PoseLandmarker`)
- **Numerical Processing:** NumPy
- **Interactive UI:** Streamlit
- **Testing:** Python `unittest`
