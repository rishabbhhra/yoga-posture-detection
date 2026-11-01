"""
Configuration constants, landmark mappings, and pose angle rules
for Real-Time Yoga Posture Detection and Correction.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# MediaPipe Pose Landmark Indices (33 Landmarks)
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# Skeleton Connections for Visualization
POSE_CONNECTIONS = [
    # Head & Face
    (LEFT_EAR, LEFT_EYE_OUTER), (LEFT_EYE_OUTER, LEFT_EYE), (LEFT_EYE, LEFT_EYE_INNER),
    (LEFT_EYE_INNER, NOSE), (NOSE, RIGHT_EYE_INNER), (RIGHT_EYE_INNER, RIGHT_EYE),
    (RIGHT_EYE, RIGHT_EYE_OUTER), (RIGHT_EYE_OUTER, RIGHT_EAR),
    # Torso
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    # Left Arm
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    # Right Arm
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    # Left Leg
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (LEFT_ANKLE, LEFT_FOOT_INDEX),
    # Right Leg
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
]

# Color Schemes (BGR for OpenCV)
COLOR_CORRECT = (46, 204, 113)      # Bright Green
COLOR_WARNING = (0, 165, 255)       # Orange
COLOR_INCORRECT = (50, 50, 235)     # Bright Red
COLOR_NEUTRAL = (220, 220, 220)     # Light Gray
COLOR_SKELETON = (240, 240, 240)    # White
COLOR_LANDMARK = (255, 191, 0)      # Deep Sky Blue / Gold
COLOR_BG_CARD = (25, 25, 25)

@dataclass
class AngleRule:
    joint_name: str
    points: Tuple[int, int, int]  # (A, B, C) where B is the vertex joint
    min_angle: float
    max_angle: float
    feedback_too_low: str
    feedback_too_high: str
    importance: float = 1.0  # Weight in total score calculation

# Defined Yoga Poses and their specific geometric angle rules
YOGA_POSES = {
    "Tadasana": {
        "name": "Tadasana (Mountain Pose)",
        "description": "Standing tall with spine erect, shoulders relaxed, arms at sides or raised, legs straight.",
        "difficulty": "Beginner",
        "benefits": "Improves posture, strengthens thighs, knees, and ankles, firms abdomen and buttocks.",
        "rules": [
            AngleRule(
                joint_name="Left Knee",
                points=(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
                min_angle=165.0,
                max_angle=180.0,
                feedback_too_low="Straighten your left knee; do not bend your leg.",
                feedback_too_high="Keep your left knee straight without hyperextending.",
                importance=1.0,
            ),
            AngleRule(
                joint_name="Right Knee",
                points=(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
                min_angle=165.0,
                max_angle=180.0,
                feedback_too_low="Straighten your right knee; do not bend your leg.",
                feedback_too_high="Keep your right knee straight without hyperextending.",
                importance=1.0,
            ),
            AngleRule(
                joint_name="Left Hip (Torso Alignment)",
                points=(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
                min_angle=165.0,
                max_angle=180.0,
                feedback_too_low="Stand upright; avoid leaning forward or hunching hips.",
                feedback_too_high="Keep torso aligned over hips.",
                importance=1.2,
            ),
            AngleRule(
                joint_name="Right Hip (Torso Alignment)",
                points=(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
                min_angle=165.0,
                max_angle=180.0,
                feedback_too_low="Keep torso tall and straight over right hip.",
                feedback_too_high="Maintain vertical spine alignment.",
                importance=1.2,
            ),
            AngleRule(
                joint_name="Left Elbow",
                points=(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
                min_angle=155.0,
                max_angle=180.0,
                feedback_too_low="Keep left arm straight and relaxed.",
                feedback_too_high="Maintain straight left arm.",
                importance=0.8,
            ),
            AngleRule(
                joint_name="Right Elbow",
                points=(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
                min_angle=155.0,
                max_angle=180.0,
                feedback_too_low="Keep right arm straight and relaxed.",
                feedback_too_high="Maintain straight right arm.",
                importance=0.8,
            ),
        ],
    },
    "Tree Pose": {
        "name": "Tree Pose (Vrikshasana)",
        "description": "Balancing on one leg with the opposite foot placed against the inner thigh, hands joined in prayer or overhead.",
        "difficulty": "Beginner - Intermediate",
        "benefits": "Strengthens thighs, calves, ankles, and spine. Improves balance and focus.",
        "rules": [
            # Evaluated dynamically for either left-leg or right-leg standing variant
            AngleRule(
                joint_name="Standing Leg Knee",
                points=(0, 0, 0),  # Assigned dynamically based on active side
                min_angle=165.0,
                max_angle=180.0,
                feedback_too_low="Keep the standing leg firmly straight for stability.",
                feedback_too_high="Engage quadriceps without locking joint.",
                importance=1.3,
            ),
            AngleRule(
                joint_name="Bent Leg Knee",
                points=(0, 0, 0),
                min_angle=25.0,
                max_angle=85.0,
                feedback_too_low="Open your bent knee outwards to open the hip.",
                feedback_too_high="Bend the lifted leg higher against the inner thigh.",
                importance=1.3,
            ),
            AngleRule(
                joint_name="Torso Upright",
                points=(0, 0, 0),
                min_angle=160.0,
                max_angle=180.0,
                feedback_too_low="Engage your core to keep your spine vertical.",
                feedback_too_high="Maintain tall upright posture.",
                importance=1.1,
            ),
        ],
    },
    "Warrior II": {
        "name": "Warrior II (Virabhadrasana II)",
        "description": "Lunge stance with front knee at 90°, back leg straight, arms extended horizontally in opposite directions.",
        "difficulty": "Intermediate",
        "benefits": "Strengthens and stretches legs and ankles, expands chest and lungs, builds stamina.",
        "rules": [
            AngleRule(
                joint_name="Front Knee",
                points=(0, 0, 0),  # Identified dynamically (Left or Right)
                min_angle=82.0,
                max_angle=105.0,
                feedback_too_low="Ease up slightly; do not let front knee collapse past 90°.",
                feedback_too_high="Bend front knee deeper towards 90° directly over ankle.",
                importance=1.5,
            ),
            AngleRule(
                joint_name="Back Leg Knee",
                points=(0, 0, 0),
                min_angle=165.0,
                max_angle=180.0,
                feedback_too_low="Straighten your back leg firmly and press through outer heel.",
                feedback_too_high="Keep back leg fully extended.",
                importance=1.3,
            ),
            AngleRule(
                joint_name="Front Shoulder (Arm Horizontal)",
                points=(0, 0, 0),
                min_angle=80.0,
                max_angle=105.0,
                feedback_too_low="Raise your front arm level with shoulders.",
                feedback_too_high="Lower your front arm to shoulder height.",
                importance=1.0,
            ),
            AngleRule(
                joint_name="Back Shoulder (Arm Horizontal)",
                points=(0, 0, 0),
                min_angle=80.0,
                max_angle=105.0,
                feedback_too_low="Raise your back arm level with shoulders.",
                feedback_too_high="Lower your back arm to shoulder height.",
                importance=1.0,
            ),
            AngleRule(
                joint_name="Front Elbow",
                points=(0, 0, 0),
                min_angle=160.0,
                max_angle=180.0,
                feedback_too_low="Extend front arm straight through fingertips.",
                feedback_too_high="Keep front arm extended.",
                importance=0.8,
            ),
            AngleRule(
                joint_name="Back Elbow",
                points=(0, 0, 0),
                min_angle=160.0,
                max_angle=180.0,
                feedback_too_low="Extend back arm straight through fingertips.",
                feedback_too_high="Keep back arm extended.",
                importance=0.8,
            ),
        ],
    },
    "Triangle Pose": {
        "name": "Triangle Pose (Trikonasana)",
        "description": "Legs wide and straight, torso laterally flexed with lower hand toward ankle and upper hand toward sky.",
        "difficulty": "Intermediate",
        "benefits": "Stretches and strengthens thighs, knees, ankles, groin, and spine. Relieves backache.",
        "rules": [
            AngleRule(
                joint_name="Front Knee",
                points=(0, 0, 0),
                min_angle=160.0,
                max_angle=180.0,
                feedback_too_low="Straighten your front leg; avoid bending knee in Triangle.",
                feedback_too_high="Keep front knee straight with a micro-softness to protect joint.",
                importance=1.2,
            ),
            AngleRule(
                joint_name="Back Knee",
                points=(0, 0, 0),
                min_angle=165.0,
                max_angle=180.0,
                feedback_too_low="Keep back leg straight and grounded.",
                feedback_too_high="Maintain back leg extension.",
                importance=1.2,
            ),
            AngleRule(
                joint_name="Arms Span (Shoulder-to-Shoulder line)",
                points=(LEFT_WRIST, LEFT_SHOULDER, RIGHT_WRIST),
                min_angle=155.0,
                max_angle=180.0,
                feedback_too_low="Open your chest and align both arms in a single vertical plane.",
                feedback_too_high="Maintain expansive arm alignment.",
                importance=1.3,
            ),
            AngleRule(
                joint_name="Lateral Torso Hip Angle",
                points=(0, 0, 0),
                min_angle=75.0,
                max_angle=125.0,
                feedback_too_low="Hinge further from hip crease without rounding spine.",
                feedback_too_high="Hinge cleanly sideways from hip crease.",
                importance=1.1,
            ),
        ],
    },
    "Downward Dog": {
        "name": "Downward Dog (Adho Mukha Svanasana)",
        "description": "Inverted V-shape with hands and feet grounded, hips lifted toward ceiling, spine lengthened.",
        "difficulty": "Beginner - Intermediate",
        "benefits": "Energizes body, stretches shoulders, hamstrings, calves, arches, and hands.",
        "rules": [
            AngleRule(
                joint_name="Left Knee",
                points=(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
                min_angle=155.0,
                max_angle=180.0,
                feedback_too_low="Work on straightening your left knee and pressing heel toward mat.",
                feedback_too_high="Maintain straight left leg.",
                importance=1.2,
            ),
            AngleRule(
                joint_name="Right Knee",
                points=(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
                min_angle=155.0,
                max_angle=180.0,
                feedback_too_low="Work on straightening your right knee and pressing heel toward mat.",
                feedback_too_high="Maintain straight right leg.",
                importance=1.2,
            ),
            AngleRule(
                joint_name="Left Elbow",
                points=(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
                min_angle=160.0,
                max_angle=180.0,
                feedback_too_low="Straighten left arm and press firmly into fingers and palm.",
                feedback_too_high="Keep left arm active.",
                importance=1.1,
            ),
            AngleRule(
                joint_name="Right Elbow",
                points=(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
                min_angle=160.0,
                max_angle=180.0,
                feedback_too_low="Straighten right arm and press firmly into palm.",
                feedback_too_high="Keep right arm active.",
                importance=1.1,
            ),
            AngleRule(
                joint_name="Hip Angle (Inverted V apex)",
                points=(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
                min_angle=60.0,
                max_angle=95.0,
                feedback_too_low="Open up the angle slightly by sending hips back and up.",
                feedback_too_high="Lift hips higher and press chest toward thighs to form sharp V shape.",
                importance=1.5,
            ),
            AngleRule(
                joint_name="Shoulder Openness",
                points=(LEFT_WRIST, LEFT_SHOULDER, LEFT_HIP),
                min_angle=150.0,
                max_angle=180.0,
                feedback_too_low="Press chest back toward thighs to lengthen shoulders and spine.",
                feedback_too_high="Maintain long, relaxed spine line.",
                importance=1.3,
            ),
        ],
    },
    "Cobra Pose": {
        "name": "Cobra Pose (Bhujangasana)",
        "description": "Prone position with chest lifted off mat, hips on floor, arms gently supporting backbend, shoulders pulled down.",
        "difficulty": "Beginner",
        "benefits": "Strengthens spine, stretches chest and lungs, shoulders, and abdomen. Stimulates abdominal organs.",
        "rules": [
            AngleRule(
                joint_name="Spine Arch (Shoulder-Hip-Knee)",
                points=(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
                min_angle=110.0,
                max_angle=155.0,
                feedback_too_low="Lift your chest higher off the ground by engaging back muscles.",
                feedback_too_high="Arch your back and draw chest forward through shoulders.",
                importance=1.5,
            ),
            AngleRule(
                joint_name="Left Elbow",
                points=(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
                min_angle=115.0,
                max_angle=170.0,
                feedback_too_low="Do not bend elbows too much; gently support upper body lift.",
                feedback_too_high="Keep a slight microbend in elbows to prevent locking.",
                importance=1.1,
            ),
            AngleRule(
                joint_name="Right Elbow",
                points=(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
                min_angle=115.0,
                max_angle=170.0,
                feedback_too_low="Support upper body lift evenly with right arm.",
                feedback_too_high="Keep a slight microbend in right elbow.",
                importance=1.1,
            ),
            AngleRule(
                joint_name="Legs Grounded (Knees Straight)",
                points=(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
                min_angle=160.0,
                max_angle=180.0,
                feedback_too_low="Keep tops of feet and thighs pressed firmly into floor.",
                feedback_too_high="Maintain active legs.",
                importance=1.0,
            ),
        ],
    },
    "Chair Pose": {
        "name": "Chair Pose (Utkatasana)",
        "description": "Standing with knees bent deep as if sitting in an imaginary chair, hips back, arms reaching up diagonally.",
        "difficulty": "Beginner - Intermediate",
        "benefits": "Strengthens thighs, spine, ankles, and calves. Stretches shoulders and chest.",
        "rules": [
            AngleRule(
                joint_name="Left Knee",
                points=(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
                min_angle=75.0,
                max_angle=125.0,
                feedback_too_low="Do not squat too low past comfort or knee safety.",
                feedback_too_high="Sit hips back deeper into the chair, bend knees more.",
                importance=1.4,
            ),
            AngleRule(
                joint_name="Right Knee",
                points=(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
                min_angle=75.0,
                max_angle=125.0,
                feedback_too_low="Do not squat too low past knee safety.",
                feedback_too_high="Bend knees deeper and distribute weight into heels.",
                importance=1.4,
            ),
            AngleRule(
                joint_name="Left Hip Flexion",
                points=(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
                min_angle=70.0,
                max_angle=115.0,
                feedback_too_low="Keep your chest lifted; avoid collapsing chest on thighs.",
                feedback_too_high="Hinge at hips and shift buttocks back.",
                importance=1.3,
            ),
            AngleRule(
                joint_name="Arms Reach (Shoulder-Elbow)",
                points=(LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW),
                min_angle=135.0,
                max_angle=180.0,
                feedback_too_low="Reach arms higher alongside your ears.",
                feedback_too_high="Maintain extended arm reach.",
                importance=1.1,
            ),
            AngleRule(
                joint_name="Left Elbow",
                points=(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
                min_angle=155.0,
                max_angle=180.0,
                feedback_too_low="Straighten your left arm through fingertips.",
                feedback_too_high="Keep arms strong and straight.",
                importance=0.8,
            ),
            AngleRule(
                joint_name="Right Elbow",
                points=(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
                min_angle=155.0,
                max_angle=180.0,
                feedback_too_low="Straighten your right arm through fingertips.",
                feedback_too_high="Keep arms strong and straight.",
                importance=0.8,
            ),
        ],
    },
}
