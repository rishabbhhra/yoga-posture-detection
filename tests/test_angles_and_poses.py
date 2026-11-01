"""
Unit tests for AngleCalculator, YogaPoseClassifier, and PostureCorrector.
"""
import unittest
import numpy as np
from src.angle_calculator import AngleCalculator
from src.classifier import YogaPoseClassifier
from src.corrector import PostureCorrector
from src.sample_generator import SamplePoseGenerator

class TestYogaPostureSystem(unittest.TestCase):

    def setUp(self):
        self.calc = AngleCalculator()
        self.classifier = YogaPoseClassifier()
        self.corrector = PostureCorrector()

    def test_angle_2d_right_angle(self):
        a = (0.0, 1.0)
        b = (0.0, 0.0)
        c = (1.0, 0.0)
        angle = self.calc.calculate_angle_2d(a, b, c)
        self.assertAlmostEqual(angle, 90.0, places=3)

    def test_angle_2d_straight_line(self):
        a = (-1.0, 0.0)
        b = (0.0, 0.0)
        c = (1.0, 0.0)
        angle = self.calc.calculate_angle_2d(a, b, c)
        self.assertAlmostEqual(angle, 180.0, places=3)

    def test_classify_and_correct_tadasana(self):
        lms = SamplePoseGenerator.generate_tadasana(correct=True)
        pose, score, _ = self.classifier.classify(lms)
        self.assertEqual(pose, "Tadasana")
        self.assertGreaterEqual(score, 0.7)

        eval_res = self.corrector.evaluate_posture("Tadasana", lms)
        self.assertTrue(eval_res.is_posture_correct)
        self.assertGreaterEqual(eval_res.overall_score, 85.0)

    def test_classify_and_correct_warrior_ii(self):
        lms = SamplePoseGenerator.generate_warrior_ii(correct=True)
        pose, score, _ = self.classifier.classify(lms)
        self.assertEqual(pose, "Warrior II")
        self.assertGreaterEqual(score, 0.7)

        eval_res = self.corrector.evaluate_posture("Warrior II", lms)
        self.assertTrue(eval_res.is_posture_correct)
        self.assertGreaterEqual(eval_res.overall_score, 80.0)

    def test_warrior_ii_incorrect_front_knee(self):
        lms = SamplePoseGenerator.generate_warrior_ii(correct=False)
        eval_res = self.corrector.evaluate_posture("Warrior II", lms)
        # Should flag under-bent front knee and drooping arms
        self.assertFalse(eval_res.is_posture_correct)
        self.assertTrue(any("Front Knee" in f or "Arm" in f for f in eval_res.feedback_summary))

    def test_classify_and_correct_tree_pose(self):
        lms = SamplePoseGenerator.generate_tree_pose(correct=True)
        pose, score, _ = self.classifier.classify(lms)
        self.assertEqual(pose, "Tree Pose")
        self.assertGreaterEqual(score, 0.6)

        eval_res = self.corrector.evaluate_posture("Tree Pose", lms)
        self.assertTrue(eval_res.is_posture_correct)

    def test_classify_and_correct_downward_dog(self):
        lms = SamplePoseGenerator.generate_downward_dog(correct=True)
        pose, score, _ = self.classifier.classify(lms)
        self.assertEqual(pose, "Downward Dog")
        self.assertGreaterEqual(score, 0.7)

        eval_res = self.corrector.evaluate_posture("Downward Dog", lms)
        self.assertTrue(eval_res.is_posture_correct)

    def test_classify_and_correct_cobra_pose(self):
        lms = SamplePoseGenerator.generate_cobra_pose(correct=True)
        pose, score, _ = self.classifier.classify(lms)
        self.assertEqual(pose, "Cobra Pose")
        self.assertGreaterEqual(score, 0.7)

        eval_res = self.corrector.evaluate_posture("Cobra Pose", lms)
        self.assertTrue(eval_res.is_posture_correct)

    def test_classify_and_correct_chair_pose(self):
        lms = SamplePoseGenerator.generate_chair_pose(correct=True)
        pose, score, _ = self.classifier.classify(lms)
        self.assertEqual(pose, "Chair Pose")
        self.assertGreaterEqual(score, 0.7)

        eval_res = self.corrector.evaluate_posture("Chair Pose", lms)
        self.assertTrue(eval_res.is_posture_correct)

    def test_classify_triangle_pose(self):
        lms = SamplePoseGenerator.generate_triangle_pose(correct=True)
        pose, score, _ = self.classifier.classify(lms)
        self.assertEqual(pose, "Triangle Pose")
        self.assertGreaterEqual(score, 0.6)

if __name__ == "__main__":
    unittest.main()
