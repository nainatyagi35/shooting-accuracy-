"""Run:  python -m unittest tests.test_core -v   (no camera / MediaPipe needed)"""
import os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
config.DB_PATH = os.path.join(tempfile.mkdtemp(), "test.db")
import database as db
from angle_utils import calculate_angle, vertical_deviation, horizontal_tilt
from posture_analyzer import analyze_landmarks, classify
from evaluation import confusion_and_metrics


def fake_landmarks(elbow_bend=0, lean_dx=0):
    """Right-handed shooter, front-ish view, everything ideal unless modified (pixel coords)."""
    V = 0.99
    lm = {n: (0, 0, 0.0) for n in ["nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
          "right_elbow", "left_wrist", "right_wrist", "left_index", "right_index", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle"]}
    lm.update({"right_ear": (300, 100, V), "left_ear": (250, 100, V),
               "right_shoulder": (320, 200 , V), "left_shoulder": (220, 200, V),
               "right_hip": (300 + lean_dx, 400, V), "left_hip": (240 + lean_dx, 400, V),
               "right_knee": (300 + lean_dx, 520, V), "left_knee": (240 + lean_dx, 520, V),
               "right_ankle": (300 + lean_dx, 640, V), "left_ankle": (240 + lean_dx, 640, V),
               "right_elbow": (440, 200 + elbow_bend, V), "right_wrist": (560, 200, V), "right_index": (620, 200, V)})
    return lm


class TestAngles(unittest.TestCase):
    def test_right_angle(self): self.assertAlmostEqual(calculate_angle((0, 1), (0, 0), (1, 0)), 90, places=3)
    def test_straight(self): self.assertAlmostEqual(calculate_angle((-1, 0), (0, 0), (1, 0)), 180, places=3)
    def test_zero_length(self): self.assertIsNone(calculate_angle((0, 0), (0, 0), (1, 1)))
    def test_vertical(self): self.assertAlmostEqual(vertical_deviation((0, 0), (0, 10)), 0, places=3)
    def test_vertical_45(self): self.assertAlmostEqual(vertical_deviation((10, 0), (0, 10)), 45, places=3)
    def test_tilt(self): self.assertAlmostEqual(horizontal_tilt((0, 0), (10, 0)), 0, places=3)


class TestAnalyzer(unittest.TestCase):
    def test_good_posture(self):
        r = analyze_landmarks(fake_landmarks(), "right")
        self.assertGreaterEqual(r["score"], 80); self.assertEqual(r["label"], "Good")
    def test_bent_elbow_detected(self):
        r = analyze_landmarks(fake_landmarks(elbow_bend=120), "right")
        self.assertIn("Elbow angle", r["errors"])
    def test_leaning_detected(self):
        r = analyze_landmarks(fake_landmarks(lean_dx=150), "right")
        self.assertTrue(any("lean" in e.lower() or "balance" in e.lower() for e in r["errors"]) or r["score"] < 80)
    def test_worse_posture_lower_score(self):
        self.assertLess(analyze_landmarks(fake_landmarks(elbow_bend=150, lean_dx=150))["score"],
                        analyze_landmarks(fake_landmarks())["score"])
    def test_classify(self): self.assertEqual([classify(x) for x in (90, 70, 40)], ["Good", "Average", "Poor"])
    def test_no_landmarks(self):
        self.assertIsNone(analyze_landmarks({k: (0, 0, 0.0) for k in fake_landmarks()}, "right"))


class TestDatabase(unittest.TestCase):
    def test_user_flow(self):
        db.init_db()
        self.assertTrue(db.create_user("alice", "secret1", "Alice"))
        self.assertFalse(db.create_user("alice", "x12345"))          # duplicate
        self.assertIsNotNone(db.verify_user("alice", "secret1"))
        self.assertIsNone(db.verify_user("alice", "wrong"))
    def test_reports(self):
        db.init_db(); u = db.verify_user("admin", "admin123")
        db.save_report(u["id"], "image", "a.jpg", "Pistol (extended arm)", analyze_landmarks(fake_landmarks()))
        self.assertGreaterEqual(len(db.get_reports(u["id"])), 1)
    def test_demo_seed(self):
        db.init_db(); u = db.verify_user("admin", "admin123"); before = len(db.get_reports(u["id"]))
        db.seed_demo_reports(u["id"], 5); self.assertEqual(len(db.get_reports(u["id"])), before + 5)


class TestMetrics(unittest.TestCase):
    def test_confusion(self):
        r = confusion_and_metrics([1, 1, 0, 0], [1, 0, 0, 1])
        self.assertEqual(r["matrix"].tolist(), [[1, 1], [1, 1]]); self.assertAlmostEqual(r["accuracy"], 0.5)

if __name__ == "__main__":
    unittest.main()
