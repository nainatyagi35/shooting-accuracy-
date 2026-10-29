"""Central configuration: paths + posture thresholds (calibrate these with a coach!)."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "shooting_posture.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

MIN_VISIBILITY = 0.5          # ignore landmarks MediaPipe is not confident about
GOOD_SCORE = 80               # score >= 80  -> "Good"
AVERAGE_SCORE = 60            # 60-79 -> "Average", below -> "Poor"
CLASSIFY_THRESHOLD = 70       # used by evaluation page: >=70 => predicted "correct"

# metric: (ideal_min, ideal_max, tolerance, weight, advice_if_below, advice_if_above)
#   tolerance = how far outside the ideal range (deg / ratio) before that metric scores 0
#   weight    = importance of that metric in the final score
PROFILES = {
    "Pistol (extended arm)": {
        "elbow":         (155, 180, 40, 1.5, "Elbow is bent - extend the shooting arm fully.", ""),
        "shoulder":      (70, 100, 25, 1.0, "Arm is raised too low - lift it to the sight line.", "Arm is raised too high - relax and lower the shoulder."),
        "neck":          (150, 180, 25, 1.0, "Head is tilted forward - keep the neck upright over the shoulders.", ""),
        "hip":           (160, 180, 25, 1.0, "Bending at the waist - stand tall with a neutral hip.", ""),
        "knee":          (165, 180, 25, 0.7, "Knees are too bent - keep a stable, nearly straight stance.", ""),
        "wrist":         (150, 180, 30, 0.6, "Wrist is bent - keep the wrist straight with a firm, aligned grip.", ""),
        "lean":          (0, 8, 12, 1.2, "", "Body is leaning - keep the torso vertical."),
        "shoulder_tilt": (0, 5, 10, 0.8, "", "Shoulders are uneven - level both shoulders."),
        "hip_tilt":      (0, 5, 10, 0.6, "", "Hips are uneven - keep weight evenly distributed."),
        "balance":       (0, 0.30, 0.35, 0.8, "", "Body imbalance - align hips over the feet."),
    },
    "Rifle (standing)": {
        "elbow":         (70, 110, 35, 1.2, "Elbow is too closed - open slightly to support the rifle.", "Elbow is too open - tuck it in to support the rifle."),
        "shoulder":      (60, 100, 25, 1.0, "Elbow is too low - raise it to form a stable shelf.", "Elbow is too high - lower it for a relaxed shoulder."),
        "neck":          (150, 180, 25, 1.0, "Head is tilted forward - keep the neck aligned with the spine.", ""),
        "hip":           (160, 180, 25, 1.0, "Bending at the waist - stand tall with a neutral hip.", ""),
        "knee":          (165, 180, 25, 0.7, "Knees are too bent - keep a stable, nearly straight stance.", ""),
        "wrist":         (150, 180, 30, 0.6, "Wrist is bent - keep the wrist straight and the grip firm.", ""),
        "lean":          (0, 8, 12, 1.2, "", "Body is leaning - keep the torso vertical."),
        "shoulder_tilt": (0, 5, 10, 0.8, "", "Shoulders are uneven - level both shoulders."),
        "hip_tilt":      (0, 5, 10, 0.6, "", "Hips are uneven - keep weight evenly distributed."),
        "balance":       (0, 0.30, 0.35, 0.8, "", "Body imbalance - align hips over the feet."),
    },
}

METRIC_LABELS = {
    "elbow": "Elbow angle", "shoulder": "Shoulder angle", "neck": "Neck angle",
    "hip": "Hip angle", "knee": "Knee angle", "wrist": "Grip/wrist angle",
    "lean": "Body lean (deg)", "shoulder_tilt": "Shoulder tilt (deg)",
    "hip_tilt": "Hip tilt (deg)", "balance": "Balance offset (ratio)",
}
