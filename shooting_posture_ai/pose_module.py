"""MediaPipe Pose wrapper. Returns landmarks in PIXEL coordinates (so angles are not
distorted by the image aspect ratio)."""
import cv2

LANDMARK_IDS = {
    "nose": 0, "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16, "left_index": 19, "right_index": 20,
    "left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}


class PoseDetector:
    def __init__(self, static_image_mode=True, model_complexity=1,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        import mediapipe as mp  # lazy import so the rest of the app can be tested without it
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode, model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)

    def detect(self, bgr_image):
        """Return list of 33 (x_px, y_px, visibility) or None if no person found."""
        h, w = bgr_image.shape[:2]
        result = self.pose.process(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        if not result.pose_landmarks:
            return None
        return [(lm.x * w, lm.y * h, lm.visibility) for lm in result.pose_landmarks.landmark]

    @staticmethod
    def to_dict(points):
        return {name: points[idx] for name, idx in LANDMARK_IDS.items()}

    def draw(self, bgr_image, points):
        out = bgr_image.copy()
        for a, b in self.mp_pose.POSE_CONNECTIONS:
            if points[a][2] > 0.5 and points[b][2] > 0.5:
                cv2.line(out, (int(points[a][0]), int(points[a][1])),
                         (int(points[b][0]), int(points[b][1])), (255, 200, 0), 2)
        for x, y, v in points:
            if v > 0.5:
                cv2.circle(out, (int(x), int(y)), 4, (0, 0, 255), -1)
        return out

    def close(self):
        self.pose.close()
