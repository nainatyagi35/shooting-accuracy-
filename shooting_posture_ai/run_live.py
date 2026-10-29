"""Real-time webcam analysis in an OpenCV window.  Run:  python run_live.py
Keys: q = quit, s = save screenshot."""
import time, cv2, os
import config
from pose_module import PoseDetector
from posture_analyzer import analyze_landmarks
from processing import annotate

SIDE, PROFILE = "right", "Pistol (extended arm)"
cap = cv2.VideoCapture(0)
det = PoseDetector(static_image_mode=False)
prev = time.time()
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    pts = det.detect(frame)
    if pts:
        res = analyze_landmarks(PoseDetector.to_dict(pts), SIDE, PROFILE)
        if res:
            frame = annotate(frame, det, pts, res)
            for i, msg in enumerate(res["feedback"][:3]):
                cv2.putText(frame, msg[:70], (10, 60 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    fps = 1 / max(time.time() - prev, 1e-6); prev = time.time()
    cv2.putText(frame, f"FPS {fps:.0f}", (frame.shape[1] - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Shooting Posture - live", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"): break
    if k == ord("s"):
        os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(config.SCREENSHOT_DIR, f"shot_{int(time.time())}.png"), frame)
cap.release(); det.close(); cv2.destroyAllWindows()
