"""Glue between OpenCV, MediaPipe and the analyzer (images, video, annotation)."""
import cv2
import numpy as np
from collections import Counter
from pose_module import PoseDetector
from posture_analyzer import analyze_landmarks, classify


def decode_image(file_bytes):
    return cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)


def annotate(img, detector, points, result):
    out = detector.draw(img, points)
    for key, val in result["metrics"].items():
        x, y = result["anchors"][key]
        cv2.putText(out, f"{val:.0f}", (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(out, f"Score {result['score']} ({result['label']})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return out


def analyze_image(img_bgr, side, profile, detector=None):
    """Returns (result_dict, annotated_bgr) or (None, None) if no pose found."""
    own = detector is None
    detector = detector or PoseDetector(static_image_mode=True)
    try:
        pts = detector.detect(img_bgr)
        if pts is None:
            return None, None
        res = analyze_landmarks(PoseDetector.to_dict(pts), side, profile)
        if res is None:
            return None, None
        return res, annotate(img_bgr, detector, pts, res)
    finally:
        if own:
            detector.close()


def analyze_video(path, side, profile, frame_step=5, progress_cb=None):
    """Analyse every `frame_step`-th frame. Returns aggregated result + best/worst frames."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    detector = PoseDetector(static_image_mode=False)
    frames, idx = [], 0
    best = worst = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_step == 0:
            pts = detector.detect(frame)
            if pts is not None:
                res = analyze_landmarks(PoseDetector.to_dict(pts), side, profile)
                if res:
                    frames.append(res)
                    ann = annotate(frame, detector, pts, res)
                    if best is None or res["score"] > best[0]: best = (res["score"], ann)
                    if worst is None or res["score"] < worst[0]: worst = (res["score"], ann)
        idx += 1
        if progress_cb: progress_cb(min(idx / total, 1.0))
    cap.release(); detector.close()
    if not frames:
        return None
    keys = {k for f in frames for k in f["metrics"]}
    avg_metrics = {k: float(np.mean([f["metrics"][k] for f in frames if k in f["metrics"]])) for k in keys}
    err_counts = Counter(e for f in frames for e in f["errors"])
    n = len(frames)
    # An error is reported if it appears in >= 30 % of analysed frames
    errors = [e for e, c in err_counts.items() if c / n >= 0.3]
    fb = Counter(m.split(": ", 1)[1] for f in frames for m in f["feedback"] if ": " in m)
    feedback = [m for m, c in fb.items() if c / n >= 0.3] or ["Excellent posture - maintain this position."]
    score = round(float(np.mean([f["score"] for f in frames])), 1)
    return {"metrics": avg_metrics, "errors": errors, "feedback": feedback, "score": score,
            "label": classify(score), "frames_analyzed": n,
            "score_series": [f["score"] for f in frames],
            "best_frame": best[1], "worst_frame": worst[1]}
