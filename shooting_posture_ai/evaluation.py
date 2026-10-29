"""Accuracy testing on dataset/test (real numbers computed from YOUR images)."""
import os, glob
import numpy as np
import cv2
import config
from pose_module import PoseDetector
from posture_analyzer import analyze_landmarks


def evaluate_split(split="test", side="right", profile="Pistol (extended arm)",
                   threshold=config.CLASSIFY_THRESHOLD):
    det = PoseDetector(static_image_mode=True)
    y_true, y_pred, scores, skipped = [], [], [], 0
    for label_id, label in enumerate(("incorrect", "correct")):   # 0 = incorrect, 1 = correct
        for f in sorted(glob.glob(os.path.join(config.DATASET_DIR, split, label, "*"))):
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            img = cv2.imread(f)
            pts = det.detect(img) if img is not None else None
            res = analyze_landmarks(PoseDetector.to_dict(pts), side, profile) if pts else None
            if res is None:
                skipped += 1
                continue
            y_true.append(label_id); scores.append(res["score"])
            y_pred.append(1 if res["score"] >= threshold else 0)
    det.close()
    return confusion_and_metrics(y_true, y_pred) | {"scores": scores, "skipped": skipped}


def confusion_and_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum()); tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum()); fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"matrix": np.array([[tn, fp], [fn, tp]]), "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "n": n}
