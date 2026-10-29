"""Dataset utilities: save labelled images, split raw -> train/val/test, extract landmark features."""
import os, glob, random, shutil, csv
import cv2
import config
from pose_module import PoseDetector
from posture_analyzer import compute_metrics
from database import METRIC_COLS

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def save_raw_image(file_bytes, filename, label):
    folder = os.path.join(config.DATASET_DIR, "raw", label)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path


def count_images():
    out = {}
    for split in ("raw", "train", "val", "test"):
        for label in ("correct", "incorrect"):
            files = [f for f in glob.glob(os.path.join(config.DATASET_DIR, split, label, "*"))
                     if f.lower().endswith(IMG_EXT)]
            out[(split, label)] = len(files)
    return out


def split_dataset(train=0.70, val=0.15, seed=42):
    """Copy raw images into train/val/test (70/15/15). Existing split folders are cleared first."""
    rnd = random.Random(seed)
    summary = {}
    for label in ("correct", "incorrect"):
        for s in ("train", "val", "test"):
            d = os.path.join(config.DATASET_DIR, s, label)
            os.makedirs(d, exist_ok=True)
            for f in glob.glob(os.path.join(d, "*")):
                if os.path.basename(f) != ".gitkeep":
                    os.remove(f)
        files = sorted(f for f in glob.glob(os.path.join(config.DATASET_DIR, "raw", label, "*"))
                       if f.lower().endswith(IMG_EXT))
        rnd.shuffle(files)
        n = len(files); n_tr = int(n * train); n_va = int(n * val)
        parts = {"train": files[:n_tr], "val": files[n_tr:n_tr + n_va], "test": files[n_tr + n_va:]}
        for s, lst in parts.items():
            for f in lst:
                shutil.copy(f, os.path.join(config.DATASET_DIR, s, label, os.path.basename(f)))
            summary[(s, label)] = len(lst)
    return summary


def extract_landmark_features(side="right"):
    """Run MediaPipe on every image in train/val/test and write dataset/landmarks/features.csv"""
    out_path = os.path.join(config.DATASET_DIR, "landmarks", "features.csv")
    det = PoseDetector(static_image_mode=True)
    rows = 0
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "label", "file"] + METRIC_COLS)
        for s in ("train", "val", "test"):
            for label in ("correct", "incorrect"):
                for f in sorted(glob.glob(os.path.join(config.DATASET_DIR, s, label, "*"))):
                    if not f.lower().endswith(IMG_EXT):
                        continue
                    img = cv2.imread(f)
                    pts = det.detect(img) if img is not None else None
                    if pts is None:
                        continue
                    m, _ = compute_metrics(PoseDetector.to_dict(pts), side)
                    w.writerow([s, label, os.path.basename(f)] + [round(m[k], 2) if k in m else "" for k in METRIC_COLS])
                    rows += 1
    det.close()
    return out_path, rows
