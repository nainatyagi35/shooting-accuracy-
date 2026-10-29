# 🎯 AI-Based Real-Time Shooting Posture Analysis (OpenCV + MediaPipe + Streamlit + SQLite)

## Quick start
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt   # Python 3.9 - 3.12
streamlit run app.py
```
Login: **admin / admin123** (or register a new account). On the Dashboard click **Load demo data** to fill charts (rows are tagged `source = demo`, i.e. synthetic).
Real-time OpenCV window: `python run_live.py` (q = quit, s = screenshot).
Tests: `python -m unittest tests.test_core -v`

## Folder structure
```
shooting_posture_ai/
├── app.py               Streamlit UI: login, dashboard, analysis pages, dataset manager, evaluation
├── config.py            paths + ideal angle ranges, tolerances, weights (calibrate!)
├── angle_utils.py       angle / tilt / vertical-deviation maths
├── pose_module.py       MediaPipe Pose wrapper (33 landmarks, pixel coords, skeleton drawing)
├── posture_analyzer.py  metrics -> per-metric score -> errors -> feedback -> final score -> class
├── processing.py        image / video pipelines + annotation
├── database.py          SQLite schema + queries (users, reports, dataset_samples)
├── dataset_tools.py     save labelled images, 70/15/15 split, landmark CSV extraction
├── evaluation.py        accuracy / precision / recall / F1 / confusion matrix on dataset/test
├── report_generator.py  downloadable text report
├── run_live.py          real-time webcam window with FPS
├── tests/test_core.py   unit tests (16)
├── data/                shooting_posture.db is created here automatically
├── dataset/             raw/ train/ val/ test/ (each with correct/ incorrect/) + landmarks/
├── uploads/  screenshots/
└── requirements.txt
```

## Where data is stored
| What | Where |
|---|---|
| Users (hashed passwords), reports, dataset records | `data/shooting_posture.db` (SQLite) |
| Labelled posture images | `dataset/raw/{correct,incorrect}/` → split into `train/ val/ test/` |
| Extracted joint-angle features | `dataset/landmarks/features.csv` |
| Uploaded videos / camera screenshots | `uploads/`, `screenshots/` |

## Database design
```
users(id PK, username UNIQUE, full_name, password_hash, salt, created_at)
reports(id PK, user_id FK->users.id, source, filename, profile, score, label,
        elbow, shoulder, neck, hip, knee, wrist, lean, shoulder_tilt, hip_tilt, balance,
        errors, feedback, created_at)
dataset_samples(id PK, user_id FK->users.id, file_path, label, split, notes, created_at)
```
One user → many reports; one user → many dataset samples (1:N).
```sql
SELECT label, COUNT(*) FROM reports WHERE user_id = 1 GROUP BY label;
SELECT date(created_at) d, AVG(score) FROM reports WHERE user_id = 1 GROUP BY d ORDER BY d;
SELECT * FROM reports WHERE user_id = 1 ORDER BY score DESC LIMIT 5;
```

## Dataset plan (suggested)
* 300–500 images (≈150–250 correct / ≈150–250 incorrect) + 20–40 short videos, several people, 2 camera views (side + front), varied lighting.
* Label by a coach or experienced shooter (note the labeller in the paper). Split 70 / 15 / 15 by **person**, not randomly by frame, to avoid leakage.
* Augmentation (for any ML classifier you add): horizontal flip, ±10° rotation, brightness/contrast jitter.
* Use the **Dataset Manager** page → upload → split → extract features → **Model Evaluation** for real accuracy numbers.

## Honest limitations (say these in the viva)
* Ideal angle ranges in `config.py` are literature-style starting values, **not** validated by coaches; tune them on your labelled data.
* 2-D angles from one camera depend on camera view; tilt/balance metrics are only computed when the shoulders are clearly apart (front-ish view).
* Grip analysis is approximate (wrist–index-finger alignment from pose landmarks, not a hand model).
* Accuracy figures must come from the Model Evaluation page on your own test set.
