# Viva cheat-sheet

**One-line pitch:** A webcam/phone app that finds 33 body landmarks with MediaPipe, converts them into joint angles, compares them with ideal ranges, scores the posture 0-100, explains mistakes and tracks progress in SQLite.

**Pipeline:** Capture → OpenCV frame → MediaPipe Pose → 33 landmarks → angles (cos θ = BA·BC / |BA||BC|) → compare with ideal range → weighted score → Good/Average/Poor → feedback → save report → dashboard.

**Score formula:** metric score = 1 inside ideal range, falls linearly to 0 at tolerance; final = 100 × Σ(w·s)/Σw. Good ≥ 80, Average 60–79, Poor < 60.

**Q: Why MediaPipe, not OpenPose?** Runs real-time on CPU/phones, pip-installable, 33 landmarks incl. fingers/feet points, no GPU/heavy build; OpenPose is more GPU-hungry.
**Q: Is it machine learning?** Pose estimation is a pretrained deep model (BlazePose); posture classification is rule/threshold-based and explainable. Future: train SVM/Random Forest/LSTM on `features.csv`.
**Q: How are passwords stored?** PBKDF2-HMAC-SHA256, per-user random salt, 120k iterations.
**Q: Why pixel coordinates for angles?** Normalised coordinates distort angles when width ≠ height.
**Q: Limitations?** Single 2-D camera, thresholds need coach calibration, occlusion, approximate grip.
**Q: Future work?** 3-D pose, sequence model for aiming steadiness/trigger-pull timing, hand model for grip, mobile app.
**Q: Where's data stored?** `data/shooting_posture.db` + `dataset/` folders (see README).

**Demo order (5 min):** Login (admin/admin123) → Dashboard (load demo data, show progress/mistake charts) → Image Analysis (skeleton + feedback) → Live Camera → History/export → Dataset Manager (folders, split) → Model Evaluation (confusion matrix) → show `data/*.db` schema.
