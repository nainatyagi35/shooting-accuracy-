"""Streamlit front-end.  Run:  streamlit run app.py   (demo login: admin / admin123)"""
import os, time
from collections import Counter
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import config, database as db
from processing import decode_image, analyze_image, analyze_video
from report_generator import build_report_text
import dataset_tools as dt

st.set_page_config(page_title="Shooting Posture AI", page_icon="🎯", layout="wide")
db.init_db()

st.markdown("""
<style>
.hero {background:linear-gradient(120deg,#0f2027,#203a43,#2c5364);padding:22px 28px;border-radius:14px;color:white;margin-bottom:18px}
.hero h1 {margin:0;font-size:1.8rem} .hero p {margin:4px 0 0;opacity:.85}
div[data-testid="stMetric"] {background:#f5f7fa;border-radius:12px;padding:14px;border:1px solid #e3e8ee}
@media (prefers-color-scheme: dark){div[data-testid="stMetric"]{background:#1f2630;border-color:#2e3744}}
</style>""", unsafe_allow_html=True)


def hero(title, sub):
    st.markdown(f'<div class="hero"><h1>🎯 {title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ LOGIN
def login_page():
    hero("AI-Based Shooting Posture Analysis", "Real-time pose estimation • joint angles • corrective feedback • progress tracking")
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        tab_in, tab_up = st.tabs(["🔐 Login", "📝 Register"])
        with tab_in:
            u = st.text_input("Username", key="li_u")
            p = st.text_input("Password", type="password", key="li_p")
            if st.button("Login", use_container_width=True, type="primary"):
                user = db.verify_user(u, p)
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            st.caption("Demo account: **admin / admin123**")
        with tab_up:
            fn = st.text_input("Full name")
            u2 = st.text_input("Choose username", key="rg_u")
            p2 = st.text_input("Choose password (min 6 chars)", type="password", key="rg_p")
            if st.button("Create account", use_container_width=True):
                if len(u2) < 3 or len(p2) < 6:
                    st.warning("Username >= 3 chars and password >= 6 chars required")
                elif db.create_user(u2, p2, fn):
                    st.success("Account created - please login")
                else:
                    st.error("Username already exists")


# ------------------------------------------------------------------ SHARED RESULT VIEW
def show_result(res, source, filename, profile, annotated=None, extra_imgs=None):
    c1, c2 = st.columns([1.3, 1])
    with c1:
        if annotated is not None:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected skeleton + joint angles", use_container_width=True)
        for cap, im in (extra_imgs or {}).items():
            st.image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), caption=cap, use_container_width=True)
    with c2:
        icon = {"Good": "🟢", "Average": "🟡", "Poor": "🔴"}[res["label"]]
        st.metric("Posture Score", f"{res['score']} / 100", f"{icon} {res['label']}", delta_color="off")
        st.progress(min(int(res["score"]), 100))
        st.markdown("**Joint measurements**")
        st.dataframe(pd.DataFrame([{"Metric": config.METRIC_LABELS[k], "Value": round(v, 1)}
                                   for k, v in res["metrics"].items()]), hide_index=True, use_container_width=True)
    st.markdown("### ❗ Detected mistakes & corrective feedback")
    if res["errors"]:
        for f in res["feedback"]:
            st.warning(f)
    else:
        st.success(res["feedback"][0])
    b1, b2 = st.columns(2)
    if b1.button("💾 Save to my history", key=f"save_{filename}_{source}"):
        db.save_report(st.session_state.user["id"], source, filename, profile, res)
        st.success("Saved to Performance History")
    b2.download_button("⬇️ Download report", build_report_text(st.session_state.user["username"], source, filename, profile, res),
                       file_name=f"posture_report_{int(time.time())}.txt")


def settings_box():
    c1, c2 = st.columns(2)
    side = c1.selectbox("Shooting arm", ["right", "left"])
    profile = c2.selectbox("Discipline profile", list(config.PROFILES.keys()))
    return side, profile


# ------------------------------------------------------------------ PAGES
def page_dashboard():
    user = st.session_state.user
    hero("Performance Dashboard", f"Welcome back, {user['full_name'] or user['username']}")
    rows = db.get_reports(user["id"])
    if not rows:
        st.info("No reports yet. Analyse an image/video, or click below to load synthetic demo data.")
        if st.button("Load demo data (synthetic)"):
            db.seed_demo_reports(user["id"]); st.rerun()
        return
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"])
    k = st.columns(5)
    k[0].metric("Total sessions", len(df))
    k[1].metric("Average score", f"{df.score.mean():.1f}")
    k[2].metric("Best score", f"{df.score.max():.1f}")
    k[3].metric("Latest score", f"{df.score.iloc[-1]:.1f}")
    n = min(3, len(df) // 2) or 1
    k[4].metric("Improvement", f"{df.score.tail(n).mean() - df.score.head(n).mean():+.1f}", help="Avg of last sessions minus avg of first sessions")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📈 Progress graph")
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.plot(df.created_at, df.score, marker="o", label="Score")
        if len(df) >= 3:
            ax.plot(df.created_at, df.score.rolling(3, min_periods=1).mean(), "--", label="3-session trend")
        ax.axhline(config.GOOD_SCORE, color="g", ls=":", lw=1); ax.set_ylim(0, 100)
        ax.set_ylabel("Posture score"); ax.legend(); fig.autofmt_xdate(); st.pyplot(fig)
    with c2:
        st.subheader("🧮 Most frequent mistakes")
        cnt = Counter(e for s in df.errors.dropna() for e in s.split("; ") if e)
        if cnt:
            fig, ax = plt.subplots(figsize=(6, 3.4))
            names, vals = zip(*cnt.most_common(6))
            ax.barh(names[::-1], vals[::-1], color="#e4572e"); ax.set_xlabel("Occurrences"); st.pyplot(fig)
        else:
            st.success("No recurring mistakes 🎉")
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("🏷️ Posture quality distribution")
        fig, ax = plt.subplots(figsize=(6, 3))
        order = ["Good", "Average", "Poor"]
        ax.bar(order, [(df.label == o).sum() for o in order], color=["#2e9e5b", "#f0b429", "#d64545"]); st.pyplot(fig)
    with c4:
        st.subheader("📐 Average joint angles")
        cols = ["elbow", "shoulder", "neck", "hip", "knee", "wrist"]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(cols, [df[c].mean() for c in cols], color="#2c5364"); ax.set_ylabel("Degrees"); st.pyplot(fig)


def page_image():
    hero("Image Analysis", "Upload a side- or front-view photo of the shooter")
    side, profile = settings_box()
    f = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if f:
        img = decode_image(f.getvalue())
        with st.spinner("Detecting pose..."):
            res, ann = analyze_image(img, side, profile)
        if res is None:
            st.error("No person / not enough visible body joints detected. Use a clearer full-body image.")
        else:
            show_result(res, "image", f.name, profile, ann)


def page_video():
    hero("Video Analysis", "Frames are sampled, analysed and averaged into one performance report")
    side, profile = settings_box()
    step = st.slider("Analyse every N-th frame", 1, 15, 5)
    f = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
    if f and st.button("▶ Analyse video", type="primary"):
        os.makedirs(config.UPLOAD_DIR, exist_ok=True)
        path = os.path.join(config.UPLOAD_DIR, f"{int(time.time())}_{f.name}")
        with open(path, "wb") as fh: fh.write(f.getvalue())
        bar = st.progress(0.0)
        res = analyze_video(path, side, profile, step, lambda p: bar.progress(p))
        if res is None:
            st.error("No pose detected in the video.")
        else:
            st.session_state.video_res = (res, f.name, profile)
    if "video_res" in st.session_state:
        res, name, prof = st.session_state.video_res
        st.caption(f"Frames analysed: {res['frames_analyzed']}")
        show_result(res, "video", name, prof, extra_imgs={"Best frame": res["best_frame"], "Worst frame": res["worst_frame"]})
        fig, ax = plt.subplots(figsize=(7, 2.8)); ax.plot(res["score_series"]); ax.set_ylim(0, 100)
        ax.set_xlabel("Analysed frame"); ax.set_ylabel("Score"); st.pyplot(fig)


def page_camera():
    hero("Live Camera", "Capture a frame from your webcam / phone camera")
    side, profile = settings_box()
    shot = st.camera_input("Take a picture in your shooting stance")
    if shot:
        img = decode_image(shot.getvalue())
        os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(config.SCREENSHOT_DIR, f"shot_{int(time.time())}.png"), img)  # screenshot capture
        res, ann = analyze_image(img, side, profile)
        if res is None: st.error("No pose detected. Step back so your full body is visible.")
        else: show_result(res, "camera", f"camera_{int(time.time())}.png", profile, ann)
    st.info("For continuous real-time analysis with FPS counter run `python run_live.py` in the terminal.")


def page_history():
    user = st.session_state.user
    hero("Performance History", "All saved reports stored in SQLite")
    rows = db.get_reports(user["id"])
    if not rows: st.info("No history yet."); return
    df = pd.DataFrame(rows).sort_values("created_at", ascending=False)
    show = df[["id", "created_at", "source", "filename", "profile", "score", "label", "errors"]]
    st.dataframe(show, hide_index=True, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Export history (CSV)", df.to_csv(index=False), "posture_history.csv")
    rid = c2.number_input("Delete report ID", min_value=0, step=1)
    if c2.button("🗑 Delete") and rid:
        db.delete_report(int(rid), user["id"]); st.rerun()
    with st.expander("Latest feedback"):
        st.write(df.iloc[0]["feedback"])


def page_dataset():
    hero("Dataset Manager", "Where your training / testing data is stored")
    st.markdown(f"**Dataset root:** `{config.DATASET_DIR}`  |  **Database:** `{config.DB_PATH}`")
    counts = dt.count_images()
    table = pd.DataFrame([{"Split": s, "Correct": counts[(s, 'correct')], "Incorrect": counts[(s, 'incorrect')],
                           "Total": counts[(s, 'correct')] + counts[(s, 'incorrect')]} for s in ("raw", "train", "val", "test")])
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.code("""dataset/
├── raw/        correct/  incorrect/     <- all collected images (upload here)
├── train/      correct/  incorrect/     <- 70 %
├── val/        correct/  incorrect/     <- 15 %
├── test/       correct/  incorrect/     <- 15 %
└── landmarks/  features.csv             <- extracted joint angles""", language="text")
    st.subheader("1️⃣ Add labelled images")
    label = st.radio("Label", ["correct", "incorrect"], horizontal=True)
    files = st.file_uploader("Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if files and st.button("Save to dataset/raw"):
        for f in files:
            path = dt.save_raw_image(f.getvalue(), f"{int(time.time()*1000)}_{f.name}", label)
            db.add_dataset_sample(st.session_state.user["id"], path, label)
        st.success(f"Saved {len(files)} image(s)"); st.rerun()
    st.subheader("2️⃣ Split & extract features")
    c1, c2 = st.columns(2)
    if c1.button("Split raw → train / val / test (70/15/15)"):
        st.write(dt.split_dataset()); st.rerun()
    if c2.button("Extract landmark features → CSV"):
        with st.spinner("Running MediaPipe on dataset..."):
            p, n = dt.extract_landmark_features(); st.success(f"{n} rows written to {p}")
    st.subheader("3️⃣ Dataset records in database")
    samples = db.get_dataset_samples()
    st.dataframe(pd.DataFrame(samples) if samples else pd.DataFrame(columns=["id", "file_path", "label"]), hide_index=True, use_container_width=True)


def page_eval():
    hero("Model Evaluation", "Accuracy, confusion matrix and score distribution on dataset/test")
    side, profile = settings_box()
    split = st.selectbox("Split", ["test", "val", "train"])
    thr = st.slider("Decision threshold (score ≥ threshold ⇒ 'correct')", 40, 95, config.CLASSIFY_THRESHOLD)
    if st.button("Run evaluation", type="primary"):
        import evaluation
        with st.spinner("Evaluating..."):
            r = evaluation.evaluate_split(split, side, profile, thr)
        if r["n"] == 0:
            st.error("No labelled images found. Add images in Dataset Manager first."); return
        m = st.columns(5)
        m[0].metric("Accuracy", f"{r['accuracy']*100:.1f}%"); m[1].metric("Precision", f"{r['precision']*100:.1f}%")
        m[2].metric("Recall", f"{r['recall']*100:.1f}%"); m[3].metric("F1-score", f"{r['f1']*100:.1f}%"); m[4].metric("Images", r["n"])
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(4.2, 3.6)); ax.imshow(r["matrix"], cmap="Blues")
            ax.set_xticks([0, 1], ["Incorrect", "Correct"]); ax.set_yticks([0, 1], ["Incorrect", "Correct"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
            for i in range(2):
                for j in range(2): ax.text(j, i, r["matrix"][i, j], ha="center", va="center", fontsize=14)
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3.6)); ax.hist(r["scores"], bins=10, color="#2c5364")
            ax.axvline(thr, color="r", ls="--"); ax.set_xlabel("Posture score"); ax.set_ylabel("Images"); ax.set_title("Score distribution")
            st.pyplot(fig)
        if r["skipped"]: st.caption(f"{r['skipped']} image(s) skipped (no pose detected)")


def page_about():
    hero("About / Architecture", "How the system works")
    st.graphviz_chart("""digraph G { rankdir=LR; node[shape=box,style=rounded];
      "Camera / Image / Video" -> "OpenCV frames" -> "MediaPipe Pose (33 landmarks)" -> "Joint angle calculation"
      -> "Threshold comparison" -> "Posture score + classification" -> "Feedback generation" -> "SQLite reports" -> "Dashboard (Streamlit)"; }""")
    st.markdown("""
**Score** = weighted average of per-metric scores (1.0 inside the ideal range, falling linearly to 0 at the tolerance limit) × 100.
**Classes:** Good ≥ 80 · Average 60–79 · Poor < 60.  Thresholds live in `config.py` and must be calibrated with expert-labelled data.""")


# ------------------------------------------------------------------ ROUTER
if "user" not in st.session_state:
    login_page()
else:
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user['full_name'] or st.session_state.user['username']}")
        page = st.radio("Navigation", ["📊 Dashboard", "🖼️ Image Analysis", "🎥 Video Analysis", "📷 Live Camera",
                                       "🗂 History", "🧪 Dataset Manager", "📈 Model Evaluation", "ℹ️ About"])
        if st.button("Logout", use_container_width=True):
            st.session_state.clear(); st.rerun()
    {"📊 Dashboard": page_dashboard, "🖼️ Image Analysis": page_image, "🎥 Video Analysis": page_video,
     "📷 Live Camera": page_camera, "🗂 History": page_history, "🧪 Dataset Manager": page_dataset,
     "📈 Model Evaluation": page_eval, "ℹ️ About": page_about}[page]()
