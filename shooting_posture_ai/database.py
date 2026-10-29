"""SQLite layer. Tables: users, reports, dataset_samples.
  users(id PK) 1---* reports(user_id FK)      users(id PK) 1---* dataset_samples(user_id FK)"""
import sqlite3, hashlib, os, random
from datetime import datetime, timedelta
import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    full_name TEXT,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source TEXT NOT NULL,            -- image | video | camera | demo
    filename TEXT,
    profile TEXT,
    score REAL NOT NULL,
    label TEXT NOT NULL,             -- Good | Average | Poor
    elbow REAL, shoulder REAL, neck REAL, hip REAL, knee REAL, wrist REAL,
    lean REAL, shoulder_tilt REAL, hip_tilt REAL, balance REAL,
    errors TEXT, feedback TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dataset_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    file_path TEXT NOT NULL,
    label TEXT NOT NULL,             -- correct | incorrect
    split TEXT DEFAULT 'raw',        -- raw | train | val | test
    notes TEXT,
    created_at TEXT NOT NULL
);
"""
METRIC_COLS = ["elbow", "shoulder", "neck", "hip", "knee", "wrist",
               "lean", "shoulder_tilt", "hip_tilt", "balance"]


def get_conn():
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_conn() as c:
        c.executescript(SCHEMA)
    if not verify_user("admin", "admin123"):
        create_user("admin", "admin123", "Demo Shooter")   # demo account for the viva


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 120_000).hex()


def create_user(username, password, full_name=""):
    salt = os.urandom(16).hex()
    try:
        with get_conn() as c:
            c.execute("INSERT INTO users(username, full_name, password_hash, salt, created_at) VALUES (?,?,?,?,?)",
                      (username.strip().lower(), full_name, _hash(password, salt), salt, _now()))
        return True
    except sqlite3.IntegrityError:
        return False


def verify_user(username, password):
    with get_conn() as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (username.strip().lower(),)).fetchone()
    if row and _hash(password, row["salt"]) == row["password_hash"]:
        return dict(row)
    return None


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_report(user_id, source, filename, profile, result, created_at=None):
    m = result["metrics"]
    cols = ", ".join(METRIC_COLS)
    marks = ", ".join("?" * len(METRIC_COLS))
    with get_conn() as c:
        c.execute(f"""INSERT INTO reports(user_id, source, filename, profile, score, label, {cols},
                      errors, feedback, created_at) VALUES (?,?,?,?,?,?,{marks},?,?,?)""",
                  (user_id, source, filename, profile, result["score"], result["label"],
                   *[m.get(k) for k in METRIC_COLS],
                   "; ".join(result["errors"]), " | ".join(result["feedback"]), created_at or _now()))


def get_reports(user_id):
    with get_conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM reports WHERE user_id=? ORDER BY created_at ASC", (user_id,))]


def delete_report(report_id, user_id):
    with get_conn() as c:
        c.execute("DELETE FROM reports WHERE id=? AND user_id=?", (report_id, user_id))


def add_dataset_sample(user_id, file_path, label, split="raw", notes=""):
    with get_conn() as c:
        c.execute("INSERT INTO dataset_samples(user_id, file_path, label, split, notes, created_at) VALUES (?,?,?,?,?,?)",
                  (user_id, file_path, label, split, notes, _now()))


def get_dataset_samples():
    with get_conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM dataset_samples ORDER BY id DESC")]


def seed_demo_reports(user_id, n=12):
    """SYNTHETIC demo rows (source='demo') so the dashboard is not empty during the viva."""
    rnd = random.Random(7)
    start = datetime.now() - timedelta(days=n * 2)
    for i in range(n):
        score = min(96, 52 + i * 3.2 + rnd.uniform(-4, 4))
        label = "Good" if score >= 80 else "Average" if score >= 60 else "Poor"
        errs = rnd.sample(["Elbow angle", "Neck angle", "Body lean (deg)", "Knee angle", "Shoulder tilt (deg)"],
                          k=max(0, 3 - i // 4))
        res = {"score": round(score, 1), "label": label, "errors": errs,
               "feedback": ["Demo data"], "metrics": {
                   "elbow": rnd.uniform(140, 178), "shoulder": rnd.uniform(65, 100),
                   "neck": rnd.uniform(140, 175), "hip": rnd.uniform(150, 178),
                   "knee": rnd.uniform(155, 178), "wrist": rnd.uniform(140, 175),
                   "lean": rnd.uniform(2, 14)}}
        save_report(user_id, "demo", f"demo_{i+1}.jpg", "Pistol (extended arm)", res,
                    (start + timedelta(days=i * 2)).strftime("%Y-%m-%d %H:%M:%S"))
