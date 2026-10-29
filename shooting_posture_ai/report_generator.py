"""Text report for download."""
from datetime import datetime
import config


def build_report_text(username, source, filename, profile, result):
    lines = ["=" * 60, "  SHOOTING POSTURE ANALYSIS REPORT", "=" * 60,
             f"Shooter   : {username}", f"Date      : {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"Source    : {source} ({filename})", f"Discipline: {profile}", "",
             f"POSTURE SCORE : {result['score']} / 100  ->  {result['label']}", "", "MEASUREMENTS"]
    for k, v in result["metrics"].items():
        lines.append(f"  - {config.METRIC_LABELS[k]:<24}: {v:.1f}")
    lines += ["", "DETECTED MISTAKES"]
    lines += [f"  - {e}" for e in result["errors"]] or ["  - None"]
    lines += ["", "CORRECTIVE FEEDBACK"] + [f"  {i+1}. {f}" for i, f in enumerate(result["feedback"])]
    return "\n".join(lines)
