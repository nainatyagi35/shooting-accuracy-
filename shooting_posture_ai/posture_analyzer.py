"""Joint angles -> threshold comparison -> classification -> errors -> feedback -> score."""
import config
from angle_utils import (calculate_angle, vertical_deviation, horizontal_tilt,
                         midpoint, distance)


def _pt(lm, name):
    """Return (x, y) if the landmark is visible enough, else None."""
    x, y, v = lm[name]
    return (x, y) if v >= config.MIN_VISIBILITY else None


def compute_metrics(lm, side="right"):
    """Compute all angle/ratio metrics. Metrics that cannot be measured are omitted.
    Returns (metrics, anchors) - anchors = pixel position where to print each value."""
    s = side
    sh, el, wr = _pt(lm, f"{s}_shoulder"), _pt(lm, f"{s}_elbow"), _pt(lm, f"{s}_wrist")
    ix, hp = _pt(lm, f"{s}_index"), _pt(lm, f"{s}_hip")
    kn, an, ear = _pt(lm, f"{s}_knee"), _pt(lm, f"{s}_ankle"), _pt(lm, f"{s}_ear")
    m, anchors = {}, {}

    def put(key, value, anchor):
        if value is not None and anchor is not None:
            m[key], anchors[key] = value, anchor

    if sh and el and wr: put("elbow", calculate_angle(sh, el, wr), el)
    if el and sh and hp: put("shoulder", calculate_angle(el, sh, hp), sh)
    if ear and sh and hp: put("neck", calculate_angle(ear, sh, hp), ear)
    if sh and hp and kn: put("hip", calculate_angle(sh, hp, kn), hp)
    if hp and kn and an: put("knee", calculate_angle(hp, kn, an), kn)
    if el and wr and ix: put("wrist", calculate_angle(el, wr, ix), wr)

    # Both-side metrics
    lsh, rsh = _pt(lm, "left_shoulder"), _pt(lm, "right_shoulder")
    lhp, rhp = _pt(lm, "left_hip"), _pt(lm, "right_hip")
    lan, ran = _pt(lm, "left_ankle"), _pt(lm, "right_ankle")
    if lsh and rsh and lhp and rhp:
        ms, mh = midpoint(lsh, rsh), midpoint(lhp, rhp)
        put("lean", vertical_deviation(ms, mh), ms)
        torso = distance(ms, mh)
        shoulder_width = distance(lsh, rsh)
        # Tilt/balance only make sense from a front-ish view (shoulders clearly apart)
        if torso > 0 and shoulder_width / torso > 0.35:
            put("shoulder_tilt", horizontal_tilt(lsh, rsh), rsh)
            put("hip_tilt", horizontal_tilt(lhp, rhp), rhp)
            if lan and ran:
                ma = midpoint(lan, ran)
                put("balance", abs(mh[0] - ma[0]) / torso, mh)
    return m, anchors


def _metric_score(value, lo, hi, tol):
    """1.0 inside the ideal range; falls linearly to 0.0 at `tol` outside the range."""
    if lo <= value <= hi:
        return 1.0
    dist = (lo - value) if value < lo else (value - hi)
    return max(0.0, 1.0 - dist / tol)


def classify(score):
    if score >= config.GOOD_SCORE: return "Good"
    if score >= config.AVERAGE_SCORE: return "Average"
    return "Poor"


def analyze_landmarks(lm, side="right", profile="Pistol (extended arm)"):
    """Full analysis of one frame. Returns dict with metrics, scores, errors, feedback, score."""
    rules = config.PROFILES[profile]
    metrics, anchors = compute_metrics(lm, side)
    if not metrics:
        return None
    per_metric, errors, feedback = {}, [], []
    weighted_sum = total_weight = 0.0
    for key, value in metrics.items():
        lo, hi, tol, weight, adv_low, adv_high = rules[key]
        sc = _metric_score(value, lo, hi, tol)
        per_metric[key] = sc
        weighted_sum += sc * weight
        total_weight += weight
        if sc < 0.75:
            msg = adv_low if value < lo else adv_high
            if msg:
                errors.append(config.METRIC_LABELS[key])
                feedback.append(f"{config.METRIC_LABELS[key]} = {value:.1f}: {msg}")
    score = 100.0 * weighted_sum / total_weight if total_weight else 0.0
    if not feedback:
        feedback.append("Excellent posture - maintain this position and breathing rhythm.")
    return {"metrics": metrics, "anchors": anchors, "metric_scores": per_metric,
            "errors": errors, "feedback": feedback, "score": round(score, 1),
            "label": classify(score)}
