"""Pure-math helpers. No ML needed, so they are easy to unit-test."""
import math
import numpy as np


def calculate_angle(a, b, c):
    """Angle at point b (degrees) formed by points a-b-c.
    Formula: cos(theta) = (BA . BC) / (|BA| * |BC|)."""
    a, b, c = np.array(a[:2], float), np.array(b[:2], float), np.array(c[:2], float)
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return None
    cos_t = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_t)))


def vertical_deviation(top, bottom):
    """Angle (degrees) between the line bottom->top and the vertical axis.
    Image y grows downward, so 'up' is (0, -1)."""
    v = np.array(top[:2], float) - np.array(bottom[:2], float)
    n = np.linalg.norm(v)
    if n == 0:
        return None
    cos_t = np.clip(np.dot(v, np.array([0.0, -1.0])) / n, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_t)))


def horizontal_tilt(left, right):
    """Tilt (degrees) of the line left->right from the horizontal axis (0 = level)."""
    dx = abs(right[0] - left[0])
    dy = right[1] - left[1]
    return float(abs(math.degrees(math.atan2(dy, dx if dx else 1e-9))))


def midpoint(p, q):
    return ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)


def distance(p, q):
    return float(math.hypot(p[0] - q[0], p[1] - q[1]))
