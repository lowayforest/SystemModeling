from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


LENGTH = 480
NOISE_SEED = 26721988
OUTPUT = Path(__file__).resolve().parent / "output" / "least_squares_recursive_480.csv"


def inverse_m_sequence(length: int) -> np.ndarray:
    state = [1, 0, 1, 0]
    bits = []
    for _ in range(15):
        bits.append(state[0])
        state = state[1:] + [state[0] ^ state[1]]
    if state != [1, 0, 1, 0]:
        raise ValueError("M sequence did not return to its initial state")
    base = 2 * np.array(bits, dtype=float) - 1
    return np.resize(np.concatenate((base, -base)), length)


def main() -> None:
    u = inverse_m_sequence(LENGTH)
    v = np.random.default_rng(NOISE_SEED).normal(0.0, 1.0, LENGTH)
    y = np.zeros(LENGTH)
    e0 = np.zeros(LENGTH)

    for i in range(LENGTH):
        y1 = y[i - 1] if i >= 1 else 0.0
        y2 = y[i - 2] if i >= 2 else 0.0
        e1 = e0[i - 1] if i >= 1 else 0.0
        e2 = e0[i - 2] if i >= 2 else 0.0
        u1 = u[i - 1] if i >= 1 else 0.0
        u2 = u[i - 2] if i >= 2 else 0.0
        y[i] = 1.5 * y1 - 0.7 * y2 + u1 + 0.5 * u2
        e0[i] = 1.5 * e1 - 0.7 * e2 + v[i]

    lam = 0.73 * np.sqrt(np.mean(y**2) / np.mean(e0**2))
    e = lam * e0
    z = y + e
    theta = np.full(4, 0.001)
    p = 1e6 * np.eye(4)
    history = []

    for i in range(LENGTH):
        z1 = z[i - 1] if i >= 1 else 0.0
        z2 = z[i - 2] if i >= 2 else 0.0
        u1 = u[i - 1] if i >= 1 else 0.0
        u2 = u[i - 2] if i >= 2 else 0.0
        row = np.array((-z1, -z2, u1, u2))
        gain = p @ row / (1.0 + row @ p @ row)
        error = z[i] - row @ theta
        theta = theta + gain * error
        p = p - np.outer(gain, row @ p)
        history.append((i + 1, u[i], v[i], y[i], e[i], z[i], *theta))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("k", "u", "v", "y", "e", "z", "a1", "a2", "b1", "b2"))
        writer.writerows(history)

    print(f"L={LENGTH}, seed={NOISE_SEED}, period=30, lambda={lam:.6f}, noise_signal_ratio=0.73")
    for k in (10, 50, 100, 200, 300, 480):
        row = history[k - 1]
        print("k={:3d}: a1={:.6f}, a2={:.6f}, b1={:.6f}, b2={:.6f}".format(row[0], *row[6:]))
    print("static_gain={:.6f}".format((theta[2] + theta[3]) / (1 + theta[0] + theta[1])))
    print(f"data={OUTPUT}")


if __name__ == "__main__":
    main()
