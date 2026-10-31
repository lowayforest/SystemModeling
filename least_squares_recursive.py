from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from least_squares_batch import LENGTH, generate_data


OUTPUT = Path(__file__).resolve().parent / "output" / "least_squares_recursive_400.csv"


def main() -> None:
    _, _, z, h = generate_data()
    theta = np.full(4, 0.001)
    p = 1e6 * np.eye(4)
    history = []

    for i in range(LENGTH):
        row = h[i]
        gain = p @ row / (1.0 + row @ p @ row)
        error = z[i] - row @ theta
        theta = theta + gain * error
        p = p - np.outer(gain, row @ p)
        history.append((i + 1, *theta))

    batch = np.linalg.solve(h.T @ h, h.T @ z)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("k", "a1", "a2", "b1", "b2"))
        writer.writerows(history)

    for k in (10, 50, 100, 200, 400):
        print("k={:3d}: a1={:.6f}, a2={:.6f}, b1={:.6f}, b2={:.6f}".format(*history[k - 1]))
    print("batch: a1={:.6f}, a2={:.6f}, b1={:.6f}, b2={:.6f}".format(*batch))
    print(f"data={OUTPUT}")


if __name__ == "__main__":
    main()
