from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from m_sequence import generate


LENGTH = 400
NOISE_SEED = 26721988
OUTPUT = Path(__file__).resolve().parent / "output" / "least_squares_data_400.csv"


def main() -> None:
    u = np.array([row["u"] for row in generate(LENGTH)], dtype=float)
    v = np.random.default_rng(NOISE_SEED).normal(0.0, 1.0, LENGTH)
    z = np.zeros(LENGTH)
    h = np.zeros((LENGTH, 4))

    for i in range(LENGTH):
        z1 = z[i - 1] if i >= 1 else 0.0
        z2 = z[i - 2] if i >= 2 else 0.0
        u1 = u[i - 1] if i >= 1 else 0.0
        u2 = u[i - 2] if i >= 2 else 0.0
        z[i] = 1.5 * z1 - 0.7 * z2 + u1 + 0.5 * u2 + v[i]
        h[i] = (-z1, -z2, u1, u2)

    if np.linalg.matrix_rank(h) != 4:
        raise ValueError("The data matrix does not have full column rank")
    theta = np.linalg.solve(h.T @ h, h.T @ z)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("k", "u", "v", "z", "minus_z_k1", "minus_z_k2", "u_k1", "u_k2"))
        for i in range(LENGTH):
            writer.writerow((i + 1, u[i], v[i], z[i], *h[i]))

    print(f"seed={NOISE_SEED}, L={LENGTH}, rank(H)={np.linalg.matrix_rank(h)}")
    print("a1={:.6f}, a2={:.6f}, b1={:.6f}, b2={:.6f}".format(*theta))
    print(f"data={OUTPUT}")


if __name__ == "__main__":
    main()
