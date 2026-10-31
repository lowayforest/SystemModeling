from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


LENGTH = 480
NOISE_SEED = 26721988
OUTPUT = Path(__file__).resolve().parent / "output" / "least_squares_recursive_480.csv"
STEP_OUTPUT = Path(__file__).resolve().parent / "output" / "least_squares_recursive_step_26.csv"
AUTOCORR_OUTPUT = Path(__file__).resolve().parent / "output" / "least_squares_recursive_autocorr_20.csv"
PARAMETER_FIGURE = Path(__file__).resolve().parent / "output" / "least_squares_recursive_parameters.png"
AUTOCORR_FIGURE = Path(__file__).resolve().parent / "output" / "least_squares_recursive_autocorr.png"
STEP_FIGURE = Path(__file__).resolve().parent / "output" / "least_squares_recursive_step.png"


def inverse_m_sequence(length: int) -> np.ndarray:
    state = [1, 0, 1, 0]
    bits = []
    for _ in range(15):
        bits.append(state[0])
        state = state[1:] + [state[0] ^ state[1]]
    if state != [1, 0, 1, 0]:
        raise ValueError("M sequence did not return to its initial state")
    base = np.resize(np.array(bits, dtype=int), length)
    square = np.arange(length) % 2
    return 2 * np.bitwise_xor(base, square) - 1


def main() -> None:
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Heiti SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei"):
        if name in installed_fonts:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

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
    h = np.zeros((LENGTH, 4))
    history = []

    for i in range(LENGTH):
        z1 = z[i - 1] if i >= 1 else 0.0
        z2 = z[i - 2] if i >= 2 else 0.0
        u1 = u[i - 1] if i >= 1 else 0.0
        u2 = u[i - 2] if i >= 2 else 0.0
        row = np.array((-z1, -z2, u1, u2))
        h[i] = row
        gain = p @ row / (1.0 + row @ p @ row)
        error = z[i] - row @ theta
        theta = theta + gain * error
        p = p - np.outer(gain, row @ p)
        history.append((i + 1, u[i], v[i], y[i], e[i], z[i], *theta))

    residual = z - h @ theta
    centered = residual - np.mean(residual)
    rho = np.array([
        centered[lag:] @ centered[:-lag] / (centered @ centered)
        for lag in range(1, 21)
    ])
    q = LENGTH * np.sum(rho**2)
    threshold = 20 + 1.65 * np.sqrt(40)

    step_true = np.zeros(26)
    step_estimated = np.zeros(26)
    for k in range(1, 26):
        step_true[k] = (
            1.5 * step_true[k - 1]
            - (0.7 * step_true[k - 2] if k >= 2 else 0.0)
            + 1.0
            + (0.5 if k >= 2 else 0.0)
        )
        step_estimated[k] = (
            -theta[0] * step_estimated[k - 1]
            - (theta[1] * step_estimated[k - 2] if k >= 2 else 0.0)
            + theta[2]
            + (theta[3] if k >= 2 else 0.0)
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("k", "u", "v", "y", "e", "z", "a1", "a2", "b1", "b2"))
        writer.writerows(history)
    with STEP_OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("k", "true", "estimated"))
        writer.writerows(zip(range(26), step_true, step_estimated))
    with AUTOCORR_OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("lag", "rho"))
        writer.writerows(zip(range(1, 21), rho))

    estimates = np.array([row[6:] for row in history])
    figure, axis = plt.subplots(figsize=(9, 5.2))
    sample_steps = np.array((100, 200, 300, 400, 480))
    for index, name, color, truth, offset in zip(
        range(4), (r"$\hat a_1$", r"$\hat a_2$", r"$\hat b_1$", r"$\hat b_2$"),
        ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"), (-1.5, 0.7, 1.0, 0.5),
        (10, -13, 10, -13)
    ):
        axis.plot(np.arange(1, LENGTH + 1), estimates[:, index], color=color, linewidth=1.35)
        axis.hlines(truth, 0, 520, color=color, linestyle="--", linewidth=1, alpha=0.8)
        axis.scatter(sample_steps, estimates[sample_steps - 1, index], color=color, s=19, zorder=3)
        for step in sample_steps:
            axis.annotate(f"({estimates[step - 1, index]:.2f})", (step, estimates[step - 1, index]),
                          xytext=(0, offset), textcoords="offset points", ha="center", fontsize=7.5, color=color)
        axis.text(527, truth, name, color=color, va="center", fontsize=11)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_position(("data", 0))
    axis.spines["left"].set_position(("data", 0))
    axis.set(xlim=(0, 560), ylim=(-1.8, 1.5), xticks=(0, 100, 200, 300, 400, 500),
             yticks=(-1.5, -1.0, -0.5, 0, 0.5, 0.7, 1.0))
    axis.text(0.01, 1.02, r"$\hat\theta(k)$", transform=axis.transAxes, fontsize=11)
    axis.text(0.98, 0.51, "$k$", transform=axis.transAxes, fontsize=11)
    figure.tight_layout()
    figure.savefig(PARAMETER_FIGURE, dpi=220)
    plt.close(figure)

    limit = 1.98 / np.sqrt(LENGTH)
    figure, axis = plt.subplots(figsize=(7.1, 4.2))
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_position(("data", 0))
    axis.spines["left"].set_position(("data", 0))
    axis.vlines(np.arange(1, 21), 0, rho, color="black", linewidth=1.25)
    axis.plot(np.arange(1, 21), rho, "ko", markersize=2.4)
    axis.hlines((limit, -limit), 0, 20.3, color="black", linestyle=(0, (5, 4)), linewidth=1)
    axis.set(xlim=(-0.6, 22.3), ylim=(-0.14, 0.14), xticks=(0, 5, 10, 15, 20), yticks=(-0.1, 0, 0.1))
    axis.text(20.45, limit, r"$+1.98/\sqrt{L}$", va="bottom", fontsize=9)
    axis.text(20.45, -limit, r"$-1.98/\sqrt{L}$", va="top", fontsize=9)
    axis.set_xlabel(r"滞后阶数 $\ell$")
    axis.xaxis.set_label_coords(0.52, -0.08)
    axis.set_ylabel("残差自相关系数")
    axis.yaxis.set_label_coords(-0.08, 0.5)
    figure.subplots_adjust(left=0.13, right=0.94, bottom=0.15, top=0.94)
    figure.savefig(AUTOCORR_FIGURE, dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.5, 3.4))
    axis.plot(range(26), step_true, "k-", marker="o", markersize=3, linewidth=1.2, label="真实对象")
    axis.plot(range(26), step_estimated, color="0.45", linestyle="--", marker="x", markersize=4, linewidth=1.2, label="辨识模型")
    axis.set(xlabel="采样步数 k", ylabel="阶跃响应", xlim=(0, 25), xticks=np.arange(0, 26, 2))
    axis.grid(color="0.88", linewidth=0.5)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(STEP_FIGURE, dpi=220)
    plt.close(figure)

    print(f"L={LENGTH}, seed={NOISE_SEED}, period=30, lambda={lam:.6f}, noise_signal_ratio=0.73")
    for k in (10, 50, 100, 200, 300, 480):
        row = history[k - 1]
        print("k={:3d}: a1={:.6f}, a2={:.6f}, b1={:.6f}, b2={:.6f}".format(row[0], *row[6:]))
    print("static_gain={:.6f}".format((theta[2] + theta[3]) / (1 + theta[0] + theta[1])))
    print("residual_mean={:.6f}, residual_variance={:.6f}".format(np.mean(residual), np.mean(centered**2)))
    print("Q20={:.6f}, threshold={:.6f}, outside_95pct={}".format(q, threshold, np.sum(np.abs(rho) > 1.98 / np.sqrt(LENGTH))))
    for k in (1, 5, 10, 15, 20, 25):
        print("step={:2d}: true={:.6f}, estimated={:.6f}".format(k, step_true[k], step_estimated[k]))
    print(f"data={OUTPUT}")
    print(f"step_data={STEP_OUTPUT}")
    print(f"autocorr_data={AUTOCORR_OUTPUT}")
    print(f"figures={PARAMETER_FIGURE}, {AUTOCORR_FIGURE}, {STEP_FIGURE}")


if __name__ == "__main__":
    main()
