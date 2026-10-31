from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from torch import nn


SEED = 26721988
LENGTH = 800
TRAJECTORIES = 5
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def simulate(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    u = rng.choice((-1.0, 1.0), size=LENGTH)
    y = np.zeros(LENGTH)
    features = np.zeros((LENGTH, 4))
    for i in range(LENGTH):
        y1 = y[i - 1] if i >= 1 else 0.0
        y2 = y[i - 2] if i >= 2 else 0.0
        u1 = u[i - 1] if i >= 1 else 0.0
        u2 = u[i - 2] if i >= 2 else 0.0
        features[i] = (y1, y2, u1, u2)
        y[i] = 1.5 * y1 - 0.7 * y2 + u1 + 0.5 * u2 + 0.4 * np.tanh(y1 * u1)
    return features, y


def main() -> None:
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)
    datasets = [simulate(rng) for _ in range(TRAJECTORIES)]
    x_train = np.concatenate([item[0] for item in datasets[:3]])
    y_train = np.concatenate([item[1] for item in datasets[:3]])
    x_valid, y_valid = datasets[3]
    x_test, y_test = datasets[4]

    basis = np.column_stack((
        -x_train[:, 0], -x_train[:, 1], x_train[:, 2], x_train[:, 3],
        np.tanh(x_train[:, 0] * x_train[:, 2])
    ))
    basis_tensor = torch.tensor(basis, dtype=torch.float64)
    target_tensor = torch.tensor(y_train, dtype=torch.float64)
    coefficients = nn.Parameter(torch.zeros(5, dtype=torch.float64))
    coefficient_optimizer = torch.optim.Adam((coefficients,), lr=0.03)
    coefficient_history = []
    for _ in range(2500):
        coefficient_loss = torch.mean((basis_tensor @ coefficients - target_tensor) ** 2)
        coefficient_optimizer.zero_grad()
        coefficient_loss.backward()
        coefficient_optimizer.step()
        coefficient_history.append(coefficients.detach().numpy().copy())
        if coefficient_loss.item() < 1e-12:
            break
    estimated_coefficients = coefficients.detach().numpy().copy()
    coefficient_history = np.array(coefficient_history)

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    y_mean = y_train.mean()
    y_std = y_train.std()
    x_train = torch.tensor((x_train - x_mean) / x_std, dtype=torch.float32)
    x_valid = torch.tensor((x_valid - x_mean) / x_std, dtype=torch.float32)
    x_test = torch.tensor((x_test - x_mean) / x_std, dtype=torch.float32)
    y_train_tensor = torch.tensor((y_train - y_mean) / y_std, dtype=torch.float32).unsqueeze(1)
    y_valid_tensor = torch.tensor((y_valid - y_mean) / y_std, dtype=torch.float32).unsqueeze(1)

    model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_losses = []
    valid_losses = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience = 0

    for epoch in range(1, 401):
        model.train()
        order = torch.randperm(len(x_train))
        for indices in order.split(128):
            prediction = model(x_train[indices])
            loss = criterion(prediction, y_train_tensor[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(x_train), y_train_tensor).item()
            valid_loss = criterion(model(x_valid), y_valid_tensor).item()
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        if valid_loss < best_loss - 1e-6:
            best_loss = valid_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= 40:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test).squeeze(1).numpy() * y_std + y_mean
    error = y_test - y_pred
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    r2 = 1 - np.sum(error**2) / np.sum((y_test - y_test.mean())**2)

    OUTPUT_DIR.mkdir(exist_ok=True)
    with (OUTPUT_DIR / "nonlinear_mlp_test.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("k", "true", "predicted", "error"))
        writer.writerows(zip(range(1, LENGTH + 1), y_test, y_pred, error))
    with (OUTPUT_DIR / "nonlinear_mlp_training.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "train_mse_normalized", "valid_mse_normalized"))
        writer.writerows(zip(range(1, len(train_losses) + 1), train_losses, valid_losses))
    with (OUTPUT_DIR / "nonlinear_coefficients.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("coefficient", "true", "estimated"))
        writer.writerows(zip(("a1", "a2", "b1", "b2", "c"), (-1.5, 0.7, 1.0, 0.5, 0.4), estimated_coefficients))
    with (OUTPUT_DIR / "nonlinear_coefficient_history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("iteration", "a1", "a2", "b1", "b2", "c"))
        writer.writerows((i, *row) for i, row in enumerate(coefficient_history, 1))

    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Heiti SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei"):
        if name in installed_fonts:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    for i, name, color, truth in zip(
        range(5), (r"$a_1$", r"$a_2$", r"$b_1$", r"$b_2$", "$c$"),
        ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"),
        (-1.5, 0.7, 1.0, 0.5, 0.4)
    ):
        axis.plot(range(1, len(coefficient_history) + 1), coefficient_history[:, i], color=color, linewidth=1.2, label=name)
        axis.axhline(truth, color=color, linestyle="--", linewidth=0.8, alpha=0.65)
    axis.set(xlabel="优化轮数", ylabel="系数估计值", ylim=(-1.8, 1.3))
    axis.legend(ncol=5, frameon=False)
    axis.grid(color="0.88", linewidth=0.5)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "nonlinear_coefficients.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(8.5, 5), sharex=True, gridspec_kw={"height_ratios": (2, 1)})
    axes[0].plot(range(1, 121), y_test[:120], color="black", linewidth=1.2, label="真实值")
    axes[0].plot(range(1, 121), y_pred[:120], color="#d62728", linestyle="--", linewidth=1.2, label="预测值")
    axes[0].set_ylabel("系统输出")
    axes[0].legend(frameon=False)
    axes[1].plot(range(1, 121), error[:120], color="#1f77b4", linewidth=1)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set(xlabel="测试序列步数 k", ylabel="真实值－预测值")
    for axis in axes:
        axis.grid(color="0.88", linewidth=0.5)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "nonlinear_mlp_prediction.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5, 4.5))
    axis.scatter(y_test, y_pred, s=9, alpha=0.55, color="#1f77b4")
    low = min(y_test.min(), y_pred.min())
    high = max(y_test.max(), y_pred.max())
    axis.plot((low, high), (low, high), "k--", linewidth=1)
    axis.set(xlabel="真实值", ylabel="预测值", xlim=(low, high), ylim=(low, high))
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="0.88", linewidth=0.5)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "nonlinear_mlp_scatter.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 3.5))
    axis.semilogy(range(1, len(train_losses) + 1), train_losses, color="black", label="训练集")
    axis.semilogy(range(1, len(valid_losses) + 1), valid_losses, color="#d62728", linestyle="--", label="验证集")
    axis.set(xlabel="训练轮数", ylabel="标准化均方误差")
    axis.legend(frameon=False)
    axis.grid(color="0.88", linewidth=0.5)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "nonlinear_mlp_loss.png", dpi=220)
    plt.close(figure)

    print(f"seed={SEED}, trajectory_length={LENGTH}, train=2400, validation=800, test=800")
    print(f"epochs={len(train_losses)}, best_epoch={best_epoch}, best_validation_mse={best_loss:.8f}")
    print(f"test_mae={mae:.6f}, test_rmse={rmse:.6f}, test_r2={r2:.6f}")
    print("coefficients=" + ", ".join(f"{value:.6f}" for value in estimated_coefficients))
    print(f"coefficient_iterations={len(coefficient_history)}")
    print(f"true_range=[{y_test.min():.4f}, {y_test.max():.4f}]")
    print(f"output={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
