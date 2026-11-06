"""Summarize the default pilot experiment after training and test evaluation."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
run = root / "runs/baseline"
records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
validation = [r for r in records if "validation_loss" in r]
best = min(validation, key=lambda r: r["validation_loss"])
training = json.loads((run / "training_report.json").read_text())
audit = json.loads((root / "runs/pilot100/audit/audit.json").read_text())
evaluation = json.loads((root / "runs/baseline-evaluation/metrics.json").read_text())
summary = {
    "complete_flights": audit["complete_flights"],
    "observations": audit["observations"],
    "simulated_seconds": audit["simulated_seconds"],
    "collision_frames": audit["collision_frames"],
    "best_checkpoint_step": best["step"],
    "training": training,
    "evaluation_windows": evaluation["windows"],
    "evaluation_flights": evaluation["flights_evaluated"],
    "summary_by_horizon": evaluation["summary_by_horizon"],
    "comparison_note": "Persistence uses the last observed RGB-D frame. Nominal physics additionally receives the exact last simulator state and known physical parameters. Action perturbation measures sensitivity only, not counterfactual correctness.",
}
for horizon, row in summary["summary_by_horizon"].items():
    values = row["mean"]
    values["rgb_mse_reduction_vs_persistence_percent"] = 100 * (
        1 - values["rgb_mse"] / values["persistence_rgb_mse"]
    )
    values["depth_rmse_reduction_vs_persistence_percent"] = 100 * (
        1 - values["depth_rmse_m"] / values["persistence_depth_rmse_m"]
    )
    selected = [r for r in evaluation["metrics"] if r["horizon_seconds"] == float(horizon)]
    row["fraction_windows_beating_rgb_persistence"] = float(
        np.mean([r["rgb_mse"] < r["persistence_rgb_mse"] for r in selected])
    )
    row["fraction_windows_beating_depth_persistence"] = float(
        np.mean([r["depth_rmse_m"] < r["persistence_depth_rmse_m"] for r in selected])
    )
(root / "runs/pilot100/experiment-summary.json").write_text(json.dumps(summary, indent=2) + "\n")

steps = [r["step"] for r in records]
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
axes[0, 0].plot(steps, [r["loss"] for r in records], lw=0.6, label="Train")
axes[0, 0].plot(
    [r["step"] for r in validation], [r["validation_loss"] for r in validation], label="Validation"
)
axes[0, 0].set(yscale="log", ylabel="Total loss", title=f"Best checkpoint: step {best['step']}")
axes[0, 0].legend()
axes[0, 1].plot(steps, [r["rgb_mse"] for r in records], lw=0.6)
axes[0, 1].set(yscale="log", ylabel="RGB reconstruction MSE")
axes[1, 0].plot(steps, [12 * np.sqrt(r["depth_mse"]) for r in records], lw=0.6)
axes[1, 0].set(yscale="log", ylabel="Depth reconstruction RMSE (m)")
axes[1, 1].plot(steps, [r["kl_dynamics"] for r in records], lw=0.6)
axes[1, 1].set(ylabel="Latent dynamics KL (nats)")
for ax in axes.flat:
    ax.set_xlabel("Training step")
    ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(run / "training-curves.png", dpi=160)
plt.close(fig)

horizons = sorted(float(h) for h in summary["summary_by_horizon"])
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, key, baseline, ylabel in zip(
    axes,
    ["rgb_mse", "depth_rmse_m", "position_error_m"],
    ["persistence_rgb_mse", "persistence_depth_rmse_m", "nominal_position_error_m"],
    ["RGB MSE", "Depth RMSE (m)", "Position error (m)"],
):
    ax.plot(
        horizons,
        [summary["summary_by_horizon"][str(h)]["mean"][key] for h in horizons],
        "o-",
        label="RSSM",
    )
    ax.plot(
        horizons,
        [summary["summary_by_horizon"][str(h)]["mean"][baseline] for h in horizons],
        "o-",
        label="Nominal physics*" if "nominal" in baseline else "Last frame",
    )
    ax.set(xlabel="Prediction horizon (s)", ylabel=ylabel, xticks=horizons)
    if "nominal" in baseline:
        ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend()
fig.suptitle(
    "Held-out prediction: 100 windows across 10 flights; *physics has exact reference state"
)
fig.tight_layout()
fig.savefig(root / "runs/baseline-evaluation/comparison.png", dpi=160)
plt.close(fig)
print(json.dumps(summary, indent=2))
