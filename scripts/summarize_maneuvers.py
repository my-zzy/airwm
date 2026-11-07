"""Summarize the completed data-only experiment and identical-flight comparisons."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def summarize(root, baseline_evaluation):
    audit = read(root / "audit/audit.json")
    pilot_audit = read(root / "pilot-audit/audit.json")
    training = read(root / "model/training_report.json")
    records = [json.loads(line) for line in (root / "model/metrics.jsonl").read_text().splitlines()]
    validation = [r for r in records if "validation_loss" in r]
    best = min(validation, key=lambda r: r["validation_loss"])
    comparisons = {}
    for dataset, left, right in (
        ("maneuvers", root / "pilot-on-maneuvers/metrics.json", root / "evaluation/metrics.json"),
        ("sweeps", baseline_evaluation / "metrics.json", root / "model-on-sweeps/metrics.json"),
    ):
        reports = [read(p) for p in (left, right)]
        keys = [
            {
                (str(Path(r["flight"]).resolve()), r["start_frame"], r["horizon_seconds"])
                for r in report["metrics"]
            }
            for report in reports
        ]
        if keys[0] != keys[1] or not keys[0]:
            raise ValueError("Flight comparisons must use exactly the same windows")
        comparisons[dataset] = {
            label: {
                "metrics_file": str(path.resolve()),
                "windows": report["windows"],
                "flights": report["flights_evaluated"],
                "summary_by_horizon": report["summary_by_horizon"],
            }
            for label, path, report in zip(("pilot", "maneuvers"), (left, right), reports)
        }
    summary = {
        "protocol": read(root / "protocol.json"),
        "collection": {
            k: audit[k]
            for k in (
                "complete_flights",
                "observations",
                "transitions",
                "simulated_seconds",
                "size_GiB",
                "collision_frames",
                "recorded_pulses_verified",
                "cancelled_pulses",
                "recovery_controls",
                "speed_percentiles_m_s",
                "angular_speed_percentiles_rad_s",
                "position_min_xyz_m",
                "position_max_xyz_m",
                "axis_command_std",
                "motor_command_at_bound_fraction",
            )
        },
        "axis_command_std_ratio_vs_pilot": {
            k: audit["axis_command_std"][k] / pilot_audit["axis_command_std"][k]
            for k in audit["axis_command_std"]
        },
        "training": training,
        "best_checkpoint_step": best["step"],
        "flight_comparisons": comparisons,
        "paired_action_comparison": read(root / "new-branches-comparison/comparison.json"),
        "replay_verification": read(root / "replay-verification.json"),
        "action_precision_diagnostic": read(root / "action-precision.json"),
        "limitations": "One room, one training seed, unchanged model/loss and fixed compute budget. Collection policy changes jointly include waypoints and motor excitation. Paired branches share eight histories and are not independent per-variant trials. New evaluation histories must be reserved for any subsequent tuning. Motion-head ablation, dedicated revisit evaluation and autonomous exploration are still outstanding.",
    }
    (root / "experiment-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    steps = [r["step"] for r in records]
    axes[0, 0].plot(steps, [r["loss"] for r in records], lw=0.6, label="Train")
    axes[0, 0].plot(
        [r["step"] for r in validation],
        [r["validation_loss"] for r in validation],
        label="Validation",
    )
    axes[0, 0].set(ylabel="Total loss", title=f"Best validation checkpoint: step {best['step']}")
    axes[0, 0].legend()
    axes[0, 1].plot(steps, [r["rgb_mse"] for r in records], lw=0.6)
    axes[0, 1].set(ylabel="RGB reconstruction MSE")
    axes[1, 0].plot(steps, [12 * np.sqrt(r["depth_mse"]) for r in records], lw=0.6)
    axes[1, 0].set(ylabel="Depth reconstruction RMSE (m)")
    axes[1, 1].plot(steps, [r["kl_dynamics"] for r in records], lw=0.6)
    axes[1, 1].set(ylabel="Latent dynamics KL (nats)")
    for ax in axes.flat:
        ax.set(xlabel="Training step", yscale="log")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "model/training-curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    horizons = [0.5, 1.0, 2.0]
    for row, dataset in enumerate(("maneuvers", "sweeps")):
        for col, (key, ylabel) in enumerate(
            (
                ("rgb_mse", "RGB MSE"),
                ("depth_rmse_m", "Depth RMSE (m)"),
                ("position_error_m", "Position error (m)"),
            )
        ):
            ax = axes[row, col]
            for model, label in (("pilot", "Sweep-trained"), ("maneuvers", "Maneuver-trained")):
                means = comparisons[dataset][model]["summary_by_horizon"]
                ax.plot(horizons, [means[str(h)]["mean"][key] for h in horizons], "o-", label=label)
            baseline = "persistence_" + key if col < 2 else "nominal_position_error_m"
            ax.plot(
                horizons,
                [means[str(h)]["mean"][baseline] for h in horizons],
                "--",
                label="Last frame" if col < 2 else "Physics (GT state)",
            )
            ax.set(
                xlabel="Horizon (s)", ylabel=ylabel, title=f"Held-out {dataset}", xticks=horizons
            )
            if col == 2:
                ax.set_yscale("log")
            ax.grid(alpha=0.2)
            ax.legend()
    fig.suptitle("Identical windows for both checkpoints: 100 per dataset, across 10 flights")
    fig.tight_layout()
    fig.savefig(root / "flight-comparison.png", dpi=160)
    plt.close(fig)
    print(f"Saved experiment summary and plots to {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default="runs/maneuvers01")
    parser.add_argument("--baseline-evaluation", type=Path, default="runs/baseline-evaluation")
    args = parser.parse_args()
    summarize(args.run, args.baseline_evaluation)
