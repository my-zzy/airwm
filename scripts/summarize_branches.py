"""Plot and summarize a completed matched-action branch experiment."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(evaluation, pilot_audit):
    report = json.loads((evaluation / "metrics.json").read_text())
    pilot = json.loads(pilot_audit.read_text())
    low, high = np.array(pilot["motor_command_min"]), np.array(pilot["motor_command_max"])
    ranges = {}
    for record in report["metrics"]:
        if record["horizon_seconds"] != 2.0:
            continue
        with h5py.File(record["branch"]) as f:
            pulse = f["actions"][:5]
        ranges.setdefault(record["variant"], []).append(
            float(np.mean((pulse < low) | (pulse > high)))
        )
    summary = {
        "checkpoint": report["checkpoint"],
        "checkpoint_step": report["checkpoint_step"],
        "common_histories": report["common_histories"],
        "branch_files": report["branch_files"],
        "censored_horizons": len(report["censored_horizons"]),
        "collision_metric_records": sum(r["collision_before_horizon"] for r in report["metrics"]),
        "maximum_reference_replay_position_error_m": max(
            r["reference_replay"]["maximum_position_error_m"] for r in report["replay_checks"]
        ),
        "maximum_repeat_position_error_m": max(
            r["repeat"]["maximum_position_error_m"] for r in report["replay_checks"]
        ),
        "mean_repeat_rgb_mse": float(
            np.mean([r["repeat_rgb_mse"] for r in report["replay_checks"]])
        ),
        "maximum_repeat_depth_rmse_m": max(
            r["repeat_depth_rmse_m"] for r in report["replay_checks"]
        ),
        "pulse_fraction_outside_pilot_per_motor_bounds": {
            k: float(np.mean(v)) for k, v in ranges.items()
        },
        "two_second_means": {
            v: rows["2.0"]["mean"]
            for v, rows in report["summary_by_variant_and_horizon"].items()
            if "2.0" in rows
        },
        "conclusion": "The pilot checkpoint underpredicts the tested action effects by orders of magnitude. Matched-command predictions are almost unchanged from predictions using the original commands. Do not use this checkpoint for autonomous exploration. This result is limited to the tested eight histories and intervention family.",
        "next_experiment": "Broaden and randomize maneuvers, investigate motor-input normalization and stronger multi-step motion/action objectives, then retrain. Keep these branches as a fixed diagnostic and reserve new histories for final testing. Motion-head ablation and revisit evaluation remain outstanding.",
    }
    (evaluation / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    means = summary["two_second_means"]
    variants = [
        "collective_up_small",
        "collective_down_small",
        "collective_up",
        "collective_down",
        "pitch_positive",
        "pitch_negative",
        "roll_positive",
        "roll_negative",
    ]
    labels = [
        "Collective\n+0.005",
        "Collective\n-0.005",
        "Collective\n+0.020",
        "Collective\n-0.020",
        "Pitch +",
        "Pitch -",
        "Roll +",
        "Roll -",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [3, 1]})
    x = np.arange(len(variants))
    for offset, prefix, label, color in (
        (-0.18, "actual", "Simulator", "#2166ac"),
        (0.18, "predicted", "RSSM", "#d6604d"),
    ):
        axes[0].bar(
            x + offset,
            [means[v][f"{prefix}_position_effect_m"] for v in variants],
            0.36,
            label=label,
            color=color,
        )
        axes[1].bar(
            np.arange(2) + offset,
            [means[v][f"{prefix}_rotation_effect_rad"] for v in ("yaw_positive", "yaw_negative")],
            0.36,
            label=label,
            color=color,
        )
    axes[0].set(
        xticks=x,
        xticklabels=labels,
        ylabel="Position change from control (m)",
        yscale="log",
        ylim=(1e-7, 1),
    )
    axes[0].set_xlabel("Collective labels are offsets in normalized rotor speed", fontsize=9)
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["Yaw +", "Yaw -"],
        ylabel="Rotation change from control (rad)",
        yscale="log",
        ylim=(1e-7, 1),
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    fig.suptitle(
        f"Action effect at 2 s: mean over {report['common_histories']} histories (log scales)"
    )
    fig.tight_layout()
    fig.savefig(evaluation / "action-effects.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, variant, key, unit in zip(
        axes,
        ("collective_up", "pitch_positive", "yaw_positive"),
        ("xyz", "xyz", "rotation"),
        ("m", "m", "rad"),
    ):
        rows = [r for r in report["trajectories"] if r["variant"] == variant]
        for prefix, label in (("actual", "Simulator"), ("predicted", "RSSM")):
            curves = np.array(
                [np.linalg.norm(np.array(r[f"{prefix}_delta_{key}"]), axis=-1) for r in rows]
            )
            ax.plot(np.arange(1, curves.shape[1] + 1) / 20, curves.mean(0), label=label)
        ax.set(
            title=variant.replace("_", " "),
            xlabel="Time after branching (s)",
            ylabel=f"Change from control ({unit})",
        )
        ax.axvspan(0, 0.25, color="gray", alpha=0.15)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Mean response; shaded interval is the 0.25 s command pulse")
    fig.tight_layout()
    fig.savefig(evaluation / "response-curves.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != "two_second_means"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--pilot-audit", default="runs/pilot100/audit/audit.json", type=Path)
    args = parser.parse_args()
    summarize(args.evaluation, args.pilot_audit)
