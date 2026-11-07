"""Compare two checkpoints on the exact same matched simulator branch bank."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compare(baseline, candidate, output):
    reports = [json.loads(p.read_text()) for p in (baseline, candidate)]

    def indexed(report):
        rows = report["metrics"]
        result = {(r["source"], r["variant"], r["horizon_seconds"]): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate branch metric")
        if report["censored_horizons"] or report["source_overlap_with_checkpoint_splits"]:
            raise ValueError("Comparison requires complete, held-out branch histories")
        if any(r["collision_before_horizon"] for r in rows):
            raise ValueError("Report collisions separately before using this comparison")
        return result

    left, right = map(indexed, reports)
    if left.keys() != right.keys():
        raise ValueError("Checkpoints must be evaluated on identical histories/variants/horizons")
    for key in left:
        if left[key]["branch"] != right[key]["branch"]:
            raise ValueError("Branch file mismatch")
        for field in ("actual_position_effect_m", "actual_rotation_effect_rad"):
            if left[key][field] != right[key][field]:
                raise ValueError("Simulator effect changed between evaluations")
    output.mkdir(parents=True, exist_ok=False)
    rows = {}
    for variant in sorted({k[1] for k in left} - {"control", "repeat"}):
        axis, unit = ("rotation", "rad") if variant.startswith("yaw") else ("position", "m")
        keys = sorted(k for k in left if k[1] == variant and k[2] == 2.0)
        values = {}
        for label, records in (("pilot", left), ("maneuvers", right)):
            actual = np.array([records[k][f"actual_{axis}_effect_{unit}"] for k in keys])
            if (actual <= 1e-5).any():
                raise ValueError("Intervention effect too small for normalized comparison")
            effect_error = np.array([records[k][f"{axis}_effect_error_{unit}"] for k in keys])
            values[label] = {
                "mean_actual_effect": float(actual.mean()),
                "mean_predicted_effect": float(
                    np.mean([records[k][f"predicted_{axis}_effect_{unit}"] for k in keys])
                ),
                "mean_signed_effect_gain": float(
                    np.mean([records[k][f"{axis}_effect_gain"] for k in keys])
                ),
                "mean_effect_error": float(effect_error.mean()),
                "mean_relative_effect_error": float(np.mean(effect_error / actual)),
            }
            for metric in ("rgb_mse", "depth_rmse_m", "position_error_m"):
                matched = np.array([records[k][metric] for k in keys])
                unchanged = np.array([records[k]["unchanged_command_" + metric] for k in keys])
                values[label][metric] = float(matched.mean())
                values[label]["unchanged_command_" + metric] = float(unchanged.mean())
                values[label]["matched_reduction_" + metric + "_percent"] = float(
                    100 * (1 - matched.mean() / unchanged.mean())
                )
        rows[variant] = {"axis": axis, "unit": unit, "histories": len(keys), **values}
    summary = {
        "baseline_metrics": str(baseline.resolve()),
        "candidate_metrics": str(candidate.resolve()),
        "checkpoints": [r["checkpoint"] for r in reports],
        "checkpoint_steps": [r["checkpoint_step"] for r in reports],
        "common_histories": reports[0]["common_histories"],
        "two_second_comparison": rows,
        "interpretation": "Relative effect error 1 equals predicting no response; 0 is exact. Signed gain 0 ignores the action and 1 reproduces its signed magnitude. Matched reduction measures error improvement when supplying changed versus unchanged commands to the same model. Report histories as the comparison unit; variants share histories. One model training seed and one intervention family do not establish repeatability or exploration readiness.",
    }
    (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    variants = list(rows)
    x = np.arange(len(variants))
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for offset, label, title in (
        (-0.18, "pilot", "Sweep-trained"),
        (0.18, "maneuvers", "Maneuver-trained"),
    ):
        for ax, key in zip(axes, ("mean_signed_effect_gain", "mean_relative_effect_error")):
            ax.bar(x + offset, [rows[v][label][key] for v in variants], 0.36, label=title)
    axes[0].axhline(1, color="black", ls="--", lw=1, label="Correct signed magnitude")
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].set(ylabel="Signed action-effect gain")
    axes[1].axhline(1, color="black", ls="--", lw=1, label="Zero-response error")
    axes[1].set(
        ylabel="Effect error / actual magnitude",
        xticks=x,
        xticklabels=[v.replace("_", "\n") for v in variants],
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    fig.suptitle(
        f"Two-second action response on {summary['common_histories']} shared histories\nYaw uses rotation; other interventions use position"
    )
    fig.tight_layout()
    fig.savefig(output / "action-comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != "two_second_comparison"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.baseline, args.candidate, args.output)
