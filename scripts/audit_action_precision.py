"""Measure input rounding only; this is not a trained-model precision ablation."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from airwm.branching import INTERVENTIONS


def audit(folder, output):
    report = {}
    for variant in INTERVENTIONS.keys() - {"control", "repeat"}:
        same, errors = [], []
        for path in sorted(folder.glob(f"*__{variant}.h5")):
            control = path.with_name(path.name.replace("__" + variant, "__control"))
            with h5py.File(path) as f, h5py.File(control) as c:
                metadata, cfg = json.loads(f.attrs["metadata"]), json.loads(f.attrs["config"])
                count = round(metadata["pulse_seconds"] * cfg["camera_hz"])
                actual, original = f["actions"][:count], c["actions"][:count]
            changed = actual != original
            if not changed.any():
                raise ValueError(f"No changed commands in {path}")
            rounded = torch.from_numpy(actual).to(torch.bfloat16).float().numpy()
            rounded_original = torch.from_numpy(original).to(torch.bfloat16).float().numpy()
            same.append(float((rounded[changed] == rounded_original[changed]).mean()))
            errors.append(
                float(np.sqrt(np.mean(((rounded - rounded_original) - (actual - original)) ** 2)))
            )
        if same:
            report[variant] = {
                "histories": len(same),
                "fraction_changed_pulse_components_indistinguishable_in_bfloat16": float(
                    np.mean(same)
                ),
                "rms_error_in_pulse_difference_from_bfloat16_rounding": float(np.mean(errors)),
            }
    if not report:
        raise ValueError("No paired intervention files")
    result = {
        "branches": str(folder.resolve()),
        "scope": "CPU input-rounding diagnostic, not a model ablation or proof of the action-response failure cause. Training uses BF16 autocast on linear layers, whereas deterministic evaluation uses FP32. Large collective/yaw pulses remain distinguishable and the pilot also fails those cases.",
        "variants": dict(sorted(report.items())),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved input-rounding diagnostic to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.branches, args.output)
