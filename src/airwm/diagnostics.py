import json
import os
import platform
import subprocess
import sys
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

EXPECTED = {
    "isaaclab": "3.0.0",
    "isaacsim": "6.2.0.0",
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "warp-lang": "1.17.0",
}


def doctor(args):
    versions = {}
    for package in EXPECTED:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    cuda = torch.cuda.is_available()
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "expected_packages": EXPECTED,
        "pins_match": all(
            versions[k] and versions[k].split("+")[0] == v for k, v in EXPECTED.items()
        ),
        "python_matches": sys.version_info[:2] == (3, 12),
        "cuda_available": cuda,
        "cuda_build": torch.version.cuda,
        "isaac_runtime_tested": False,
    }
    if cuda:
        props = torch.cuda.get_device_properties(0)
        report.update(
            gpu=props.name,
            vram_GiB=props.total_memory / 1024**3,
            cuda_calculation=float(torch.ones(4, device="cuda").sum()),
        )
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        report["nvidia_smi"] = (r.stdout if r.returncode == 0 else r.stderr or r.stdout).strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        report["nvidia_smi"] = str(error)
    print(json.dumps(report, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["pins_match"] and report["python_matches"] and cuda else 1


class GpuMonitor:
    """Sample device-wide VRAM, including Kit allocations outside PyTorch.

    Device-wide totals include other applications. The report labels this
    explicitly; they must not be presented as model tensor memory.
    """

    def __init__(self, index=0):
        self.index = index
        self.samples = []
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.index}",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    self.samples.append((time.time(), float(result.stdout.strip())))
                else:
                    self.error = result.stderr.strip()
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                self.error = str(error)
            self.stop_event.wait(0.5)

    def start(self):
        self.thread.start()

    def stop(self, path):
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self.thread.join(timeout=6)
        report = {
            "scope": "device-wide VRAM, including other processes",
            "pid": os.getpid(),
            "sample_interval_seconds": 0.5,
            "samples": len(self.samples),
            "error": self.error,
            "peak_used_MiB": max((s[1] for s in self.samples), default=None),
            "first_used_MiB": self.samples[0][1] if self.samples else None,
        }
        Path(path).write_text(json.dumps(report, indent=2) + "\n")
