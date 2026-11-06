# Aerial World Model (airwm)

A simulation research project for a drone that explores an indoor room and learns to predict what it will see and how it will move under future flight commands.

The goal is to learn visible room geometry, appearance, and flight dynamics from RGB-D, IMU measurements, and motor-command histories, then use those predictions to guide further exploration.

**Status:** the first pilot experiment is complete. Collected and audited 100 flights, trained the RSSM for 10,000 steps, and evaluated the selected checkpoint on held-out flights. At a two-second prediction horizon, test RGB MSE is 93.9% lower and depth RMSE is 79.9% lower than repeating the last frame. These results cover gentle scripted flights in one room; autonomous exploration and persistent room memory remain future work. All 15 CPU tests pass.

See [SIMULATION_PLAN.md](SIMULATION_PLAN.md) for the detailed experiment design, evaluation protocol, and references.

## Approach

Start with a **DreamerV3-style recurrent state-space model (RSSM)**. It combines current observations with recurrent memory, then predicts future observations and motion conditioned on action sequences.

```mermaid
flowchart LR
    Sensors["RGB-D + IMU history"] --> Encoder["Sensor encoders"]
    Encoder --> Model["RSSM: latent state + memory"]
    Actions["Motor-command sequences"] --> Model
    Model --> Images["Future RGB-D"]
    Model --> Motion["Future drone motion"]
```

The proposed model adds depth reconstruction and an auxiliary motion-prediction head to the Dreamer-style baseline. Simulator state provides motion targets and evaluation ground truth; it is excluded from the main world-model inputs.

After validating prediction, add **Plan2Explore-style exploration** using disagreement between learned predictors to seek informative experience. Persistent spatial memory follows as a separate extension for retaining room geometry across long turns, occlusions, and revisits.

## Planned simulation

| Component | Initial choice |
|---|---|
| Simulator | Isaac Lab 3.0.0 with Isaac Sim 6.2.0/PhysX and RTX rendering |
| Python environment | Conda `airwm`, Python 3.12, PyTorch 2.11.0 with CUDA 13.0 |
| Scene | One static furnished room |
| Robot | Quadrotor with per-motor thrust/torque, motor response, drag, and collisions |
| Sensors | RGB, depth, and IMU |
| Camera / world-model cadence | 64 × 64 images at 20 Hz |
| Physics / motor control | 200 Hz / 100 Hz |
| Local hardware | NVIDIA GeForce RTX 5090, 32 GB VRAM; driver 580.178.04 |
| Parallelism | One environment initially; increase after throughput and VRAM profiling |

CUDA 13.0 and single-drone PhysX/RTX flight collection pass on this machine with the existing driver 580.178.04. Initial data collection and neural-model training run separately. See the [Conda setup and pinned versions](SIMULATION_PLAN.md#0-conda-environment-and-pinned-software) for the installation rationale.

## Setup and commands

Create a dedicated environment once, then install the pinned stack:

```bash
conda create -n airwm python=3.12 pip -y
conda activate airwm
AIRWM_DIRECT=1 bash setup/install.sh
bash setup/run.sh airwm doctor --output setup/environment/doctor.json
```

The environment is already installed on this machine; the commands above reproduce it. The installer uses a [version-and-hash lock](setup/environment/airwm-python-linux-64.lock) and [resolved overrides](setup/environment/isaaclab-resolved-overrides.txt), derived from the official [Isaac Lab overrides](setup/isaaclab-3.0-overrides.txt). Existing Conda environments are preserved.

`AIRWM_DIRECT=1` disables proxy variables only inside the installer. Omit it to use the current proxy configuration. An optional mirror can speed up ordinary PyPI packages while retaining official PyPI as a fallback and enforcing the locked hashes:

```bash
AIRWM_DIRECT=1 AIRWM_PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple bash setup/install.sh
```

This setup used direct downloads first, proxy retries where needed, and the Tsinghua mirror for remaining dependencies. Large wheels were downloaded with resume support and verified against official SHA-256 hashes. Global proxy and package-index settings were not changed.

This host's shell adds a CUDA 12.8 library directory to `LD_LIBRARY_PATH`. Run commands through `bash setup/run.sh airwm ...` to select `airwm` and remove that inherited path for the subprocess, using the new environment's packaged CUDA libraries. For example: `bash setup/run.sh airwm doctor --output setup/environment/doctor.json`.

The following commands reproduce the validated short workflow, then start the larger experiment. Use fresh output directories when repeating collection or training:

```bash
# Required license acknowledgement for noninteractive Isaac Sim launches.
export OMNI_KIT_ACCEPT_EULA=YES

# One drone, local room geometry, no remote scene assets. Headless by default.
airwm collect --output data/hover --episodes 1 --seconds 5 --trajectory hover
airwm collect --output data/pilot --episodes 3 --seconds 8 --trajectory sweep

# Measure training VRAM independently of the simulator.
airwm profile --batch-size 16 --length 64 --output runs/profile.json

# Short pipeline check; use a fresh --run directory for each run.
airwm train --data data/pilot --run runs/smoke --steps 10 --batch-size 2
airwm evaluate --checkpoint runs/smoke/best.pt --output runs/smoke-evaluation

# Once flights and predictions are validated, collect the planned pilot dataset.
airwm collect --output data/pilot100 --episodes 100 --seconds 45 --seed 10
python scripts/audit_flights.py --data data/pilot100 --output runs/pilot100/audit --expected-flights 100
airwm train --data data/pilot100 --run runs/baseline --steps 10000 --batch-size 16 --workers 4
airwm evaluate --checkpoint runs/baseline/best.pt --output runs/baseline-evaluation --windows 100
python scripts/summarize_pilot.py
```

Add `--gui` to `collect` for a viewport. Collection and training are separate processes. A short training run verifies the pipeline; it does not establish useful learned predictions. Training-step count is a starting budget and must be adjusted from validation results. Test predictions remain independent of training/validation flight splits.

The collector saves one HDF5 file per flight, camera intrinsics and conventions, tracking/depth diagnostics, and sampled device-wide VRAM. Physics and camera warmup occur before the episode clock starts, and every recorded camera pose is checked against the drone state. An interrupted file retains `.partial` and is excluded from training. Evaluation saves `prediction.png`, raw prediction arrays, and RGB/depth/motion errors at 0.5, 1, and 2 seconds, including a last-frame baseline and an action-change sensitivity check.

The audit checks every completed flight for finite sensor values, action/timestamp alignment, quaternion norms, depth validity, and collisions, then saves a file-hash manifest, RGB preview, and flight-coverage plot. Evaluation balances its window budget across held-out flights and samples throughout each flight; each metric retains its source flight and frame offset. Pose occupancy describes the drone's path, not the fraction of room surfaces reconstructed.

## Data and implementation boundaries

- Each flight contains `N` RGB-D/IMU observations and `N-1` motor-command blocks. `actions[t]` has shape `(5, 4)` and drives observation `t` to `t+1`.
- RGB is uint8; depth is metric distance to the image plane with a validity mask. IMU contains ideal body-frame specific acceleration and angular velocity, sampled at each image timestamp.
- State targets use world position, **xyzw** body-to-world quaternion, world linear velocity, and world angular velocity. Ground-truth state is excluded from the encoder inputs.
- The initial drone uses a box collision hull, 0.5 kg mass, explicit inertia, four motor force points, first-order motor lag, and linear/angular drag. These are assumed parameters, not a calibrated real drone.
- The RSSM has 21.4M parameters, categorical stochastic latents, unimix, and balanced KL/free-nats losses. RGB-D reconstruction and a motion head are project adaptations. This is a Dreamer-inspired predictor; actor/critic and Plan2Explore are not implemented yet.
- Evaluation reports velocity, per-frame relative pose, and integrated pose errors. The nominal rigid-body baseline receives the last ground-truth state and known physical parameters; it omits contacts and PhysX solver details. Learned pose increments are anchored at the last ground-truth pose only for evaluation. Motion-head ablations and revisit experiments remain future work.

## Development checks

```bash
python -m pytest -q
```

The CPU test suite covers hover/motor mixing, motor lag, IMU conventions, timestamp/action alignment, flight splitting, gradient flow, action sensitivity, pose composition, nominal dynamics, and offline checkpoint/evaluation plumbing. Synthetic fixtures are test data only.

The [pinned-environment training profile](setup/environment/profile.json) measured **1.89 GiB peak tensor memory / 2.25 GiB reserved** for the 21.4M-parameter model, BF16, batch 16, and sequence length 64. This synthetic forward/backward/Adam benchmark excludes Isaac Sim and does not measure prediction quality. Installed versions and CUDA checks are recorded in [doctor.json](setup/environment/doctor.json).

The [local runtime validation report](setup/environment/validation.json) records:

- A five-second hover with 101 observations, no collisions, and measured wall depth 2.820000 m versus the expected 2.82 m.
- Three eight-second moving flights with 483 observations, no collisions, valid depth throughout, and 50 ms observation intervals. Mean tracking errors were 0.20–0.31 m.
- A controlled floor-contact probe detected a collision and ended the flight at 1.9 seconds, with the final collision flag saved.
- About 21.7 camera frames per wall-clock second during moving-flight collection, excluding simulator startup. Peak **device-wide** GPU usage was 6.78 GiB, including other applications.
- A 10-step training smoke test and finite held-out RGB/depth/motion metrics. This is a pipeline check, not evidence of useful learned predictions.

Local generated artifacts: [flight preview](runs/validation/sweep-preview.png), [checkpoint](runs/validation/smoke01/best.pt), and [prediction preview](runs/validation/evaluation01/prediction.png). The validated flight data is in `data/validation/sweep04`. Earlier diagnostic collections with reset issues are quarantined or incomplete and are excluded from training.

Simulator validation used the documented CLI and script-file launches. One inline `python -c` diagnostic crashed before simulator startup; its cause was not isolated, and the script-file version passed.

## First experiment

1. Validate stable flight and synchronized sensor/action logging in one room.
2. Collect varied trajectories with a conventional controller, including turns, acceleration, braking, altitude changes, and revisits.
3. Split complete flights into training, validation, and test sets.
4. Train the RSSM to reconstruct observations and predict motion.
5. Evaluate predictions on held-out flights without intermediate observation corrections.

The immediate learning milestone is:

> Given one second of camera/IMU history and a future motor-command sequence, predict the next two seconds of RGB-D observations and drone motion.

Compare against visual persistence and nominal physics, and change future commands from the same starting state to verify that the model responds to actions. The pilot dataset is approximately 100 flights of 30–60 seconds, with further collection guided by validation results.

## Pilot results (2026-11-06)

The dataset contains **100 flights × 45 seconds**, 90,100 RGB-D/IMU observations and 90,000 motor-command blocks, occupying 2.07 GiB. All files passed the audit; no collisions or unfinished files remain. Collection took 65.8 minutes at 22.8 camera frames per wall-clock second, with 6.58 GiB peak device-wide GPU usage, including other applications.

The flight split is **80 training / 10 validation / 10 test**. The 21.4M-parameter RSSM trained for 10,000 steps with BF16, batch 16, length 64, and four loading workers in **13.0 minutes**. Peak tensor memory was **1.93 GiB**, with **2.25 GiB reserved**. All 140 validation windows were used for checkpoint selection; the best checkpoint is step **9,800**, validation loss **2.633**.

Test evaluation used 100 windows, ten per test flight, with one second of sensor history followed by two seconds of predictions without observation corrections. Values below are means over the test windows; lower is better.

| Horizon | RSSM RGB MSE | Last-frame RGB MSE | RSSM depth RMSE | Last-frame depth RMSE | RSSM position error |
|---|---:|---:|---:|---:|---:|
| 0.5 s | 0.000390 | 0.004830 | 0.0767 m | 0.2099 m | 0.0137 m |
| 1.0 s | 0.000560 | 0.009423 | 0.0847 m | 0.3361 m | 0.0264 m |
| 2.0 s | 0.000954 | 0.015645 | 0.1027 m | 0.5109 m | 0.0528 m |

At two seconds, nominal physics has 0.000438 m mean position error and is substantially more accurate than learned motion. It receives the exact last simulator state and known physical parameters; the RSSM receives RGB-D, IMU, and actions. Learned relative pose is anchored at the last reference pose only when computing evaluation errors. Predictions respond to changed motor commands, but this sensitivity test does **not** establish that counterfactual predictions are correct.

The drone visited a central 2.2 × 1.8 m region at altitudes 1.1–1.5 m, with speeds up to 0.389 m/s and all heading bins represented. Training and test flights share the same room and scripted trajectory family. The results establish prediction on this distribution, not a complete room map or reliable dynamics under unfamiliar actions. Next, broaden maneuvers and test predictions against simulator branches with changed commands, plus motion-head ablation and revisit checks, before learned exploration.

Local artifacts: [prediction preview](runs/baseline-evaluation/prediction.png), [training curves](runs/baseline/training-curves.png), [baseline comparison](runs/baseline-evaluation/comparison.png), [dataset coverage](runs/pilot100/audit/coverage.png), [experiment summary](runs/pilot100/experiment-summary.json), [test metrics](runs/baseline-evaluation/metrics.json), and [best checkpoint](runs/baseline/best.pt). The [audit manifest](runs/pilot100/audit/audit.json) includes each flight's SHA-256; the source snapshot and hashes are under `runs/pilot100/source` and `runs/pilot100/source-manifest.json`. Generated data, checkpoints, and plots remain local under ignored `data/` and `runs/` directories.

## Roadmap

- [x] Implement the collection, storage, RSSM training, and prediction interfaces.
- [x] Verify CPU tests and measure synthetic training memory on the RTX 5090.
- [x] Finish the pinned simulator installation and validate the new CUDA environment.
- [x] Validate the Isaac Lab room, drone dynamics, and RGB-D/IMU logging.
- [x] Run a short training/checkpoint/held-out evaluation smoke test.
- [x] Collect and inspect the pilot flight dataset.
- [x] Train and evaluate the RSSM prediction baseline.
- [ ] Broaden maneuvers; validate changed-action predictions, motion-head ablation, and revisits.
- [ ] Add learned exploration and compare it with scripted sweeps at equal flight time.
- [ ] Add persistent spatial memory and evaluate revisits.
- [ ] Investigate adaptation to new rooms and changed drone parameters.

The initial model is specific to one room. Unobserved regions remain unknown, and generalization to new rooms requires separate experiments.

## Documentation

- [Simulation experiment plan](SIMULATION_PLAN.md): architecture, data collection, training, evaluation, exploration, and source references.

Setup results and remaining research milestones are recorded in the [simulation plan](SIMULATION_PLAN.md).
