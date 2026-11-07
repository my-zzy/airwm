# Aerial World Model: Simulation Experiment Plan

[Project overview](README.md)

## Objective

Train a drone world model that learns a room's visible geometry and appearance, together with how flight commands affect the drone's motion. After exploration, the model should predict future camera observations and motion under a proposed action sequence.

The initial approach is a **DreamerV3-style recurrent state-space model (RSSM)** for prediction, followed by **Plan2Explore-style autonomous exploration**.

Learning prediction rules across many flights and remembering a room during one flight are different capabilities. The first experiment tests prediction. Persistent spatial memory and adaptation to new rooms come afterward.

This document defines the experiment and its milestones. Implementation began on 2026-11-06; see the status below and [README.md](README.md) for commands.

## Implementation status

- Installed the `airwm` Conda environment with Python 3.12.15, Isaac Lab 3.0.0, Isaac Sim 6.2.0, and PyTorch 2.11.0/CUDA 13.0. Exact versions and GPU execution pass [the environment check](setup/environment/doctor.json). The [runtime validation report](setup/environment/validation.json) records successful PhysX/RTX flight collection and a training/evaluation smoke test.
- Confirmed the local NVIDIA GeForce RTX 5090 (32 GB VRAM), driver 580.178.04, and working CUDA in the new environment. Hover and moving-flight RGB-D/IMU checks passed with this existing driver; no driver change was needed.
- Added a single-drone collector with a procedural furnished room, motor/controller model, synchronized RGB-D/IMU recording, and HDF5 flight storage.
- Added the categorical RSSM, offline training, whole-flight splits, checkpoints, open-loop prediction evaluation, matched simulator branching, and randomized maneuver collection. All 24 CPU unit/integration tests pass. A 10-step smoke run first verified the pipeline; the full experiments are recorded in Section 4 below.
- [Synthetic training profile](setup/environment/profile.json) with pinned PyTorch 2.11.0/CUDA 13.0: 21,427,536 parameters, BF16, batch 16, length 64; peak tensor memory **1.89 GiB** and PyTorch reserved memory **2.25 GiB**. This includes forward/backward/Adam state and excludes the simulator. The earlier 8–12 GB training estimate was a conservative planning budget, not a measured requirement.
- Validated a five-second hover (101 observations) and three eight-second moving flights (483 observations), with no collisions and valid depth throughout. Fixed first-frame reset synchronization by initializing PhysX/Fabric before episode resets, refreshing camera poses, and capturing warmup frames before time zero. Recorded camera positions agree with the drone state within 0.000001 m; initial wall-depth checks agree within 0.0002 m.
- Moving-flight collection ran at about 21.7 camera frames per wall-clock second, excluding startup. Peak device-wide GPU use was 6.78 GiB, including other applications. The first launch compiled shaders; subsequent validation launches reused that cache.
- A controlled descent into the floor confirmed contact sensing and early episode termination at 1.9 seconds. The saved flight ends with its collision flag set. CLI and script-file launches passed; one inline `python -c` diagnostic crashed before startup, with the cause unisolated.
- Completed Section 3's first pilot: 100 flights × 45 seconds, 90,100 observations, no collisions, all data checks passed. Trained 10,000 steps on the 80/10/10 flight split; checkpoint selection used all 140 validation windows. Evaluated 100 windows across all ten test flights, evenly sampled within each flight. The first Section 4 prediction comparison passes the last-frame baseline.
- Completed the paired action-response test: eight new histories and 96 simulator branches, with exact control-state/depth replay and no collisions. The fixed checkpoint almost ignores the tested action changes. Improving action-response learning, ablations, and revisits remain necessary before autonomous exploration; persistent memory is still a later milestone.
- Completed the controlled data-only follow-up on 2026-11-07: 100 varied flights, all 2,200 pulses verified, and 10,000 training steps with unchanged settings. Both checkpoints were compared on identical held-out flights and 96 branches from eight new histories. The new model improves maneuver prediction relative to the pilot but does not beat two-second RGB-D persistence; sweep prediction worsens. Pitch/roll action response remains negligible and yaw is inconsistent. The action-response gate remains unmet.

## 0. Conda environment and pinned software

Use a dedicated Conda environment named **`airwm`**, with the released Isaac Lab wheel installed into that environment. Conda manages Python; `uv pip` resolves and installs the simulator packages into the same interpreter. Our drone environment and world-model code will live in this project.

### Selected versions

| Component | Installed version / selected setting |
|---|---|
| Target platform | Ubuntu 24.04, Linux x86_64 |
| Conda environment | `airwm` |
| Python | 3.12 |
| Isaac Lab | **3.0.0**, corresponding to release tag `v3.0.0` |
| Isaac Sim | **6.2.0**, Python package version `6.2.0.0` |
| PyTorch / TorchVision | `2.11.0` / `0.26.0`, CUDA 13.0 wheels |
| NVIDIA Warp | `1.17.0`, supplied by the Isaac Lab dependency set |
| Physics backend | `isaacsim_physx` |
| Renderer | `isaacsim_rtx` |

Version selection checked on **2026-11-06**. Isaac Lab 3.0.0 is a general-availability release and publishes this Python / Isaac Sim / PyTorch combination. Pin the release rather than tracking `main`, a moving release branch, or an early-access build. [Release notes](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0), [version metadata](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/pyproject.toml)

The existing workspace uses Python 3.11 in a separate environment. The new `airwm` environment uses Python 3.12 and preserves that existing environment. The RSSM implementation uses PyTorch; using the original JAX-based Dreamer implementation later would warrant a separate training environment.

### Host prerequisites

- The local GPU is an RTX 5090 with 32 GB VRAM; the machine reports about 60 GiB system RAM. Single-drone rendering and training memory have been measured above; reprofile when increasing scene, camera, model, or batch size.
- Verify `nvidia-smi` works on the target machine. Use an NVIDIA driver supported by Isaac Sim 6.2 and the CUDA 13.0 PyTorch build. The CUDA stack's Linux floor is `580.65.06`; RTX rendering can impose additional driver requirements, so that number alone is not a compatibility guarantee.
- Use GLIBC 2.35 or newer for the Linux Isaac Sim wheels. This workstation runs Ubuntu 24.04 x86_64 with GLIBC 2.39, and GPU execution is validated. Repeat the checks on another machine.

Use the [Isaac Lab 3.0 installation guide](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/installation) and [Isaac Sim 6.2 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html) for host checks. Standard CUDA-enabled wheels supply their runtime dependencies; a separate CUDA Toolkit is not part of this initial setup.

### Create and install

The pinned stack is installed on this machine. These commands reproduce it on Linux x86_64. On this workstation, activate the existing environment; do not recreate it. The installer uses the full resolved version/hash lock and matching dependency overrides. On another machine, create the dedicated environment first.

```bash
conda create --name airwm python=3.12 pip -y
conda activate airwm
AIRWM_DIRECT=1 bash setup/install.sh
bash setup/run.sh airwm doctor --output setup/environment/doctor.json
```

This adapts the official released-wheel installation to Conda. The release-specific overrides reconcile Isaac Sim and Isaac Lab dependency pins. Keep them when resolving this stack. The [resolved overrides](setup/environment/isaaclab-resolved-overrides.txt) pin the applicable official overrides to the [full lock](setup/environment/airwm-python-linux-64.lock), allowing hash enforcement without relaxing version pins. Explicit `--python` targets the activated Conda interpreter; no nested virtual environment is needed. [Installation instructions](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/installation#installing-the-isaacsim-extra), [uv interpreter selection](https://docs.astral.sh/uv/pip/environments/)

`AIRWM_DIRECT=1` removes proxy variables for the installer subprocess only; omit it to use the configured proxy. For slow PyPI downloads, set `AIRWM_PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple`. Official PyPI remains a fallback, NVIDIA retains its own index, and required hashes prevent the mirror from changing accepted artifacts. The current installation used direct connections first, proxy retries, and the mirror; downloaded large wheels were verified against official SHA-256 hashes. No global network or package-index settings were changed. [Tsinghua mirror documentation](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)

Use the pip-distributed Isaac Sim package with this Conda workflow. The separately downloaded Isaac Sim application uses its bundled interpreter and is a different installation path. Select `physics=isaacsim_physx` and `renderer=isaacsim_rtx` in the future task configuration because Isaac Lab 3.0's default backend is Newton.

### Verify before collecting data

Check the interpreter, installed distributions, and GPU execution:

```bash
conda activate airwm
nvidia-smi
python - <<'PY'
import sys
from importlib.metadata import version
import torch

assert sys.version_info[:2] == (3, 12), sys.version
print("Python:", sys.executable)
expected = {
    "isaaclab": "3.0.0",
    "isaacsim": "6.2.0.0",
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "warp-lang": "1.17.0",
}
for package, wanted in expected.items():
    installed = version(package)
    print(package, installed)
    assert installed.split("+")[0] == wanted, (package, installed)
assert torch.version.cuda == "13.0", torch.version.cuda
assert torch.cuda.is_available(), "CUDA is not available in this environment"
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA operation:", torch.ones(4, device="cuda").sum().item())
PY
isaaclab --help
```

On a workstation with a display, launch `isaacsim` once and handle its first-run license prompt, then close it. The first startup may download extensions. A successful CUDA check is only the first check: before dataset collection, validate a one-environment RGB-D camera run using the selected physics and rendering backends, including valid depth, timestamps, and peak VRAM.

For noninteractive collection, the runtime requires its license acknowledgement via `OMNI_KIT_ACCEPT_EULA=YES`, for example `OMNI_KIT_ACCEPT_EULA=YES bash setup/run.sh airwm collect ...`. The completed pilot used this process-local setting.

After validation, record the resolved environment from the project root:

```bash
mkdir -p setup/environment
conda env export --from-history > setup/environment/airwm-conda.yml
conda list --explicit > setup/environment/airwm-conda-linux-64.txt
python -m uv pip freeze --python "$CONDA_PREFIX/bin/python" \
  > setup/environment/airwm-python-linux-64.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv \
  > setup/environment/gpu.csv
```

Environment snapshots, the version/hash lock, and diagnostic reports are saved under `setup/environment/`. The raw freeze is a machine snapshot and contains local wheel/editable paths; use the portable lock and installer for reproduction. Revalidate camera output and GPU execution after any dependency change.

## 1. Initial simulation experiment

Use **Isaac Lab with Isaac Sim/PhysX and RTX rendering** as the primary simulation stack. The planned machine has an NVIDIA RTX GPU with at least 16 GB VRAM. This choice prioritizes furnished-room assets, RGB-D rendering, and parallel data collection while retaining the same RSSM learning method.

Use the pinned **Isaac Lab 3.0.0 / Isaac Sim 6.2.0** stack and Conda workflow above. GPU execution and the single-drone physics/camera pipeline are validated in the procedural room. Larger scenes and parallel environments require separate profiling.

| Item | Initial choice |
|---|---|
| Simulation stack | Isaac Lab 3.0.0 + Isaac Sim 6.2.0/PhysX + RTX rendering |
| Environment | One static furnished room, using USD assets |
| Parallel environments | Start with one; trial 4 and then 8 after profiling |
| Drone | Quadrotor with per-motor thrust/torque, motor response, gravity, drag, and collisions |
| Observations | RGB, depth, and IMU |
| Camera resolution | 64 × 64 initially |
| Physics rate | 200 Hz |
| Motor-control rate | 100 Hz |
| World-model update rate | 20 Hz |
| Training data | Repeated flights with varied trajectories and starting poses |
| Prediction horizons | 0.5, 1, and 2 seconds |

These are starting configurations to validate, not established optimal settings.

A conventional controller generates the initial flights. Record the actual motor commands and preserve all five command samples between consecutive camera frames. The model can then associate motion with the commands that caused it.

Synchronize RGB-D, IMU, motor commands, and simulator state using simulation timestamps. The initial collection controller may use simulator state to generate stable flights. For world-model learning, simulator state is reserved for auxiliary training targets and evaluation; it is not an input to the main sensor-based world model.

### Rendering and memory budget

Use Isaac Lab's batched camera pipeline for RGB and depth. Render at the 20 Hz observation cadence rather than at every physics step. Verify metric depth conventions, camera intrinsics/extrinsics, and image/action timestamp alignment before collecting the dataset. See the [camera documentation](https://isaac-sim.github.io/IsaacLab/main/source/concepts/sensors/camera.html).

Start with a single environment and measure rendering throughput, peak VRAM, and valid RGB-D output. Trial 4 and then 8 environments only while retaining memory headroom and correct observations. These are profiling points, not guaranteed capacities; 16 GB VRAM does not establish how many camera environments and training batches will fit together.

For the first offline experiment, collect data and train the RSSM in separate runs, releasing simulator GPU allocations before training. Profile their combined footprint when introducing online exploration and model updates. Increase image resolution only after the 64 × 64 baseline works.

### Drone dynamics implementation

The initial implementation uses a 0.5 kg rigid body with a box collision hull and specified inertia. Four rotor force points, reaction torques, motor response, and drag produce the body wrench applied through Isaac Lab 3.0. Replace the assumed geometry and parameters with a calibrated quadrotor asset in a later fidelity experiment. The older [Isaac Lab v2.3.2 quadcopter example](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/quadcopter_env.py) illustrates a collective-thrust/body-moment interface; it is a historical reference, not our installation target.

Validate hover, commanded translation and rotation, motor-command timing, and collisions before gathering learning data. Document assumed physical parameters and aerodynamic approximations.

MuJoCo remains a fallback for a lightweight diagnostic experiment if RTX setup is blocked. The primary plan does not require implementing both simulator backends.

## 2. Neural world-model architecture

Use an RSSM with recurrent memory and a stochastic latent state. During observation, an encoder updates that state from sensor measurements. During imagination, the dynamics model advances it using actions alone. This is the central [Dreamer mechanism](https://arxiv.org/html/2301.04104v2).

| Module | Purpose |
|---|---|
| RGB-D encoder | Extract appearance and geometry features |
| IMU encoder | Extract motion information |
| RSSM | Combine history, observations, and actions |
| RGB decoder | Reconstruct and predict camera appearance |
| Depth decoder | Reconstruct and predict visible geometry |
| Motion head | Predict velocity, angular velocity, and relative pose change |

The depth and motion heads are proposed adaptations to the Dreamer-style baseline.

Simulator state supplies targets for the auxiliary motion head and evaluation. The main model receives sensor observations and actions, without ground-truth pose as an input.

## 3. Data collection and training

Start with approximately **100 flights of 30–60 seconds** as a pilot dataset. Increase collection according to validation results rather than assuming that this amount is sufficient.

Flights should include:

- Room sweeps from different starting poses.
- Altitude changes and turns.
- Acceleration, braking, and lateral motion.
- Revisits to previously observed areas.

Train on sequences using an objective that combines:

- RGB reconstruction.
- Depth reconstruction.
- Dreamer's latent prediction and representation losses.
- Auxiliary motion prediction.

Keep Dreamer's loss-balancing mechanisms initially. The depth and motion losses require separate weighting and ablation.

Split the dataset by complete flights into training, validation, and test sets. Adjacent frames from one flight must stay in the same split.

### Controlled varied-maneuver experiment

The implemented `collect --trajectory maneuvers` behavior policy samples a waypoint
and heading every four seconds, returns to the initial position periodically, and
injects an independent motor-axis pulse every two seconds. Axis order, sign,
amplitude, and pulse duration are seeded. Collective offsets are 0.005/0.010/0.020,
roll and pitch offsets 0.0005/0.0010/0.0015, and yaw offsets 0.005/0.010/0.015,
with both signs. Units are normalized desired rotor speed. Pulse durations are
0.15/0.20/0.25 seconds. Feedback is held at its pre-pulse command during a pulse
and resumes afterwards; all applied motor commands are logged at 100 Hz.

Waypoints stay within x/y ±1.1 m and altitude 1.1–1.9 m. A privileged collection
supervisor cancels an active pulse and steers toward the room centre outside
x/y ±1.65 m, altitude 0.9–2.3 m, or 30 degrees of tilt. This envelope is a recovery
rule, not a collision guarantee. Recovery data and collisions are retained;
collisions terminate the episode. The learner never receives simulator state as
an observation. Pulse schedules and actual application counts are stored in each
flight's metadata and included in the audit.

The first comparison uses 100 × 45-second independent flights (seed 302), matching
the pilot's collection duration and training budget. Keep model.py, loss,
optimizer, 10,000 steps, batch 16, length 64, and seed 0 unchanged. Select the
checkpoint by whole-validation loss. The three seed-301 smoke flights are excluded.
Record data, code, checkpoint, and environment hashes. Source code for the model,
training, data loader, physical configuration, and motor/controller dynamics was
verified identical to the pilot before starting this experiment.

Evaluate both checkpoints on identical held-out maneuver flights and eight new
seed-401 sweep histories with matched simulator interventions. The existing
seed-101 bank stays a fixed diagnostic. Compare signed action-effect gain and
error with the zero-effect baseline, and compare forecasts supplied changed
versus unchanged commands. Report shared-history counts rather than treating
all branches as independent samples. No architecture/loss tuning uses the new
bank before this comparison; future tuning requires another reserved bank.

This experiment tests the combined collection-policy change. It does not isolate
which data component helps, and one training seed does not establish repeatability.
If action prediction still fails, test action normalization and explicit multi-step
RGB-D/motion objectives separately. Training multiple branches from a common
history is another planned intervention; those branches must be split by source
history and use enough context plus future frames for the training sequence length.

## 4. Prediction evaluation

The first demonstration should show actual and predicted RGB-D sequences side by side, with motion errors underneath.

| Test | What it establishes |
|---|---|
| Predict held-out flight trajectories | Generalization beyond training sequences |
| Compare against repeating the last frame | Improvement over visual persistence |
| Compare motion against nominal physics | Whether learned dynamics add value |
| Change future commands from the same starting state | Whether predictions respond correctly to actions |
| Remove the motion-prediction head | Whether auxiliary supervision helps |
| Turn away, then revisit an area | How well recurrent memory preserves earlier observations |

Evaluate predictions without feeding intermediate observations back into the model. Repeated observation corrections can otherwise hide rollout drift.

Report RGB and depth prediction errors alongside position, orientation, velocity, and angular-velocity errors at each prediction horizon. Record any privileged state inputs used by comparison baselines so their results are interpreted appropriately.

### First pilot result (2026-11-06)

The [full results and artifact links](README.md#pilot-results-2026-11-06) record 100 complete 45-second flights (2.07 GiB), an 80/10/10 split, and 10,000 training steps. Collection took 65.8 minutes with 6.58 GiB peak device-wide GPU use. Training took 13.0 minutes with 1.93 GiB peak tensor memory / 2.25 GiB reserved. The best checkpoint was step 9,800, selected by validation loss 2.633.

With one second of history and a two-second open-loop horizon, mean test RGB MSE was **0.000954** versus **0.015645** for the last-frame baseline (93.9% lower). Mean depth RMSE was **0.1027 m** versus **0.5109 m** (79.9% lower). Both image metrics beat persistence on all 100 sampled test windows. Integrated position error averaged **0.0528 m**; nominal physics, initialized with exact simulator state and known parameters, achieved **0.000438 m**. The learned model does not improve on that privileged physics baseline.

These results are limited to the same room and gentle scripted sweeps: a central 2.2 × 1.8 m region, 1.1–1.5 m altitude, and maximum speed 0.389 m/s. All heading bins are covered, but motor commands occupy a narrow range. The initial changed-command test established sensitivity only; its +0.15 perturbation had no paired simulator ground truth. The follow-up below supplies that comparison. Motion-head ablations and revisit retention remain untested. Do not treat this pilot as a complete room map or sufficient evidence for autonomous exploration.

### Paired action-response result (2026-11-06)

The [protocol, reproduction commands, and artifacts](README.md#action-response-test-2026-11-06) cover eight new seed-101 histories and 96 two-second branches. Each branch restores body pose/velocity and rotor lag, applies a 0.25-second command pulse, and then follows the original recorded commands without controller feedback. Control/repeat physics states and depth match exactly. RGB repeats have mean MSE 0.000108 from rendering variation. All horizons completed without collisions.

The step-9,800 pilot checkpoint substantially underpredicts action effects. At two seconds, a collective +0.020 pulse changes simulated position by 0.233 m on average, but predicted position changes by only 0.000402 m. Pitch/roll pulses change position by about 0.534 m, versus about 0.000001 m predicted. A positive yaw pulse changes rotation by 0.0904 rad, versus 0.0000026 rad predicted. These are changes relative to the matched unchanged-action control, averaged over eight histories. Supplying changed versus unchanged commands produces almost the same forecast.

Pitch and roll commands stay within the pilot's per-motor ranges, although their joint sequences and resulting motion differ from the scripted training policy. This is evidence against trusting the current checkpoint to select unfamiliar actions. The exact contributions of limited excitation, action scaling, and training objectives are hypotheses to test, not established causes. Preserve this branch bank as a diagnostic and reserve new histories for evaluating any tuned model.

### Varied-data result (2026-11-07)

The [full result and artifacts](README.md#varied-maneuver-experiment-2026-11-07)
record 100 × 45-second seed-302 maneuver flights, 90,100 observations (2.04 GiB),
and 2,200 independently randomized axis pulses. All pulses were verified against
recorded states/commands; no collisions, cancellations or recovery interventions
occurred. About 0.17% of actual command components reached actuator bounds;
none of the added pulses required clipping. Maximum speed was 2.10 m/s, angular
speed 3.01 rad/s, and altitude 1.06–1.92 m. Collection took 75.3 minutes with 7.35 GiB
peak device-wide VRAM. Training took 13.0 minutes with 1.93 GiB peak tensor memory
and 2.25 GiB reserved. Validation selected step 9,700 (loss 7.4913).

On 100 identical maneuver test windows, two-second RGB MSE improved from the
pilot's 0.017039 to 0.011076, and depth RMSE from 0.6971 to 0.5100 m. Both still
lose to persistence (0.009963 and 0.4113 m). On the original sweep test windows,
the new model's RGB/depth/position errors worsen to 0.015111 / 0.5620 m / 0.5893 m,
versus 0.000954 / 0.1027 m / 0.0528 m for the pilot. Neither result establishes
reliable dynamics under unfamiliar commands.

The eight new seed-401 histories produced 96 complete, collision-free branches.
Control/repeat states and depth matched exactly; mean repeat RGB MSE was 0.0000924.
At two seconds, pitch/roll pulses changed actual position by about 0.535 m, but the
new model predicted changes below 0.000008 m. Collective +0.020 produced a predicted
change of 0.0271 m versus 0.2329 m actual, with mean signed gain −0.0104 and effect
error worse than predicting no response. Positive/negative yaw gains were 0.1871 /
0.0617 (1 is correct), with substantial variation across histories, including a
wrong-sign negative-yaw case. The original fixed diagnostic also remains unsolved.

At this compute budget and seed, broader data alone is insufficient. This experiment
does not identify a sole failure cause or establish that varied data cannot help.
The [input-rounding diagnostic](runs/maneuvers01/action-precision.json) shows BF16
rounding erases 82.5% of changed components in the old bank's small positive
pitch/roll pulses, while larger failed interventions survive rounding. Test
normalization before BF16 projection next, using training-only statistics saved
with the checkpoint and preserving old-checkpoint behavior. Then test explicit
multi-step prediction and motion weighting separately. Retain the same data/split
for these ablations; investigate mixed sweep/maneuver training as a separate data
experiment if needed. Reserve fresh evaluation histories, because both existing
branch banks are now inspected. Motion-head and dedicated revisit tests remain.

## 5. Autonomous exploration

After prediction is useful, introduce **Plan2Explore-style exploration**. Train an ensemble of predictors and reward actions whose predicted outcomes produce disagreement. This disagreement approximates where additional experience may improve the model. See [Plan2Explore](https://proceedings.mlr.press/v119/sekar20a.html).

For this aerial adaptation, combine predicted information gain with collision and excessive-control penalties. Ensemble disagreement is a heuristic: measure whether it actually improves coverage and held-out prediction accuracy.

The learning loop is:

1. Collect flights.
2. Update the world model and predictor ensemble.
3. Improve the exploration policy using imagined trajectories.
4. Execute the policy in simulation to collect more informative flights.
5. Repeat while retaining earlier experience in the replay dataset.

Compare learned exploration against scripted room sweeps using the same flight-time budget.

The exploration policy's action representation must match the world model's recorded motor-command sequences. Any move to a higher-level command interface requires explicitly modeling the controller's response.

## 6. Persistent room memory

Once the RSSM baseline works, add a spatial map that stores geometry and features outside the current camera view. Test whether it improves predictions after long turns, occlusion, and revisits.

Unobserved regions should remain unknown. Recurrent memory alone does not guarantee a complete or geometrically consistent room map.

The first model is specific to the training room. Adapting to a new room during exploration is a subsequent experiment requiring separate evaluation.

## Immediate milestone

The pilot, matched simulator branching, and controlled varied-data experiment are complete. Broader data improves maneuver prediction relative to the pilot, but does not solve action response and reduces sweep prediction quality. Section 4's action-response requirement is not yet met.

Next, implement checkpointed action normalization using training-only statistics, and compare it with the unchanged-data-only model on the same split and training budget. Then test explicit multi-step RGB-D/motion losses separately. Keep both checkpoints and both branch banks fixed; use new histories for final evaluation after tuning. Run the motion-head ablation and revisit tests as well. Section 5's learned exploration remains gated on reliable action-response predictions.

## References

- [DreamerV3: Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)
- [Dreamer RSSM and training method](https://arxiv.org/html/2301.04104v2)
- [Planning to Explore via Self-Supervised World Models](https://proceedings.mlr.press/v119/sekar20a.html)
- [Isaac Lab 3.0.0 release](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0)
- [Isaac Lab 3.0 installation](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/installation)
- [Isaac Lab 3.0.0 dependency overrides](https://raw.githubusercontent.com/isaac-sim/IsaacLab/v3.0.0/tools/wheel_builder/uv-overrides.txt)
- [Isaac Lab camera documentation](https://isaac-sim.github.io/IsaacLab/main/source/concepts/sensors/camera.html)
- [Isaac Lab renderer options](https://isaac-sim.github.io/IsaacLab/main/source/concepts/renderers.html)
- [Isaac Sim 6.2 hardware requirements](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html)
- [Isaac Sim environment assets](https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_environments.html)
- [Isaac Lab v2.3.2 quadcopter example](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/quadcopter_env.py)
- [MuJoCo Python documentation (fallback)](https://mujoco.readthedocs.io/en/stable/python.html)
