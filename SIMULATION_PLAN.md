# Aerial World Model: Simulation Experiment Plan

[Project overview](README.md)

## Objective

Train a drone world model that learns a room's visible geometry and appearance, together with how flight commands affect the drone's motion. After exploration, the model should predict future camera observations and motion under a proposed action sequence.

The initial approach is a **DreamerV3-style recurrent state-space model (RSSM)** for prediction, followed by **Plan2Explore-style autonomous exploration**.

Learning prediction rules across many flights and remembering a room during one flight are different capabilities. The first experiment tests prediction. Persistent spatial memory and adaptation to new rooms come afterward.

This document is a simulation plan. It does not assume that the system has been implemented or trained.

## 0. Conda environment and pinned software

Use a dedicated Conda environment named **`airwm`**, with the released Isaac Lab wheel installed into that environment. Conda manages Python; `uv pip` resolves and installs the simulator packages into the same interpreter. Our drone environment and world-model code will live in this project.

### Selected versions

| Component | Planned version / setting |
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

The existing workspace uses Python 3.11 in a separate environment. Create `airwm` with Python 3.12 rather than upgrading that existing environment. The planned RSSM implementation will use PyTorch; using the original JAX-based Dreamer implementation later would warrant a separate training environment.

### Host prerequisites

- Confirm the target RTX GPU is supported, with at least 16 GB VRAM and 32 GB system RAM; profile additional memory needed for training.
- Verify `nvidia-smi` works on the target machine. Use an NVIDIA driver supported by Isaac Sim 6.2 and the CUDA 13.0 PyTorch build. The CUDA stack's Linux floor is `580.65.06`; RTX rendering can impose additional driver requirements, so that number alone is not a compatibility guarantee.
- Use GLIBC 2.35 or newer for the Linux Isaac Sim wheels. This workspace reports Ubuntu 24.04 x86_64 and GLIBC 2.39, but GPU access still needs validation on the execution machine.

Use the [Isaac Lab 3.0 installation guide](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/installation) and [Isaac Sim 6.2 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html) for host checks. Standard CUDA-enabled wheels supply their runtime dependencies; a separate CUDA Toolkit is not part of this initial setup.

### Create and install

The following commands are **planned setup instructions, not commands already executed**. Run them on the target Linux machine in a shell with Conda initialized. Use a new `airwm` environment; inspect any existing environment with that name before reusing it.

```bash
conda create --name airwm python=3.12 pip -y
conda activate airwm
python -m pip install --upgrade pip uv

python -m uv pip install --python "$CONDA_PREFIX/bin/python" \
  "isaaclab[isaacsim]==3.0.0" \
  "isaacsim[all,extscache]==6.2.0.0" \
  "torch==2.11.0" "torchvision==0.26.0" \
  --overrides https://raw.githubusercontent.com/isaac-sim/IsaacLab/v3.0.0/tools/wheel_builder/uv-overrides.txt \
  --extra-index-url https://pypi.nvidia.com \
  --torch-backend cu130 \
  --index-strategy unsafe-best-match
```

This adapts the official released-wheel installation to Conda. The release-specific overrides reconcile Isaac Sim and Isaac Lab dependency pins. Keep them when resolving this stack. Explicit `--python` ensures installation targets the activated Conda interpreter; no nested virtual environment is needed. [Installation instructions](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/installation#installing-the-isaacsim-extra), [uv interpreter selection](https://docs.astral.sh/uv/pip/environments/)

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

These files will record the tested environment; they do not exist yet. Preserve the install indexes and override file alongside these package snapshots when reproducing the setup. Revalidate camera output and GPU execution after any dependency change.

## 1. Initial simulation experiment

Use **Isaac Lab with Isaac Sim/PhysX and RTX rendering** as the primary simulation stack. The planned machine has an NVIDIA RTX GPU with at least 16 GB VRAM. This choice prioritizes furnished-room assets, RGB-D rendering, and parallel data collection while retaining the same RSSM learning method.

Use the pinned **Isaac Lab 3.0.0 / Isaac Sim 6.2.0** stack and Conda workflow above. The exact GPU, driver, and available memory still require validation; the hardware description is a planning assumption, not a completed compatibility check.

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

Use a quadcopter asset as a starting point, then explicitly implement or verify per-motor thrust, reaction torque, force application points, motor response, and drag. The older [Isaac Lab v2.3.2 quadcopter example](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/quadcopter_env.py) illustrates a collective-thrust/body-moment interface; it is a historical reference, not our installation target. Inspect the 3.0.0 task and actuator APIs before implementing the per-motor model.

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

First validate one simulated drone hovering and following a short trajectory in the furnished room, with synchronized RGB-D, IMU, and motor-command logging. Record rendering throughput and peak VRAM on the target RTX machine.

Provide **one second of camera/IMU history** and a future motor-command sequence, then predict the next **two seconds of RGB-D and drone motion**.

Use held-out flights, action-sensitivity checks, and comparisons with simple baselines to determine whether the learned world model is ready to guide exploration.

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
