"""Single-room Isaac Lab 3 / PhysX / RTX collector. Imported only by collect.

The initial drone is a single rigid body with a conservative box collision hull,
specified mass/inertia and four rotor force application points. The room uses
local USD primitives, so collection does not depend on remote asset servers.
"""

import json
import math
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from airwm.config import FlightConfig
from airwm.data import EpisodeWriter
from airwm.dynamics import Motors, body_vector, controller, imu_sample, rotation_matrix


def _tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


def create_room(sim_utils):
    def box(name, size, pos, color):
        cfg = sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )
        cfg.func("/World/Room/" + name, cfg, translation=pos)

    sim_utils.create_prim("/World/Room", "Xform")
    box("Floor", (6.0, 6.0, 0.1), (0.0, 0.0, -0.05), (0.35, 0.28, 0.20))
    box("Ceiling", (6.0, 6.0, 0.1), (0.0, 0.0, 3.05), (0.85, 0.85, 0.85))
    box("East", (0.1, 6.0, 3.0), (3.05, 0.0, 1.5), (0.75, 0.82, 0.88))
    box("West", (0.1, 6.0, 3.0), (-3.05, 0.0, 1.5), (0.85, 0.72, 0.65))
    box("North", (6.0, 0.1, 3.0), (0.0, 3.05, 1.5), (0.75, 0.80, 0.65))
    box("South", (6.0, 0.1, 3.0), (0.0, -3.05, 1.5), (0.82, 0.75, 0.85))
    box("Table", (1.2, 0.8, 0.12), (1.8, 1.8, 0.8), (0.28, 0.15, 0.08))
    for i, (x, y) in enumerate([(1.3, 1.5), (2.3, 1.5), (1.3, 2.1), (2.3, 2.1)]):
        box(f"TableLeg{i}", (0.1, 0.1, 0.75), (x, y, 0.375), (0.15, 0.12, 0.1))
    box("Cabinet", (0.5, 1.4, 1.8), (-2.7, -1.6, 0.9), (0.2, 0.35, 0.48))
    box("Sofa", (1.8, 0.7, 0.6), (0.0, -2.5, 0.3), (0.5, 0.15, 0.12))
    box("SofaBack", (1.8, 0.15, 0.5), (0.0, -2.8, 0.8), (0.5, 0.15, 0.12))
    box("WallArt", (0.04, 1.0, 0.7), (2.98, -1.2, 1.8), (0.12, 0.4, 0.18))
    for i in range(4):
        box(
            f"Book{i}",
            (0.15, 0.35, 0.3),
            (1.5 + 0.18 * i, 1.8, 1.01),
            ((i + 1) / 5, 0.2, 1 - (i + 1) / 5),
        )
    light = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.9, 0.9, 0.9))
    light.func("/World/Light", light)
    # Interior lights are necessary because the room has a closed ceiling.
    for i, position in enumerate(((1.2, 0.8, 2.7), (-1.2, -0.8, 2.7))):
        light = sim_utils.SphereLightCfg(radius=0.12, intensity=1800.0, color=(1.0, 0.95, 0.9))
        light.func(f"/World/CeilingLight{i}", light, translation=position)


def target_at(t, phase, hover, device):
    if hover:
        pos = [0.0, 0.0, 1.3]
        yaw = 0.0
    else:
        pos = [
            1.1 * math.sin(0.3 * t + phase),
            0.9 * math.sin(0.23 * t + phase),
            1.3 + 0.2 * math.sin(0.37 * t + phase),
        ]
        # Sweep through all headings over a 30–60 s flight, including revisits.
        yaw = phase + 0.2 * t + 0.25 * math.sin(0.25 * t + phase)
    return torch.tensor([pos], device=device), torch.tensor([yaw], device=device)


def collect(args):
    from airwm.diagnostics import GpuMonitor

    plans = None
    if getattr(args, "command", None) == "branch-collect":
        from airwm.branching import prepare_branches

        plans = prepare_branches(args)
        args.episodes = len(plans)
        args.seconds = args.horizon / FlightConfig().camera_hz
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=plans is None)
    monitor = GpuMonitor(int(args.device.split(":")[-1]) if ":" in args.device else 0)
    monitor.start()
    try:
        _collect(args, monitor, plans)
    finally:
        monitor.stop(root / f"gpu_memory_s{args.seed:04d}.json")


def _collect(args, monitor, plans=None):
    # Delay all Isaac imports: training and data tools never load Kit.
    import isaaclab.sim as sim_utils
    from isaaclab.app import launch_simulation
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sensors.camera import CameraCfg
    from isaaclab.utils import instantiate
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    from isaaclab_physx.sensors.contact_sensor import ContactSensorCfg

    cfg = FlightConfig()
    if args.episodes < 1 or args.seconds < 1 / cfg.camera_hz:
        raise ValueError("Request at least one episode with one camera interval")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sim_cfg = sim_utils.SimulationCfg(
        device=args.device,
        dt=1 / cfg.physics_hz,
        physics=PhysxCfg(),
        render_interval=cfg.physics_hz // cfg.camera_hz,
        gravity=(0.0, 0.0, -cfg.gravity),
    )
    launcher = {
        "device": args.device,
        "headless": not args.gui,
        "enable_cameras": True,
        "physics": "isaacsim_physx",
        "visualizer": "kit" if args.gui else "none",
    }
    with launch_simulation(sim_cfg, launcher), ExitStack() as cleanup:
        # Kit may terminate the process during launcher shutdown. Save the
        # device-wide memory report before leaving the simulation context.
        cleanup.callback(monitor.stop, root / f"gpu_memory_s{args.seed:04d}.json")
        sim = sim_utils.SimulationContext(sim_cfg)
        create_room(sim_utils)
        drone = RigidObject(
            RigidObjectCfg(
                prim_path="/World/Drone",
                spawn=sim_utils.CuboidCfg(
                    size=(0.32, 0.32, 0.07),
                    rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                    mass_props=sim_utils.MassCfg(mass=cfg.mass),
                    collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                    activate_contact_sensors=True,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.12)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.3)),
            )
        )
        # Use only our explicit aerodynamic drag, avoiding hidden PhysX damping.
        import omni.usd
        from pxr import PhysxSchema

        drone_prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/Drone")
        body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(drone_prim)
        body_api.CreateLinearDampingAttr(0.0)
        body_api.CreateAngularDampingAttr(0.0)
        camera = instantiate(
            CameraCfg(
                prim_path="/World/Drone/Camera",
                height=cfg.image_size,
                width=cfg.image_size,
                update_period=0.0,
                update_latest_camera_pose=True,
                data_types=["rgb", "distance_to_image_plane"],
                renderer_cfg=IsaacRtxRendererCfg(),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.18, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"
                ),
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=12.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.05, cfg.depth_far),
                ),
            )
        )
        contacts = instantiate(
            ContactSensorCfg(prim_path="/World/Drone", update_period=0.0, history_length=1)
        )
        sim.reset()
        # Approximate 0.5 kg quad inertia (kg m^2), not identified from real hardware.
        inertia = torch.diag(torch.tensor([0.003, 0.003, 0.005], device=sim.device)).reshape(
            1, 1, 9
        )
        drone.set_inertias_index(inertias=inertia)
        sim.set_camera_view([2.6, 2.6, 2.6], [0.0, 0.0, 1.0])
        # Initialize PhysX/Fabric's first dynamic transform publication before
        # any episode teleport. This unrecorded step is discarded by the pose
        # and velocity reset below, before the episode clock starts at zero.
        sim.step(render=False)
        drone.update(1 / cfg.physics_hz)
        motors = Motors(cfg, sim.device)
        started = time.perf_counter()
        reports = []
        for episode in range(args.episodes):
            plan = plans[episode] if plans is not None else None
            phase = (
                plan["metadata"]["source_phase"] if plan else float(rng.uniform(-math.pi, math.pi))
            )
            pos, yaw = target_at(0, phase, args.trajectory == "hover", sim.device)
            pose = torch.cat(
                (
                    pos,
                    torch.zeros(1, 2, device=sim.device),
                    torch.sin(yaw / 2)[:, None],
                    torch.cos(yaw / 2)[:, None],
                ),
                -1,
            )
            velocity = torch.zeros(1, 6, device=sim.device)
            if plan:
                initial = torch.as_tensor(plan["initial_state"], device=sim.device)[None]
                pose, velocity = initial[:, :7], initial[:, 7:]
                replay_actions = torch.as_tensor(plan["actions"], device=sim.device)
            drone.write_root_pose_to_sim_index(root_pose=pose)
            drone.write_root_velocity_to_sim_index(root_velocity=velocity)
            drone.reset()
            # Publish teleported physics transforms before resetting camera views.
            sim.forward()
            camera.reset()
            contacts.reset()
            motors.reset()
            if plan:
                motors.speed.copy_(torch.as_tensor(plan["motor_speed"], device=sim.device)[None])
            drone.update(0.0)
            for _ in range(6):
                sim.render()
                # In Lab 3, camera reads drive capture; sim.render() alone does
                # not warm a headless camera or clear its previous-pose image.
                camera.update(0.0, force_recompute=True)
            state = _tensor(drone.data.root_state_w).clone()
            imu = imu_sample(
                state[:, 3:7],
                state[:, 7:10],
                state[:, 7:10],
                state[:, 10:13],
                1 / cfg.physics_hz,
            )
            if plan:
                # This is the last pre-branch IMU sample from the common history.
                imu = torch.as_tensor(plan["initial_imu"], device=sim.device)[None]
            filename = root / (
                plan["filename"] if plan else f"flight_s{args.seed:04d}_{episode:04d}.h5"
            )
            metadata = {
                "seed": args.seed,
                "episode": episode,
                "phase": phase,
                "trajectory": args.trajectory,
                "physics": "isaacsim_physx",
                "renderer": "isaacsim_rtx",
                "room": "procedural_6x6x3_v1",
                "camera_parent_offset_xyz": [0.18, 0.0, 0.0],
                "camera_forward": "body +X; up +Z",
                "depth": "distance to image plane in metres; invalid pixels masked",
                "imu": "ideal, latest physics-step sample at camera timestamp",
                "inertia_kg_m2": [0.003, 0.003, 0.005],
                "camera_intrinsic_matrix": _tensor(camera.data.intrinsic_matrices)[0]
                .cpu()
                .tolist(),
            }
            if plan:
                metadata.update(plan["metadata"])
            frame_errors = []
            collision_count = 0
            valid_fractions = []
            rgb_contrasts = []
            center_depths = []
            camera_position_errors = []

            def snapshot(timestamp, collided):
                expected_camera_pos = state[:, :3] + 0.18 * rotation_matrix(state[:, 3:7])[:, :, 0]
                camera_error = float(
                    (_tensor(camera.data.pos_w) - expected_camera_pos).norm(dim=-1).max()
                )
                camera_position_errors.append(camera_error)
                if camera_error > 0.001:
                    raise RuntimeError(
                        f"Camera pose and drone state differ by {camera_error:.6f} m "
                        f"at t={timestamp:.3f} s: camera={_tensor(camera.data.pos_w).tolist()}, "
                        f"expected={expected_camera_pos.tolist()}"
                    )
                rgb = _tensor(camera.data.output["rgb"])[0, :, :, :3].cpu().numpy().astype(np.uint8)
                depth = (
                    _tensor(camera.data.output["distance_to_image_plane"])[0]
                    .squeeze(-1)
                    .cpu()
                    .numpy()
                )
                valid = np.isfinite(depth) & (depth > 0) & (depth < cfg.depth_far)
                if not valid.any():
                    raise RuntimeError("Camera produced no valid metric depth; recording aborted")
                valid_fractions.append(float(valid.mean()))
                rgb_contrasts.append(float(rgb.std()))
                center = depth[
                    cfg.image_size // 2 - 2 : cfg.image_size // 2 + 2,
                    cfg.image_size // 2 - 2 : cfg.image_size // 2 + 2,
                ]
                center_depths.append(float(np.median(center)))
                if timestamp == 0.0:
                    if rgb_contrasts[-1] < 1.0:
                        raise RuntimeError(
                            "Initial RGB image is blank; inspect lighting/render initialization"
                        )
                    if args.trajectory == "hover" and abs(center_depths[-1] - 2.82) > 0.1:
                        raise RuntimeError(
                            "Hover camera depth disagrees with the known front wall distance (2.82 m)"
                        )
                return dict(
                    rgb=rgb,
                    depth=np.where(valid, depth, 0).astype(np.float32),
                    depth_valid=valid,
                    imu=imu[0].cpu().numpy().astype(np.float32),
                    state=state[0].cpu().numpy().astype(np.float32),
                    time=np.float64(timestamp),
                    collision=np.bool_(collided),
                )

            with EpisodeWriter(filename, cfg, metadata) as writer:
                writer.append(snapshot(0.0, False))
                frames = round(args.seconds * cfg.camera_hz)
                for frame in range(frames):
                    actions = []
                    collided = False
                    for control in range(cfg.controls_per_frame):
                        t = frame / cfg.camera_hz + control / cfg.control_hz
                        if plan:
                            command = replay_actions[frame, control][None]
                        else:
                            target, yaw = target_at(
                                t, phase, args.trajectory == "hover", sim.device
                            )
                            command = controller(
                                motors,
                                state[:, :3],
                                state[:, 3:7],
                                state[:, 7:10],
                                state[:, 10:13],
                                target,
                                yaw,
                            )
                        actions.append(command[0].cpu().numpy().copy())
                        for _ in range(cfg.physics_per_control):
                            old_velocity = state[:, 7:10].clone()
                            force, torque = motors.step(command, 1 / cfg.physics_hz)
                            force -= cfg.linear_drag * body_vector(state[:, 3:7], state[:, 7:10])
                            torque -= cfg.angular_drag * body_vector(state[:, 3:7], state[:, 10:13])
                            drone.instantaneous_wrench_composer.add_forces_and_torques(
                                forces=force[:, None, :],
                                torques=torque[:, None, :],
                                is_global=False,
                            )
                            drone.write_data_to_sim()
                            sim.step(render=False)
                            drone.update(1 / cfg.physics_hz)
                            contacts.update(1 / cfg.physics_hz)
                            state = _tensor(drone.data.root_state_w).clone()
                            imu = imu_sample(
                                state[:, 3:7],
                                state[:, 7:10],
                                old_velocity,
                                state[:, 10:13],
                                1 / cfg.physics_hz,
                            )
                            collided = collided or bool(
                                (_tensor(contacts.data.net_forces_w).norm(dim=-1) > 0.1).any()
                            )
                    sim.render()
                    camera.update(1 / cfg.camera_hz, force_recompute=True)
                    writer.append(
                        snapshot((frame + 1) / cfg.camera_hz, collided),
                        np.stack(actions),
                    )
                    if plan:
                        target_at_frame = torch.as_tensor(
                            plan["reference_states"][frame + 1, :3], device=sim.device
                        )[None]
                    else:
                        target_at_frame, _ = target_at(
                            (frame + 1) / cfg.camera_hz,
                            phase,
                            args.trajectory == "hover",
                            sim.device,
                        )
                    frame_errors.append(float((state[:, :3] - target_at_frame).norm()))
                    if collided:
                        collision_count += 1
                        print(f"Ending {filename.name}: contact detected", flush=True)
                        break
                    if not sim.is_running():
                        raise RuntimeError("Simulator closed during collection")
                writer.file.attrs["termination"] = "collision" if collision_count else "time_limit"
            report = {
                "path": str(filename),
                "frames": writer.count,
                "collision_frames": collision_count,
                "mean_tracking_error_m": float(np.mean(frame_errors)),
                "max_tracking_error_m": float(np.max(frame_errors)),
                "mean_valid_depth_fraction": float(np.mean(valid_fractions)),
                "minimum_rgb_std_uint8": float(np.min(rgb_contrasts)),
                "initial_center_depth_m": center_depths[0],
                "maximum_camera_position_error_m": max(camera_position_errors),
            }
            if plan:
                report["variant"] = plan["metadata"]["variant"]
                report["source"] = plan["metadata"]["source"]
                report["mean_reference_deviation_m"] = report.pop("mean_tracking_error_m")
                report["max_reference_deviation_m"] = report.pop("max_tracking_error_m")
            reports.append(report)
            print(json.dumps(report), flush=True)
        summary = {
            "episodes": reports,
            "wall_seconds": time.perf_counter() - started,
            "recorded_camera_frames": sum(r["frames"] for r in reports),
        }
        summary["camera_frames_per_wall_second"] = (
            summary["recorded_camera_frames"] / summary["wall_seconds"]
        )
        (root / f"collection_s{args.seed:04d}.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)
