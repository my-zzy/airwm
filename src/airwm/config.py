"""Physical units are SI; quaternions use Isaac Lab 3's xyzw convention."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FlightConfig:
    physics_hz: int = 200
    control_hz: int = 100
    camera_hz: int = 20
    image_size: int = 64
    mass: float = 0.5
    arm_length: float = 0.17  # centre to rotor, X configuration
    max_rotor_thrust: float = 3.0  # N at normalized motor speed 1
    yaw_moment_ratio: float = 0.018  # reaction torque / thrust, metres
    motor_time_constant: float = 0.025  # seconds, normalized rotor speed
    linear_drag: float = 0.12  # N / (m/s)
    angular_drag: float = 0.002  # Nm / (rad/s)
    gravity: float = 9.81
    depth_far: float = 12.0

    def __post_init__(self):
        if min(self.physics_hz, self.control_hz, self.camera_hz) <= 0:
            raise ValueError("Rates must be positive")
        if self.physics_hz % self.control_hz or self.control_hz % self.camera_hz:
            raise ValueError("Physics, control, and camera rates must divide exactly")
        if self.mass <= 0 or self.motor_time_constant <= 0:
            raise ValueError("Mass and motor response time must be positive")

    @property
    def controls_per_frame(self):
        return self.control_hz // self.camera_hz

    @property
    def physics_per_control(self):
        return self.physics_hz // self.control_hz

    def to_dict(self):
        return asdict(self)
