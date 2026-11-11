"""Versioned raw physics context for the optional continuous Flat DR path.

The legacy 33-value prefix is unchanged. The extension stores effective mass
ratios, multiplier samples, material coefficients, delay seconds, the command
held before this transition, and this transition's world-frame velocity push.
It is privileged dynamics/critic input, not an additional policy observation.
"""

import torch

LEGACY_XI_DIM = 33
XI_DIM = 293
SCHEMA = "scaler_supermimic_flat_context_v1"
SLICES = {
    "body_mass_ratio": slice(33, 68),
    "kp_scale": slice(68, 92),
    "kd_scale": slice(92, 116),
    "joint_friction_scale": slice(116, 140),
    "armature_scale": slice(140, 164),
    "dynamic_friction": slice(164, 210),
    "restitution": slice(210, 256),
    "delay_seconds": slice(256, 260),
    "previous_action": slice(260, 284),
    "push_delta_velocity": slice(284, 290),
    "reference_frame": slice(290, 293),
}


def pack_context(legacy, *, body_mass_ratio, kp_scale, kd_scale,
                 joint_friction_scale, armature_scale, dynamic_friction,
                 restitution, delay_seconds, previous_action=None,
                 push_delta_velocity=None, reference_frame=None):
    """Pack actual applied physics, with exact batch/field shapes (no broadcasting).

    Body order includes the 35 MuJoCo bodies (the massless world's ratio is 1);
    geom order includes all 46 geoms. Joint order is the 24 controlled joints.
    Delay order is legs, feet, shoulders, arms. The three COM offsets in the
    legacy prefix are shared by base and torso. Calibration is zero in Flat DR.
    reference_frame is episode-fixed [yaw radians, translation x/y metres],
    applied to the complete clip, not a per-transition physical disturbance.
    Call this at sampling/reset; hot-path evolution uses ``advance_context``.
    """
    if legacy.shape[-1] != LEGACY_XI_DIM or not legacy.is_floating_point():
        raise ValueError("Expected floating legacy context with 33 values")
    if not bool(torch.isfinite(legacy).all()) or bool((legacy[..., 1:25] != 0).any()):
        raise ValueError("Flat context requires finite values and zero calibration")
    supplied = locals()
    fields = [legacy]
    for name, sl in SLICES.items():
        value = supplied[name]
        shape = legacy.shape[:-1] + (sl.stop - sl.start,)
        value = legacy.new_zeros(shape) if value is None else torch.as_tensor(
            value, device=legacy.device, dtype=legacy.dtype
        )
        if value.shape != shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid {name} shape or nonfinite value")
        if name == "body_mass_ratio" or name.endswith("_scale"):
            if bool((value <= 0).any()):
                raise ValueError(f"{name} must be positive")
        if name in ("dynamic_friction", "restitution") and bool((value < 0).any()):
            raise ValueError(f"{name} must be nonnegative")
        if name == "restitution" and bool((value > 1).any()):
            raise ValueError("Restitution coefficient must not exceed one")
        if name == "delay_seconds":
            valid = (value[..., None] - value.new_tensor([0.0, 0.005, 0.010])).abs().amin(-1)
            if bool((valid > 1e-8).any()):
                raise ValueError("Delay must be one of 0/5/10 ms")
        fields.append(value)
    return torch.cat(fields, -1)


def normalize_context(x):
    """Fixed unit transforms; the model's RunningNorm remains unchanged."""
    if x.shape[-1] not in (LEGACY_XI_DIM, XI_DIM):
        raise ValueError("Unsupported physics context dimension")
    prefix = torch.cat(((x[..., :1] - 0.75) / 0.45,
                        x[..., 1:25] / 0.01,
                        x[..., 25:28] / 0.05,
                        x[..., 28:33] / 0.05), -1)
    if x.shape[-1] == LEGACY_XI_DIM:
        return prefix
    return torch.cat((prefix,
                      x[..., 33:68] - 1.0,
                      (x[..., 68:164] - 1.0) / 0.2,
                      (x[..., 164:210] - 0.75) / 0.45,
                      x[..., 210:256] / 0.5,
                      x[..., 256:260] / 0.01,
                      x[..., 260:284],
                      x[..., 284:290],
                      x[..., 290:291] / torch.pi,
                      x[..., 291:293]), -1)


def advance_context(x, action):
    """Next imagined transition holds this command; future pushes are unknown.

    No morphology/material/PD sample is redrawn. The zero push is the declared
    unforced imagined rollout, not a prediction of the random push schedule.
    Keep the action's gradient, since delayed dynamics depend on that command.
    """
    if x.shape[-1] == LEGACY_XI_DIM:
        return x
    if x.shape[-1] != XI_DIM or action.shape != x.shape[:-1] + (24,):
        raise ValueError("Invalid context/action shape")
    return torch.cat((x[..., :260], action, torch.zeros_like(x[..., 284:290]),
                      x[..., 290:293]), -1)
