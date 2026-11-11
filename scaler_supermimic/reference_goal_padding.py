"""Stationary velocity channels for a goal's synthetic post-clip tail.

This is not yet wired into MotionLib. The caller must already clamp pose and
joint-position targets to the clip's last recorded frame. The idea follows
zita-ch/supermimic_vibed commit 2e8b623 (MIT); this helper is an original
implementation for Scaler's batched goal layout.
"""

import torch


def stationary_tail_goal(
    flat_goal: torch.Tensor,
    unclamped_frames: torch.Tensor,
    clip_end: torch.Tensor,
    *,
    num_joints: int = 24,
) -> torch.Tensor:
    """Return a new goal with only synthetic tail velocities set to zero.

    Shapes are exactly ``[..., K * (15 + 2*N)]``, ``[..., K]``, and
    ``[...]``; broadcasting batch dimensions is deliberately disallowed.
    Each frame is ``[position3, rotation6, linear_velocity3,
    angular_velocity3, joint_positionN, joint_velocityN]``. Only requested
    frames strictly greater than ``clip_end`` have velocity channels cleared.
    The actual final frame, positions, rotations and joint positions are kept.

    Goals must be strided float16/bfloat16/float32/float64 tensors. Indices
    must be strided int32/int64 tensors with matching dtype and device.
    Validation reads tensor metadata only: no host scalar conversion, device
    synchronization, value reduction or mutation. Reference-index validity and
    finite input goals remain the caller's responsibility.
    """
    if type(num_joints) is not int or num_joints < 1:
        raise ValueError("num_joints must be a positive integer, not bool")
    values = (flat_goal, unclamped_frames, clip_end)
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("goal, unclamped_frames and clip_end must be tensors")
    if any(value.layout != torch.strided for value in values):
        raise TypeError("goal and frame indices must use strided tensors")
    if flat_goal.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("goal must use a supported real floating dtype")
    if unclamped_frames.dtype not in (torch.int32, torch.int64):
        raise TypeError("frame indices must use int32 or int64, not bool/float")
    if clip_end.dtype != unclamped_frames.dtype:
        raise TypeError("clip_end and unclamped_frames must share integer dtype")
    if any(value.device != flat_goal.device for value in values[1:]):
        raise ValueError("goal and frame indices must be on the same device")
    if flat_goal.ndim < 1 or unclamped_frames.ndim < 1:
        raise ValueError("goal and unclamped_frames need a final feature/frame axis")
    batch_shape = unclamped_frames.shape[:-1]
    horizon = unclamped_frames.shape[-1]
    width = 15 + 2 * num_joints
    if horizon < 1 or tuple(flat_goal.shape) != (*batch_shape, horizon * width):
        raise ValueError("goal shape must be exactly [..., K * (15 + 2*num_joints)]")
    if clip_end.shape != batch_shape:
        raise ValueError("clip_end shape must exactly match the goal batch shape")

    framed_goal = flat_goal.reshape(*batch_shape, horizon, width)
    channel = torch.arange(width, device=flat_goal.device)
    velocity_channel = ((channel >= 9) & (channel < 15)) | (channel >= 15 + num_joints)
    tail = unclamped_frames > clip_end[..., None]
    clear = tail[..., None] & velocity_channel
    # where preserves identity gradients on retained channels and avoids 0*NaN
    # in channels intentionally replaced; it does not certify input finiteness.
    return torch.where(clear, torch.zeros_like(framed_goal), framed_goal).reshape_as(flat_goal)
