"""Paper reward/observations with explicit Scaler shape conditioning (metres / .05)."""

import torch

from .reference_goal_mode import LEGACY_GOAL, validate_goal_mode
from . import dr_context
from .robot import DT
from .rotations import mat_to_6d, mat_to_quat, quat_to_mat, rot_angle_sq
from .heading_frame_mode import LEGACY_HEADING, validate_heading_frame_mode, heading_function
from .reward_alignment_mode import LEGACY_REWARD, ANCHOR_REWARD, validate_reward_alignment_mode

NUM_J = 24
STATE_DIM = 61
INV_DIM = 61
OBS_DIM = 83  # q24, qd24, omega3, gravity3, previous action24, shape5
XI_DIM = 33  # friction1, calibration24, torso COM3, shape5


def flatten_state(S):
    return torch.cat((S[0], mat_to_quat(S[1]), *S[2:]), -1)


def unflatten_state(s):
    return (
        s[..., :3],
        quat_to_mat(s[..., 3:7]),
        s[..., 7:10],
        s[..., 10:13],
        s[..., 13:37],
        s[..., 37:61],
    )


def invariant_features(S, *, heading_frame_mode=LEGACY_HEADING):
    p, R, v, w, q, qd = S
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(heading_function(heading_frame_mode)(R).transpose(-1, -2) @ R),
            (Rt @ v.unsqueeze(-1)).squeeze(-1),
            (Rt @ w.unsqueeze(-1)).squeeze(-1),
            q,
            qd,
        ),
        -1,
    )


def normalize_xi(x):
    return dr_context.normalize_context(x)


def aligned_body_reward_targets(p, R, root_p, root_R, body_p, body_R):
    """Differentiable XY/relative-yaw reanchoring, independent of heading mode.

    At relative projected-forward radius <1e-6 only, define identity yaw.
    Select safe raw atan2 coordinates BEFORE atan2 so its unused pole cannot
    create NaN backward values. This extension is discontinuous at the cutoff;
    it is not a global yaw chart or a guarantee for arbitrary nonfinite input.
    """
    relative = R @ root_R.transpose(-1, -2)
    x, y = relative[..., 0, 0], relative[..., 1, 0]
    pole = (x.square() + y.square()) < 1e-12
    safe_x = torch.where(pole, torch.ones_like(x), x)
    safe_y = torch.where(pole, torch.zeros_like(y), y)
    angle = torch.atan2(safe_y, safe_x)
    c, s = torch.cos(angle), torch.sin(angle)
    z, one = torch.zeros_like(c), torch.ones_like(c)
    yaw = torch.stack((c, -s, z, s, c, z, z, z, one), -1).reshape(*c.shape, 3, 3)
    anchor = torch.cat((p[..., :2], root_p[..., 2:3]), -1)
    position = anchor.unsqueeze(-2) + (
        yaw.unsqueeze(-3) @ (body_p - root_p.unsqueeze(-2)).unsqueeze(-1)
    ).squeeze(-1)
    rotation = yaw.unsqueeze(-3) @ body_R
    return position, rotation, pole


class MDP:
    def __init__(self, motion, dr_config=None, *, heading_frame_mode=None,
                 reward_alignment_mode=LEGACY_REWARD, tracking_std_scale=1.0):
        self.motion = motion
        self.reward_alignment_mode = validate_reward_alignment_mode(reward_alignment_mode)
        motion_heading = validate_heading_frame_mode(getattr(motion, "heading_frame_mode", LEGACY_HEADING))
        self.heading_frame_mode = motion_heading if heading_frame_mode is None else validate_heading_frame_mode(heading_frame_mode)
        if self.heading_frame_mode != motion_heading:
            raise ValueError("Motion and MDP heading modes differ")
        self.reference_goal_mode = validate_goal_mode(getattr(motion, "reference_goal_mode", LEGACY_GOAL))
        self.kin = motion.kin
        self.dr_config = dr_config
        if dr_config is not None and not isinstance(dr_config, dict):
            raise ValueError("dr_config must be a configuration dictionary or None")
        self.xi_dim = XI_DIM if dr_config is None else dr_context.XI_DIM
        self.continuous_morphology = bool(getattr(motion, "continuous_morphology", False))
        if dr_config is not None and not self.continuous_morphology:
            raise ValueError("Flat DR requires continuous morphology reference adaptation")
        device = motion.device
        self.default_q = torch.tensor(
            motion.robot.default_q, dtype=torch.float32, device=device
        )
        self.noise = torch.tensor(
            [0.01] * 24 + [0.5] * 24 + [0.2] * 3 + [0.05] * 3 + [0.0] * 29,
            device=device,
        )
        if dr_config is not None:
            scales = dr_config["nuisance"]["observation_noise_scales"]
            self.noise = torch.tensor(
                [scales["joint_pos"]] * 24 + [scales["joint_vel"]] * 24
                + [scales["ang_vel"]] * 3 + [scales["projected_gravity"]] * 3
                + [0.0] * 29, device=device,
            )
        self.soft_lo = self.kin.q_lo
        self.soft_hi = (
            self.kin.q_hi
        )  # actual hardware limits; no unexplained 0.9 scaling
        from .partymimic_recipe import tracking_stds
        self.tracking_std_scale = float(tracking_std_scale)
        self.tracking_std = tracking_stds(self.tracking_std_scale)
        self.canonical_tracking_std = tracking_stds(1.0)

    def normalize_context(self, xi):
        if xi.shape[-1] != self.xi_dim:
            raise ValueError("Context dimension differs from this MDP")
        return dr_context.normalize_context(xi)

    def advance_context(self, xi, action):
        if xi.shape[-1] != self.xi_dim:
            raise ValueError("Context dimension differs from this MDP")
        return dr_context.advance_context(xi, action)

    def morphology_delta(self, xi):
        if not self.continuous_morphology:
            return None
        if xi is None or xi.shape[-1] != self.xi_dim:
            raise ValueError("Continuous reference requires this sample's actual context")
        return xi[..., 28:33]

    def goal(self, f, S=None, xi=None):
        if getattr(self.motion, "heading_frame_mode", LEGACY_HEADING) != self.heading_frame_mode:
            raise ValueError("Heading mode changed after MDP construction")
        if getattr(self.motion, "reference_goal_mode", LEGACY_GOAL) != self.reference_goal_mode:
            raise ValueError("Reference-goal mode changed after MDP construction")
        delta = self.morphology_delta(xi)
        if delta is None:
            return self.motion.goal(f, S=S)
        return self.motion.goal(f, S=S, delta=delta, frame=self.reference_frame(xi))

    def reference_frame(self, xi):
        return xi[..., 290:293] if self.xi_dim == dr_context.XI_DIM else None

    def obs(self, S, a_prev, xi, f):
        _p, R, _v, w, q, qd = S
        return torch.cat(
            (
                q - self.default_q - xi[..., 1:25],
                qd,
                (R.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1),
                -R[..., 2, :],
                a_prev,
                xi[..., 28:33] / 0.05,
            ),
            -1,
        )

    def add_noise(self, o):
        return o + (2 * torch.rand_like(o) - 1) * self.noise

    def reward(self, S, f, a, previous, terms=False, *, xi=None, canonical_evaluation=False):
        if type(canonical_evaluation) is not bool:
            raise ValueError("canonical_evaluation must be boolean")
        mode = validate_reward_alignment_mode(self.reward_alignment_mode)
        m = self.motion
        p, R, _v, _w, q, _qd = S
        delta = self.morphology_delta(xi)
        if delta is None:
            bp, bR, bv, bw = self.kin.forward(*S, m.shape[f])
            targets = m.body_p[f], m.body_R[f], m.body_v[f], m.body_w[f]
            root_p, root_R = m.root_p[f], m.root_R[f]
        else:
            bp, bR, bv, bw = self.kin.forward(*S, delta=delta)
            frame = self.reference_frame(xi)
            targets = m.reference_bodies(f, delta=delta, frame=frame)
            root_p, root_R = m.ref_state(f, delta=delta, frame=frame)[:2]
        errors = {
            "body_position": ((bp - targets[0]) ** 2).sum(-1).mean(-1),
            "body_rotation": rot_angle_sq(bR, targets[1]).mean(-1),
            "body_velocity": ((bv - targets[2]) ** 2).sum(-1).mean(-1),
            "body_angular_velocity": ((bw - targets[3]) ** 2).sum(-1).mean(-1),
            "root_position": ((p - root_p) ** 2).sum(-1),
            "root_rotation": rot_angle_sq(R, root_R),
        }
        # These six world errors are the immutable evaluation/diagnostic keys.
        # terms=True does not change reward selection. Canonical evaluation does.
        tracking_errors = errors.values()
        auxiliary = {}
        if mode == ANCHOR_REWARD and not canonical_evaluation:
            aligned_p, aligned_R, pole = aligned_body_reward_targets(
                p, R, root_p, root_R, targets[0], targets[1]
            )
            aligned_position = ((bp - aligned_p) ** 2).sum(-1).mean(-1)
            aligned_rotation = rot_angle_sq(bR, aligned_R).mean(-1)
            tracking_errors = (
                aligned_position, aligned_rotation, errors["body_velocity"],
                errors["body_angular_velocity"], errors["root_position"], errors["root_rotation"],
            )
            if terms:
                auxiliary = {
                    "aligned_body_reward_position": aligned_position,
                    "aligned_body_reward_rotation": aligned_rotation,
                    "aligned_body_reward_identity_yaw_fallback": pole,
                }
        scales = self.canonical_tracking_std if canonical_evaluation else self.tracking_std
        weights = (1.0, 1.0, 1.0, 1.0, 0.5, 0.5)
        tracking = sum(
            wt * torch.exp(-e / sd**2)
            for e, sd, wt in zip(tracking_errors, scales, weights)
        )
        limit = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        rate = ((a - previous) ** 2).sum(-1)
        reward = DT * (tracking - 0.1 * rate - 2 * limit)
        if auxiliary:
            auxiliary["aligned_training_reward"] = reward
        return (
            (reward, {**errors, "action_rate": rate, "joint_limit": limit, **auxiliary})
            if terms
            else reward
        )

    def termination(self, S, f, soft=False, *, xi=None):
        delta = self.morphology_delta(xi)
        root_p, root_R = (self.motion.root_p[f], self.motion.root_R[f]) if delta is None else self.motion.ref_state(f, delta=delta, frame=self.reference_frame(xi))[:2]
        dz = (S[0][..., 2] - root_p[..., 2]).abs()
        tilt = (S[1][..., 2, 2] - root_R[..., 2, 2]).abs()
        if not soft:
            return (dz > 0.25) | (tilt > 0.8)
        return 1 - (1 - torch.sigmoid((dz - 0.25) / 0.02)) * (
            1 - torch.sigmoid((tilt - 0.8) / 0.04)
        )

    def critic_features(self, S, f, xi):
        p, R, _v, _w, q, _qd = S
        m = self.motion
        delta = self.morphology_delta(xi)
        root_p, root_R = (m.root_p[f], m.root_R[f]) if delta is None else m.ref_state(f, delta=delta, frame=self.reference_frame(xi))[:2]
        Ht = heading_function(self.heading_frame_mode)(R).transpose(-1, -2)
        return torch.cat(
            (
                invariant_features(S, heading_frame_mode=self.heading_frame_mode),
                (Ht @ (root_p - p).unsqueeze(-1)).squeeze(-1),
                mat_to_6d(R.transpose(-1, -2) @ root_R),
                m.q[f] - q,
                self.normalize_context(xi),
                self.goal(f, S=S, xi=xi),
            ),
            -1,
        )
