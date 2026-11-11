"""Typed Scaler references, 50 Hz interpolation, and per-shape reference FK."""

import json
from pathlib import Path

import numpy as np
import torch
import mujoco

from .kinematics import ScalerKinematics
from .prepare import sha
from .robot import DT, Robot
from .reference_goal_mode import LEGACY_GOAL, STATIONARY_GOAL, validate_goal_mode
from .reference_goal_padding import stationary_tail_goal
from .rotations import mat_to_6d, quat_slerp, quat_to_mat, so3_log
from .heading_frame_mode import LEGACY_HEADING, validate_heading_frame_mode, heading_function

CONTROL_DT = DT


def gradient(x):
    g = torch.empty_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * DT)
    g[0] = (x[1] - x[0]) / DT
    g[-1] = (x[-1] - x[-2]) / DT
    return g


class MotionLib:
    def __init__(self, pack, device="cpu", role="train", continuous_morphology=False,
                 *, reference_goal_mode=LEGACY_GOAL, heading_frame_mode=LEGACY_HEADING):
        self.reference_goal_mode = validate_goal_mode(reference_goal_mode)
        self.heading_frame_mode = validate_heading_frame_mode(heading_frame_mode)
        self.path = Path(pack).resolve()
        manifest = json.loads((self.path / "manifest.json").read_text())
        self.manifest = manifest
        if (
            manifest["schema"] != "scaler_supermimic_motion_pack_v1"
            or manifest["role"] != role
        ):
            raise ValueError("Split mismatch")
        for name, digest in manifest["files"].items():
            p = (self.path / name).resolve()
            if not p.is_relative_to(self.path) or sha(p) != digest:
                raise ValueError(f"Pack input changed: {name}")
        self.robot = Robot.from_yaml(self.path / manifest["robot"])
        self.models = [
            self.robot.load_model(self.path / v["model"]) for v in manifest["variants"]
        ]
        self.kin = ScalerKinematics(self.models, self.robot, device)
        self.deltas = torch.tensor(
            [v["delta_m"] for v in manifest["variants"]],
            dtype=torch.float32,
            device=device,
        )
        self.continuous_morphology = continuous_morphology
        if continuous_morphology:
            self.kin.enable_continuous(self.deltas)
        self.device = device
        self.names = []
        lengths = []
        states = []
        shapes = []
        for row in manifest["motions"]:
            with np.load(self.path / row["file"], allow_pickle=False) as z:
                p = torch.tensor(z["root_p"], dtype=torch.float32)
                quat = torch.tensor(z["root_quat"], dtype=torch.float32)
                q = torch.tensor(z["joint_pos"], dtype=torch.float32)
                fps = float(z["fps"])
            n = int(np.floor((len(q) - 1) / fps / DT)) + 1
            if n < 40:
                raise ValueError("Reference too short for 32-step imagination")
            t = torch.arange(n) * DT * fps
            i = t.floor().long().clamp(max=len(q) - 2)
            a = (t - i).clamp(0, 1)
            p = p[i] * (1 - a[:, None]) + p[i + 1] * a[:, None]
            q = q[i] * (1 - a[:, None]) + q[i + 1] * a[:, None]
            R = quat_to_mat(quat_slerp(quat[i], quat[i + 1], a))
            v = gradient(p)
            qd = gradient(q)
            w = torch.empty_like(p)
            w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * DT)
            w[0] = so3_log(R[1] @ R[0].T) / DT
            w[-1] = so3_log(R[-1] @ R[-2].T) / DT
            states.append((p, R, v, w, q, qd))
            lengths.append(n)
            self.names.append(row["name"])
            shapes.append(row["shape"])
        self.root_p, self.root_R, self.root_v, self.root_w, self.q, self.qd = [
            torch.cat([s[i] for s in states]).to(device) for i in range(6)
        ]
        self.clip_len = torch.tensor(lengths, device=device)
        self.clip_start = self.clip_len.cumsum(0) - self.clip_len
        self.clip_end = self.clip_start + self.clip_len - 1
        self.clip_shape = torch.tensor(shapes, device=device)
        self.clip_id = torch.repeat_interleave(
            torch.arange(len(lengths), device=device), self.clip_len
        )
        self.shape = self.clip_shape[self.clip_id]
        self.num_frames = sum(lengths)
        with torch.no_grad():
            if continuous_morphology:
                self._prepare_continuous_feet()
            else:
                result = [
                    self.kin.forward(
                        *(s[a : a + 2048] for s in self.ref_state()),
                        self.shape[a : a + 2048],
                    )
                    for a in range(0, self.num_frames, 2048)
                ]
                self.body_p, self.body_R, self.body_v, self.body_w = [
                    torch.cat([r[i] for r in result]) for i in range(4)
                ]
        self.bin_start = torch.cat(
            [
                torch.arange(int(s), int(e), 50, device=device)
                for s, e in zip(self.clip_start, self.clip_end)
            ]
        )
        self.num_bins = len(self.bin_start)
        self.frame_bin = (
            torch.searchsorted(
                self.bin_start, torch.arange(self.num_frames, device=device), right=True
            )
            - 1
        )

    def _prepare_continuous_feet(self):
        """Cache exact foot-height sensitivities, not a bank of sampled bodies."""
        body_ids, geom_p, geom_R, geom_size = [], [], [], []
        for side in ("left", "right"):
            geometries = [m.geom(side + "_foot_col") for m in self.models]
            first = geometries[0]
            if int(first.type[0]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                raise ValueError("Continuous references require the qualified foot boxes")
            for g in geometries[1:]:
                if not all(np.allclose(getattr(g, key), getattr(first, key), atol=1e-8, rtol=0)
                           for key in ("pos", "quat", "size", "bodyid")):
                    raise ValueError("Foot geometry changed across morphology witnesses")
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, first.quat)
            body_ids.append(int(first.bodyid[0]) - 1)
            geom_p.append(first.pos.copy()); geom_R.append(matrix.reshape(3, 3))
            geom_size.append(first.size.copy())
        tensor = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        self.foot_geom_p, self.foot_geom_R, self.foot_geom_size = map(tensor, (geom_p, geom_R, geom_size))
        self.foot_body_ids = body_ids
        self.source_foot_z = torch.empty(self.num_frames, 2, device=self.device)
        self.foot_z_jacobian = torch.empty(self.num_frames, 2, 5, device=self.device)
        for start in range(0, self.num_frames, 2048):
            end = min(start + 2048, self.num_frames)
            p, R, _, _, jac = self.kin.forward(
                *(s[start:end] for s in self.ref_state()), self.shape[start:end],
                all_bodies=True, return_position_jacobian=True,
            )
            feet_R = R[:, body_ids]
            centers = p[:, body_ids] + (feet_R @ self.foot_geom_p[..., None]).squeeze(-1)
            box_R = feet_R @ self.foot_geom_R
            self.source_foot_z[start:end] = centers[..., 2] - (
                box_R[..., 2, :].abs() * self.foot_geom_size
            ).sum(-1)
            self.foot_z_jacobian[start:end] = jac[:, body_ids, 2]

    def root_height_shift(self, f, delta):
        """Preserve source minimum sole height, including jumps and source errors.

        This is angle-preserving kinematic adaptation. It preserves neither
        both-foot contact labels nor horizontal no-slip constraints.
        """
        difference = delta - self.deltas[self.shape[f]]
        target = self.source_foot_z[f] + torch.einsum(
            "...sf,...f->...s", self.foot_z_jacobian[f], difference
        )
        return self.source_foot_z[f].amin(-1) - target.amin(-1)

    def ref_state(self, f=None, delta=None, frame=None):
        s = (self.root_p, self.root_R, self.root_v, self.root_w, self.q, self.qd)
        result = s if f is None else tuple(x[f] for x in s)
        p, R, v, w, q, qd = result
        if delta is not None:
            if f is None or not self.continuous_morphology:
                raise ValueError("Continuous reference adaptation requires frame indices")
            shift = self.root_height_shift(f, delta)
            before = torch.maximum(f - 1, self.clip_start[self.clip_id[f]])
            after = torch.minimum(f + 1, self.clip_end[self.clip_id[f]])
            speed = (self.root_height_shift(after, delta) - self.root_height_shift(before, delta)) / (
                (after - before).clamp_min(1) * DT
            )
            offset = torch.stack((torch.zeros_like(shift), torch.zeros_like(shift), shift), -1)
            velocity = torch.stack((torch.zeros_like(speed), torch.zeros_like(speed), speed), -1)
            p, v = p + offset, v + velocity
        if frame is not None:
            c, s = frame[..., 0].cos(), frame[..., 0].sin()
            zero, one = torch.zeros_like(c), torch.ones_like(c)
            yaw = torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(*c.shape, 3, 3)
            translation = torch.stack((frame[..., 1], frame[..., 2], zero), -1)
            p = (yaw @ p[..., None]).squeeze(-1) + translation
            R = yaw @ R
            v = (yaw @ v[..., None]).squeeze(-1)
            w = (yaw @ w[..., None]).squeeze(-1)
        return p, R, v, w, q, qd

    def reference_bodies(self, f, delta=None, frame=None):
        if delta is None:
            return self.body_p[f], self.body_R[f], self.body_v[f], self.body_w[f]
        return self.kin.forward(*self.ref_state(f, delta, frame), delta=delta)

    def next_frame(self, f):
        return torch.minimum(f + 1, self.clip_end[self.clip_id[f]])

    def goal(self, f, K=10, S=None, delta=None, frame=None):
        end = self.clip_end[self.clip_id[f]]
        requested = f[..., None] + torch.arange(1, K + 1, device=f.device)
        future = torch.minimum(requested, end[..., None])
        target = self.ref_state(
            future, None if delta is None else delta[..., None, :],
            None if frame is None else frame[..., None, :],
        )
        current = self.ref_state(f, delta, frame) if S is None else S
        anchor_p, anchor_R = current[:2]
        Ht = heading_function(self.heading_frame_mode)(anchor_R).transpose(-1, -2).unsqueeze(-3)
        dp = (
            Ht @ (target[0] - anchor_p.unsqueeze(-2)).unsqueeze(-1)
        ).squeeze(-1)
        rot = mat_to_6d(Ht @ target[1])
        Rt = target[1].transpose(-1, -2)
        lv = (Rt @ target[2].unsqueeze(-1)).squeeze(-1)
        av = (Rt @ target[3].unsqueeze(-1)).squeeze(-1)
        goal = torch.cat(
            (dp, rot, lv, av, target[4], target[5] * 0.1), -1
        ).flatten(-2)
        if self.reference_goal_mode == STATIONARY_GOAL:
            return stationary_tail_goal(goal, requested, end, num_joints=target[4].shape[-1])
        return goal
