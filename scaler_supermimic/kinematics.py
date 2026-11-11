"""Differentiable Scaler FK, including fixed length links and nonzero hinge anchors."""

import mujoco
import numpy as np
import torch

from .robot import TRACKED
from .rotations import axis_angle_to_mat, quat_to_mat


class ScalerKinematics:
    def __init__(self, models, robot, device="cpu"):
        m = models[0]
        self.models = models
        self.device = device
        self.names = [m.body(b).name for b in range(1, m.nbody)]
        self.parent = [int(m.body_parentid[b]) - 1 for b in range(1, m.nbody)]
        self.tracked_idx = [self.names.index(n) for n in TRACKED]
        self.joint_idx = []
        axes = []
        anchors = []
        for b in range(1, m.nbody):
            if m.body_jntnum[b] > 1:
                raise ValueError("Multiple joints per body unsupported")
            if m.body_jntnum[b] == 0:
                j = -1
                axis = [0, 0, 1]
                anchor = [0, 0, 0]
            else:
                k = m.body_jntadr[b]
                if m.jnt_type[k] == mujoco.mjtJoint.mjJNT_FREE:
                    j = -2
                elif m.jnt_type[k] == mujoco.mjtJoint.mjJNT_HINGE:
                    j = robot.names.index(m.joint(k).name)
                else:
                    raise ValueError("Morphology must be baked; unexpected joint")
                axis = m.jnt_axis[k]
                anchor = m.jnt_pos[k]
                if j >= 0 and abs(m.qpos0[m.jnt_qposadr[k]]) > 1e-10:
                    raise ValueError("Nonzero hinge reference unsupported")
            self.joint_idx.append(j)
            axes.append(axis)
            anchors.append(anchor)
        for other in models:
            if [
                other.body(b).name for b in range(1, other.nbody)
            ] != self.names or not np.array_equal(other.body_parentid, m.body_parentid):
                raise ValueError("Morphology topology differs")
            for name in robot.names:
                if not np.allclose(other.joint(name).axis, m.joint(name).axis):
                    raise ValueError("Morphology changed joint axes")
                if not np.allclose(other.joint(name).pos, m.joint(name).pos, atol=1e-9, rtol=0):
                    raise ValueError("Morphology changed hinge anchors")
        tensor = lambda x: torch.tensor(
            np.asarray(x), dtype=torch.float32, device=device
        )
        self.off_p = tensor([x.body_pos[1:] for x in models])
        self.off_R = quat_to_mat(tensor([x.body_quat[1:] for x in models]))
        self.axes = tensor(axes)
        self.anchors = tensor(anchors)
        self.q_lo = tensor([m.joint(n).range[0] for n in robot.names])
        self.q_hi = tensor([m.joint(n).range[1] for n in robot.names])

    def enable_continuous(self, deltas):
        """Recover the exact affine baked-link offsets; reject a nonlinear bank.

        Length changes translate fixed link origins; hinge axes and local body
        rotations stay unchanged. This uses the qualified models as witnesses,
        and never selects the nearest discrete morphology.
        """
        d = torch.as_tensor(deltas, device=self.off_p.device, dtype=self.off_p.dtype)
        design = torch.cat((torch.ones_like(d[:, :1]), d), -1)
        if int(torch.linalg.matrix_rank(design)) != 6:
            raise ValueError("Continuous FK needs six independent affine witnesses")
        coeff = torch.linalg.lstsq(design.double(), self.off_p.flatten(1).double()).solution
        error = (design.double() @ coeff - self.off_p.flatten(1).double()).abs().max()
        if float(error) > 2e-7 or not torch.allclose(
            self.off_R, self.off_R[:1].expand_as(self.off_R), atol=2e-6, rtol=0
        ):
            raise ValueError("Qualified morphologies do not share affine link kinematics")
        self.continuous_offsets = coeff.to(self.off_p.dtype).reshape(6, *self.off_p.shape[1:])

    def offsets(self, delta):
        if not hasattr(self, "continuous_offsets"):
            raise ValueError("Continuous morphology FK was not configured")
        return self.continuous_offsets[0] + torch.einsum(
            "...f,fbi->...bi", delta, self.continuous_offsets[1:]
        )

    def forward(
        self, p, R, v, w, q, qd, shape=None, all_bodies=False, delta=None,
        return_position_jacobian=False,
    ):
        offsets = self.off_p[shape] if delta is None else self.offsets(delta)
        rotations = self.off_R[shape] if delta is None else self.off_R[0]
        P = []
        Rs = []
        V = []
        W = []
        JP = []
        for i, par in enumerate(self.parent):
            j = self.joint_idx[i]
            if j == -2:
                pi, Ri, vi, wi = p, R, v, w
                if return_position_jacobian:
                    ji = p.new_zeros(*p.shape[:-1], 3, 5)
            else:
                if par < 0:
                    raise ValueError("Expected one free-root tree")
                Rp = Rs[par]
                R0 = Rp @ rotations[..., i, :, :]
                origin = P[par] + (Rp @ offsets[..., i, :].unsqueeze(-1)).squeeze(-1)
                if j >= 0:
                    Ri = R0 @ axis_angle_to_mat(self.axes[i], q[..., j])
                    a0 = (R0 @ self.anchors[i].unsqueeze(-1)).squeeze(-1)
                    a1 = (Ri @ self.anchors[i].unsqueeze(-1)).squeeze(-1)
                    pi = origin + a0 - a1
                    axis = (R0 @ self.axes[i].unsqueeze(-1)).squeeze(-1)
                    wrel = axis * qd[..., j : j + 1]
                    wi = W[par] + wrel
                    vi = (
                        V[par]
                        + torch.cross(W[par], pi - P[par], dim=-1)
                        - torch.cross(wrel, a1, dim=-1)
                    )
                else:
                    pi, Ri = origin, R0
                    wi = W[par]
                    vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1)
                if return_position_jacobian:
                    ji = JP[par] + Rp @ self.continuous_offsets[1:, i].T
            P.append(pi)
            Rs.append(Ri)
            V.append(vi)
            W.append(wi)
            if return_position_jacobian:
                JP.append(ji)
        ids = range(len(P)) if all_bodies else self.tracked_idx
        result = tuple(
            torch.stack([xs[i] for i in ids], dim=dim)
            for xs, dim in [(P, -2), (Rs, -3), (V, -2), (W, -2)]
        )
        if return_position_jacobian:
            result += (torch.stack([JP[i] for i in ids], dim=-3),)
        return result
