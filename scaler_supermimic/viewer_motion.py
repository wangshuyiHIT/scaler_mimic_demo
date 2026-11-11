"""Lazy UFO references and continuous morphology for interactive inference only.

This module is deliberately separate from the training MotionLib. Intermediate
shapes are an exhibition construction, not qualified retargeting or evaluation.
"""
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .kinematics import ScalerKinematics
from .motion import MotionLib, gradient
from .reference_goal_mode import LEGACY_GOAL, validate_goal_mode
from .heading_frame_mode import LEGACY_HEADING, validate_heading_frame_mode
from .prepare import sha
from .robot import DT, Robot
from .rotations import quat_slerp, quat_to_mat, so3_log


def preserve_constants(model, data):
    q, v, t = data.qpos.copy(), data.qvel.copy(), data.time
    mujoco.mj_setConst(model, data)
    data.qpos[:], data.qvel[:], data.time = q, v, t
    mujoco.mj_forward(model, data)


def collision_bottom(model, data):
    """Lowest active primitive collision surface, including prone/fallen poses."""
    lows = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0 or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        kind, size = model.geom_type[g], model.geom_size[g]
        z = data.geom_xpos[g, 2]
        axis = data.geom_xmat[g].reshape(3, 3)[2]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            radius = np.abs(axis) @ size
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            radius = size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius = size[0] + abs(axis[2]) * size[1]
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            radius = size[0] * np.linalg.norm(axis[:2]) + abs(axis[2]) * size[1]
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            radius = np.linalg.norm(axis * size)
        elif kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            radius = -(model.mesh_vert[start:start+count] @ axis).min()
        else:
            raise ValueError("Live morphology requires primitive collision geometry")
        lows.append(z - radius)
    return min(lows)


def refresh_body_bounds(model):
    """Refit each body BVH in its (possibly changed) principal inertia frame.

    mj_setConst does not rebuild these compiler bounds. Mesh-local BVHs are
    unchanged; only body trees and telescopic box bounds need refitting.
    """
    for body in range(1, model.nbody):
        start, count = model.body_bvhadr[body], model.body_bvhnum[body]
        Ri = np.empty(9)
        mujoco.mju_quat2Mat(Ri, model.body_iquat[body])
        Ri = Ri.reshape(3, 3).T
        for node in range(start+count-1, start-1, -1):
            geom = model.bvh_nodeid[node]
            if geom >= 0:
                Rg = np.empty(9)
                mujoco.mju_quat2Mat(Rg, model.geom_quat[geom])
                Rg = Rg.reshape(3, 3)
                center, half = model.geom_aabb[geom, :3], model.geom_aabb[geom, 3:]
                model.bvh_aabb[node, :3] = Ri @ (model.geom_pos[geom] + Rg @ center - model.body_ipos[body])
                model.bvh_aabb[node, 3:] = np.abs(Ri @ Rg) @ half
            else:
                children = [model.bvh_aabb[start+c] for c in model.bvh_child[node] if c >= 0]
                low = np.min([c[:3]-c[3:] for c in children], axis=0)
                high = np.max([c[:3]+c[3:] for c in children], axis=0)
                model.bvh_aabb[node, :3], model.bvh_aabb[node, 3:] = (low+high)/2, (high-low)/2


class LiveMorphology:
    """Piecewise continuous compact -> nominal -> extended exhibition path."""
    fields = ("body_pos", "body_ipos", "body_mass", "geom_pos", "geom_size")

    def __init__(self, models, variants):
        by_name = {v["name"]: (m, np.asarray(v["delta_m"])) for m, v in zip(models, variants)}
        self.knots = [by_name[n] for n in ("compact", "nominal", "extended")]
        self.fraction = self.target = 0.0
        base = models[0]
        for model in models:
            for field in ("body_quat", "geom_quat", "jnt_pos", "jnt_axis", "geom_bodyid", "geom_type", "body_parentid"):
                if not np.allclose(getattr(model, field), getattr(base, field)):
                    raise ValueError("Incompatible morphology topology/axes: " + field)
        self.inertias = []
        for model, _ in self.knots:
            tensors = []
            for q, diag in zip(model.body_iquat, model.body_inertia):
                R = np.empty(9)
                mujoco.mju_quat2Mat(R, q)
                R = R.reshape(3, 3)
                tensors.append((R * diag) @ R.T)
            self.inertias.append(np.asarray(tensors))

    def nudge(self, direction):
        self.target = float(np.clip(self.target + direction * 0.1, -1, 1))

    def advance(self, dt):
        value = self.fraction + np.clip(self.target - self.fraction, -dt * 0.8, dt * 0.8)
        changed = abs(value - self.fraction) > 1e-8
        self.fraction = float(value)
        return changed

    def weights(self):
        return (0, 1, self.fraction + 1) if self.fraction < 0 else (1, 2, self.fraction)

    @property
    def delta(self):
        a, b, t = self.weights()
        return (1-t) * self.knots[a][1] + t * self.knots[b][1]

    def apply(self, model, data=None):
        a, b, t = self.weights()
        ma, mb = self.knots[a][0], self.knots[b][0]
        old_bottom = collision_bottom(model, data) if data is not None else None
        for field in self.fields:
            getattr(model, field)[:] = (1-t) * getattr(ma, field) + t * getattr(mb, field)
        # Blend tensors in the body frame, not principal axes/quaternions.
        inertia = (1-t) * self.inertias[a] + t * self.inertias[b]
        for body in range(1, model.nbody):
            values, vectors = np.linalg.eigh(inertia[body])
            if np.linalg.det(vectors) < 0:
                vectors[:, 0] *= -1
            model.body_inertia[body] = values
            mujoco.mju_mat2Quat(model.body_iquat[body], vectors.reshape(9))
        # Box profiles vary in size; mesh visuals retain their original bounds.
        boxes = model.geom_type == mujoco.mjtGeom.mjGEOM_BOX
        model.geom_rbound[boxes] = np.linalg.norm(model.geom_size[boxes], axis=1)
        model.geom_aabb[boxes, 3:] = model.geom_size[boxes]
        refresh_body_bounds(model)
        if data is not None:
            preserve_constants(model, data)
            data.qpos[2] += old_bottom - collision_bottom(model, data)
            mujoco.mj_forward(model, data)


class ViewerMotion(MotionLib):
    """Same reference interpolation/goal contract, one selected clip in memory."""
    def __init__(self, pack, *, reference_goal_mode=LEGACY_GOAL, heading_frame_mode=LEGACY_HEADING):
        self.reference_goal_mode = validate_goal_mode(reference_goal_mode)
        self.heading_frame_mode = validate_heading_frame_mode(heading_frame_mode)
        self.path = Path(pack).resolve()
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest["role"] != "train":
            raise ValueError("Exhibition uses UFO TRAIN only")
        for name, digest in self.manifest["files"].items():
            path = (self.path / name).resolve()
            if not path.is_relative_to(self.path) or sha(path) != digest:
                raise ValueError("Changed viewer pack input: " + name)
        self.device = "cpu"
        self.robot = Robot.from_yaml(self.path / self.manifest["robot"])
        self.templates = [self.robot.load_model(self.path / v["model"]) for v in self.manifest["variants"]]
        self.morph = LiveMorphology(self.templates, self.manifest["variants"])
        self.models = [copy.copy(self.morph.knots[1][0])]
        self.kin = ScalerKinematics(self.models, self.robot)
        self.deltas = torch.zeros(1, 5)
        self.selected = None
        self.select(0)

    def select(self, index):
        if index == self.selected:
            return
        row = self.manifest["motions"][index]
        with np.load(self.path / row["file"], allow_pickle=False) as z:
            p, quat, q = [torch.tensor(z[k], dtype=torch.float32) for k in ("root_p", "root_quat", "joint_pos")]
            fps = float(z["fps"])
        n = int(np.floor((len(q)-1) / fps / DT)) + 1
        time = torch.arange(n) * DT * fps
        i = time.floor().long().clamp(max=len(q)-2)
        a = (time-i).clamp(0, 1)
        self.base_root = p[i]*(1-a[:, None]) + p[i+1]*a[:, None]
        self.q = q[i]*(1-a[:, None]) + q[i+1]*a[:, None]
        self.qd = gradient(self.q)
        self.root_R = quat_to_mat(quat_slerp(quat[i], quat[i+1], a))
        self.root_w = torch.empty_like(self.base_root)
        self.root_w[1:-1] = so3_log(self.root_R[2:] @ self.root_R[:-2].transpose(-1,-2))/(2*DT)
        self.root_w[0] = so3_log(self.root_R[1] @ self.root_R[0].T)/DT
        self.root_w[-1] = so3_log(self.root_R[-1] @ self.root_R[-2].T)/DT
        self.clip_len, self.clip_start, self.clip_end = torch.tensor([n]), torch.tensor([0]), torch.tensor([n-1])
        self.clip_shape = torch.zeros(1, dtype=torch.long)
        self.clip_id = self.shape = torch.zeros(n, dtype=torch.long)
        self.num_frames, self.names = n, [row["name"]]
        self.bin_start = torch.arange(0, n-1, 50)
        self.num_bins = len(self.bin_start)
        self.frame_bin = torch.searchsorted(self.bin_start, torch.arange(n), right=True)-1
        self.source_model = self.templates[row["shape"]]
        source_kin = ScalerKinematics([self.source_model], self.robot)
        self.source_floor = self.soles(source_kin, self.source_model)
        self.selected = index
        self.update_morph()

    def soles(self, kin, model):
        P, R, _, _ = kin.forward(self.base_root, self.root_R, torch.zeros_like(self.base_root),
                                 self.root_w, self.q, self.qd, self.shape, all_bodies=True)
        lows = []
        for name in ("left_foot_col", "right_foot_col"):
            g = model.geom(name).id
            b = int(model.geom_bodyid[g])-1
            local = torch.tensor(model.geom_pos[g], dtype=torch.float32)
            size = torch.tensor(model.geom_size[g], dtype=torch.float32)
            gr = quat_to_mat(torch.tensor(model.geom_quat[g], dtype=torch.float32))
            world = R[:, b] @ gr
            lows.append((P[:, b] + (R[:, b] @ local[:, None]).squeeze(-1))[:, 2] - (world[:, 2].abs()*size).sum(-1))
        return torch.minimum(*lows)

    def update_morph(self):
        model = self.models[0]
        self.morph.apply(model)
        self.kin.off_p[0] = torch.tensor(model.body_pos[1:], dtype=torch.float32)
        self.deltas[0] = torch.tensor(self.morph.delta, dtype=torch.float32)
        self.root_p = self.base_root.clone()
        self.root_p[:, 2] += self.source_floor - self.soles(self.kin, model)
        self.root_v = gradient(self.root_p)
        self.body_p, self.body_R, self.body_v, self.body_w = self.kin.forward(*self.ref_state(), self.shape)

    @property
    def display_model_path(self):
        variant = next(v for v in self.manifest["variants"] if v["name"] == "nominal")
        return self.path / variant["model"]
