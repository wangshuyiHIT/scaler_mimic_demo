"""Continuous morphology collectors with the frozen Flat DR sampling schedule.

One physical morphology is sampled per world at startup. Reference angles are
adapted by exact same-morphology FK and minimum-sole-height correction. The
MuJoCo contact model is explicit: dynamic sliding friction and restitution
damping mapping, rather than claiming PhysX contact-solver equivalence.
"""

import copy
import dataclasses
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .domain_randomization import (
    MorphologyModelFields, load_config, sample_morphology, sample_nuisance,
    sample_push, sample_reset,
)
from .dr_context import SLICES, pack_context
from .mdp import flatten_state
from .robot import DT
from .reset_sampling import ResetSampler
from .rotations import mat_to_quat, quat_to_mat


def refresh_cpu_body_bounds(model):
    """CPU collision BVHs use each body's current principal-inertia frame.

    MuJoCo-Warp instead uses the per-world geom_aabb/rbound during broadphase;
    its public Model has no CPU body BVH array to overwrite.
    """
    for body in range(1, model.nbody):
        start, count = model.body_bvhadr[body], model.body_bvhnum[body]
        inertia_rotation = np.empty(9)
        mujoco.mju_quat2Mat(inertia_rotation, model.body_iquat[body])
        inverse = inertia_rotation.reshape(3, 3).T
        for node in range(start + count - 1, start - 1, -1):
            geom = model.bvh_nodeid[node]
            if geom >= 0:
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
                rotation = rotation.reshape(3, 3)
                center, half = model.geom_aabb[geom, :3], model.geom_aabb[geom, 3:]
                model.bvh_aabb[node, :3] = inverse @ (
                    model.geom_pos[geom] + rotation @ center - model.body_ipos[body]
                )
                model.bvh_aabb[node, 3:] = np.abs(inverse @ rotation) @ half
            else:
                children = [model.bvh_aabb[start + child] for child in model.bvh_child[node] if child >= 0]
                low = np.min([child[:3] - child[3:] for child in children], axis=0)
                high = np.max([child[:3] + child[3:] for child in children], axis=0)
                model.bvh_aabb[node, :3] = (low + high) / 2
                model.bvh_aabb[node, 3:] = (high - low) / 2


class _ContinuousTracking:
    def __init__(self, motion, mdp, num_envs=1, seed=0, randomize=True,
                 auto_reset=True, dr_config=None, fixed_deltas=None, reset_sampling_config=None):
        if not motion.continuous_morphology or not mdp.continuous_morphology:
            raise ValueError("Continuous collector requires matching motion and MDP")
        self.motion, self.mdp, self.robot = motion, mdp, motion.robot
        self.N, self.device = num_envs, motion.device
        self.randomize, self.auto_reset, self.seed = randomize, auto_reset, seed
        self.config = dr_config or load_config()
        self.rng = np.random.default_rng(seed)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.physics_dt = 0.001
        nominal = (motion.deltas.abs().amax(-1) < 1e-9).nonzero().flatten()
        if len(nominal) != 1:
            raise ValueError("Require exactly one qualified nominal morphology")
        self.nominal = copy.copy(motion.models[int(nominal[0])])
        self.nominal.opt.timestep = self.physics_dt
        # Make the robot's sampled contact material effective against ground.
        self.nominal.geom_priority[self.nominal.geom_bodyid != 0] = 1
        builder = MorphologyModelFields(
            self.nominal, self.robot.config["training"]["morphology"]
        )
        c = self.config["morphology"]
        sampled = sample_morphology(
            num_envs, self.rng, builder.bounds, sigma_m=c["sigma_m"],
            clip_m=c["clip_m"], anchor_fraction=c["anchor_fraction"],
        )
        if fixed_deltas is not None:
            fixed = np.asarray(fixed_deltas, dtype=float)
            if fixed.shape != (num_envs, 5):
                raise ValueError("Fixed morphology must have one five-vector per world")
            sampled = dict(requested_delta_m=fixed, applied_delta_m=fixed,
                           anchor_mask=np.zeros(num_envs, dtype=bool))
        elif not randomize:
            sampled = dict(requested_delta_m=np.zeros((num_envs, 5)),
                           applied_delta_m=np.zeros((num_envs, 5)),
                           anchor_mask=np.ones(num_envs, dtype=bool))
        self.morphology_sample = sampled
        self.delta = self.tensor(sampled["applied_delta_m"])
        geometry = builder.fields(sampled["applied_delta_m"])
        # The analytical FK witnesses and actual field constructor must agree.
        fk_offsets = motion.kin.offsets(self.delta).cpu().numpy()
        if not np.allclose(fk_offsets, geometry["body_pos"][:, 1:], atol=3e-7, rtol=0):
            raise ValueError("Continuous FK/physical morphology offsets differ")
        body_names = [self.nominal.body(i).name for i in range(self.nominal.nbody)]
        parameters = sample_nuisance(
            num_envs, self.rng, body_names, self.robot.names,
            self.nominal.geom_bodyid, config=self.config,
        )
        if not randomize:
            for name in ("body_mass_add_kg", "body_com_offset_m", "restitution"):
                parameters[name].fill(0)
            for name in ("body_mass_scale", "kp_scale", "kd_scale", "joint_friction_scale",
                         "armature_scale", "dynamic_friction", "static_friction"):
                parameters[name].fill(1)
        initial_mass = geometry["body_mass"].copy()
        mass = initial_mass * parameters["body_mass_scale"] + parameters["body_mass_add_kg"]
        if np.any(mass[:, 1:] <= 0):
            raise ValueError("Flat mass randomization produced nonpositive physical mass")
        ratio = np.divide(mass, initial_mass, out=np.ones_like(mass), where=initial_mass != 0)
        geometry["body_mass"] = mass
        geometry["body_inertia"] *= ratio[..., None]
        geometry["body_ipos"] += parameters["body_com_offset_m"]
        geometry["geom_friction"] = np.broadcast_to(
            self.nominal.geom_friction, (num_envs, self.nominal.ngeom, 3)
        ).copy()
        geometry["geom_friction"][..., 0] = parameters["dynamic_friction"]
        geometry["geom_solref"] = np.broadcast_to(
            self.nominal.geom_solref, (num_envs, self.nominal.ngeom, 2)
        ).copy()
        restitution = parameters["restitution"]
        loge = np.log(np.maximum(restitution, 1e-8))
        damping = -loge / np.sqrt(np.pi**2 + loge**2)
        damping[restitution == 0] = 1.0
        geometry["geom_solref"][..., 1] = damping
        self.qa, self.va = self.robot.addresses(self.nominal)
        for field, key in (("dof_frictionloss", "joint_friction_scale"),
                           ("dof_armature", "armature_scale")):
            values = np.broadcast_to(getattr(self.nominal, field), (num_envs, self.nominal.nv)).copy()
            values[:, self.va] *= parameters[key]
            geometry[field] = values
        self.fields, self.parameters = geometry, parameters
        self.kp = self.tensor(parameters["kp_scale"] * self.robot.kp)
        self.kd = self.tensor(parameters["kd_scale"] * self.robot.kd)
        self.default_q = self.tensor(self.robot.default_q)
        self.effort = self.tensor(self.robot.effort)
        self.f = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.ep_step = torch.zeros_like(self.f)
        self.a_prev = torch.zeros(num_envs, 24, device=self.device)
        legacy = torch.zeros(num_envs, 33, device=self.device)
        legacy[:, 0] = self.tensor(parameters["dynamic_friction"].mean(-1))
        legacy[:, 25:28] = self.tensor(parameters["body_com_offset_m"][:, self.nominal.body("base_link").id])
        legacy[:, 28:33] = self.delta
        self.xi = pack_context(
            legacy, body_mass_ratio=ratio, kp_scale=parameters["kp_scale"],
            kd_scale=parameters["kd_scale"], joint_friction_scale=parameters["joint_friction_scale"],
            armature_scale=parameters["armature_scale"], dynamic_friction=parameters["dynamic_friction"],
            restitution=restitution, delay_seconds=np.zeros((num_envs, 4)),
        )
        self._reset_sampler = ResetSampler(motion.num_bins, reset_sampling_config, device=self.device)
        self.bin_return = self._reset_sampler.bin_return
        self.bin_history = self._reset_sampler.bin_history
        self.start_bin = torch.zeros_like(self.f)
        self.episode_return = torch.zeros(num_envs, device=self.device)
        self.max_return = torch.ones_like(self.episode_return)
        self.delay_steps = torch.zeros(num_envs, 24, dtype=torch.long, device=self.device)
        self.elapsed_physics = 0.0
        push = sample_push(num_envs, self.rng, config=self.config)
        self.next_push = push["push_interval_s"] if randomize else np.full(num_envs, np.inf)
        self.push_events = 0
        self._create_physics()
        self.reset(torch.arange(num_envs, device=self.device))

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def sampler_state_dict(self):
        return self._reset_sampler.state_dict()

    def load_sampler_state_dict(self, state):
        self._reset_sampler.load_state_dict(state)

    @torch.no_grad()
    def reset(self, ids, frame_indices=None):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        if not len(ids):
            return
        m = self.motion
        if frame_indices is None:
            prob = self._reset_sampler.probabilities(record_reset=True)
            bins = torch.multinomial(prob, len(ids), replacement=True, generator=self.generator)
            start = m.bin_start[bins]
            width = torch.minimum(torch.full_like(start, 50), m.clip_end[m.clip_id[start]] - start)
            f = start + (torch.rand(len(ids), device=self.device, generator=self.generator) * width).long()
        else:
            f = torch.as_tensor(frame_indices, dtype=torch.long, device=self.device)
            if f.shape != ids.shape or bool(((f < 0) | (f >= m.num_frames)).any()):
                raise ValueError("Invalid explicit reference start frames")
            bins = m.frame_bin[f]
        reset = sample_reset(len(ids), self.rng, self.robot.names, config=self.config, physics_dt=self.physics_dt)
        if not self.randomize:
            for key in ("source_delay_seconds", "delay_physics_steps", "reset_root_pose",
                        "reset_root_velocity", "reset_joint_offset_rad", "reset_joint_velocity_rad_s"):
                reset[key].fill(0)
        pose = self.tensor(reset["reset_root_pose"])
        if bool((pose[:, 2:5] != 0).any()):
            raise ValueError("Flat tracking frame supports its zero z/roll/pitch reset ranges")
        raw = m.ref_state(f, self.delta[ids])
        angle = pose[:, 5]
        c, s = angle.cos(), angle.sin()
        rotated_xy = torch.stack((c * raw[0][:, 0] - s * raw[0][:, 1],
                                  s * raw[0][:, 0] + c * raw[0][:, 1]), -1)
        frame = torch.cat((angle[:, None], raw[0][:, :2] + pose[:, :2] - rotated_xy), -1)
        self.xi[ids, SLICES["reference_frame"]] = frame
        S = list(m.ref_state(f, self.delta[ids], frame))
        velocity = self.tensor(reset["reset_root_velocity"])
        S[2] = S[2] + velocity[:, :3]
        S[3] = S[3] + velocity[:, 3:]
        S[4] = (S[4] + self.tensor(reset["reset_joint_offset_rad"])).clamp(m.kin.q_lo, m.kin.q_hi)
        S[5] = S[5] + self.tensor(reset["reset_joint_velocity_rad_s"])
        self.delay_steps[ids] = torch.as_tensor(
            reset["delay_physics_steps"][:, reset["joint_delay_group"]], device=self.device
        )
        self.xi[ids, SLICES["delay_seconds"]] = self.tensor(reset["source_delay_seconds"])
        self.xi[ids, SLICES["previous_action"]] = 0
        self.xi[ids, SLICES["push_delta_velocity"]] = 0
        self.f[ids], self.start_bin[ids] = f, bins
        self.ep_step[ids] = 0
        self.a_prev[ids] = 0
        self.episode_return[ids] = 0
        self.max_return[ids] = .1 * (1 - .99 ** (m.clip_end[m.clip_id[f]] - f)) / (1 - .99)
        self._write_state(ids, tuple(S), reset=True)

    @torch.no_grad()
    def step(self, action):
        a = action.clamp(-self.robot.action_clip, self.robot.action_clip)
        S, f, previous, ep = self.state(), self.f.clone(), self.a_prev.clone(), self.ep_step.clone()
        self.xi[:, SLICES["previous_action"]] = previous
        self.xi[:, SLICES["push_delta_velocity"]] = 0
        due = np.flatnonzero(self.next_push <= self.elapsed_physics)
        if len(due):
            event = sample_push(len(due), self.rng, config=self.config)
            ids = torch.as_tensor(due, device=self.device)
            dv = self.tensor(event["push_velocity"])
            self.xi[ids, SLICES["push_delta_velocity"]] = dv
            pushed = tuple(x[ids] for x in S)
            pushed = (*pushed[:2], pushed[2] + dv[:, :3], pushed[3] + dv[:, 3:], *pushed[4:])
            self._write_state(ids, pushed, reset=False)
            self.next_push[due] = self.elapsed_physics + event["push_interval_s"]
            self.push_events += len(due)
        xi = self.xi.clone()
        torque = self._integrate(a, previous)
        S1 = self.state()
        if not all(bool(torch.isfinite(s).all()) for s in S1):
            raise FloatingPointError("Nonfinite continuous DR simulator state")
        f1 = self.motion.next_frame(f)
        reward = self.mdp.reward(S1, f1, a, previous, xi=xi)
        failure = self.mdp.termination(S1, f1, xi=xi)
        done = failure | (f1 >= self.motion.clip_end[self.motion.clip_id[f]])
        result = dict(s=flatten_state(S), s1=flatten_state(S1), a=a, a_prev=previous,
                      xi=xi, r=reward, f=f, ep_step=ep, done=done,
                      torque=torque, failure=failure)
        self.episode_return += .99**ep * reward
        ids = done.nonzero().flatten()
        if len(ids):
            bins = self.start_bin[ids]
            val = self.episode_return[ids] / self.max_return[ids].clamp_min(1e-8)
            self._reset_sampler.update_returns(bins, val)
        self.f, self.a_prev = f1, a.clone()
        self.xi = self.mdp.advance_context(xi, a)
        self.ep_step += 1
        self.elapsed_physics += DT
        if self.auto_reset:
            self.reset(ids)
        return result

    def save_dr_evidence(self, output):
        output = Path(output)
        np.savez(output / "dr_world_samples.npz", **self.morphology_sample,
                 initial_physics_context=self.xi.detach().cpu().numpy())
        receipt = dict(
            schema="scaler_supermimic_continuous_dr_receipt_v1", seed=self.seed,
            worlds=self.N, startup_fixed_morphology=True,
            unique_applied_morphologies=len(np.unique(self.morphology_sample["applied_delta_m"], axis=0)),
            anchor_worlds=int(self.morphology_sample["anchor_mask"].sum()),
            minimum_delta_m=self.delta.amin(0).cpu().tolist(), maximum_delta_m=self.delta.amax(0).cpu().tolist(),
            physics_dt_s=self.physics_dt, policy_dt_s=DT, delay_support_s=[0, .005, .010],
            context_dim=self.mdp.xi_dim, reference_adaptation="same-delta exact FK; source minimum sole height preserved; angle-preserving, not fresh IK or no-slip contact preservation",
            reset_frame="whole reference yaw/xy transform; source-relative joint and velocity perturbations",
            material_mapping="sampled dynamic sliding friction; robot contact priority=1; restitution mapped to damping ratio -log(e)/sqrt(pi^2+log(e)^2), not PhysX solver equivalence",
            joint_friction_mapping="sampled scale multiplies MuJoCo frictionloss; not numerical PhysX joint-friction equivalence",
            interval_clock="elapsed physics across reference clip resets", config=self.config,
        )
        (output / "dr_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


class ContinuousCPUTrackingSim(_ContinuousTracking):
    def _create_physics(self):
        if self.N > 64 or str(self.device) != "cpu":
            raise ValueError("CPU continuous collector is limited to 64 CPU worlds")
        self.models, self.data = [], []
        for i in range(self.N):
            model = copy.copy(self.nominal)
            for key, values in self.fields.items():
                getattr(model, key)[:] = values[i]
            refresh_cpu_body_bounds(model)
            data = mujoco.MjData(model)
            mujoco.mj_setConst(model, data)
            self.models.append(model); self.data.append(data)

    def _write_state(self, ids, S, reset=False):
        for row, world in enumerate(ids.cpu().tolist()):
            data, model = self.data[world], self.models[world]
            p, R, v, w, q, qd = (x[row].cpu().numpy() for x in S)
            data.qpos[:3], data.qpos[3:7], data.qpos[self.qa] = p, mat_to_quat(torch.from_numpy(R)).numpy(), q
            data.qvel[:3], data.qvel[3:6], data.qvel[self.va] = v, R.T @ w, qd
            if reset:
                data.qacc_warmstart[:] = 0; data.time = 0
            mujoco.mj_forward(model, data)

    def state(self):
        values = []
        for model, data in zip(self.models, self.data):
            R = data.xmat[model.body("base_link").id].reshape(3, 3).copy()
            values.append((data.qpos[:3].copy(), R, data.qvel[:3].copy(),
                           R @ data.qvel[3:6], data.qpos[self.qa].copy(), data.qvel[self.va].copy()))
        return tuple(self.tensor(np.stack([s[j] for s in values])) for j in range(6))

    def _integrate(self, action, previous):
        torque = []
        for i, (model, data) in enumerate(zip(self.models, self.data)):
            for tick in range(round(DT / self.physics_dt)):
                applied = torch.where(tick >= self.delay_steps[i], action[i], previous[i])
                target = self.default_q + self.robot.action_scale * applied
                effort = self.kp[i] * (target - self.tensor(data.qpos[self.qa])) - self.kd[i] * self.tensor(data.qvel[self.va])
                data.ctrl[:] = effort.clamp(-self.effort, self.effort).cpu().numpy()
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            torque.append(data.ctrl.copy())
        return self.tensor(np.stack(torque))


class ContinuousWarpTrackingSim(_ContinuousTracking):
    def _create_physics(self):
        import mujoco_warp as mw
        import warp as wp
        self.mw, self.wp = mw, wp
        wp.init()
        self.wp_device = wp.get_device(self.device)
        with wp.ScopedDevice(self.wp_device):
            self.model = mw.put_model(self.nominal)
            for field in dataclasses.fields(self.model):
                spec = getattr(field.type, "shape", ())
                if not spec or spec[0] != "*":
                    continue
                original = getattr(self.model, field.name)
                if original is None:
                    continue
                values = self.fields.get(field.name)
                if values is None:
                    values = np.repeat(original.numpy(), self.N, axis=0)
                values = values.reshape((self.N,) + original.numpy().shape[1:])
                setattr(self.model, field.name, wp.array(values, dtype=original.dtype,
                        shape=(self.N,) + original.shape[1:], device=self.wp_device))
            self.data = mw.make_data(self.nominal, nworld=self.N, nconmax=128, njmax=256)
            self.qpos = wp.to_torch(self.data.qpos)
            self.qvel = wp.to_torch(self.data.qvel)
            self.ctrl = wp.to_torch(self.data.ctrl)
            mw.set_const(self.model, self.data)
        wp.synchronize_device(self.wp_device)
        self.qa = torch.as_tensor(self.qa, device=self.device)
        self.va = torch.as_tensor(self.va, device=self.device)

    def _forward(self):
        with self.wp.ScopedStream(self.wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            self.mw.forward(self.model, self.data)

    def _write_state(self, ids, S, reset=False):
        self.qpos[ids, :3] = S[0]
        self.qpos[ids, 3:7] = mat_to_quat(S[1])
        self.qpos[ids[:, None], self.qa] = S[4]
        self.qvel[ids, :3] = S[2]
        self.qvel[ids, 3:6] = (S[1].transpose(-1, -2) @ S[3][..., None]).squeeze(-1)
        self.qvel[ids[:, None], self.va] = S[5]
        if reset:
            self.wp.to_torch(self.data.qacc_warmstart)[ids] = 0
            self.wp.to_torch(self.data.time)[ids] = 0
        self._forward()

    def state(self):
        R = quat_to_mat(self.qpos[:, 3:7])
        return (self.qpos[:, :3].clone(), R, self.qvel[:, :3].clone(),
                (R @ self.qvel[:, 3:6, None]).squeeze(-1),
                self.qpos[:, self.qa].clone(), self.qvel[:, self.va].clone())

    def _integrate(self, action, previous):
        with self.wp.ScopedStream(self.wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            for tick in range(round(DT / self.physics_dt)):
                applied = torch.where(tick >= self.delay_steps, action, previous)
                target = self.default_q + self.robot.action_scale * applied
                effort = self.kp * (target - self.qpos[:, self.qa]) - self.kd * self.qvel[:, self.va]
                self.ctrl.copy_(effort.clamp(-self.effort, self.effort))
                self.mw.step(self.model, self.data)
        return self.ctrl.clone()
