"""Headless MuJoCo collector. CPU physics is the correctness/reference backend."""

import copy

import mujoco
import numpy as np
import torch

from .mdp import flatten_state
from .robot import DT
from .reset_sampling import ResetSampler


class TrackingSim:
    def __init__(
        self, motion, mdp, num_envs=4, seed=0, randomize=True, auto_reset=True,
        reset_sampling_config=None
    ):
        self.motion, self.mdp, self.N, self.device = (
            motion,
            mdp,
            num_envs,
            motion.device,
        )
        self.robot = motion.robot
        self.rng = np.random.default_rng(seed)
        self.randomize = randomize
        self.auto_reset = auto_reset
        self.models = [None] * num_envs
        self.data = [None] * num_envs
        self.f = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.ep_step = torch.zeros_like(self.f)
        self.a_prev = torch.zeros(num_envs, 24, device=self.device)
        self.xi = torch.zeros(num_envs, 33, device=self.device)
        self._reset_sampler = ResetSampler(motion.num_bins, reset_sampling_config, backend="numpy")
        self.bin_return = self._reset_sampler.bin_return
        self.bin_history = self._reset_sampler.bin_history
        self.start_bin = np.zeros(num_envs, dtype=int)
        self.episode_return = np.zeros(num_envs)
        self.max_return = np.ones(num_envs)
        self.reset(range(num_envs))

    def _probabilities(self):
        return self._reset_sampler.probabilities()

    def sampler_state_dict(self):
        return self._reset_sampler.state_dict()

    def load_sampler_state_dict(self, state):
        self._reset_sampler.load_state_dict(state)

    def reset(self, ids):
        ids = list(ids)
        if not ids:
            return
        m = self.motion
        probabilities = self._reset_sampler.probabilities(record_reset=True)
        for e in ids:
            b = int(self.rng.choice(m.num_bins, p=probabilities))
            start = int(m.bin_start[b])
            end = min(start + 50, int(m.clip_end[m.clip_id[start]]))
            f = int(self.rng.integers(start, end))
            shape = int(m.shape[f])
            model = copy.copy(m.models[shape])
            data = mujoco.MjData(model)
            x = np.zeros(33)
            x[0] = self.rng.uniform(0.3, 1.2) if self.randomize else 1
            if self.randomize:
                x[1:25] = self.rng.uniform(-0.01, 0.01, 24)
                x[25:28] = self.rng.uniform([-0.025, -0.05, -0.05], [0.025, 0.05, 0.05])
            x[28:] = m.deltas[shape].cpu().numpy()
            model.geom_friction[:, 0] = x[0]
            model.body_ipos[model.body("torso_pitch_link").id] += x[25:28]
            mujoco.mj_setConst(model, data)
            qadr, vadr = self.robot.addresses(model)
            S = [a[f].cpu().numpy() for a in m.ref_state()]
            from .rotations import mat_to_quat

            data.qpos[:3] = S[0]
            data.qpos[3:7] = mat_to_quat(m.root_R[f]).cpu().numpy()
            data.qpos[qadr] = S[4]
            data.qvel[:3] = S[2]
            data.qvel[3:6] = S[1].T @ S[3]
            data.qvel[vadr] = S[5]
            mujoco.mj_forward(model, data)
            self.models[e], self.data[e] = model, data
            self.f[e] = f
            self.ep_step[e] = 0
            self.a_prev[e] = 0
            self.xi[e] = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            self.start_bin[e] = b
            self.episode_return[e] = 0
            length = int(m.clip_end[m.clip_id[f]]) - f
            self.max_return[e] = 0.1 * (1 - 0.99**length) / (1 - 0.99)

    def state(self):
        rows = []
        for model, data in zip(self.models, self.data):
            qadr, vadr = self.robot.addresses(model)
            R = data.xmat[model.body("base_link").id].reshape(3, 3)
            rows.append(
                (
                    data.qpos[:3].copy(),
                    R.copy(),
                    data.qvel[:3].copy(),
                    R @ data.qvel[3:6],
                    data.qpos[qadr].copy(),
                    data.qvel[vadr].copy(),
                )
            )
        return tuple(
            torch.as_tensor(
                np.stack([r[i] for r in rows]), dtype=torch.float32, device=self.device
            )
            for i in range(6)
        )

    @torch.no_grad()
    def step(self, action):
        action = action.clamp(-self.robot.action_clip, self.robot.action_clip)
        S = self.state()
        f = self.f.clone()
        xi = self.xi.clone()
        previous = self.a_prev.clone()
        ep = self.ep_step.clone()
        a = action.cpu().numpy()
        x = xi.cpu().numpy()
        torques = []
        for e, (model, data) in enumerate(zip(self.models, self.data)):
            qa, va = self.robot.addresses(model)
            target = self.robot.default_q + self.robot.action_scale * a[e] + x[e, 1:25]
            for _ in range(round(DT / model.opt.timestep)):
                torque = (
                    self.robot.kp * (target - data.qpos[qa])
                    - self.robot.kd * data.qvel[va]
                )
                data.ctrl[:] = np.clip(torque, -self.robot.effort, self.robot.effort)
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            torques.append(data.ctrl.copy())
        S1 = self.state()
        f1 = self.motion.next_frame(f)
        if not all(torch.isfinite(s).all() for s in S1):
            raise FloatingPointError("Nonfinite simulator state")
        reward = self.mdp.reward(S1, f1, action, previous)
        failure = self.mdp.termination(S1, f1)
        done = failure | (f1 >= self.motion.clip_end[self.motion.clip_id[f]])
        self.episode_return += (0.99 ** ep.cpu().numpy()) * reward.cpu().numpy()
        result = {
            "s": flatten_state(S),
            "s1": flatten_state(S1),
            "a": action,
            "a_prev": previous,
            "xi": xi,
            "r": reward,
            "f": f,
            "ep_step": ep,
            "done": done,
            "torque": torch.tensor(np.stack(torques), device=self.device),
            "failure": failure,
        }
        self.f = f1
        self.a_prev = action.clone()
        self.ep_step += 1
        ids = done.nonzero().flatten().cpu().tolist()
        if ids:
            values = self.episode_return[ids] / np.maximum(self.max_return[ids], 1e-8)
            self._reset_sampler.update_returns(self.start_bin[ids], values, legacy_sequential=True)
        if self.auto_reset:
            self.reset(ids)
        return result
