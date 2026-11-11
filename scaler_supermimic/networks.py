# Adapted from zita-ch/supermimic_vibed a1508d4 (MIT); see LICENSE.upstream.
"""World model, policy (motion encoder + shared decoder), and critic (paper Table A.4)."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .mdp import INV_DIM, NUM_J, XI_DIM, invariant_features
from .rotations import so3_exp
from .heading_frame_mode import LEGACY_HEADING, validate_heading_frame_mode, heading_function


def mlp(
    inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False
) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if self.count >= self.max_count:
            return
        n = x.shape[0]
        if torch.distributed.is_initialized():
            packed = torch.cat((x.sum(0), x.square().sum(0), x.new_tensor([n])))
            torch.distributed.all_reduce(packed)
            total, square, n_tensor = (
                packed[:-1].chunk(2)[0],
                packed[:-1].chunk(2)[1],
                packed[-1],
            )
            n = float(n_tensor)
            m = total / n
            v = (square / n - m.square()).clamp_min(0)
        else:
            m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (
            self.var * self.count + v * n + delta**2 * self.count * n / tot
        ) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


WM_OUT = 3 + 3 + 3 + 3 + NUM_J + NUM_J

# bf16 autocast for MLP matmuls only (state integration, FK, rewards stay fp32)
AMP_DTYPE: torch.dtype | None = None


def run_mlp(
    net: nn.Module, x: torch.Tensor, *, activation_checkpointing: bool = False
) -> torch.Tensor:
    def forward(features):
        if AMP_DTYPE is None:
            return net(features)
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            return net(features).float()

    # Only the pure MLP is recomputed. Normalization updates, noise, state
    # integration, FK and termination stay outside this function.
    if activation_checkpointing and torch.is_grad_enabled():
        return checkpoint(forward, x, use_reentrant=False)
    return forward(x)


class WorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi) predicting frame-local state increments."""

    def __init__(self, hidden=(2048, 2048, 2048), *, activation_checkpointing=False, xi_dim=XI_DIM,
                 heading_frame_mode=LEGACY_HEADING):
        super().__init__()
        self.heading_frame_mode = validate_heading_frame_mode(heading_frame_mode)
        self.activation_checkpointing = activation_checkpointing
        if isinstance(xi_dim, bool) or not isinstance(xi_dim, int) or xi_dim < XI_DIM:
            raise ValueError("xi_dim must retain the 33-value legacy prefix")
        self.xi_dim = xi_dim
        self.in_dim = INV_DIM + NUM_J + xi_dim
        self.net = mlp(self.in_dim, list(hidden), WM_OUT, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(WM_OUT, eps=1e-4)

    def inputs(self, S, a, xi_n):
        if xi_n.shape[-1] != self.xi_dim:
            raise ValueError("WorldModel physics context dimension mismatch")
        return torch.cat((invariant_features(S, heading_frame_mode=self.heading_frame_mode), a, xi_n), -1)

    @staticmethod
    def targets(S, S1, *, heading_frame_mode=LEGACY_HEADING):
        """Frame-local increments from S to S1 (the regression targets)."""
        from .rotations import so3_log

        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = heading_function(heading_frame_mode)(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ R1),
                (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    def step(self, S, a, xi_n):
        p, R, v, w, q, qd = S
        y = run_mlp(
            self.net,
            self.in_norm(self.inputs(S, a, xi_n)),
            activation_checkpointing=self.activation_checkpointing,
        )
        d = y * self.out_norm.std() + self.out_norm.mean
        H = heading_function(self.heading_frame_mode)(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, NUM_J, NUM_J), -1)
        p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
        R1 = R @ so3_exp(dth)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        return (p1, R1, v1, w1, q + dq, qd + dqd)

    @staticmethod
    def state_error(S_hat, S, *, heading_frame_mode=LEGACY_HEADING):
        """Frame-local state error components (same layout as ``targets``)."""
        from .rotations import so3_log

        p, R, v, w, q, qd = S
        ph, Rh, vh, wh, qh, qdh = S_hat
        H = heading_function(heading_frame_mode)(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (ph - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ Rh),
                (Rt @ (vh - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (wh - w).unsqueeze(-1)).squeeze(-1),
                qh - q,
                qdh - qd,
            ),
            -1,
        )


class Policy(nn.Module):
    """a = dec(enc(g), o_{t-9:t}); deterministic, latent z in R^32 bounded by tanh."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        hist: int = 10,
        z_dim: int = 32,
        hidden=(2048, 1024, 512, 256),
        activation_checkpointing=False,
        ref_skip=None,
    ):
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        self.hist = hist
        self.ref_off = None
        self.enc = mlp(goal_dim, list(hidden), z_dim)
        self.dec = mlp(z_dim + obs_dim * hist, list(hidden), NUM_J)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)
        if ref_skip is not None:
            offset, q0, alpha = ref_skip
            if type(offset) is not int or offset < 0:
                raise ValueError("ref_skip offset must be a nonnegative integer")
            q0 = torch.as_tensor(q0, dtype=torch.float32)
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
            if q0.shape != (NUM_J,) or alpha.shape != (NUM_J,) or not bool((alpha > 0).all()):
                raise ValueError("ref_skip q0 and alpha must be 24 values, alpha positive")
            self.ref_off = offset
            self.register_buffer("ref_q0", q0)
            self.register_buffer("ref_alpha", alpha)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """obs_hist: (B, hist, obs_dim) noisy observations, newest last."""
        z = torch.tanh(
            run_mlp(
                self.enc,
                self.goal_norm(goal),
                activation_checkpointing=self.activation_checkpointing,
            )
        )
        o = self.obs_norm(obs_hist).flatten(-2)
        action = run_mlp(
            self.dec,
            torch.cat((z, o), -1),
            activation_checkpointing=self.activation_checkpointing,
        )
        if self.ref_off is None:
            return action
        q_ref = goal[..., self.ref_off:self.ref_off + action.shape[-1]]
        return action + (q_ref - self.ref_q0) / self.ref_alpha


class Critic(nn.Module):
    def __init__(self, in_dim: int, *, activation_checkpointing=False):
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        self.net = mlp(in_dim, [256, 128], 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(
            self.net,
            self.norm(x),
            activation_checkpointing=self.activation_checkpointing,
        ).squeeze(-1)
