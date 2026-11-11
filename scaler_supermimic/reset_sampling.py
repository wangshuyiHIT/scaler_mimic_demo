"""Versioned adaptive reset-bin sampling; no physics or random draws.

The opt-in probability mode stores an EMA of past *probabilities*, not returns.
Its .99 history decay is an explicit implementation choice. A nonempty reset
uses the history from earlier resets, then records the current softmax once.
Legacy keeps each collector's original arithmetic, including the CPU
collector's sequential handling of duplicate finished bins.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch
from .reset_sampling_config import DEFAULTS, LEGACY, PROBABILITY, reset_sampling_config


SCHEMA = "scaler_supermimic_reset_sampling_v1"


class ResetSampler:
    """Rank-local curriculum state. RNG/replay/physics state is deliberately absent."""

    def __init__(self, bin_count, config=None, *, backend="torch", device=None,
                 rank=None, world_size=None):
        if type(bin_count) is not int or bin_count <= 0:
            raise ValueError("bin_count must be a positive integer")
        if backend not in ("torch", "numpy"):
            raise ValueError("Unknown reset sampling backend")
        if rank is None and world_size is None:
            distributed = torch.distributed.is_initialized()
            rank = torch.distributed.get_rank() if distributed else 0
            world_size = torch.distributed.get_world_size() if distributed else 1
        if (type(rank) is not int or type(world_size) is not int
                or world_size <= 0 or not 0 <= rank < world_size):
            raise ValueError("Invalid rank/world_size")
        self.bin_count, self.rank, self.world_size = bin_count, rank, world_size
        self.config = reset_sampling_config(config)
        self.mode, self.backend = self.config["mode"], backend
        self.bin_return = (torch.ones(bin_count, device=device) if backend == "torch"
                           else np.ones(bin_count))
        self.bin_history = self._clone(self.bin_return) if self.mode == LEGACY else None
        self.history_probability = (None if self.mode == LEGACY
                                    else self._clone(self.bin_return) / bin_count)
        self.sampling_updates = 0

    @staticmethod
    def _clone(value):
        return value.detach().clone() if isinstance(value, torch.Tensor) else value.copy()

    def _finite(self, value):
        return bool(torch.isfinite(value).all()) if self.backend == "torch" else bool(np.isfinite(value).all())

    def _softmax(self, returns):
        if self.backend == "torch":
            return torch.softmax(-25 * returns, 0)
        weights = np.exp(-25 * (returns - returns.min()))
        return weights / weights.sum()

    @torch.no_grad()
    def probabilities(self, *, record_reset=False):
        """No RNG. Set record_reset only once for a nonempty, sampled reset call."""
        if type(record_reset) is not bool:
            raise ValueError("record_reset must be bool")
        if not self._finite(self.bin_return):
            raise ValueError("Nonfinite current reset returns")
        current = self._softmax(self.bin_return)
        if self.mode == LEGACY:
            if not self._finite(self.bin_history):
                raise ValueError("Nonfinite legacy history returns")
            history = self._softmax(self.bin_history)
        else:
            history = self.history_probability
            self._validate_probability(history)
        probability = 0.5 / self.bin_count + 0.4 * current + 0.1 * history
        if not self._finite(probability):
            raise ValueError("Nonfinite reset probabilities")
        if record_reset and self.mode == PROBABILITY:
            # Assignment into the existing vector keeps collector aliases valid.
            self.history_probability[:] = 0.99 * history + 0.01 * current
            self.sampling_updates += 1
        return probability

    def _observations(self, bins, values):
        if self.backend == "torch":
            bins = torch.as_tensor(bins, device=self.bin_return.device)
            if bins.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError("Finished bins must be integer indices")
            values = torch.as_tensor(values, device=self.bin_return.device)
            if values.dtype == torch.bool or values.is_complex():
                raise ValueError("Finished returns must be real numbers")
            bins = bins.long()
            values = values.to(self.bin_return.dtype)
        else:
            bins, values = np.asarray(bins), np.asarray(values)
            if bins.dtype.kind not in "iu" or values.dtype.kind not in "fiu":
                raise ValueError("Finished bins/returns must be integer/real numbers")
            values = values.astype(self.bin_return.dtype, copy=False)
        if (bins.ndim != 1 or values.shape != bins.shape
                or not self._finite(values) or bool((bins < 0).any())
                or bool((bins >= self.bin_count).any())):
            raise ValueError("Invalid finished-bin observations")
        return bins, values

    @torch.no_grad()
    def update_returns(self, bins, values, *, legacy_sequential=False):
        """Update current returns with .1 EMA of the simultaneous per-bin mean.

        The legacy CPU collector alone retains its original sequential EMA;
        opt-in probability mode always aggregates duplicate indices first.
        """
        if type(legacy_sequential) is not bool:
            raise ValueError("legacy_sequential must be bool")
        bins, values = self._observations(bins, values)
        if not len(bins):
            return
        if self.mode == LEGACY and legacy_sequential:
            if self.backend != "numpy":
                raise ValueError("Sequential legacy update is only the CPU numpy contract")
            for b, value in zip(bins, values):
                self.bin_history[b] = 0.99 * self.bin_history[b] + 0.01 * self.bin_return[b]
                self.bin_return[b] = 0.9 * self.bin_return[b] + 0.1 * value
            return
        if self.backend == "torch":
            sums = torch.zeros_like(self.bin_return).scatter_add_(0, bins, values)
            counts = torch.zeros_like(self.bin_return).scatter_add_(0, bins, torch.ones_like(values))
        else:
            sums, counts = np.zeros_like(self.bin_return), np.zeros_like(self.bin_return)
            np.add.at(sums, bins, values)
            np.add.at(counts, bins, np.ones_like(values))
        mask = counts > 0
        if self.mode == LEGACY:
            self.bin_history[mask] = 0.99 * self.bin_history[mask] + 0.01 * self.bin_return[mask]
        self.bin_return[mask] = 0.9 * self.bin_return[mask] + 0.1 * sums[mask] / counts[mask]

    def state_dict(self):
        return {
            "schema": SCHEMA, "mode": self.mode, "bin_count": self.bin_count,
            "rank": self.rank, "world_size": self.world_size,
            "config": dict(self.config), "sampling_updates": self.sampling_updates,
            "current_return": self._clone(self.bin_return),
            "legacy_history_return": None if self.bin_history is None else self._clone(self.bin_history),
            "history_probability": None if self.history_probability is None else self._clone(self.history_probability),
        }

    def _state_vector(self, value, name):
        try:
            if self.backend == "torch":
                vector = torch.as_tensor(value, device=self.bin_return.device)
                if not vector.is_floating_point():
                    raise ValueError("State vector must use a floating dtype")
                vector = vector.to(self.bin_return.dtype)
            else:
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                vector = np.asarray(value)
                if vector.dtype.kind != "f":
                    raise ValueError("State vector must use a floating dtype")
                vector = vector.astype(self.bin_return.dtype, copy=False)
            if vector.shape != (self.bin_count,) or not self._finite(vector):
                raise ValueError("State vector shape/finite mismatch")
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Invalid reset sampler {name}") from exc
        return self._clone(vector)

    def _validate_probability(self, vector):
        # Float32 EMA has small summation error; this validates, never renormalizes.
        total = float(vector.sum())
        if (not self._finite(vector) or bool((vector < 0).any())
                or not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-5)):
            raise ValueError("History probabilities must be finite, nonnegative and sum to one")

    @torch.no_grad()
    def load_state_dict(self, state):
        """Validate every field before mutation; never migrate mode, bin count or rank."""
        expected = {"schema", "mode", "bin_count", "rank", "world_size", "config",
                    "sampling_updates", "current_return", "legacy_history_return", "history_probability"}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Invalid reset sampler state fields")
        if state["schema"] != SCHEMA or state["mode"] != self.mode:
            raise ValueError("Reset sampler schema/mode mismatch")
        for name in ("bin_count", "rank", "world_size"):
            if type(state[name]) is not int or state[name] != getattr(self, name):
                raise ValueError(f"Reset sampler {name} mismatch")
        if (not isinstance(state["config"], Mapping)
                or reset_sampling_config(state["config"]) != self.config
                or dict(state["config"]) != self.config):
            raise ValueError("Reset sampler config mismatch")
        updates = state["sampling_updates"]
        if type(updates) is not int or updates < 0 or (self.mode == LEGACY and updates != 0):
            raise ValueError("Invalid reset sampler update count")
        current = self._state_vector(state["current_return"], "current_return")
        if self.mode == LEGACY:
            if state["history_probability"] is not None:
                raise ValueError("Legacy history returns cannot be interpreted as probabilities")
            history = self._state_vector(state["legacy_history_return"], "legacy_history_return")
        else:
            if state["legacy_history_return"] is not None:
                raise ValueError("Probability history cannot contain legacy history returns")
            history = self._state_vector(state["history_probability"], "history_probability")
            self._validate_probability(history)
        self.bin_return[:] = current
        if self.mode == LEGACY:
            self.bin_history[:] = history
        else:
            self.history_probability[:] = history
        self.sampling_updates = updates
