"""Opt-in PartyMimic training recipe. Missing block keeps the historical learner."""

from __future__ import annotations

import math
from types import SimpleNamespace

from .mdp import OBS_DIM
from .networks import Policy

GOAL_Q_OFFSET = 15  # first future frame: dp3 + rot6 + lv3 + av3, then joint positions
_OPTS = ("adam", "muon")
_POSE_STD_INDEX = (0, 1, 4, 5)
_BASE_TRACKING_STD = (0.3, 0.4, 1.0, 3.14, 0.3, 0.4)


def legacy_recipe():
    return SimpleNamespace(
        enabled=False,
        tracking_std_scale=1.0,
        pi_ref_skip=False,
        pi_ema=0.0,
        pi_opt="adam",
        wm_opt="adam",
        pi_muon_io=False,
        pi_lr_mult=1.0,
    )


def _number(value, name):
    if type(value) is bool or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"partymimic.{name} must be a finite number")
    return float(value)


def _bool(value, name):
    if type(value) is not bool:
        raise ValueError(f"partymimic.{name} must be a boolean")
    return value


def partymimic_settings(config):
    """Return the recipe. Omit ``partymimic`` to keep Adam, unit tracking kernels and no skip."""
    if not isinstance(config, dict):
        raise ValueError("Training config must be a dictionary")
    block = config.get("partymimic")
    if block is None:
        return legacy_recipe()
    if not isinstance(block, dict):
        raise ValueError("partymimic must be a dictionary")
    required = (
        "enabled", "tracking_std_scale", "pi_ref_skip", "pi_ema",
        "pi_opt", "wm_opt", "pi_muon_io", "pi_lr_mult",
    )
    missing = [key for key in required if key not in block]
    if missing:
        raise ValueError(f"partymimic is missing {missing}")
    if _bool(block["enabled"], "enabled") is not True:
        raise ValueError("partymimic.enabled must be true, or omit the block")
    scale = _number(block["tracking_std_scale"], "tracking_std_scale")
    if scale <= 0:
        raise ValueError("partymimic.tracking_std_scale must be positive")
    ema = _number(block["pi_ema"], "pi_ema")
    if not 0 <= ema < 1:
        raise ValueError("partymimic.pi_ema must be in [0, 1)")
    pi_opt, wm_opt = block["pi_opt"], block["wm_opt"]
    if pi_opt not in _OPTS or wm_opt not in _OPTS:
        raise ValueError("partymimic pi_opt and wm_opt must be adam or muon")
    mult = _number(block["pi_lr_mult"], "pi_lr_mult")
    if mult <= 0:
        raise ValueError("partymimic.pi_lr_mult must be positive")
    return SimpleNamespace(
        enabled=True,
        tracking_std_scale=scale,
        pi_ref_skip=_bool(block["pi_ref_skip"], "pi_ref_skip"),
        pi_ema=ema,
        pi_opt=pi_opt,
        wm_opt=wm_opt,
        pi_muon_io=_bool(block["pi_muon_io"], "pi_muon_io"),
        pi_lr_mult=mult,
    )


def tracking_stds(scale=1.0):
    """PartyMimic narrows position and orientation kernels only; velocity kernels stay."""
    scale = _number(scale, "tracking_std_scale")
    if scale <= 0:
        raise ValueError("tracking_std_scale must be positive")
    return tuple(
        std * scale if index in _POSE_STD_INDEX else std
        for index, std in enumerate(_BASE_TRACKING_STD)
    )


def reward_tracking_stds(scale, *, canonical_evaluation):
    """Training may narrow the kernels. Canonical evaluation keeps the historical widths."""
    if type(canonical_evaluation) is not bool:
        raise ValueError("canonical_evaluation must be boolean")
    return tracking_stds(1.0 if canonical_evaluation else scale)


def reference_skip(robot):
    scale = robot.action_scale
    if isinstance(scale, (int, float)):
        alpha = [float(scale)] * 24
    else:
        alpha = [float(x) for x in scale]
        if len(alpha) != 24:
            raise ValueError("action_scale must be a scalar or 24 values")
    return (GOAL_Q_OFFSET, [float(x) for x in robot.default_q], alpha)


def make_policy(config, robot, *, activation_checkpointing=False):
    recipe = partymimic_settings(config)
    ref_skip = reference_skip(robot) if recipe.pi_ref_skip else None
    return Policy(
        OBS_DIM,
        63 * 10,
        hidden=config["policy_hidden"],
        activation_checkpointing=activation_checkpointing,
        ref_skip=ref_skip,
    )


def load_policy(checkpoint, robot, *, activation_checkpointing=False):
    policy = make_policy(
        checkpoint["config"], robot, activation_checkpointing=activation_checkpointing,
    )
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    return policy
