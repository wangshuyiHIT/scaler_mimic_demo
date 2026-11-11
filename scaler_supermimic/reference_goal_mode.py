"""Versioned goal/heading semantics and strict checkpoint config admission.

Pure stdlib admission. Missing legacy keys are interpreted without changing
either input config. Heading migration is distinct from goal, sampler and
batch migrations; it does not authorize changes to any other science field.
"""
import json

from .heading_frame_mode import (
    LEGACY_HEADING, TWO_CHART_HEADING, config_heading_frame_mode,
)
from .reset_sampling_config import LEGACY, PROBABILITY, reset_sampling_config
from .reward_alignment_mode import LEGACY_REWARD, ANCHOR_REWARD, config_reward_alignment_mode

LEGACY_GOAL = "legacy_clamped"
STATIONARY_GOAL = "stationary_tail_v1"
BATCH_MIGRATION_FIELDS = {"num_envs", "wm_batch", "pi_batch", "activation_checkpointing"}
RUN_FIELDS = {"purpose", "iterations", "checkpoint_every", "logger"}


def validate_goal_mode(mode):
    if type(mode) is not str or mode not in (LEGACY_GOAL, STATIONARY_GOAL):
        raise ValueError("reference_goal_mode must be legacy_clamped or stationary_tail_v1")
    return mode


def config_goal_mode(config):
    if not isinstance(config, dict):
        raise ValueError("Checkpoint/training config must be a dictionary")
    return validate_goal_mode(config.get("reference_goal_mode", LEGACY_GOAL))


def validate_resume_configuration(previous, current, *, checkpoint_world, world,
                                  batch_migration=False, reference_goal_migration=False,
                                  reset_sampling_migration=False,
                                  heading_frame_migration=False,
                                  reward_alignment_migration=False):
    """Admit config before network/Adam loads; allow one declared migration."""
    old_mode, new_mode = config_goal_mode(previous), config_goal_mode(current)
    old_heading = config_heading_frame_mode(previous)
    new_heading = config_heading_frame_mode(current)
    old_reward = config_reward_alignment_mode(previous)
    new_reward = config_reward_alignment_mode(current)
    if (type(world) is not int or world < 1 or type(checkpoint_world) is not int
            or checkpoint_world != world):
        raise ValueError("Resume world size differs from checkpoint or is invalid")
    flags = (batch_migration, reference_goal_migration, reset_sampling_migration,
             heading_frame_migration, reward_alignment_migration)
    if any(type(flag) is not bool for flag in flags):
        raise ValueError("Migration selections must be boolean")
    if sum(flags) > 1:
        raise ValueError("Scientific and batch migrations cannot be combined")
    if reference_goal_migration:
        if (old_mode, new_mode) != (LEGACY_GOAL, STATIONARY_GOAL):
            raise ValueError("Reference-goal migration permits legacy to stationary only")
    elif old_mode != new_mode:
        raise ValueError("Reference-goal change requires --resume-reference-goal-migration")
    if heading_frame_migration:
        if (old_heading, new_heading) != (LEGACY_HEADING, TWO_CHART_HEADING):
            raise ValueError("Heading migration permits legacy to two_chart_v1 only")
    elif old_heading != new_heading:
        raise ValueError("Heading change requires --resume-heading-frame-migration")
    if reward_alignment_migration:
        if (old_reward, new_reward) != (LEGACY_REWARD, ANCHOR_REWARD):
            raise ValueError("Reward migration permits legacy world to anchor_xy_yaw_v1 only")
    elif old_reward != new_reward:
        raise ValueError("Reward change requires --resume-reward-alignment-migration")
    ignored = RUN_FIELDS | {"reference_goal_mode", "heading_frame_mode", "reward_alignment_mode"}
    if reset_sampling_migration:
        before = reset_sampling_config(previous.get("reset_sampling"))
        after = reset_sampling_config(current.get("reset_sampling"))
        if before["mode"] != LEGACY or after["mode"] != PROBABILITY:
            raise ValueError("Reset-sampling migration requires legacy to probability EMA (.99) only")
        ignored |= {"reset_sampling"}
    if batch_migration:
        if world != 1:
            raise ValueError("Batch migration requires unchanged single-rank world size 1")
        for config in (previous, current):
            for field in ("num_envs", "wm_batch", "pi_batch"):
                if type(config.get(field)) is not int or config[field] <= 0:
                    raise ValueError(f"Batch migration requires positive integer {field}")
            if type(config.get("activation_checkpointing", False)) is not bool:
                raise ValueError("activation_checkpointing must be boolean")
        ignored |= BATCH_MIGRATION_FIELDS

    def scientific(config):
        return json.dumps({k: v for k, v in config.items() if k not in ignored},
                          sort_keys=True, separators=(",", ":"), allow_nan=False)

    if scientific(previous) != scientific(current):
        raise ValueError("Resume learner/collector configuration differs from checkpoint")
    result = {"previous_effective_mode": old_mode, "current_effective_mode": new_mode,
              "reference_goal_migration": reference_goal_migration,
              "missing_reference_goal_mode_default": LEGACY_GOAL}
    if heading_frame_migration or "heading_frame_mode" in previous or "heading_frame_mode" in current:
        result.update(previous_heading_frame_mode=old_heading,
                      current_heading_frame_mode=new_heading,
                      heading_frame_migration=heading_frame_migration,
                      missing_heading_frame_mode_default=LEGACY_HEADING)
    if reward_alignment_migration or "reward_alignment_mode" in previous or "reward_alignment_mode" in current:
        result.update(previous_reward_alignment_mode=old_reward,
                      current_reward_alignment_mode=new_reward,
                      reward_alignment_migration=reward_alignment_migration,
                      missing_reward_alignment_mode_default=LEGACY_REWARD)
    return result
