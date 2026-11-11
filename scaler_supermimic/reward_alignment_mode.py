"""Versioned reward semantics; missing checkpoint/config key preserves world reward."""
LEGACY_REWARD = "legacy_world_v1"
ANCHOR_REWARD = "anchor_xy_yaw_v1"


def validate_reward_alignment_mode(mode):
    if type(mode) is not str or mode not in (LEGACY_REWARD, ANCHOR_REWARD):
        raise ValueError("reward_alignment_mode must be legacy_world_v1 or anchor_xy_yaw_v1")
    return mode


def config_reward_alignment_mode(config):
    if not isinstance(config, dict):
        raise ValueError("Checkpoint/training config must be a dictionary")
    return validate_reward_alignment_mode(config.get("reward_alignment_mode", LEGACY_REWARD))
