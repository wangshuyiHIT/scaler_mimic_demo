"""JSON-only sampling contract shared by checkpoint admission and collectors."""
from collections.abc import Mapping
import math

LEGACY = "legacy_return_ema_v1"
PROBABILITY = "probability_ema_v1"
DEFAULTS = {
    "mode": LEGACY, "uniform_weight": 0.5, "current_weight": 0.4,
    "history_weight": 0.1, "temperature": 25.0,
    "return_update_rate": 0.1, "history_decay": 0.99,
}


def reset_sampling_config(value=None):
    if value is None:
        value = {}
    if not isinstance(value, Mapping) or set(value) - set(DEFAULTS):
        raise ValueError("Invalid reset_sampling_config keys/type")
    config = {**DEFAULTS, **value}
    if type(config["mode"]) is not str or config["mode"] not in (LEGACY, PROBABILITY):
        raise ValueError("Unknown reset sampling mode")
    for name, expected in DEFAULTS.items():
        if name == "mode":
            continue
        actual = config[name]
        if (type(actual) not in (int, float) or not math.isfinite(actual)
                or actual <= 0 or actual != expected):
            raise ValueError(f"Reset sampling {name} must remain {expected}")
        config[name] = expected
    return config
