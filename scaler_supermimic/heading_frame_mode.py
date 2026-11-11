"""Explicit heading coordinates; absent checkpoint fields retain legacy behavior.

Configuration admission is pure stdlib. Tensor code is resolved only when a
consumer requests its validated, process-local heading function.
"""

LEGACY_HEADING = "legacy_projected_x_v1"
TWO_CHART_HEADING = "two_chart_v1"


def validate_heading_frame_mode(mode):
    if type(mode) is not str or mode not in (LEGACY_HEADING, TWO_CHART_HEADING):
        raise ValueError("heading_frame_mode must be legacy_projected_x_v1 or two_chart_v1")
    return mode


def config_heading_frame_mode(config):
    if not isinstance(config, dict):
        raise ValueError("Checkpoint/training config must be a dictionary")
    return validate_heading_frame_mode(config.get("heading_frame_mode", LEGACY_HEADING))


def heading_function(mode):
    mode = validate_heading_frame_mode(mode)
    if mode == LEGACY_HEADING:
        from .rotations import yaw_mat

        return yaw_mat
    from .rotations import yaw_mat_two_chart

    return yaw_mat_two_chart
