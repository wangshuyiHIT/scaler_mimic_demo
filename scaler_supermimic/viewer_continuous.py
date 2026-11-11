"""CPU playback for checkpoints trained with continuous Flat DR.

The target shape may be any permitted five-vector, independently of the source
clip's baked witness shape. Reference adaptation is the training MotionLib FK
and sole-height correction, not fresh IK. Optional live height updates are a
viewer intervention: the training collector remains startup-fixed. Fixed mode
retains the original rebuild-and-reset behavior.
"""

import numpy as np
import torch

FAMILIES = ("thigh", "calf", "torso", "upperarm", "forearm")


class ContinuousHeightPath:
    """Piecewise linear custom-origin -> physical lower/upper five-vector."""

    def __init__(self, origin, bounds):
        self.origin, self.bounds = np.asarray(origin).copy(), np.asarray(bounds).copy()
        self.fraction = self.target = 0.0

    @property
    def delta(self):
        if self.fraction == -1:
            return self.bounds[:, 0].copy()
        if self.fraction == 1:
            return self.bounds[:, 1].copy()
        end = self.bounds[:, 0 if self.fraction < 0 else 1]
        return np.clip(self.origin + abs(self.fraction) * (end - self.origin),
                       self.bounds[:, 0], self.bounds[:, 1])

    def nudge(self, direction):
        if not np.isfinite(direction):
            raise ValueError("Nonfinite height direction")
        self.target = float(np.clip(self.target + direction * 0.1, -1, 1))

    def advance(self, dt):
        if not np.isfinite(dt) or dt < 0 or not np.isfinite([self.fraction, self.target]).all():
            raise ValueError("Height time and positions must be finite, with nonnegative time")
        if not -1 <= self.fraction <= 1 or not -1 <= self.target <= 1:
            raise ValueError("Height fraction/target must be within [-1,1]")
        value = self.fraction + np.clip(self.target - self.fraction, -dt * .8, dt * .8)
        changed = abs(value - self.fraction) > 1e-12
        self.fraction = float(value)
        return changed


def checkpoint_dr_config(checkpoint):
    config = checkpoint["config"].get("domain_randomization")
    if config is None:
        return None
    if (
        not isinstance(config, dict)
        or config.get("schema") != "scaler_supermimic_flat_dr_v1"
    ):
        raise ValueError("Unsupported checkpoint domain_randomization configuration")
    morphology = config.get("morphology", {})
    if (
        morphology.get("mode") != "startup_fixed_per_world_continuous"
        or tuple(morphology.get("family_names", ())) != FAMILIES
    ):
        raise ValueError(
            "Checkpoint does not declare the supported continuous morphology"
        )
    return config


class ContinuousPlaybackBackend:
    def __init__(self, motion, mdp, dr_config, shape_delta=None, live_height=False):
        if not motion.continuous_morphology or mdp.xi_dim != 293:
            raise ValueError(
                "Continuous playback requires the matching 293-context MDP"
            )
        self.motion, self.mdp, self.config = motion, mdp, dr_config
        metadata = motion.robot.config["training"]["morphology"]
        if tuple(metadata["family_names"]) != FAMILIES:
            raise ValueError("Physical morphology family order differs")
        bounds = np.asarray(metadata["family_bounds_m"], dtype=np.float64)
        clip = dr_config["morphology"]["clip_m"]
        if (
            isinstance(clip, bool)
            or not np.isfinite(clip)
            or clip <= 0
            or bounds.shape != (5, 2)
            or not np.isfinite(bounds).all()
        ):
            raise ValueError("Invalid training or physical morphology bounds")
        self.bounds = np.column_stack(
            (np.maximum(bounds[:, 0], -clip), np.minimum(bounds[:, 1], clip))
        )
        if np.any(self.bounds[:, 0] > self.bounds[:, 1]):
            raise ValueError("Empty intersection of training and mechanical bounds")
        self.custom_delta = (
            None if shape_delta is None else self._validate_delta(shape_delta)
        )
        self.live_height = bool(live_height)
        self.morph = ContinuousHeightPath(
            np.zeros(5) if self.custom_delta is None else self.custom_delta, self.bounds
        ) if self.live_height else None
        self.env = None
        self._target_delta = None
        self._field_builder = self._constant_data = None

    def _validate_delta(self, value):
        delta = np.asarray(value, dtype=np.float64)
        if (
            delta.shape != (5,)
            or not np.isfinite(delta).all()
            or np.any(delta < self.bounds[:, 0])
            or np.any(delta > self.bounds[:, 1])
        ):
            raise ValueError(
                "--shape-delta must be five finite meter values inside training and mechanical bounds"
            )
        return delta.copy()

    def reset(self, selection):
        from .continuous_env import ContinuousCPUTrackingSim

        clip, source_shape = selection["clip"], selection["shape"]
        if (
            not 0 <= clip < len(self.motion.clip_start)
            or int(self.motion.clip_shape[clip]) != source_shape
        ):
            raise ValueError("Selection must identify an actual source clip/shape pair")
        delta = self.morph.delta if self.live_height else self.custom_delta
        if delta is None:
            delta = self._validate_delta(
                self.motion.manifest["variants"][source_shape]["delta_m"]
            )
        if self.env is None or (not self.live_height and not np.array_equal(delta, self._target_delta)):
            self.env = ContinuousCPUTrackingSim(
                self.motion,
                self.mdp,
                1,
                seed=0,
                randomize=False,
                auto_reset=False,
                dr_config=self.config,
                fixed_deltas=delta[None],
            )
            self._target_delta = delta.copy()
            self._field_builder = self._constant_data = None
        elif self.live_height and not np.array_equal(delta, self._target_delta):
            self._apply_live_delta(delta)
        # Constructor's random initial frame is never used for playback.
        self.env.reset(
            torch.tensor([0]), frame_indices=self.motion.clip_start[clip : clip + 1]
        )
        return self.env

    @torch.no_grad()
    def _apply_live_delta(self, value):
        """Mutate this CPU model in place, preserving clip and integrated state.

        Only root Z is corrected to preserve the lowest collision clearance.
        This keeps hinge angles/velocities and commands, but does not claim
        energy-conserving telescopic actuation or validated dynamic stability.
        """
        import mujoco
        from .continuous_env import refresh_cpu_body_bounds
        from .domain_randomization import MorphologyModelFields
        from .viewer_motion import collision_bottom

        delta = self._validate_delta(value)
        env = self.env
        if env is None or env.N != 1 or str(env.device) != "cpu" or env.randomize:
            raise ValueError("Live height requires one non-randomized CPU playback world")
        model, data = env.models[0], env.data[0]
        if self._field_builder is None:
            self._field_builder = MorphologyModelFields(
                env.nominal, self.motion.robot.config["training"]["morphology"]
            )
            self._constant_data = mujoco.MjData(model)
        geometry = self._field_builder.fields(delta[None])
        offsets = self.motion.kin.offsets(env.tensor(delta[None])).cpu().numpy()
        if not np.allclose(offsets, geometry["body_pos"][:, 1:], atol=3e-7, rtol=0):
            raise ValueError("Live physical geometry and reference FK offsets differ")
        bottom = collision_bottom(model, data)
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        integration = np.empty(mujoco.mj_stateSize(model, state_spec))
        mujoco.mj_getState(model, data, integration, state_spec)
        warmstart = data.qacc_warmstart.copy()
        for name, values in geometry.items():
            getattr(model, name)[:] = values[0]
        refresh_cpu_body_bounds(model)
        # setConst uses a private scratch MjData; it cannot reset live qpos/time.
        mujoco.mj_setConst(model, self._constant_data)
        mujoco.mj_setState(model, data, integration, state_spec)
        mujoco.mj_forward(model, data)
        data.qpos[2] += bottom - collision_bottom(model, data)
        mujoco.mj_forward(model, data)
        data.qacc_warmstart[:] = warmstart
        env.fields.update(geometry)
        env.delta.copy_(env.tensor(delta[None]))
        env.xi[:, 28:33] = env.delta
        env.morphology_sample = {
            "requested_delta_m": delta[None].copy(), "applied_delta_m": delta[None].copy(),
            "anchor_mask": np.zeros(1, dtype=bool),
        }
        self._target_delta = delta.copy()

    def advance_height(self, dt):
        """Advance the held-key target without reset, model replacement or steps."""
        if not self.live_height:
            return False
        previous = self.morph.fraction
        if not self.morph.advance(dt):
            return False
        try:
            self._apply_live_delta(self.morph.delta)
        except BaseException:
            self.morph.fraction = previous
            raise
        return True

    def goal(self, state):
        return self.mdp.goal(self.env.f, S=state, xi=self.env.xi)

    def reference_state(self):
        state = self.motion.ref_state(
            self.env.f,
            delta=self.mdp.morphology_delta(self.env.xi),
            frame=self.mdp.reference_frame(self.env.xi),
        )
        return tuple(component[0] for component in state)
