"""CPU policy playback with the UFO native layout and a reference skeleton.

The player owns physics. UI callbacks enqueue commands, and rendering never steps
simulation. This is reference-library visualization, not acceptance evaluation.
"""

import argparse
import copy
import json
import math
import os
import time
import numpy as np
from pathlib import Path
from queue import Empty, SimpleQueue

MODULE = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = MODULE / "policy/snapshot_iteration43100.pt"
DEFAULT_SHA = "57f026fdfffeaa7579d9fed8a3ebdd16df3b9f99215038dd02d248c972c09f68"
DEFAULT_ITERATION = 43100
DEFAULT_PACK = MODULE / "data"
FAMILIES = ("thigh", "calf", "torso", "upperarm", "forearm")
KEYS = {
    263: "prev_motion",
    262: "next_motion",
    265: "next_morph",
    264: "prev_morph",
    82: "reset",
    114: "reset",
    32: "pause",
    76: "loop",
    108: "loop",
    256: "close",
}


class Catalog:
    """Only manifest-declared motion/shape pairs; no fabricated nominal shape."""

    def __init__(self, manifest):
        if (
            manifest["schema"] != "scaler_supermimic_motion_pack_v1"
            or manifest["role"] != "train"
        ):
            raise ValueError("Viewer requires the explicitly labelled training pack")
        self.manifest = manifest
        self.variants = manifest["variants"]
        self.motion_names, self.pairs = [], {}
        for index, row in enumerate(manifest["motions"]):
            shape = row["shape"]
            variant = self.variants[shape]
            delta = variant["delta_m"]
            if len(delta) != 5 or not all(math.isfinite(v) for v in delta):
                raise ValueError("Invalid five-dimensional morphology")
            suffix = "__" + variant["name"]
            name = row["name"].removesuffix(suffix)
            if name not in self.pairs:
                self.motion_names.append(name)
                self.pairs[name] = {}
            if shape in self.pairs[name]:
                raise ValueError("Ambiguous duplicate motion/shape pair")
            self.pairs[name][shape] = index
        if not self.motion_names:
            raise ValueError("Empty viewer catalog")

    def shape_order(self, motion):
        return sorted(
            self.pairs[motion],
            key=lambda s: (
                sum(self.variants[s]["delta_m"][:3]),
                *self.variants[s]["delta_m"],
                s,
            ),
        )

    def select(self, motion, preferred_shape=None):
        if motion not in self.pairs:
            raise ValueError(f"Unknown motion: {motion}")
        choices = self.shape_order(motion)
        fallback = preferred_shape is not None and preferred_shape not in choices
        if preferred_shape in choices:
            shape = preferred_shape
        elif preferred_shape is None:
            shape = choices[0]
        else:
            wanted = self.variants[preferred_shape]["delta_m"]
            shape = min(
                choices,
                key=lambda s: (
                    sum(
                        (a - b) ** 2
                        for a, b in zip(self.variants[s]["delta_m"], wanted)
                    ),
                    s,
                ),
            )
        return {
            "motion": motion,
            "shape": shape,
            "clip": self.pairs[motion][shape],
            "fallback": fallback,
            "requested_shape": preferred_shape,
        }

    def description(self):
        return {
            "role": "train",
            "motion_groups": len(self.motion_names),
            "morphologies": len(self.variants),
            "actual_pairs": len(self.manifest["motions"]),
            "height_order": "sum(thigh, calf, torso extension); relative preset, not absolute height",
            "dimensions": list(FAMILIES),
            "variants": self.variants,
            "motions": [
                {
                    "name": name,
                    "available_variants": [
                        self.variants[s]["name"] for s in self.shape_order(name)
                    ],
                }
                for name in self.motion_names
            ],
        }


class Playback:
    def __init__(
        self, pack, checkpoint, expected_sha, expected_iteration=DEFAULT_ITERATION, threads=2,
        exhibition=False, shape_delta=None,
    ):
        # Lazy imports keep --list completely independent of torch/CUDA/model loading.
        import mujoco
        import torch

        from .env import TrackingSim
        from .mdp import MDP, OBS_DIM
        from .motion import MotionLib
        from .reference_goal_mode import config_goal_mode
        from .heading_frame_mode import config_heading_frame_mode
        from .reward_alignment_mode import LEGACY_REWARD, config_reward_alignment_mode
        from .partymimic_recipe import load_policy, partymimic_settings
        from .prepare import sha
        from .viewer_continuous import checkpoint_dr_config, ContinuousPlaybackBackend

        self.mujoco, self.torch, self.obs_dim = mujoco, torch, OBS_DIM
        torch.set_num_threads(threads)
        torch.manual_seed(0)
        checkpoint = Path(checkpoint).resolve()
        if checkpoint.name in ("last.pt", "last.pt.tmp"):
            raise ValueError("Use an immutable checkpoint copy, never live last.pt")
        before = checkpoint.stat()
        if sha(checkpoint) != expected_sha:
            raise ValueError("Checkpoint SHA differs from the selected fixed model")
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        after = checkpoint.stat()
        if any(
            getattr(before, k) != getattr(after, k)
            for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ):
            raise ValueError("Checkpoint changed while loading")
        if ck["iteration"] != expected_iteration:
            raise ValueError(
                "Checkpoint iteration differs from the selected fixed model"
            )
        goal_mode = config_goal_mode(ck["config"])
        reward_mode = config_reward_alignment_mode(ck["config"])
        if ck.get("reward_alignment_mode", reward_mode) != reward_mode:
            raise ValueError("Checkpoint reward metadata differs from its config")
        heading_mode = config_heading_frame_mode(ck["config"])
        if ck.get("heading_frame_mode", heading_mode) != heading_mode:
            raise ValueError("Checkpoint heading metadata differs from its config")
        dr_config = checkpoint_dr_config(ck)
        self.continuous = dr_config is not None
        self.exhibition = exhibition and not self.continuous
        self.live_height = bool(exhibition)
        if shape_delta is not None and not self.continuous:
            raise ValueError("--shape-delta requires a continuous-DR checkpoint")
        if self.exhibition:
            from .viewer_motion import ViewerMotion
            self.motion = ViewerMotion(pack, reference_goal_mode=goal_mode, heading_frame_mode=heading_mode)
            if tuple(ck["pack_manifest"]["joint_names"]) != self.motion.robot.names:
                raise ValueError("Checkpoint and exhibition motor order differ")
        else:
            self.motion = MotionLib(pack, "cpu", role="train", continuous_morphology=self.continuous,
                                    reference_goal_mode=goal_mode, heading_frame_mode=heading_mode)
        if not self.exhibition and ck["pack_manifest"] != self.motion.manifest:
            raise ValueError("Viewer pack differs from the checkpoint training pack")
        self.catalog = Catalog(self.motion.manifest)
        recipe = partymimic_settings(ck["config"])
        self.mdp = MDP(self.motion, dr_config=dr_config, heading_frame_mode=heading_mode,
                       reward_alignment_mode=reward_mode,
                       tracking_std_scale=recipe.tracking_std_scale)
        self.continuous_backend = (
            ContinuousPlaybackBackend(
                self.motion, self.mdp, dr_config, shape_delta,
                live_height=self.live_height,
            )
            if self.continuous else None
        )
        self.policy = load_policy(ck, self.motion.robot)
        self.identity = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": expected_sha,
            "iteration": expected_iteration,
            "reference_goal_mode": goal_mode,
            **({"heading_frame_mode": heading_mode} if "heading_frame_mode" in ck["config"] else {}),
            "reward_alignment_mode": reward_mode,
            "canonical_evaluation": LEGACY_REWARD,
            "device": "cpu",
            "pack": str(Path(pack).resolve()),
            "pack_manifest_sha256": sha(Path(pack) / "manifest.json"),
            "reference_matches_training_pack": ck["pack_manifest"] == self.motion.manifest,
            "checkpoint_training_pairs": len(ck["pack_manifest"]["motions"]),
            "reference_scope": "UFO LAFAN TRAIN; continuous exhibition retarget, not qualified evaluation" if self.exhibition else "checkpoint training pack",
        }
        if self.continuous:
            self.identity.update(
                playback_mode="continuous_checkpoint_live_height" if self.live_height else "continuous_checkpoint_fixed_shape",
                context_dim=self.mdp.xi_dim, nuisance_randomization=False,
                reference_adaptation="training same-delta FK and sole-height correction; angle-preserving, not fresh IK",
                shape_delta_requested_m=None if shape_delta is None else list(shape_delta),
                shape_bounds_m=self.continuous_backend.bounds.tolist(),
                shape_change=(
                    "in-place display-time morphology; preserve pose, velocity, action history and clip time; training remains startup-fixed"
                    if self.live_height else "fixed target; preset changes rebuild physics and restart source clip"
                ),
            )
        del ck
        self.env = None if self.continuous else TrackingSim(
            self.motion, self.mdp, 1, randomize=False, auto_reset=False
        )
        self.reset_serial = 0
        self.reset(self.catalog.select(self.catalog.motion_names[0]))

    def reset(self, selection):
        from .rotations import mat_to_quat

        torch, mj, m = self.torch, self.mujoco, self.motion
        c, shape = selection["clip"], selection["shape"]
        if getattr(self, "continuous", False):
            env = self.env = self.continuous_backend.reset(selection)
            with torch.no_grad():
                obs = self.mdp.obs(env.state(), env.a_prev, env.xi, env.f)
            self.history = obs[:, None].expand(1, 10, self.obs_dim).clone()
            self.model, self.data, self.selection = env.models[0], env.data[0], dict(selection)
            self.steps, self.finished, self.failed = 0, False, False
            self.first_failure_frame = None
            self.body_rmse, self.root_rmse = 0.0, 0.0
            self.reset_serial += 1
            return self.status()
        if self.exhibition:
            m.select(c)
            # The full catalog is lazy; only the active clip/shape occupies RAM.
            c = shape = 0
            self.env.bin_return = np.ones(m.num_bins)
            self.env.bin_history = np.ones(m.num_bins)
        if int(m.clip_shape[c]) != shape:
            raise ValueError("Reference and physical morphology differ")
        model = copy.copy(m.models[shape])
        data = mj.MjData(model)
        model.geom_friction[:, 0] = 1.0
        mj.mj_setConst(model, data)
        f = int(m.clip_start[c])
        S = m.ref_state(torch.tensor([f]))
        qa, va = m.robot.addresses(model)
        data.qpos[:3], data.qpos[3:7] = S[0][0], mat_to_quat(S[1][0])
        data.qpos[qa] = S[4][0]
        data.qvel[:3], data.qvel[3:6], data.qvel[va] = (
            S[2][0],
            S[1][0].T @ S[3][0],
            S[5][0],
        )
        mj.mj_forward(model, data)
        env = self.env
        env.models[0], env.data[0] = model, data
        env.f[0], env.ep_step[0] = f, 0
        env.a_prev.zero_()
        env.xi.zero_()
        env.xi[0, 0], env.xi[0, 28:] = 1.0, m.deltas[shape]
        env.start_bin[0] = int(m.frame_bin[f])
        env.episode_return[0] = 0
        env.max_return[0] = 0.1 * (1 - 0.99 ** (int(m.clip_len[c]) - 1)) / (1 - 0.99)
        with torch.no_grad():
            obs = self.mdp.obs(env.state(), env.a_prev, env.xi, env.f)
        self.history = obs[:, None].expand(1, 10, self.obs_dim).clone()
        self.model, self.data, self.selection = model, data, dict(selection)
        self.steps, self.finished, self.failed = 0, False, False
        self.first_failure_frame = None
        self.body_rmse, self.root_rmse = 0.0, 0.0
        self.reset_serial += 1
        return self.status()

    @property
    def height_morph(self):
        return self.continuous_backend.morph if self.continuous else self.motion.morph

    def nudge_height(self, direction):
        self.height_morph.nudge(direction)

    def advance_height(self, dt):
        if getattr(self, "continuous", False) and getattr(self, "live_height", False):
            if not self.continuous_backend.advance_height(dt):
                return False
            # The current morphology conditions every history slot immediately;
            # the recorded joint/action observations and playback clock stay put.
            self.history[..., -5:] = self.env.delta[:, None] / .05
            return True
        if not self.exhibition or not self.motion.morph.advance(dt):
            return False
        m = self.motion
        # Keep live joint pose, velocity, action/history and clip time. Only the
        # vertical root correction preserves the existing collision clearance.
        m.morph.apply(self.model, self.data)
        m.update_morph()
        self.env.xi[0, 28:] = m.deltas[0]
        # Morph conditioning must agree in every history slot immediately.
        self.history[..., -5:] = m.deltas[0] / .05
        return True

    def step(self):
        from .mdp import unflatten_state

        if self.finished:
            return False
        with self.torch.no_grad():
            S = self.env.state()
            # The selected collector owns PD/substeps. DR uses the same context
            # for its policy goal, physical transition and reward/termination.
            continuous = getattr(self, "continuous", False)
            goal = self.continuous_backend.goal(S) if continuous else self.motion.goal(self.env.f, S=S)
            action = self.policy(self.history, goal).clamp(-4, 4)
            previous = self.env.a_prev.clone()
            tr = self.env.step(action)
            state = unflatten_state(tr["s1"])
            kwargs = {"xi": tr["xi"]} if continuous else {}
            _, terms = self.mdp.reward(state, self.env.f, tr["a"], previous, terms=True, canonical_evaluation=True, **kwargs)
            obs = self.mdp.obs(state, tr["a"], self.env.xi, self.env.f)
            self.history = self.torch.cat((self.history[:, 1:], obs[:, None]), 1)
        self.steps += 1
        self.body_rmse = math.sqrt(float(terms["body_position"][0]))
        self.root_rmse = math.sqrt(float(terms["root_position"][0]))
        self.record_completion(bool(tr["failure"][0]), bool(tr["done"][0]))
        return True

    def record_completion(self, failure, done):
        # GUI playback may continue after a failure; scientific termination and
        # the first failure remain visible. Headless checks retain strict done.
        if failure and self.first_failure_frame is None:
            self.first_failure_frame = self.steps
        self.failed = self.failed or failure
        if getattr(self, "play_through_failures", False):
            c = 0 if getattr(self, "exhibition", False) else self.selection["clip"]
            self.finished = self.steps >= int(self.motion.clip_len[c]) - 1
        else:
            self.finished = done

    def command_selection(self, command):
        name, shape = self.selection["motion"], self.selection["shape"]
        if command in ("prev_motion", "next_motion"):
            names = self.catalog.motion_names
            name = names[
                (names.index(name) + (1 if command == "next_motion" else -1))
                % len(names)
            ]
        elif command in ("prev_morph", "next_morph"):
            if getattr(self, "continuous", False) and self.continuous_backend.custom_delta is not None:
                return dict(self.selection)
            choices = self.catalog.shape_order(name)
            shape = choices[
                (choices.index(shape) + (1 if command == "next_morph" else -1))
                % len(choices)
            ]
        return self.catalog.select(name, shape)

    def status(self):
        c, shape = self.selection["clip"], self.selection["shape"]
        variant = self.catalog.variants[shape]
        live = getattr(self, "live_height", getattr(self, "exhibition", False))
        continuous = getattr(self, "continuous", False)
        custom = continuous and not live and self.continuous_backend.custom_delta is not None
        if self.exhibition:
            c = 0
        return {
            **self.selection,
            "variant": "live height" if live else "custom fixed delta" if custom else variant["name"],
            "delta_m": self.env.delta[0].tolist() if continuous else self.motion.morph.delta.tolist() if live else variant["delta_m"],
            "custom_shape_fixed": custom,
            "continuous_checkpoint": continuous,
            "height_fraction": self.height_morph.fraction if live else None,
            "height_target": self.height_morph.target if live else None,
            "reference_variant": variant["name"],
            "frame": self.steps,
            "reference_steps": int(self.motion.clip_len[c]) - 1,
            "time_seconds": self.steps * 0.02,
            "finished": self.finished,
            "failed": self.failed,
            "first_failure_frame": self.first_failure_frame,
            "body_position_error_m": self.body_rmse,
            "root_position_error_m": self.root_rmse,
            "reset_serial": self.reset_serial,
            "physics": "MuJoCo CPU 1000Hz / control50Hz" if continuous else "MuJoCo CPU 500Hz / control50Hz",
            "display_only": True,
            "formal_acceptance": False,
        }


def write_status(path, player, *, paused, loop, terminal=None):
    if path is None:
        return
    record = {
        "schema": "scaler_supermimic_viewer_status_v1",
        "observed_epoch": time.time(),
        "pid": os.getpid(),
        "identity": player.identity,
        **player.status(),
        "paused": paused,
        "loop": loop,
        "terminal": terminal,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp, path)


def overlay_status(player, paused, loop, reference_skeleton):
    """World-space reference is neither aligned to policy nor applied to physics."""
    status = player.status()
    kwargs = {"reference_state": player.continuous_backend.reference_state()} if getattr(player, "continuous", False) else {}
    names, reference = reference_skeleton.sample(player.motion, int(player.env.f[0]), **kwargs)
    actual_xy, reference_xy = player.data.qpos[:2], reference[0][:2]
    return {
        "motion_name": status["motion"],
        "motion_index": player.catalog.motion_names.index(status["motion"]),
        "motion_count": len(player.catalog.motion_names),
        "morphology_name": status["variant"],
        "morphology_delta_m": status["delta_m"],
        "frame_index": status["frame"],
        "frame_count": status["reference_steps"] + 1,
        "world_xy_error_m": math.hypot(
            *(a - b for a, b in zip(actual_xy, reference_xy))
        ),
        "reference_body_positions_m": reference,
        "reference_body_names": names,
        "actual_root_xy": actual_xy.tolist(),
        "ref_root_xy": reference_xy,
        "reset_serial": status["reset_serial"],
        "dr_enabled": False,
        "physics_backend": status["physics"] if status["continuous_checkpoint"] else "native MuJoCo CPU / live policy",
        "paused": paused,
        "loop_enabled": loop,
        "model_iteration": player.identity["iteration"],
        "failed": status["failed"],
        "fallback": status["fallback"],
        "height_fraction": status.get("height_fraction"),
        "height_target": status.get("height_target"),
        "custom_shape_fixed": status.get("custom_shape_fixed", False),
        "reference_variant": status.get("reference_variant"),
        "external_reference": not player.identity.get("reference_matches_training_pack", True),
    }


class JoinedPassiveViewer:
    """Own the native render thread through scene destruction.

    MuJoCo's public passive handle.close() only requests exit; its daemon
    thread can still be rendering when Python/GLFW tears down. The bundled
    runtime's own launcher is used unchanged, with an explicit thread join.
    """

    def __init__(self, model, data, **options):
        import threading
        from queue import Queue, Full

        import mujoco.viewer

        ready = Queue(1)
        self._render_error = None

        def render():
            try:
                mujoco.viewer._launch_internal(
                    model, data, run_physics_thread=False, handle_return=ready,
                    **options,
                )
            except BaseException as error:
                self._render_error = error
                try:
                    ready.put_nowait(error)
                except Full:
                    pass

        self._render_thread = threading.Thread(
            target=render, name="scaler-sm-native-render", daemon=False
        )
        self._render_thread.start()
        self._handle = ready.get()
        if isinstance(self._handle, BaseException):
            self._render_thread.join()
            raise RuntimeError("Native viewer failed to open") from self._handle

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def close(self):
        self._handle.close()
        self._render_thread.join()
        if self._render_error is not None:
            raise RuntimeError("Native viewer render thread failed") from self._render_error


class PlaybackCycle:
    """A visible neutral intermission, independent of manual pause and failure."""

    def __init__(self, hold_seconds=0.8):
        self.hold_seconds = hold_seconds
        self.elapsed = None

    def tick(self, player, *, paused, loop, dt):
        if not player.finished:
            self.elapsed = None
            if not paused:
                player.step()
        if not player.finished:
            return "playing"
        if self.elapsed is None:
            self.elapsed = 0.0
        elif not paused:
            self.elapsed += max(0.0, dt)
        if loop and not paused and self.elapsed >= self.hold_seconds:
            self.elapsed = None
            return "restart"
        return "t_pose"


def run_gui(
    player,
    *,
    paused=False,
    loop=False,
    stop_file=None,
    status_file=None,
    mono_end=float("inf"),
    epoch_end=float("inf"),
    visual_assets=None,
    font_scale=200,
    command_source=None,
):
    import mujoco.viewer

    from .reference_overlay import (
        ReferenceSkeleton,
        WorldTrace,
        _draw_world_trace,
        format_lafan_source,
        format_status,
    )
    from .visuals import Display
    from .viewer_hud import LargeHUD
    from .viewer_keys import HeldHeightKeys, NativeKeyBridge

    commands = command_source if command_source is not None else SimpleQueue()
    handle = display = reference_skeleton = None
    trace = WorldTrace()
    closing = False
    terminal = "window_closed"
    last_status = 0.0
    last_tick = time.monotonic()
    cycle = PlaybackCycle()
    hud = LargeHUD(font_scale)
    held_height = HeldHeightKeys()
    key_bridge = None
    player.play_through_failures = True

    def rebind():
        nonlocal handle, display, reference_skeleton, key_bridge
        if handle is not None:
            handle.close()
        held_height.clear()
        model_path = player.motion.display_model_path if player.exhibition else (
            Path(player.identity["pack"])
            / player.catalog.variants[player.selection["shape"]]["model"]
        )
        display = Display(visual_assets, model_path, player.model, player.data,
                          live_morphology=player.exhibition or getattr(player, "continuous", False))
        reference_skeleton = ReferenceSkeleton(player.model, player.motion.robot)
        names = {
            mujoco.mj_id2name(display.model, mujoco.mjtObj.mjOBJ_BODY, i): i
            for i in range(1, display.model.nbody)
        }
        trace.set_topology(names, display.model.body_parentid)
        on_press = lambda key: commands.put(KEYS[key]) if key in KEYS else None
        key_bridge = NativeKeyBridge(held_height, on_press) if player.live_height else on_press
        handle = JoinedPassiveViewer(
            display.model,
            display.data,
            key_callback=key_bridge,
            show_left_ui=False,
            show_right_ui=False,
        )
        with handle.lock():
            # Match UFO/MJLab native auto-track defaults; mouse controls remain native.
            handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            handle.cam.trackbodyid = display.model.body("base_link").id
            handle.cam.distance, handle.cam.azimuth, handle.cam.elevation = (
                5.0,
                90.0,
                -45.0,
            )
            handle.cam.lookat[:] = player.data.qpos[:3]

    def select(selection):
        nonlocal handle, paused, reference_skeleton
        was_finished = player.finished
        new_shape = not player.live_height and selection["shape"] != player.selection["shape"]
        if new_shape and handle is not None:
            handle.close()
            handle = None
        player.reset(selection)
        reference_skeleton = ReferenceSkeleton(player.model, player.motion.robot)
        cycle.elapsed = None
        if new_shape:
            rebind()
        if was_finished:
            paused = False
        print(json.dumps({"event": "selection", **player.status()}), flush=True)

    try:
        rebind()
        while not closing:
            tick = time.monotonic()
            if tick >= mono_end or time.time() >= epoch_end:
                terminal = "max_seconds"
                break
            if stop_file is not None and stop_file.exists():
                terminal = "stop_file"
                break
            while True:
                try:
                    command = commands.get_nowait()
                except Empty:
                    break
                if command == "close":
                    closing = True
                elif command == "pause":
                    paused = not paused
                    print(json.dumps({
                        "event": "playback_pause_command", "paused": paused,
                        "frame": player.steps, "reset_serial": player.reset_serial,
                    }), flush=True)
                elif command == "loop":
                    loop = not loop
                elif player.live_height and command in ("prev_morph", "next_morph"):
                    player.nudge_height(1 if command == "next_morph" else -1)
                else:
                    select(player.command_selection(command))
            if closing or not handle.is_running():
                break
            height_dt = max(0., min(tick-last_tick, 0.05))
            if player.live_height:
                held_height.drive(player.height_morph, height_dt)
            player.advance_height(height_dt)
            phase = cycle.tick(player, paused=paused, loop=loop, dt=tick-last_tick)
            last_tick = tick
            if phase == "restart":
                print(json.dumps({"event": "clip_replay", **player.status()}), flush=True)
                select(player.selection)
                phase = "playing"
            status = overlay_status(player, paused, loop, reference_skeleton)
            status["playback_phase"] = phase
            status["first_failure_frame"] = player.first_failure_frame
            if phase == "t_pose":
                # Intermission is explicitly display-only, never a policy pose.
                status["reference_body_positions_m"] = None
            trace.update(status)
            left1, left2 = format_status(status)
            right1, right2 = format_lafan_source(status)
            hud.update(handle, (left1, left2), (right1, right2))
            with handle.lock():
                if player.live_height:
                    display.sync_morphology(player.model)
                if phase == "t_pose":
                    display.show_t_pose(player.data.qpos[:2])
                else:
                    display.sync(player.data)
                # Stabilize tracking camera on the root origin, as UFO does.
                if handle.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                    bid = handle.cam.trackbodyid
                    display.data.subtree_com[bid] = display.data.xpos[bid]
                handle.user_scn.ngeom = 0
                if phase != "t_pose":
                    _draw_world_trace(mujoco, handle.user_scn, trace)
            handle.sync()
            if tick - last_status >= 1.0:
                write_status(status_file, player, paused=paused, loop=loop)
                last_status = tick
            time.sleep(
                max(0.0, (1 / 30 if paused or phase == "t_pose" else 0.02) - (time.monotonic() - tick))
            )
    finally:
        if handle is not None:
            handle.close()
        # Window closure is a terminal event, not an implicit pause command.
        write_status(status_file, player, paused=paused, loop=loop, terminal=terminal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--exhibition", action=argparse.BooleanOptionalAction, default=True,
                        help="Lazy full UFO library and continuous height (inference only)")
    parser.add_argument("--font-scale", type=int, choices=(100, 150, 200, 250, 300), default=200)
    parser.add_argument("--height", type=float, default=0.0,
                        help="Initial live height fraction: -1 compact, 0 nominal, +1 extended")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-sha256", default=DEFAULT_SHA)
    parser.add_argument("--expected-iteration", type=int, default=DEFAULT_ITERATION)
    parser.add_argument("--motion")
    parser.add_argument("--morph")
    parser.add_argument("--shape-delta", type=float, nargs=5,
                        metavar=("THIGH", "CALF", "TORSO", "UPPERARM", "FOREARM"),
                        help="Continuous-DR checkpoint only: initial extensions in meters; --no-exhibition keeps them fixed")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=0,
                        help="Optional display time limit; 0 keeps the window open until closed")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument(
        "--visual-urdf",
        type=Path,
        default=MODULE / "assets/urdf/scaler.urdf",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--commands",
        default="",
        help="Headless only: zero-based step:next_motion,next_morph,reset events separated by commas",
    )
    args = parser.parse_args()
    if not math.isfinite(args.height) or not -1 <= args.height <= 1:
        parser.error("--height must be in [-1, 1]")
    catalog = Catalog(json.loads((args.pack / "manifest.json").read_text()))
    if args.list:
        print(json.dumps(catalog.description(), indent=2))
        return
    start_mono, start_epoch = time.monotonic(), time.time()
    if not math.isfinite(args.max_seconds) or args.max_seconds < 0:
        parser.error("--max-seconds must be nonnegative and finite")
    if args.max_seconds == 0:
        args.max_seconds = float("inf")
    if args.status_file is not None and args.status_file.exists():
        parser.error("Use a new --status-file path for this viewer session")
    if args.stop_file is not None and args.stop_file.exists():
        parser.error("--stop-file already exists; no model loaded")
    if not 1 <= args.threads <= 8:
        parser.error("--threads must be between 1 and 8")
    if args.headless and not 1 <= args.steps <= 500:
        parser.error("Headless connection check is limited to 1..500 control steps")
    if args.commands and not args.headless:
        parser.error("--commands is only for headless checks")
    motion = args.motion or catalog.motion_names[0]
    shape = next((i for i, v in enumerate(catalog.variants) if v["name"] == "nominal"), None) if args.exhibition else None
    if args.morph:
        matches = [i for i, v in enumerate(catalog.variants) if v["name"] == args.morph]
        if not matches:
            parser.error("Unknown --morph; use --list")
        shape = matches[0]
    selection = catalog.select(motion, shape)
    player = Playback(
        args.pack,
        args.checkpoint,
        args.checkpoint_sha256,
        args.expected_iteration,
        args.threads,
        exhibition=args.exhibition,
        shape_delta=args.shape_delta,
    )
    if args.height != 0 and not player.live_height:
        parser.error("--height requires live exhibition mode")
    if player.live_height:
        player.height_morph.fraction = player.height_morph.target = args.height
        if player.exhibition:
            player.motion.update_morph()
    player.reset(selection)
    print(
        json.dumps(
            {"event": "loaded", **player.identity, "selection": player.status()}
        ),
        flush=True,
    )
    if args.headless:
        script = {}
        for event in filter(None, args.commands.split(",")):
            when, command = event.split(":", 1)
            if command not in (
                "next_motion",
                "prev_motion",
                "next_morph",
                "prev_morph",
                "reset",
            ):
                raise ValueError("Unknown headless command")
            script.setdefault(int(when), []).append(command)
        events = []
        terminal = "steps_completed"
        steps_executed = 0
        for step in range(args.steps):
            if (
                time.monotonic() >= start_mono + args.max_seconds
                or time.time() >= start_epoch + args.max_seconds
            ):
                terminal = "max_seconds"
                break
            if args.stop_file is not None and args.stop_file.exists():
                terminal = "stop_file"
                break
            for cmd in script.get(step, []):
                if player.live_height and cmd in ("prev_morph", "next_morph"):
                    player.nudge_height(1 if cmd == "next_morph" else -1)
                else:
                    player.reset(player.command_selection(cmd))
                events.append(
                    {"event": cmd, "at_step": step, "status": player.status()}
                )
            player.advance_height(0.02)
            steps_executed += int(player.step())
        print(
            json.dumps(
                {
                    "schema": "scaler_supermimic_viewer_headless_v1",
                    "identity": player.identity,
                    "steps_requested": args.steps,
                    "steps_executed": steps_executed,
                    "terminal": terminal,
                    "events": events,
                    "final": player.status(),
                    "gui_started": False,
                    "gpu_used": False,
                    "formal_acceptance": False,
                }
            ),
            flush=True,
        )
    else:
        from .visuals import VisualAssets

        visual_assets = VisualAssets(args.visual_urdf)
        if player.exhibition:
            visual_assets.identity["physics_source"] = "UFO TRAIN model with continuous exhibition morphology; training unchanged"
        player.identity["visuals"] = visual_assets.identity
        run_gui(
            player,
            paused=args.paused,
            loop=args.loop,
            stop_file=args.stop_file,
            status_file=args.status_file,
            mono_end=start_mono + args.max_seconds,
            epoch_end=start_epoch + args.max_seconds,
            visual_assets=visual_assets,
            font_scale=args.font_scale,
        )


if __name__ == "__main__":
    main()
