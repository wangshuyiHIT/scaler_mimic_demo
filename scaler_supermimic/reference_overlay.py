"""UFO-compatible decorative reference overlay; never steps a physics model.

Pure display helpers adapted from scaler_ufo_adapter/interactive_viewer.py.
No UFO runtime or CUDA dependency is required by the standalone module.
"""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Mapping
from typing import Any

FAMILY_ORDER = ("thigh", "calf", "torso", "upperarm", "forearm")
REFERENCE_RGBA = (0.12, 0.82, 1.0, 0.8)
ACTUAL_TRAIL_RGBA = (1.0, 0.48, 0.12, 0.95)
UFO_DISPLAY_SOURCE_SHA256 = (
    "af043ba979ee0829e9633ce43bddc6207147f129cb5eb53ec383013b5a5627b0"
)


def _text(value: Any, *, limit: int = 96) -> str:
    if value is None:
        return "unavailable"
    return " ".join(str(value).split())[:limit]


def _number(value: Any, *, scale: float = 1.0, signed: bool = False) -> str:
    try:
        number = float(value) * scale
    except (TypeError, ValueError, OverflowError):
        return "unavailable"
    if not math.isfinite(number):
        return "unavailable"
    return f"{number:+.3f}" if signed else f"{number:.3f}"


def _position(index: Any, count: Any) -> str:
    # Public indices are zero based; only the display is one based.
    try:
        valid = (
            not isinstance(index, bool)
            and not isinstance(count, bool)
            and int(index) == index
            and int(count) == count
            and 0 <= index < count
        )
    except (TypeError, ValueError, OverflowError):
        valid = False
    return f"{int(index) + 1}/{int(count)}" if valid else "unavailable"


def format_lafan_source(status: Mapping[str, Any]) -> tuple[str, str]:
    """Describe the selected source clip, independently of resampled playback.

    MotionCatalog admits recording_subject_start_end identities. These ranges
    are zero-based, end-exclusive BVH slices, not the player's control frames.
    """
    match = re.fullmatch(
        r"(?P<recording>[A-Za-z][A-Za-z0-9_]*)_subject(?P<subject>\d+)_(?P<start>\d+)_(?P<end>\d+)",
        str(status.get("motion_name", "")),
    )
    if match is None:
        full = re.fullmatch(
            r"(?P<recording>[A-Za-z][A-Za-z0-9_]*)_subject(?P<subject>\d+)",
            str(status.get("motion_name", "")),
        )
        if full is None:
            return "LAFAN source", "unavailable"
        return (
            "LAFAN file\nRecording\nPerformer\nBVH range",
            f"{full['recording']}_subject{full['subject']}.bvh\n{full['recording']}\nSubject {int(full['subject'])}\nunspecified in prepared clip name",
        )
    if int(match["end"]) <= int(match["start"]):
        return "LAFAN source", "unavailable"
    start, end = int(match["start"]), int(match["end"])
    return (
        "LAFAN file\nRecording\nPerformer\nBVH frames",
        (
            f"{match['recording']}_subject{match['subject']}.bvh\n"
            f"{match['recording']}\nSubject {int(match['subject'])}\n"
            f"[{start}, {end})  (0-based)"
        ),
    )


def format_status(status: Mapping[str, Any]) -> tuple[str, str]:
    """Return native overlay columns, without querying the environment.

    ``family_lengths_m`` is optional absolute physical family length. Without
    it, ``morphology_delta_m`` is explicitly labelled as extension, never height.
    Both vectors use the authoritative ``contracts.FAMILY_ORDER``. Callers pass
    already copied CPU values so display formatting does not synchronize CUDA.
    """
    absolute = status.get("family_lengths_m") is not None
    lengths = status.get("family_lengths_m" if absolute else "morphology_delta_m")
    try:
        values = list(lengths)
        if len(values) != 5:
            values = [None] * 5
    except TypeError:
        values = [None] * 5
    unit_label = "Length (m)" if absolute else "Extension (mm)"
    length_text = "  ".join(
        f"{family}={_number(value, scale=1.0 if absolute else 1000.0, signed=not absolute)}"
        for family, value in zip(FAMILY_ORDER, values)
    )
    dr = status.get("dr_enabled")
    dr_text = "ON" if dr is True else "OFF" if dr is False else "unavailable"
    state = "PAUSED" if status.get("paused") else "RUNNING"
    if status.get("playback_phase") == "t_pose":
        state += " | T-POSE intermission"
    state += " | loop ON" if status.get("loop_enabled") else " | loop OFF"
    rows = [
        (
            "Motion",
            f"{_position(status.get('motion_index'), status.get('motion_count'))}  {_text(status.get('motion_name'))}",
        ),
        ("Morphology", _text(status.get("morphology_name"))),
        (unit_label, length_text),
        (
            "Playback frame",
            _position(status.get("frame_index"), status.get("frame_count")),
        ),
        ("World XY error (m)", _number(status.get("world_xy_error_m"))),
        ("DR / physics", f"{dr_text} / {_text(status.get('physics_backend'))}"),
        ("Playback", state),
        ("Keys", "Left/Right motion | custom shape fixed | R reset" if status.get("custom_shape_fixed")
         else "Left/Right motion | Up taller / Down shorter | R reset"),
        ("", "Space pause | L loop | Esc close | mouse camera"),
    ]
    if status.get("reference_body_positions_m") is not None:
        rows += [
            ("Reference", "cyan skeleton (world coordinates)"),
            ("Root XY trails", "orange actual | cyan reference | last 1024 frames"),
        ]
    rows += [
        ("Policy robot", "display-only neutral pose (not policy output)" if status.get("playback_phase") == "t_pose"
         else "solid V1.1 mesh; CPU policy + physical dynamics"),
        (
            "Model / scope",
            f"iteration {_text(status.get('model_iteration'))} | training-set display; NOT acceptance",
        ),
    ]
    if status.get("failed"):
        rows.append(("Tracking", f"FAILED at frame {status.get('first_failure_frame')}; display continues, not success"))
    if status.get("fallback"):
        rows.append(
            ("Pair selection", "requested shape absent; nearest actual 5D pair")
        )
    if status.get("height_fraction") is not None:
        # A large-font HUD must stay short enough for a normal laptop window.
        rows = [
            ("Motion", f"{_position(status.get('motion_index'), status.get('motion_count'))}  {_text(status.get('motion_name'))}"),
            ("Height", f"{status['height_fraction']:+.2f} -> {status['height_target']:+.2f}  (compact -1 / tall +1)"),
            ("Frame", _position(status.get("frame_index"), status.get("frame_count"))),
            ("Playback", state),
            ("Tracking", f"FAILED at {status.get('first_failure_frame')}; inference continues" if status.get("failed") else "Tracking"),
            ("Reference", "cyan: live-height target | orange: actual path"),
            ("Policy", "T-pose intermission; not policy output" if status.get("playback_phase") == "t_pose" else f"SuperMimic {status.get('model_iteration')} | CPU inference"),
            ("Scope", "UFO TRAIN exhibition; not trained on every clip" if status.get("external_reference") else "UFO TRAIN exhibition; not quality acceptance"),
            ("Keys", "Left/Right motion | Hold Up/Down height"),
            ("", "Space pause | R replay | L loop | Esc exit"),
        ]
    return "\n".join(row[0] for row in rows), "\n".join(row[1] for row in rows)


def skeleton_edges(names, native_ids, parent_ids):
    """Connect named reference bodies through the actual model's parent tree.

    Intermediate native bodies absent from the reference are traversed, never
    mistaken for a reference index. An unknown name fails instead of inventing
    a skeleton with matching array length but a different body order.
    """
    if len(set(names)) != len(names) or any(name not in native_ids for name in names):
        raise ValueError("Reference body names must be unique actual robot body names")
    reference_indices = {native_ids[name]: i for i, name in enumerate(names)}
    edges = []
    for child, name in enumerate(names):
        ancestor = int(parent_ids[native_ids[name]])
        visited = set()
        while ancestor != 0 and ancestor not in reference_indices:
            if ancestor in visited:
                raise ValueError("Invalid native body parent cycle")
            visited.add(ancestor)
            ancestor = int(parent_ids[ancestor])
        if ancestor in reference_indices:
            edges.append((reference_indices[ancestor], child))
    return tuple(edges)


def _finite_point(value, dimensions):
    try:
        result = tuple(float(x) for x in value)
    except (TypeError, ValueError, OverflowError):
        return None
    return (
        result
        if len(result) == dimensions and all(map(math.isfinite, result))
        else None
    )


class WorldTrace:
    """Small CPU drawing state. Coordinates are copied without alignment."""

    def __init__(self, max_points=1024):
        self.actual = deque(maxlen=max_points)
        self.reference = deque(maxlen=max_points)
        self.reference_points = ()
        self.edges = ()
        self._native_ids = None
        self._parent_ids = None
        self._names = None
        self._selection = None
        self._frame = None

    def set_topology(self, native_ids, parent_ids):
        self._native_ids, self._parent_ids = dict(native_ids), tuple(parent_ids)
        self._names = None

    def update(self, status):
        selection = (
            status.get("reset_serial"),
            status.get("motion_name"),
            status.get("morphology_name"),
        )
        frame = status.get("frame_index")
        if selection != self._selection or (
            frame is not None and self._frame is not None and frame < self._frame
        ):
            self.actual.clear()
            self.reference.clear()
            self._frame = None
        self._selection = selection
        # Paused redraws are not new samples. A reset_serial handles reset at
        # the same frame without joining unrelated episodes with a long line.
        if frame is not None and frame != self._frame:
            for key, path, z in (
                ("actual_root_xy", self.actual, 0.014),
                ("ref_root_xy", self.reference, 0.018),
            ):
                point = _finite_point(status.get(key), 2)
                # Break a path when a sample is unavailable; do not bridge it.
                path.append((*point, z) if point is not None else None)
            self._frame = frame
        raw = status.get("reference_body_positions_m")
        names = tuple(status.get("reference_body_names", ()))
        if raw is None:
            self.reference_points, self.edges = (), ()
            self._names = None
            return
        points = tuple(_finite_point(point, 3) for point in raw)
        if (
            len(names) != len(points)
            or not names
            or any(point is None for point in points)
        ):
            raise ValueError(
                "Reference overlay requires finite N x 3 positions and N body names"
            )
        if self._native_ids is None:
            raise RuntimeError(
                "Configure the native topology before drawing a reference"
            )
        if names != self._names:
            self.edges = skeleton_edges(names, self._native_ids, self._parent_ids)
            self._names = names
        self.reference_points = points

    def segments(self):
        for a, b in self.edges:
            yield (
                self.reference_points[a],
                self.reference_points[b],
                0.006,
                REFERENCE_RGBA,
            )
        for path, color in (
            (self.actual, ACTUAL_TRAIL_RGBA),
            (self.reference, REFERENCE_RGBA),
        ):
            for a, b in zip(path, list(path)[1:]):
                if a is not None and b is not None:
                    yield a, b, 0.003, color


def _draw_world_trace(mujoco, scene, trace):
    """Add decorative geoms only; called after native clears the user scene."""
    import numpy as np

    def allocate(kind, size, position, color):
        if scene.ngeom >= scene.maxgeom:
            return None
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom=geom,
            type=kind,
            size=np.asarray(size),
            pos=np.asarray(position),
            mat=np.eye(3).reshape(-1),
            rgba=np.asarray(color, dtype=np.float32),
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR.value
        scene.ngeom += 1
        return geom

    for point in trace.reference_points:
        allocate(
            mujoco.mjtGeom.mjGEOM_SPHERE.value, [0.009, 0.0, 0.0], point, REFERENCE_RGBA
        )
    for a, b, radius, color in trace.segments():
        if sum((x - y) ** 2 for x, y in zip(a, b)) < 1e-16:
            continue
        geom = allocate(
            mujoco.mjtGeom.mjGEOM_CAPSULE.value, [radius, 0.0, 0.0], a, color
        )
        if geom is None:
            break
        mujoco.mjv_connector(
            geom=geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE.value,
            width=radius,
            from_=np.asarray(a),
            to=np.asarray(b),
        )


class ReferenceSkeleton:
    """Full named-body kinematic reference on an independent data object."""

    def __init__(self, model, robot):
        import mujoco

        self.model = model
        self.data = mujoco.MjData(model)
        self.qadr, _ = robot.addresses(model)
        self.names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(1, model.nbody)
        )

    def sample(self, motion, frame, *, reference_state=None):
        import mujoco

        from .rotations import mat_to_quat

        if reference_state is None:
            p, R, q = motion.root_p[frame], motion.root_R[frame], motion.q[frame]
        else:
            # Already adapted by the active collector's exact delta/frame;
            # the independent MjData uses the same physical model morphology.
            p, R, _, _, q, _ = reference_state
        self.data.qpos[:3] = p
        self.data.qpos[3:7] = mat_to_quat(R)
        self.data.qpos[self.qadr] = q
        # No mj_step: this data is a pure target pose, never a physics robot.
        mujoco.mj_forward(self.model, self.data)
        return self.names, self.data.xpos[1:].tolist()
