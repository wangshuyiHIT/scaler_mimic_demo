"""Explicit Flat morphology sampling and MuJoCo field construction.

No simulator step, mesh copy, nearest-shape lookup, reference retargeting or GPU
operation occurs here. The caller writes all returned per-world fields before
MuJoCo-Warp ``set_const``. Samples are startup-fixed, except fields explicitly
labelled reset/interval. Pose-conditioned reference FK is a separate concern.
"""

import json
import re
from pathlib import Path

import numpy as np

FAMILIES = ("thigh", "calf", "torso", "upperarm", "forearm")


def load_config(path=None):
    path = (
        path
        or Path(__file__).resolve().parents[1]
        / "configs/domain_randomization_flat.json"
    )
    return json.loads(Path(path).read_text())


def _count(n):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("world count must be a positive integer")


def sample_morphology(
    n, rng, family_bounds_m, *, sigma_m=0.02, clip_m=0.05, anchor_fraction=0.15
):
    """Flat's independent Gaussian + compact/nominal/extended anchors, in metres.

    ``requested_delta_m`` is the Gaussian/anchor result; ``applied_delta_m`` is
    additionally clamped to real family travel. Conditioning must use the latter.
    RNG is caller-owned; the helper never reseeds or samples at reset implicitly.
    """
    _count(n)
    bounds = np.asarray(family_bounds_m, dtype=np.float64)
    if (
        bounds.shape != (5, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] > 0)
        or np.any(bounds[:, 1] < 0)
    ):
        raise ValueError(
            "Expected five finite physical travel intervals containing zero"
        )
    if (
        not np.isfinite([sigma_m, clip_m, anchor_fraction]).all()
        or sigma_m < 0
        or clip_m < 0
        or not 0 <= anchor_fraction <= 1
    ):
        raise ValueError("Invalid morphology sampling parameters")
    requested = np.clip(rng.normal(0, sigma_m, (n, 5)), -clip_m, clip_m)
    anchor_mask = np.zeros(n, dtype=bool)
    count = min(n, round(n * anchor_fraction))
    if count:
        rows = rng.permutation(n)[:count]
        counts = [count // 3 + int(i < count % 3) for i in range(3)]
        start = 0
        for value, size in zip((-clip_m, 0.0, clip_m), counts):
            selected = rows[start : start + size]
            requested[selected] = value
            anchor_mask[selected] = True
            start += size
    return {
        "requested_delta_m": requested,
        "applied_delta_m": np.clip(requested, bounds[:, 0], bounds[:, 1]),
        "anchor_mask": anchor_mask,
    }


def _parallel_axis(mass, displacement):
    return mass[..., None, None] * (
        np.sum(displacement * displacement, axis=-1)[..., None, None] * np.eye(3)
        - displacement[..., :, None] * displacement[..., None, :]
    )


def profile_inertial(profile, delta):
    """Vectorized Scaler carrier + resized uniform profile, about the true COM.

    This is the same parallel-axis construction as
    scaler_description.morphology.compute_length_link_inertial. Metadata is the
    asset's exact UFO morphology metadata, not a different Flat asset's masses.
    """
    delta = np.asarray(delta, dtype=np.float64)
    length = float(profile["nominal_length_m"]) + delta
    if not np.isfinite(length).all() or np.any(length <= 0):
        raise ValueError("Nonpositive or nonfinite physical profile length")
    axis = np.asarray(profile["axis"], dtype=np.float64)
    rotation = np.asarray(profile["rotation_matrix"], dtype=np.float64)
    if (
        not np.isclose(np.linalg.norm(axis), 1)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
        or not np.isclose(np.linalg.det(rotation), 1)
    ):
        raise ValueError("Invalid profile axes")
    center = np.asarray(profile["nominal_center_m"]) - 0.5 * delta[..., None] * axis
    pmass = float(profile["linear_density_kg_m"]) * length
    cmass = float(profile["carrier_mass_kg"])
    mass = cmass + pmass
    if cmass <= 0 or np.any(pmass <= 0):
        raise ValueError("Nonpositive carrier/profile mass")
    ccom = np.asarray(profile["carrier_com_m"])
    com = (cmass * ccom + pmass[..., None] * center) / mass[..., None]
    x, y = profile["section_xy_m"]
    diagonal = (
        pmass[..., None]
        / 12
        * np.stack(
            (
                y * y + length * length,
                x * x + length * length,
                np.full_like(length, x * x + y * y),
            ),
            -1,
        )
    )
    pinertia = np.einsum("ij,...j,kj->...ik", rotation, diagonal, rotation)
    inertia = (
        np.asarray(profile["carrier_inertia_kg_m2"])
        + _parallel_axis(np.asarray(cmass), ccom - com)
        + pinertia
        + _parallel_axis(pmass, center - com)
    )
    return mass, com, inertia, length


def _matrix_quaternion(rotation):
    """Proper rotation matrices -> normalized MuJoCo wxyz quaternions."""
    shape = rotation.shape[:-2]
    out = np.empty(shape + (4,))
    for r, q in zip(rotation.reshape(-1, 3, 3), out.reshape(-1, 4)):
        trace = np.trace(r)
        if trace > 0:
            s = 2 * np.sqrt(trace + 1)
            q[:] = [
                s / 4,
                (r[2, 1] - r[1, 2]) / s,
                (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s,
            ]
        else:
            i = int(np.argmax(np.diag(r)))
            j, k = (i + 1) % 3, (i + 2) % 3
            s = 2 * np.sqrt(max(0.0, 1 + r[i, i] - r[j, j] - r[k, k]))
            q[0] = (r[k, j] - r[j, k]) / s
            q[i + 1] = s / 4
            q[j + 1] = (r[j, i] + r[i, j]) / s
            q[k + 1] = (r[k, i] + r[i, k]) / s
        q /= np.linalg.norm(q)
        if q[0] < 0:
            q *= -1
    return out


def _quaternion_matrix(q):
    w, x, y, z = np.moveaxis(np.asarray(q), -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(q.shape[:-1] + (3, 3))


class MorphologyModelFields:
    """Small per-world arrays from one verified nominal model; meshes are shared.

    The nominal model must already contain the baked, fixed length bodies and
    nine physical profile boxes. Its profile geometry and inertials are checked
    against metadata at delta=0 to reject double baking. No MjModel is cloned.
    """

    FIELD_NAMES = (
        "body_pos",
        "body_mass",
        "body_ipos",
        "body_inertia",
        "body_iquat",
        "geom_pos",
        "geom_size",
        "geom_rbound",
        "geom_aabb",
    )

    def __init__(self, nominal_model, metadata):
        if tuple(metadata["family_names"]) != FAMILIES:
            raise ValueError("Morphology family order differs")
        self.bounds = np.asarray(metadata["family_bounds_m"], dtype=np.float64)
        if self.bounds.shape != (5, 2) or not np.isfinite(self.bounds).all():
            raise ValueError("Invalid physical family bounds")
        self.base = {
            name: np.asarray(getattr(nominal_model, name)).copy()
            for name in self.FIELD_NAMES
        }
        self.offset_coefficients = np.zeros((5, nominal_model.nbody, 3))
        offsets = {x["body_name"]: x for x in metadata["body_offsets"]}
        self.profiles = []
        for profile in metadata["profiles"]:
            b = nominal_model.body(profile["body_name"]).id
            g = nominal_model.geom(profile["geom_name"]).id
            f = int(profile["family_index"])
            offset = offsets[profile["body_name"]]
            if not 0 <= f < 5 or int(offset["family_index"]) != f:
                raise ValueError("Profile/offset family mismatch")
            if (
                nominal_model.body_jntnum[b] != 0
                or nominal_model.geom_bodyid[g] != b
                or nominal_model.geom_type[g] != 6
            ):
                raise ValueError("Expected a fixed length body with an attached box")
            mass, com, inertia, length = profile_inertial(profile, np.asarray(0.0))
            r = _quaternion_matrix(self.base["body_iquat"][b])
            actual_inertia = (r * self.base["body_inertia"][b]) @ r.T
            checks = (
                (self.base["body_mass"][b], mass),
                (self.base["body_ipos"][b], com),
                (actual_inertia, inertia),
                (self.base["geom_pos"][g], profile["nominal_center_m"]),
                (
                    self.base["geom_size"][g],
                    np.array([*profile["section_xy_m"], length]) / 2,
                ),
            )
            if any(
                not np.allclose(a, expected, atol=2e-7, rtol=2e-6)
                for a, expected in checks
            ):
                raise ValueError(
                    "Model is not the metadata's nominal carrier; refusing double bake"
                )
            if not np.allclose(
                _quaternion_matrix(nominal_model.geom_quat[g]),
                profile["rotation_matrix"],
                atol=2e-6,
            ):
                raise ValueError("Profile box rotation differs from metadata")
            self.offset_coefficients[f, b] = offset["axis"]
            self.profiles.append((b, g, f, profile))
        if len({b for b, _, _, _ in self.profiles}) != len(self.profiles) or len(
            self.profiles
        ) != len(offsets):
            raise ValueError("Duplicate/missing profile bodies")

    def fields(self, applied_delta_m):
        """Return batched fields. Caller must subsequently recompute set_const.

        geom_aabb is in each geom's local frame, not its body's frame. Profile
        changes update both AABB and bounding sphere; mesh bounds are unchanged.
        body_subtreemass/inverse inertias/etc. are derived by set_const, not here.
        """
        delta = np.asarray(applied_delta_m, dtype=np.float64)
        if (
            delta.ndim != 2
            or delta.shape[1] != 5
            or not len(delta)
            or not np.isfinite(delta).all()
        ):
            raise ValueError("Expected finite [world,5] applied deltas in metres")
        if np.any(delta < self.bounds[:, 0] - 1e-12) or np.any(
            delta > self.bounds[:, 1] + 1e-12
        ):
            raise ValueError("Applied delta exceeds physical joint travel")
        fields = {
            k: np.broadcast_to(v, (len(delta),) + v.shape).copy()
            for k, v in self.base.items()
        }
        fields["body_pos"] += np.einsum("nf,fbi->nbi", delta, self.offset_coefficients)
        for b, g, f, profile in self.profiles:
            d = delta[:, f]
            mass, com, inertia, length = profile_inertial(profile, d)
            eigen, axes = np.linalg.eigh(inertia)
            if np.any(eigen <= 0) or np.any(eigen[:, 2] > eigen[:, :2].sum(-1) + 1e-10):
                raise ValueError("Nonphysical principal inertia")
            axes[:, :, 2] *= np.linalg.det(axes)[:, None]
            fields["body_mass"][:, b] = mass
            fields["body_ipos"][:, b] = com
            fields["body_inertia"][:, b] = eigen
            fields["body_iquat"][:, b] = _matrix_quaternion(axes)
            fields["geom_pos"][:, g] = np.asarray(
                profile["nominal_center_m"]
            ) - 0.5 * d[:, None] * np.asarray(profile["axis"])
            half = np.broadcast_to(
                np.array([*profile["section_xy_m"], 0.0]) / 2, (len(d), 3)
            ).copy()
            half[:, 2] = length / 2
            fields["geom_size"][:, g] = half
            fields["geom_rbound"][:, g] = np.linalg.norm(half, axis=-1)
            fields["geom_aabb"][:, g] = np.concatenate((np.zeros_like(half), half), -1)
        return fields


def sample_nuisance(
    n, rng, body_names, joint_names, geom_bodyid, *, config=None, physics_dt=0.001
):
    """Return startup Flat nuisance samples, not a simulator mutation.

    Startup material buckets are generated once per call, selected per geom.
    Call sample_reset/sample_push at their own event times; never redraw this
    startup physics merely because a tracking clip reset.
    Unsupported restitution/contact semantics remain visible in ``semantics``.
    """
    _count(n)
    c = (config or load_config())["nuisance"]
    B, J = len(body_names), len(joint_names)
    geom_bodyid = np.asarray(geom_bodyid, dtype=int)
    if (
        J != 24
        or len(set(joint_names)) != J
        or geom_bodyid.ndim != 1
        or np.any(geom_bodyid < 0)
        or np.any(geom_bodyid >= B)
    ):
        raise ValueError("Expected 24 ordered controls and valid geom-to-body mapping")
    if not np.isfinite(physics_dt) or physics_dt <= 0:
        raise ValueError("Invalid physics timestep")
    uniform = lambda bounds, shape: rng.uniform(*bounds, size=shape)
    out = {
        "body_mass_add_kg": np.zeros((n, B)),
        "body_mass_scale": np.ones((n, B)),
        "body_com_offset_m": np.zeros((n, B, 3)),
    }
    selected = [i for i, name in enumerate(body_names) if name in c["mass_com_bodies"]]
    if len(selected) != len(c["mass_com_bodies"]):
        raise ValueError("Missing Flat base/torso nuisance body")
    out["body_mass_add_kg"][:, selected] = uniform(
        c["base_mass_add_kg"], (n, len(selected))
    )
    out["body_com_offset_m"][:, selected] = uniform(c["com_offset_m"], (n, 1, 3))
    limbs = [
        i
        for i, name in enumerate(body_names)
        if any(re.fullmatch(pattern, name or "") for pattern in c["link_mass_patterns"])
    ]
    out["body_mass_scale"][:, limbs] = uniform(c["link_mass_scale"], (n, len(limbs)))
    for name in ("kp_scale", "kd_scale", "joint_friction_scale", "armature_scale"):
        out[name] = uniform(c[name], (n, J))
    material = c["material"]
    buckets = np.stack(
        [
            uniform(material[key], material["buckets"])
            for key in ("static_friction", "dynamic_friction", "restitution")
        ],
        -1,
    )
    buckets[:, 1] = np.minimum(buckets[:, 1], buckets[:, 0])
    idx = rng.integers(0, material["buckets"], (n, len(geom_bodyid)))
    values = buckets[idx]
    values[:, geom_bodyid == 0] = [1.0, 1.0, 0.0]
    out.update(
        material_buckets=buckets,
        material_bucket_index=idx,
        static_friction=values[:, :, 0],
        dynamic_friction=values[:, :, 1],
        restitution=values[:, :, 2],
    )
    out["semantics"] = c["semantics"]
    out["observation_noise_scales"] = c["observation_noise_scales"]
    return out


def sample_reset(n, rng, joint_names, *, config=None, physics_dt=0.001):
    """Fresh reference-relative reset perturbations and group delay, no physics redraw."""
    _count(n)
    c = (config or load_config())["nuisance"]
    if len(joint_names) != 24 or len(set(joint_names)) != 24:
        raise ValueError("Expected 24 ordered control joints")
    if not np.isfinite(physics_dt) or physics_dt <= 0:
        raise ValueError("Invalid physics timestep")
    uniform = lambda bounds, shape: rng.uniform(*bounds, size=shape)
    groups = c["delay"]["groups"]
    mapping = []
    for name in joint_names:
        found = [
            i
            for i, patterns in enumerate(groups.values())
            if any(re.fullmatch(p, name) for p in patterns)
        ]
        if len(found) != 1:
            raise ValueError(f"Missing/ambiguous actuator delay group: {name}")
        mapping.append(found[0])
    support = (
        np.asarray(c["delay"]["source_lag_steps"])
        * c["delay"]["source_physics_dt_s"]
        / physics_dt
    )
    if not np.allclose(support, np.rint(support), atol=1e-10, rtol=0):
        raise ValueError(
            "Physics timestep cannot exactly represent Flat 0/5/10ms delays"
        )
    ticks = rng.choice(c["delay"]["source_lag_steps"], (n, len(groups)))
    seconds = ticks * c["delay"]["source_physics_dt_s"]
    steps = np.rint(seconds / physics_dt).astype(np.int64)
    out = {
        "joint_delay_group": np.array(mapping),
        "source_delay_seconds": seconds,
        "delay_physics_steps": steps,
    }
    out["reset_root_pose"] = np.stack(
        [
            uniform(c["reset_pose"][k], n)
            for k in ("x", "y", "z", "roll", "pitch", "yaw")
        ],
        -1,
    )
    out["reset_root_velocity"] = np.stack(
        [
            uniform(c["velocity_ranges"][k], n)
            for k in ("x", "y", "z", "roll", "pitch", "yaw")
        ],
        -1,
    )
    out["reset_joint_offset_rad"] = uniform(
        c["reset_joint_offset_rad"], (n, len(joint_names))
    )
    out["reset_joint_velocity_rad_s"] = np.zeros((n, len(joint_names)))
    return out


def sample_push(n, rng, *, config=None):
    """World additive delta-v/delta-omega plus next per-world physical-time interval.

    Caller maintains the clock across tracking resets, as the actual Flat class
    does. Sampling an event is not applying a force or overwriting root velocity.
    """
    _count(n)
    c = (config or load_config())["nuisance"]
    uniform = lambda bounds, shape: rng.uniform(*bounds, size=shape)
    out = {
        "push_velocity": np.stack(
            [
                uniform(c["velocity_ranges"][k], n)
                for k in ("x", "y", "z", "roll", "pitch", "yaw")
            ],
            -1,
        )
    }
    out["push_interval_s"] = uniform(c["push_interval_s"], n)
    return out
