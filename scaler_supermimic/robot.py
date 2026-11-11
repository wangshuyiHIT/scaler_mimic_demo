"""Scaler-specific motor and rigid-body contracts; no G1 motor substitutions."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yaml

FAMILIES = ("thigh", "calf", "torso", "upperarm", "forearm")
DT = 0.02
TRACKED = (
    "base_link",
    "torso_pitch_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "left_knee_link",
    "right_knee_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_forearm_length_link",
    "right_forearm_length_link",
)


class Robot:
    def __init__(self, config):
        self.config = config
        self.names = tuple(config["control_joints"]["names"])
        if len(self.names) != 24 or len(set(self.names)) != 24:
            raise ValueError("Expected 24 explicitly ordered Scaler motor joints")
        c = config["training"]["control"]
        motor = config["training"]["actuator"]["joints"]
        self.default_q = np.array([config["default_dof_pos"][n] for n in self.names])

        def gains(key):
            values = []
            for name in self.names:
                matches = [v for k, v in c[key].items() if k in name]
                if len(matches) != 1:
                    raise ValueError(f"Ambiguous motor gain: {name}")
                values.append(matches[0])
            return np.array(values)

        self.kp, self.kd = gains("stiffness"), gains("damping")
        self.effort = np.array([motor[n]["effort_limit"] for n in self.names])
        self.speed = np.array([motor[n]["velocity_limit"] for n in self.names])
        self.armature = np.array([motor[n]["armature"] for n in self.names])
        self.friction = np.array([motor[n]["friction"] for n in self.names])
        self.action_scale, self.action_clip = c["action_scale"], c["action_clip_value"]

    @classmethod
    def from_yaml(cls, path):
        return cls(yaml.safe_load(Path(path).read_text()))

    def load_model(self, path):
        """Keep original bodies, inertias and collisions; replace actuator plumbing with torques."""
        path = Path(path).resolve()
        root = ET.parse(path).getroot()
        comp = root.find("compiler")
        comp.set("meshdir", str((path.parent / comp.get("meshdir", ".")).resolve()))
        opt = root.find("option")
        if opt is None:
            opt = ET.SubElement(root, "option")
        opt.set("timestep", "0.002")
        opt.set("integrator", "implicitfast")
        if root.find("worldbody/geom[@name='supermimic_ground']") is None:
            ET.SubElement(
                root.find("worldbody"),
                "geom",
                name="supermimic_ground",
                type="plane",
                size="0 0 0.05",
                friction="1 0.005 0.0001",
                rgba="0.3 0.3 0.3 1",
            )
        old = root.find("actuator")
        if old is not None:
            root.remove(old)
        actuators = ET.SubElement(root, "actuator")
        for i, n in enumerate(self.names):
            ET.SubElement(
                actuators,
                "motor",
                name=n + "_motor",
                joint=n,
                gear="1",
                ctrllimited="true",
                ctrlrange=f"{-self.effort[i]} {self.effort[i]}",
            )
        m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        if m.nq != 31 or m.nv != 30 or m.nu != 24:
            raise ValueError("Expected baked 24-DOF Scaler")
        for i, n in enumerate(self.names):
            j = m.joint(n).id
            adr = m.jnt_dofadr[j]
            m.dof_armature[adr] = self.armature[i]
            m.dof_frictionloss[adr] = self.friction[i]
        return m

    def addresses(self, m):
        ids = np.array([m.joint(n).id for n in self.names])
        return m.jnt_qposadr[ids], m.jnt_dofadr[ids]
