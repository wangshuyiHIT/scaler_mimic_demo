"""Current URDF appearance on a separate, never-stepped MuJoCo display model."""

import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


class VisualAssets:
    def __init__(self, urdf):
        self.urdf = Path(urdf).resolve(strict=True)
        raw = self.urdf.read_bytes()
        root = ET.fromstring(raw)
        self.links, self.assets = {}, {}
        for link in root.findall("link"):
            visual = link.find("visual")
            if visual is None:
                continue
            if len(link.findall("visual")) != 1:
                raise ValueError("Ambiguous URDF visuals: " + link.get("name"))
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                raise ValueError("Expected a mesh visual: " + link.get("name"))
            path = (self.urdf.parent / mesh.get("filename")).resolve(strict=True)
            body = path.read_bytes()
            stat = path.stat()
            self.assets[str(path)] = {
                "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body),
                "stat": [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns],
            }
            origin, color = visual.find("origin"), visual.find("material/color")
            self.links[link.get("name")] = {
                "path": str(path),
                "scale": mesh.get("scale", "1 1 1"),
                "pos": origin.get("xyz", "0 0 0") if origin is not None else "0 0 0",
                "euler": origin.get("rpy", "0 0 0") if origin is not None else "0 0 0",
                "rgba": color.get("rgba", "0.7 0.7 0.7 1") if color is not None else "0.7 0.7 0.7 1",
            }
        self.identity = {
            "urdf": str(self.urdf),
            "urdf_sha256": hashlib.sha256(raw).hexdigest(),
            "mesh_assets": self.assets,
            "visual_links": len(self.links),
            "physics_source": "unchanged checkpoint training pack",
            "render_model_stepped": False,
        }

    def model(self, baked_xml):
        root = ET.parse(baked_xml).getroot()
        compiler = root.find("compiler")
        compiler.set("meshdir", str(Path(baked_xml).parent.resolve()))
        asset = root.find("asset")
        by_name = {m.get("name"): m for m in asset.findall("mesh")}
        mapped = set()
        for body in root.findall(".//body"):
            name = body.get("name")
            visual = self.links.get(name)
            if visual is None:
                continue
            geoms = [g for g in body.findall("geom") if g.get("name") == name + "_source_geom_0"]
            if len(geoms) != 1:
                raise ValueError("Cannot map current URDF visual to baked body: " + name)
            geom = geoms[0]
            mesh = by_name[geom.get("mesh")]
            mesh.set("file", visual["path"])
            mesh.set("scale", visual["scale"])
            for key in ("quat", "axisangle", "xyaxes", "zaxis", "euler"):
                geom.attrib.pop(key, None)
            for key in ("pos", "rgba"):
                geom.set(key, visual[key])
            # URDF rpy is fixed-axis Rz(yaw) Ry(pitch) Rx(roll).
            r, p, y = (v / 2 for v in map(float, visual["euler"].split()))
            cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
            quat = (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)
            geom.set("quat", " ".join(map(str, quat)))
            mapped.add(name)
        if mapped != set(self.links):
            raise ValueError("URDF links absent from baked model: " + str(set(self.links) - mapped))
        # The display includes current mesh visuals plus the baked telescopic
        # profiles. Collision proxies remain exclusively in the physical model.
        for geom in root.findall(".//geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            if geom.get("name", "").endswith("_foot_col"):
                geom.set("rgba", "0 0 0 0")
        ET.SubElement(root.find("worldbody"), "geom", name="display_ground", type="plane",
                      size="0 0 0.05", rgba="0.3 0.3 0.3 1", contype="0", conaffinity="0")
        for path, pin in self.assets.items():
            st = Path(path).stat()
            if [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns] != pin["stat"]:
                raise ValueError("Visual asset changed during this viewer session")
        return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class Display:
    def __init__(self, assets, baked_xml, physics_model, physics_data, live_morphology=False):
        self.model = assets.model(baked_xml)
        self.data = mujoco.MjData(self.model)
        if (self.model.nq, self.model.nv, self.model.nbody) != (
            physics_model.nq, physics_model.nv, physics_model.nbody
        ):
            raise ValueError("Display and physical state topology differ")
        for kind, count in ((mujoco.mjtObj.mjOBJ_BODY, self.model.nbody),
                            (mujoco.mjtObj.mjOBJ_JOINT, self.model.njnt)):
            if any(mujoco.mj_id2name(self.model, kind, i) != mujoco.mj_id2name(physics_model, kind, i)
                   for i in range(count)):
                raise ValueError("Display and physical body/joint order differ")
        if live_morphology:
            self.sync_morphology(physics_model)
        self.sync(physics_data)
        if not np.allclose(self.data.xpos, physics_data.xpos, rtol=0, atol=1e-9):
            raise ValueError("Display link poses differ from physical morphology")

    def sync_morphology(self, physical):
        """Update native display arrays in place; never replace the GL model."""
        import numpy as np
        if np.array_equal(self.model.body_pos, physical.body_pos):
            return
        self.model.body_pos[:] = physical.body_pos
        # Preserve V1.1 mesh origins/colors. Only telescopic profile boxes vary.
        for g in range(self.model.ngeom):
            if self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            name = self.model.geom(g).name
            source = mujoco.mj_name2id(physical, mujoco.mjtObj.mjOBJ_GEOM, name)
            if source >= 0:
                for field in ("geom_size", "geom_pos", "geom_quat", "geom_rbound", "geom_aabb"):
                    getattr(self.model, field)[g] = getattr(physical, field)[source]

    def sync(self, physics_data):
        self.data.qpos[:] = physics_data.qpos
        self.data.qvel[:] = physics_data.qvel
        self.data.time = physics_data.time
        mujoco.mj_forward(self.model, self.data)

    def show_t_pose(self, root_xy):
        """Show Scaler's zero-joint T-pose without touching or stepping physics."""
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[7:] = 0
        self.data.qpos[:2] = root_xy
        self.data.qpos[2] = 0
        self.data.qpos[3:7] = (1, 0, 0, 0)
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        # Rest the foot collision boxes on the floor for each baked morphology.
        bottom = []
        for name in ("left_foot_col", "right_foot_col"):
            gid = self.model.geom(name).id
            rotation = self.data.geom_xmat[gid].reshape(3, 3)
            bottom.append(self.data.geom_xpos[gid, 2] - abs(rotation[2]) @ self.model.geom_size[gid])
        self.data.qpos[2] -= min(bottom)
        mujoco.mj_forward(self.model, self.data)
