"""Export an existing qualified Scaler split to a portable, non-pickle motion pack."""

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

from .robot import FAMILIES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(report_path, output, limit=None, role="train", mesh_dir=None, variant_names=None):
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    if (
        report.get("schema") != "scaler_ufo_qualified_dataset_v2"
        or report.get("role") != role
        or report.get("passed") is not True
        or report.get("temporal_qualified") is not True
        or report.get("review_only") is not False
        or (role == "train" and report.get("valid_for_training") is not True)
    ):
        raise ValueError("Require a qualified report in the requested split")
    source = Path(report["manifest_snapshot"])
    if sha(source) != report["manifest_sha256"]:
        raise ValueError("Changed source manifest")
    manifest = json.loads(source.read_text())
    config_path = Path(manifest["asset_manifest"]["robot_config"])
    config = yaml.safe_load(config_path.read_text())
    names = config["control_joints"]["names"]
    entries = {
        f"{c['name']}__{v['name']}": (c, v)
        for c in manifest["clips"]
        for v in c["variants"]
    }
    selected_variants = None if variant_names is None else sorted(set(variant_names))
    rows = report["records"]
    if selected_variants is not None:
        available = {entries[row["motion"]][1]["name"] for row in rows}
        if not selected_variants or set(selected_variants) - available:
            raise ValueError("Requested variants must be a nonempty qualified subset")
        rows = [row for row in rows if entries[row["motion"]][1]["name"] in selected_variants]
    rows = rows[:limit]
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "meshes").mkdir()
    (out / "models").mkdir()
    (out / "motions").mkdir()
    (out / "robot.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    variants = {}
    motions = []
    copied = {}
    for row in rows:
        if (
            row["role"] != role
            or row["passed"] is not True
            or row["temporal_qualified"] is not True
        ):
            raise ValueError("Unqualified motion in report")
        _clip, var = entries[row["motion"]]
        motion_path = Path(var["motion"])
        xml = Path(var["mjcf"])
        if (
            sha(motion_path) != row["inputs"]["motion"]
            or sha(xml) != row["inputs"]["mjcf"]
        ):
            raise ValueError(f"Changed input for {row['motion']}")
        delta = np.array(var["morphology_delta_m"])
        with np.load(motion_path, allow_pickle=False) as z:
            if list(z["joint_names"]) != names or not np.allclose(
                z["morphology_delta_m"], delta, atol=1e-9, rtol=0
            ):
                raise ValueError("Joint order/morphology mismatch")
            p, q, j = z["root_pos"], z["root_quat"], z["dof_pos"]
            fps = float(z["fps"])
            if (
                p.shape != (len(j), 3)
                or q.shape != (len(j), 4)
                or j.shape[1] != 24
                or len(j) < 3
                or not np.isfinite(fps)
                or fps <= 0
            ):
                raise ValueError("Invalid motion shapes/FPS")
            if not all(
                np.isfinite(a).all() for a in (p, q, j, delta)
            ) or not np.allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-5):
                raise ValueError("Invalid finite pose/quaternion")
            filename = f"motions/{len(motions):05d}.npz"
            np.savez(out / filename, root_p=p, root_quat=q, joint_pos=j, fps=fps)
        if var["name"] not in variants:
            tree = ET.parse(xml)
            root = tree.getroot()
            compiler = root.find("compiler")
            meshdir = (
                Path(mesh_dir).resolve()
                if mesh_dir
                else (xml.parent / compiler.get("meshdir", ".")).resolve()
            )
            for mesh in root.findall("asset/mesh"):
                src = (meshdir / mesh.get("file")).resolve()
                key = src.name
                digest = sha(src)
                expected = {
                    a["name"]: a["sha256"]
                    for a in report["asset_contract"]["mesh_assets"]
                }
                if expected.get(mesh.get("name")) != digest:
                    raise ValueError("Mesh differs from qualified asset")
                if key in copied and copied[key] != digest:
                    raise ValueError("Mesh filename collision")
                if key not in copied:
                    shutil.copyfile(src, out / "meshes" / key)
                    copied[key] = digest
                mesh.set("file", key)
            compiler.set("meshdir", "../meshes")
            model_name = f"models/{len(variants):03d}.xml"
            tree.write(out / model_name)
            variants[var["name"]] = {
                "index": len(variants),
                "name": var["name"],
                "delta_m": delta.tolist(),
                "model": model_name,
                "source_xml_sha256": sha(xml),
            }
        v = variants[var["name"]]
        if v["delta_m"] != delta.tolist() or v["source_xml_sha256"] != sha(xml):
            raise ValueError("Inconsistent variant")
        motions.append(
            {
                "name": row["motion"],
                "file": filename,
                "shape": v["index"],
                "recording_group": row["recording_group"],
                "source_sha256": row["inputs"]["motion"],
                "sha256": sha(out / filename),
                "frames": len(j),
                "fps": fps,
            }
        )
    if not motions:
        raise ValueError("Empty split")
    pack = {
        "schema": "scaler_supermimic_motion_pack_v1",
        "role": role,
        "families": FAMILIES,
        "source_report": str(report_path),
        "source_report_sha256": sha(report_path),
        "source_manifest_sha256": sha(source),
        "robot": "robot.yaml",
        "joint_names": names,
        "variants": list(variants.values()),
        "motions": motions,
        "mesh_sha256": copied,
        "dynamics_validated": False,
        "training_subset": limit is not None,
    }
    if selected_variants is not None:
        pack["qualification_selection"] = {
            "variant_names": selected_variants,
            "source_qualified_motion_count": len(report["records"]),
            "selected_motion_count": len(motions),
            "limit": limit,
            "purpose": "Predeclared morphology split; no selection by model evaluation results",
        }
    pack["files"] = {
        str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()
    }
    (out / "manifest.json").write_text(json.dumps(pack, indent=2) + "\n")
    return pack


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qualification", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument(
        "--role", choices=["train", "validation", "heldout"], default="train"
    )
    p.add_argument("--mesh-dir")
    p.add_argument("--variant-names", nargs="+", help="Explicit qualified variants; selected before any model evaluation")
    a = p.parse_args()
    m = prepare(a.qualification, a.output, a.limit, a.role, a.mesh_dir, a.variant_names)
    print(
        json.dumps(
            {
                "motions": len(m["motions"]),
                "shapes": len(m["variants"]),
                "role": m["role"],
            }
        )
    )


if __name__ == "__main__":
    main()
