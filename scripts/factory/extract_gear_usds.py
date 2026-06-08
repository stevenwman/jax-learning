"""Phase 0 (GearMesh): pull factory_gear_*.usd from S3 + extract to OBJ.

Mirrors `extract_usds.py` (PegInsert pipeline) but targets the GearMesh
asset set and SKIPS CoACD decomposition — gears use SDF-vs-SDF contacts
via `<geom type="sdf">`, so the raw triangulated mesh is what we want.
Tooth-on-tooth contact through CoACD would produce hundreds of convex
pieces per gear.

Outputs:
  jax_rl/envs/manipulation/factory/assets/gear_mesh/extracted/{gear}.obj
  jax_rl/envs/manipulation/factory/assets/gear_mesh/extracted/metadata.json
  .tmp/factory_usds/factory_gear_*.usd                              (cache)

Usage:
  uv run python scripts/factory/extract_gear_usds.py
"""

import json
import urllib.request
from pathlib import Path

import numpy as np
import trimesh

from pxr import Usd, UsdGeom

REPO_ROOT = Path(__file__).parents[2]
USD_CACHE = REPO_ROOT / ".tmp" / "factory_usds"
EXTRACTED = REPO_ROOT / "jax_rl" / "envs" / "manipulation" / "factory" / "assets" / "gear_mesh" / "extracted"
USD_CACHE.mkdir(parents=True, exist_ok=True)
EXTRACTED.mkdir(parents=True, exist_ok=True)

S3_BASE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/IsaacLab/Factory"

FILES = {
    "gear_base":   "factory_gear_base.usd",
    "gear_small":  "factory_gear_small.usd",
    "gear_medium": "factory_gear_medium.usd",
    "gear_large":  "factory_gear_large.usd",
}


def download_usd(filename: str) -> Path:
    local = USD_CACHE / filename
    if local.exists():
        print(f"  [cache hit] {filename}")
        return local
    url = f"{S3_BASE}/{filename}"
    print(f"  Downloading {url}")
    urllib.request.urlretrieve(url, local)
    print(f"  Saved {local} ({local.stat().st_size // 1024} KB)")
    return local


def _fan_triangulate(face_vertex_counts, face_vertex_indices):
    tris = []
    idx = 0
    for count in face_vertex_counts:
        v0 = face_vertex_indices[idx]
        for k in range(1, count - 1):
            tris.append([v0,
                         face_vertex_indices[idx + k],
                         face_vertex_indices[idx + k + 1]])
        idx += count
    return np.array(tris, dtype=np.int32)


def extract_mesh_from_usd(usd_path: Path) -> tuple[trimesh.Trimesh, dict]:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Stage.Open returned None for {usd_path}")

    up_axis = UsdGeom.GetStageUpAxis(stage)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)

    all_verts, all_tris, prim_infos = [], [], []
    vert_offset = 0
    # USD Factory assets ship BOTH /visuals and /collisions Mesh prims for
    # the same geometry (identical point counts). Merging both into one OBJ
    # produces duplicate co-located faces, which makes type="sdf" geom
    # behaviour fall apart (sign discontinuities → NaN during contacts).
    # Filter to /collisions only if any path includes that suffix; else
    # keep all (fallback for assets without the split).
    all_mesh_paths = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    has_collisions = any("/collisions" in p for p in all_mesh_paths)
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if has_collisions and "/collisions" not in str(prim.GetPath()):
            continue  # skip /visuals when both exist
        mesh = UsdGeom.Mesh(prim)
        pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        fvc = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        fvi = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        if pts.size == 0 or fvc.size == 0:
            continue
        tris_local = _fan_triangulate(fvc, fvi)
        # Apply any local-to-world transform
        xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        m = np.asarray(xform, dtype=np.float64).reshape(4, 4)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts_world = (pts_h @ m)[:, :3]
        all_verts.append(pts_world)
        all_tris.append(tris_local + vert_offset)
        prim_infos.append({
            "prim_path": str(prim.GetPath()),
            "n_verts": int(len(pts)),
            "n_tris": int(len(tris_local)),
        })
        vert_offset += len(pts)

    if not all_verts:
        raise RuntimeError(f"No UsdGeom.Mesh prims found in {usd_path}")

    verts = np.concatenate(all_verts, axis=0)
    tris = np.concatenate(all_tris, axis=0)
    mesh = trimesh.Trimesh(vertices=verts, faces=tris, process=True)

    # Scale to metres if USD declares mm or cm
    if abs(meters_per_unit - 1.0) > 1e-9:
        mesh.apply_scale(meters_per_unit)

    # Rotate Y-up → Z-up if needed
    if up_axis == "Y":
        R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        mesh.apply_transform(R)

    meta = {
        "up_axis": up_axis,
        "meters_per_unit": meters_per_unit,
        "prim_meshes": prim_infos,
        "bounds_after_scale_m": mesh.bounds.tolist(),
        "extents_m": mesh.extents.tolist(),
        "volume_m3": float(mesh.volume) if mesh.is_watertight else None,
        "watertight": bool(mesh.is_watertight),
        "n_verts_total": int(len(mesh.vertices)),
        "n_tris_total": int(len(mesh.faces)),
    }
    return mesh, meta


def main():
    print(f"USD cache:   {USD_CACHE}")
    print(f"Extracted:   {EXTRACTED}")
    print()

    metadata: dict[str, dict] = {}
    for key, filename in FILES.items():
        print(f"━━━ {key} ━━━")
        usd_path = download_usd(filename)
        mesh, meta = extract_mesh_from_usd(usd_path)
        out_obj = EXTRACTED / f"{key}.obj"
        mesh.export(out_obj)
        meta["obj_path"] = str(out_obj.relative_to(REPO_ROOT))
        metadata[key] = meta
        print(f"  bounds (m):  {np.asarray(meta['bounds_after_scale_m']).round(4).tolist()}")
        print(f"  extents (m): {np.asarray(meta['extents_m']).round(4).tolist()}")
        print(f"  verts/tris:  {meta['n_verts_total']} / {meta['n_tris_total']}")
        print(f"  watertight:  {meta['watertight']}")
        print(f"  → {out_obj}")
        print()

    with open(EXTRACTED / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata: {EXTRACTED / 'metadata.json'}")


if __name__ == "__main__":
    main()
