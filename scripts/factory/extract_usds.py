"""Phase 0a: USD mesh-only extract for Factory peg + hole.

Downloads factory_peg_8mm.usd and factory_hole_8mm.usd from NVIDIA's
public S3 mirror. Traverses each USD stage with pxr.Usd to extract
triangle meshes directly to OBJ. CoACD-decomposes the hole (non-convex
annulus). Harvests USD metadata to metadata.json for audit vs. IsaacLab
factory_tasks_cfg.py values.

No URDF middleman — pxr.Usd directly to OBJ.

Outputs:
  jax_rl/envs/manipulation/factory/assets/peg_insert/extracted/peg_8mm.obj
  jax_rl/envs/manipulation/factory/assets/peg_insert/extracted/hole_8mm_raw.obj
  jax_rl/envs/manipulation/factory/assets/peg_insert/extracted/hole_decomp/hole_*.obj
  jax_rl/envs/manipulation/factory/assets/peg_insert/extracted/metadata.json
  .tmp/factory_usds/factory_peg_8mm.usd       (cache; gitignored)
  .tmp/factory_usds/factory_hole_8mm.usd      (cache; gitignored)

Usage:
  uv run python scripts/factory/extract_usds.py
"""

import json
import urllib.request
from pathlib import Path

import numpy as np
import trimesh

from pxr import Usd, UsdGeom, UsdPhysics, Gf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parents[2]
USD_CACHE = REPO_ROOT / ".tmp" / "factory_usds"
EXTRACTED = REPO_ROOT / "jax_rl" / "envs" / "manipulation" / "factory" / "assets" / "peg_insert" / "extracted"
HOLE_DECOMP = EXTRACTED / "hole_decomp"

USD_CACHE.mkdir(parents=True, exist_ok=True)
EXTRACTED.mkdir(parents=True, exist_ok=True)
HOLE_DECOMP.mkdir(parents=True, exist_ok=True)

S3_BASE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/IsaacLab/Factory"

FILES = {
    "peg": "factory_peg_8mm.usd",
    "hole": "factory_hole_8mm.usd",
}

# ---------------------------------------------------------------------------
# Step 1: Download USDs (cached)
# ---------------------------------------------------------------------------

def download_usd(key: str) -> Path:
    """Download USD file if not cached; return local path."""
    filename = FILES[key]
    local = USD_CACHE / filename
    if local.exists():
        print(f"  [cache hit] {filename}")
        return local
    url = f"{S3_BASE}/{filename}"
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, local)
    size_kb = local.stat().st_size // 1024
    print(f"  Saved {local} ({size_kb} KB)")
    return local


# ---------------------------------------------------------------------------
# Step 2: USD traversal → trimesh
# ---------------------------------------------------------------------------

def _fan_triangulate(face_vertex_counts, face_vertex_indices):
    """Fan-triangulate n-gons. Returns (N_tris, 3) int array."""
    tris = []
    idx = 0
    for count in face_vertex_counts:
        # Fan from first vertex: (v0, v1, v2), (v0, v2, v3), ...
        v0 = face_vertex_indices[idx]
        for k in range(1, count - 1):
            v1 = face_vertex_indices[idx + k]
            v2 = face_vertex_indices[idx + k + 1]
            tris.append([v0, v1, v2])
        idx += count
    return np.array(tris, dtype=np.int32)


def extract_mesh_from_usd(usd_path: Path) -> tuple[trimesh.Trimesh, dict]:
    """Open USD, traverse for UsdGeom.Mesh prims, combine into one Trimesh.

    Returns (mesh, metadata_dict).
    """
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"pxr.Usd.Stage.Open returned None for {usd_path}")

    # Stage-level metadata
    up_axis = UsdGeom.GetStageUpAxis(stage)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)

    prim_infos = []
    all_verts = []
    all_faces = []
    vert_offset = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh_prim = UsdGeom.Mesh(prim)

        # Points attribute
        points_attr = mesh_prim.GetPointsAttr()
        face_count_attr = mesh_prim.GetFaceVertexCountsAttr()
        face_idx_attr = mesh_prim.GetFaceVertexIndicesAttr()

        points = points_attr.Get()
        face_counts = face_count_attr.Get()
        face_indices = face_idx_attr.Get()

        if points is None or face_counts is None or face_indices is None:
            print(f"    [warn] prim {prim.GetPath()} has None geometry attributes, skipping")
            continue

        # Convert Vt.Vec3fArray → numpy
        verts = np.array(points, dtype=np.float64)  # shape (N, 3)
        face_counts_np = np.array(face_counts, dtype=np.int32)
        face_indices_np = np.array(face_indices, dtype=np.int32)

        # NOTE: These Isaac Factory USDs report meters_per_unit=0.01 but vertex
        # coordinates are already authored in meters (peg bbox shows 7.986mm diameter
        # and 50mm height in native coords). Do NOT scale — the metadata is a legacy
        # artifact from the Isaac pipeline. We record meters_per_unit in metadata.json
        # for audit but do not apply it to geometry.

        # Fan-triangulate
        tris = _fan_triangulate(face_counts_np, face_indices_np)
        tris = tris + vert_offset

        all_verts.append(verts)
        all_faces.append(tris)
        vert_offset += len(verts)

        # Collect physics schemas if present
        mass_info = {}
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI(prim)
            m_attr = mass_api.GetMassAttr()
            if m_attr and m_attr.Get() is not None:
                mass_info["mass_kg"] = float(m_attr.Get())
            d_attr = mass_api.GetDensityAttr()
            if d_attr and d_attr.Get() is not None:
                mass_info["density"] = float(d_attr.Get())

        has_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)

        prim_infos.append({
            "path": str(prim.GetPath()),
            "n_verts": len(verts),
            "n_tris": len(tris),
            "mass_info": mass_info,
            "has_rigid_body_api": has_rigid,
            "has_collision_api": has_collision,
        })

    if not all_verts:
        raise RuntimeError(
            f"No UsdGeom.Mesh prims found in {usd_path}. "
            "Stage may use a different prim type or be empty."
        )

    combined_verts = np.concatenate(all_verts, axis=0)
    combined_faces = np.concatenate(all_faces, axis=0)

    mesh = trimesh.Trimesh(vertices=combined_verts, faces=combined_faces, process=True)

    metadata = {
        "source_usd": str(usd_path.name),
        "up_axis": str(up_axis),
        "meters_per_unit": float(meters_per_unit),
        "meters_per_unit_note": (
            "Isaac Factory USDs report meters_per_unit=0.01 but vertex coords "
            "are already in meters. Scale NOT applied — geometry is correct as-is."
        ),
        "prims": prim_infos,
        "combined_n_verts": int(len(combined_verts)),
        "combined_n_tris": int(len(combined_faces)),
    }

    return mesh, metadata


# ---------------------------------------------------------------------------
# Step 3: CoACD decomp via Python API
# ---------------------------------------------------------------------------

def run_coacd(obj_path: Path, out_dir: Path, threshold: float = 0.05) -> list[Path]:
    """Run CoACD on a mesh OBJ via the Python API.

    Returns list of output OBJ paths (one per convex piece).
    """
    import coacd

    mesh = trimesh.load(str(obj_path), force="mesh")
    coacd_mesh = coacd.Mesh(
        vertices=np.array(mesh.vertices, dtype=np.float64),
        indices=np.array(mesh.faces, dtype=np.int32),
    )

    print(f"  Running CoACD (threshold={threshold}) on {obj_path.name} ...")
    parts = coacd.run_coacd(coacd_mesh, threshold=threshold)
    print(f"  CoACD produced {len(parts)} convex pieces")

    out_paths = []
    for i, part in enumerate(parts):
        verts = np.array(part[0], dtype=np.float64)
        faces = np.array(part[1], dtype=np.int32)
        piece_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        out_path = out_dir / f"hole_{i:03d}.obj"
        piece_mesh.export(str(out_path))
        out_paths.append(out_path)

    return out_paths


# ---------------------------------------------------------------------------
# Step 4: Harvest metadata + also check parent prims for physics APIs
# ---------------------------------------------------------------------------

def harvest_all_metadata(usd_path: Path, mesh_meta: dict) -> dict:
    """Walk the full stage (not just Mesh prims) to find physics APIs on any prim."""
    stage = Usd.Stage.Open(str(usd_path))
    rigid_prims = []
    collision_prims = []
    mass_prims = []

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_prims.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_prims.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI(prim)
            info = {"path": str(prim.GetPath())}
            m_attr = mass_api.GetMassAttr()
            if m_attr and m_attr.Get() is not None:
                info["mass_kg"] = float(m_attr.Get())
            d_attr = mass_api.GetDensityAttr()
            if d_attr and d_attr.Get() is not None:
                info["density"] = float(d_attr.Get())
            mass_prims.append(info)

    mesh_meta["rigid_body_prims"] = rigid_prims
    mesh_meta["collision_prims"] = collision_prims
    mesh_meta["mass_prims"] = mass_prims
    return mesh_meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n=== Phase 0a: USD Mesh Extraction ===\n")

    all_metadata = {}

    # --- PEG ---
    print("[1/4] Downloading peg USD ...")
    peg_usd = download_usd("peg")

    print("[2/4] Extracting peg mesh ...")
    peg_mesh, peg_meta = extract_mesh_from_usd(peg_usd)
    peg_meta = harvest_all_metadata(peg_usd, peg_meta)

    peg_obj = EXTRACTED / "peg_8mm.obj"
    peg_mesh.export(str(peg_obj))
    print(f"  Wrote {peg_obj} ({len(peg_mesh.vertices)} verts, {len(peg_mesh.faces)} tris)")
    print(f"  watertight={peg_mesh.is_watertight}, volume={peg_mesh.volume:.6e} m³")
    all_metadata["peg_8mm"] = peg_meta

    # --- HOLE ---
    print("[3/4] Downloading hole USD ...")
    hole_usd = download_usd("hole")

    print("[4/4] Extracting hole mesh ...")
    hole_mesh, hole_meta = extract_mesh_from_usd(hole_usd)
    hole_meta = harvest_all_metadata(hole_usd, hole_meta)

    hole_raw_obj = EXTRACTED / "hole_8mm_raw.obj"
    hole_mesh.export(str(hole_raw_obj))
    print(f"  Wrote {hole_raw_obj} ({len(hole_mesh.vertices)} verts, {len(hole_mesh.faces)} tris)")
    print(f"  watertight={hole_mesh.is_watertight}, volume={hole_mesh.volume:.6e} m³")
    all_metadata["hole_8mm"] = hole_meta

    # --- CoACD decomp on hole ---
    print("\n[CoACD] Decomposing hole ...")
    # Use threshold=0.05; if >32 pieces, retry with 0.08
    coacd_threshold_used = 0.05
    decomp_pieces = run_coacd(hole_raw_obj, HOLE_DECOMP, threshold=coacd_threshold_used)
    if len(decomp_pieces) > 32:
        print(f"  [warn] {len(decomp_pieces)} pieces > 32 limit, retrying with threshold=0.08 ...")
        coacd_threshold_used = 0.08
        # Clear old pieces
        for p in HOLE_DECOMP.glob("hole_*.obj"):
            p.unlink()
        decomp_pieces = run_coacd(hole_raw_obj, HOLE_DECOMP, threshold=coacd_threshold_used)
    print(f"  Final: {len(decomp_pieces)} convex pieces in {HOLE_DECOMP}")
    all_metadata["hole_8mm"]["coacd_pieces"] = len(decomp_pieces)
    all_metadata["hole_8mm"]["coacd_threshold"] = coacd_threshold_used

    # --- Write metadata.json ---
    meta_path = EXTRACTED / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    print(f"\nMetadata written to {meta_path}")

    # --- Audit vs. IsaacLab expected values ---
    print("\n=== Metadata Audit vs. IsaacLab factory_tasks_cfg.py ===")
    print(f"  Peg up_axis:          {peg_meta['up_axis']} (expected Z)")
    print(f"  Peg meters_per_unit:  {peg_meta['meters_per_unit']} (expected 1.0 or 0.01)")
    print(f"  Hole up_axis:         {hole_meta['up_axis']} (expected Z)")
    print(f"  Hole meters_per_unit: {hole_meta['meters_per_unit']} (expected 1.0 or 0.01)")

    # Cylinder volume check: peg should be ~π*(0.003993)²*0.050 ≈ 2.51e-6 m³
    expected_peg_vol = np.pi * (0.003993**2) * 0.050
    print(f"\n  Peg volume:     {peg_mesh.volume:.4e} m³  (expected ~{expected_peg_vol:.4e} m³)")
    if peg_mesh.volume > 0:
        ratio = peg_mesh.volume / expected_peg_vol
        print(f"  Peg vol ratio:  {ratio:.3f}  (1.0 = exact match; >0.8 = acceptable)")
    else:
        print(f"  [warn] Peg volume is non-positive — mesh may not be watertight")

    print(f"\n  Hole pieces:    {len(decomp_pieces)} (≤32 required)")

    # Check mass from USD (if present)
    if peg_meta["mass_prims"]:
        for mp in peg_meta["mass_prims"]:
            m_val = mp.get("mass_kg", "not found")
            print(f"  Peg mass (USD): {m_val} kg  (IsaacLab: 0.019 kg)")
    else:
        print("  Peg mass (USD): not authored — will use analytical 0.019 kg in MJCF")

    print("\n=== Done ===")
    print(f"  peg_8mm.obj:      {peg_obj}")
    print(f"  hole_8mm_raw.obj: {hole_raw_obj}")
    print(f"  hole_decomp/:     {len(decomp_pieces)} OBJ files")
    print(f"  metadata.json:    {meta_path}")


if __name__ == "__main__":
    main()
