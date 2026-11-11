"""
02_render_thumbnails.py
Renderează thumbnails PNG texturate pentru fiecare obiect din catalog.jsonl.
Folosește PyVista/VTK cu backend Cocoa/CGL (GPU macOS M1, headless, fără EGL/X11).

Textura este baked în vertex colors via trimesh → fără transparență MTL.

Instalare:
    pip install pyvista trimesh Pillow numpy tqdm

Rulare:
    python 02_render_thumbnails.py \\
        --catalog    ./catalog/catalog.jsonl \\
        --output_dir ./catalog/thumbnails \\
        --img_size   512 \\
        --n_views    3 \\
        --workers    1
"""

import json
import argparse
import traceback
import os
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Suppress VTK/CoreAnalytics noise pe macOS
os.environ["VTK_SILENCE_GET_VOID_POINTER_WARNINGS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
from tqdm import tqdm
import trimesh
import pyvista as pv

pv.OFF_SCREEN = True
pv.global_theme.transparent_background = False


# Unghiuri cameră
CAMERA_VIEWS = [
    (45,  30),
    (135, 25),
    (225, 20),
]


# Render

def _camera_position(center, radius, azimuth_deg, elevation_deg, up_axis="z"):
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    if up_axis == "y":
        # 3D-FUTURE (normalized_model.obj) este Y-up: elevația se măsoară pe axa Y
        x = center[0] + radius * np.cos(el) * np.sin(az)
        y = center[1] + radius * np.sin(el)
        z = center[2] + radius * np.cos(el) * np.cos(az)
        return (x, y, z)
    x = center[0] + radius * np.cos(el) * np.cos(az)
    y = center[1] + radius * np.cos(el) * np.sin(az)
    z = center[2] + radius * np.sin(el)
    return (x, y, z)


def load_mesh_pyvista(obj_path: str, tex_path: str) -> pv.PolyData:
    """
    Încarcă OBJ cu trimesh, bake textură → vertex colors, returnează PyVista PolyData.
    Ignoră MTL complet → fără probleme de transparență.
    """
    tm = trimesh.load(str(obj_path), force="mesh", process=True)
    if isinstance(tm, trimesh.Scene):
        tm = tm.dump(concatenate=True)
    if not isinstance(tm, trimesh.Trimesh):
        raise RuntimeError("Nu s-a putut obține un Trimesh valid")

    # Bake UV textură → culori per vertex
    if Path(tex_path).exists() and hasattr(tm.visual, "uv") and tm.visual.uv is not None:
        try:
            tm.visual = tm.visual.to_color()
        except Exception:
            pass

    # Conversie la PyVista PolyData
    faces_pv = np.hstack([
        np.full((len(tm.faces), 1), 3, dtype=np.int64),
        tm.faces.astype(np.int64),
    ])
    mesh = pv.PolyData(tm.vertices.astype(np.float32), faces_pv)

    if hasattr(tm.visual, "vertex_colors") and tm.visual.vertex_colors is not None:
        mesh.point_data["RGB"] = tm.visual.vertex_colors[:, :3].astype(np.uint8)

    return mesh


def render_object(obj_path: str, out_dir: Path, img_size: int, n_views: int, up_axis: str = "z") -> list[str]:
    """Renderează n_views PNG-uri texturate. Returnează lista de căi."""
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(obj_path).parent
    tex_path  = str(model_dir / "texture.png")

    mesh = load_mesh_pyvista(obj_path, tex_path)

    center = np.array(mesh.center)
    bounds = mesh.bounds
    diag   = np.linalg.norm([
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    ])
    radius = diag * 1.6 if diag > 0 else 2.0

    has_colors = "RGB" in mesh.point_data
    views      = CAMERA_VIEWS[:n_views]
    out_paths  = []

    for i, (az, el) in enumerate(views):
        plotter = pv.Plotter(off_screen=True, window_size=[img_size, img_size])
        plotter.set_background("white")

        add_kwargs = dict(smooth_shading=True, opacity=1.0)
        if has_colors:
            plotter.add_mesh(mesh, scalars="RGB", rgb=True, **add_kwargs)
        else:
            plotter.add_mesh(mesh, color="lightgray", **add_kwargs)

        plotter.enable_lightkit()
        if up_axis == "y":
            plotter.add_light(pv.Light(position=(1, 2, 1),  intensity=0.6))
            plotter.add_light(pv.Light(position=(-1, 1, -1), intensity=0.4))
        else:
            plotter.add_light(pv.Light(position=(1, 1, 2),  intensity=0.6))
            plotter.add_light(pv.Light(position=(-1, -1, 1), intensity=0.4))

        cam_pos = _camera_position(center, radius, az, el, up_axis)
        up_vec = (0, 1, 0) if up_axis == "y" else (0, 0, 1)
        plotter.camera_position = [cam_pos, tuple(center), up_vec]

        out_path = out_dir / f"view_{i}.png"
        plotter.screenshot(str(out_path), window_size=[img_size, img_size])
        plotter.close()
        out_paths.append(str(out_path))

    return out_paths


# Worker

def process_entry(args_tuple):
    entry, thumbnails_dir, img_size, n_views, up_axis = args_tuple
    uid      = entry["uid"]
    obj_path = entry.get("obj_path", "")
    out_dir  = Path(thumbnails_dir) / uid

    if (out_dir / "view_0.png").exists():
        return uid, [str(out_dir / f"view_{i}.png") for i in range(n_views)], None

    if not obj_path or not Path(obj_path).exists():
        return uid, [], f"OBJ not found: {obj_path}"

    try:
        paths = render_object(obj_path, out_dir, img_size, n_views, up_axis)
        return uid, paths, None
    except Exception:
        return uid, [], traceback.format_exc()


# Main

def main():
    parser = argparse.ArgumentParser(description="Render thumbnails texturate — PyVista/VTK")
    parser.add_argument("--catalog",     required=True)
    parser.add_argument("--output_dir",  default="./catalog/thumbnails")
    parser.add_argument("--img_size",    type=int, default=512)
    parser.add_argument("--n_views",     type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--workers",     type=int, default=1,
                        help="Procese paralele. PyVista/VTK pe macOS e stabil cu 1.")
    parser.add_argument("--limit",       type=int, default=0)
    parser.add_argument("--up_axis",     default="z", choices=["z", "y"],
                        help="Axa verticală a mesh-urilor. 3D-FUTURE este Y-up → folosiți 'y' "
                             "(default 'z' = comportamentul inițial, care randa obiectele culcate).")
    args = parser.parse_args()

    thumbnails_dir = Path(args.output_dir)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    with open(args.catalog) as f:
        entries = [json.loads(line) for line in f]

    if args.limit > 0:
        entries = entries[:args.limit]
        print(f"[DEBUG] Limitat la {args.limit} obiecte")

    print(f"Total obiecte de rendat: {len(entries)}")

    tasks         = [(e, str(thumbnails_dir), args.img_size, args.n_views, args.up_axis) for e in entries]
    uid_to_entry  = {e["uid"]: e for e in entries}
    updated_catalog = []
    errors        = []

    if args.workers == 1:
        for task in tqdm(tasks, desc="Render"):
            uid, paths, err = process_entry(task)
            if err:
                errors.append((uid, err[:300]))
            entry = uid_to_entry[uid]
            entry["thumbnail_path"]  = paths[0] if paths else None
            entry["thumbnail_paths"] = paths
            updated_catalog.append(entry)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(process_entry, t): t[0]["uid"] for t in tasks}
            pbar = tqdm(as_completed(future_map), total=len(tasks), desc="Render")
            for future in pbar:
                uid, paths, err = future.result()
                if err:
                    errors.append((uid, err[:300]))
                entry = uid_to_entry[uid]
                entry["thumbnail_path"]  = paths[0] if paths else None
                entry["thumbnail_paths"] = paths
                updated_catalog.append(entry)
            pbar.close()

    catalog_out = thumbnails_dir.parent / "catalog_rendered.jsonl"
    with open(catalog_out, "w") as f:
        for e in updated_catalog:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    n_ok = sum(1 for e in updated_catalog if e["thumbnail_path"])
    print(f"\nCatalog actualizat: {catalog_out}")
    print(f"  OK:    {n_ok}/{len(entries)}")
    print(f"  Erori: {len(errors)}")
    if errors:
        for uid, err in errors[:3]:
            print(f"  {uid[:8]}: {err[:200]}")


if __name__ == "__main__":
    main()
