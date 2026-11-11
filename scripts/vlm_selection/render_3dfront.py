"""
render_3dfront.py
Randare automatĂ a scenelor 3D-FRONT cu BlenderProc, producând formatul intermediar
consumat de `build_eval_set.py`:
    <out>/<scene>_<view>/rgb.jpg
    <out>/<scene>_<view>/instance_segmap.npy   # HxW int (idx de instanță)
    <out>/<scene>_<view>/instances.json        # {"<idx>": "<3D-FUTURE model_id (jid)>"}

Ground-truth-ul (categorie coarse, stil canonic, stil cameră) nu se scrie aici — se
derivă ulterior automat din model_info.json în build_eval_set.py prin taxonomy.py.

CERINȚE (date 3D-FRONT, descărcate separat de 3D-FUTURE):
  - 3D-FRONT/                (json-uri de scenă)
  - 3D-FRONT-texture/        (texturi)
  - 3D-FUTURE-model/
  - blenderproc instalat:    pip install blenderproc

Rulare (atenție: prin `blenderproc run`, nu `python`!):
    blenderproc run render_3dfront.py -- \
        --front   /cale/3D-FRONT \
        --future  ../../3D-FUTURE-model \
        --texture /cale/3D-FRONT-texture \
        --out     ./eval_set_tierA/renders \
        --n_scenes 12 --views_per_scene 2

Note de versiune: API-ul BlenderProc poate varia ușor între versiuni. Scriptul
folosește `load_front3d`, `enable_segmentation_output(map_by=[...])` și
eșantionare de cameră cu verificare de acoperire — în linia exemplului oficial
BlenderProc pentru 3D-FRONT. Dacă numele proprietății modelului diferă, ajustează
`JID_KEYS` mai jos.
"""

import blenderproc as bproc  # TREBUIE primul import (inițializează Blender)

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

# Chei posibile pentru ID-ul modelului 3D-FUTURE pe obiectele încărcate (după versiune)
JID_KEYS = ["jid", "uid", "3D_future_model_id"]


def get_model_id(obj):
    for k in JID_KEYS:
        try:
            v = obj.get_cp(k)
            if v:
                return str(v)
        except Exception:
            continue
    # fallback: din nume (BlenderProc denumește adesea obiectele cu jid-ul)
    name = obj.get_name()
    return name.split(".")[0] if name else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", required=True, help="Director 3D-FRONT (json-uri scenă)")
    ap.add_argument("--future", required=True, help="Director 3D-FUTURE-model")
    ap.add_argument("--texture", required=True, help="Director 3D-FRONT-texture")
    ap.add_argument("--out", required=True, help="Director output (renders)")
    ap.add_argument("--n_scenes", type=int, default=12)
    ap.add_argument("--views_per_scene", type=int, default=2)
    ap.add_argument("--res", type=int, default=640, help="Latura imaginii")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    scene_jsons = sorted(Path(args.front).glob("*.json"))
    random.shuffle(scene_jsons)
    scene_jsons = scene_jsons[:args.n_scenes]
    print(f"[render] {len(scene_jsons)} scene selectate din {args.front}")

    bproc.init()
    bproc.camera.set_resolution(args.res, args.res)

    # Mapping de etichete (necesar load_front3d). Folosim resursa implicită BlenderProc.
    try:
        mapping_file = bproc.utility.resolve_resource(
            os.path.join("front_3D", "3D_front_mapping.csv"))
        mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)
    except Exception as e:
        print(f"[render] avertisment: mapping implicit indisponibil ({e}); continui fără.")
        mapping = None

    for si, sj in enumerate(scene_jsons):
        try:
            kwargs = dict(json_path=str(sj),
                          future_model_path=args.future,
                          front_3D_texture_path=args.texture)
            if mapping is not None:
                kwargs["label_mapping"] = mapping
            loaded = bproc.loader.load_front3d(**kwargs)
        except Exception as e:
            print(f"  [skip] {sj.name}: {e}")
            continue

        # Segmentare pe instanță + jid (id-ul modelului 3D-FUTURE)
        bproc.renderer.enable_segmentation_output(map_by=["instance", "jid"])

        # Eșantionare cameră: poziții la înălțime umană, privind spre interior,
        # acceptate dacă "văd" suficiente obiecte (acoperire de scenă).
        # BVH pentru verificări de coliziune/vizibilitate.
        bvh = bproc.object.create_bvh_tree_multi_objects(
            [o for o in loaded if o.has_cp("category_id") or True])

        # bbox-ul aproximativ al scenei pentru eșantionarea pozițiilor
        all_pts = np.concatenate([o.get_bound_box() for o in loaded], axis=0)
        lo, hi = all_pts.min(0), all_pts.max(0)
        center = (lo + hi) / 2.0

        poses = 0
        tries = 0
        while poses < args.views_per_scene and tries < 200:
            tries += 1
            loc = np.array([
                random.uniform(lo[0], hi[0]),
                random.uniform(lo[1], hi[1]),
                random.uniform(1.4, 1.8),  # înălțime cameră
            ])
            # privește spre un punct de interes lângă centru
            poi = center + np.array([random.uniform(-1, 1),
                                     random.uniform(-1, 1), 0.0])
            rot = bproc.camera.rotation_from_forward_vec(poi - loc)
            cam2world = bproc.math.build_transformation_mat(loc, rot)
            # acceptă dacă scena are acoperire rezonabilă din această poziție
            if bproc.camera.scene_coverage_score(cam2world) > 0.5 and \
               bproc.camera.perform_obstacle_in_view_check(cam2world, {"min": 0.5}, bvh):
                bproc.camera.add_camera_pose(cam2world)
                poses += 1
        if poses == 0:
            print(f"  [skip] {sj.name}: nicio poziție de cameră validă")
            continue

        data = bproc.renderer.render()
        rgbs = data["colors"]
        segs = data["instance_segmaps"]
        attrs = data["instance_attribute_maps"]

        from PIL import Image
        scene_id = sj.stem
        for vi in range(len(rgbs)):
            vdir = out_root / f"{scene_id}_view{vi}"
            vdir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgbs[vi]).convert("RGB").save(vdir / "rgb.jpg", quality=90)
            np.save(vdir / "instance_segmap.npy", segs[vi].astype(np.int32))
            # mapping idx → jid (model_id 3D-FUTURE)
            idx2jid = {}
            for a in attrs[vi]:
                jid = a.get("jid")
                if jid:
                    idx2jid[str(a["idx"])] = str(jid)
            (vdir / "instances.json").write_text(json.dumps(idx2jid, ensure_ascii=False))
        print(f"  [{si+1}/{len(scene_jsons)}] {scene_id}: {len(rgbs)} vederi scrise")

        bproc.utility.reset_keyframes()

    print(f"Randare gata în {out_root}. Continuă cu build_eval_set.py.")


if __name__ == "__main__":
    main()
