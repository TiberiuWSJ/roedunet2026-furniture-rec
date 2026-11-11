"""
build_eval_set.py
Construiește setul de evaluare Tier A (submulțime 3D-FRONT) din output-ul de randare
BlenderProc, producând `eval_manifest.jsonl` — o linie per instanță click-uită:

  {
    "instance_id":   "scene0123_view0_inst7",
    "frame_path":    ".../rgb.jpg",
    "crop_path":     ".../crops/scene0123_view0_inst7.jpg",
    "click_x": 412, "click_y": 268,
    "gt_model_id":   "<uid 3D-FUTURE>",
    "gt_category_coarse": "Sofa",      # mapat 34→14 (taxonomy.py)
    "gt_style_canonical": "Modern",    # normalizat ~19→11 (taxonomy.py)
    "gt_room_style":      "Modern"     # stil dominant derivat al camerei
  }

FORMAT DE INTRARE AȘTEPTAT (--renders_dir)
------------------------------------------
Un subdirector per vedere, fiecare conținând:
  - rgb.jpg | rgb.png            : cadrul RGB randat
  - instance_segmap.npy          : hartă HxW de ID-uri întregi de instanță
  - instances.json               : { "<instance_id>": "<3D-FUTURE model_id>", ... }

Acesta este un format intermediar minimal. Output-ul nativ BlenderProc-3dfront
(COCO / HDF5) trebuie convertit la el printr-un mic adaptor.

Bugetul de rulare (≤1–2h) impune o submulțime mică: vezi --max_instances /
--max_per_class. Eșantionarea este stratificată pe categoria coarse GT.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from taxonomy import coarse_category, normalize_style, derive_room_style, OTHER, AMBIGUOUS


def _load_rgb(view_dir: Path) -> Image.Image:
    for name in ("rgb.jpg", "rgb.png"):
        p = view_dir / name
        if p.exists():
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(f"Niciun rgb.jpg/png în {view_dir}")


def _instance_bbox_centroid(segmap: np.ndarray, inst_id: int):
    """Returnează (x0, y0, x1, y1, cx, cy, area) pentru o instanță."""
    ys, xs = np.where(segmap == inst_id)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cx, cy = int(xs.mean()), int(ys.mean())
    return x0, y0, x1, y1, cx, cy, len(xs)


def main():
    ap = argparse.ArgumentParser(description="Builder set evaluare Tier A (3D-FRONT)")
    ap.add_argument("--renders_dir", required=True,
                    help="Director cu subdirectoare per vedere (vezi format în docstring)")
    ap.add_argument("--model_info", required=True,
                    help="3D-FUTURE-model/model_info.json (pentru GT categorie/stil)")
    ap.add_argument("--output_dir", default="./eval_set_tierA")
    ap.add_argument("--min_area_frac", type=float, default=0.01,
                    help="Aria minimă a măștii ca fracție din cadru (filtrare obiecte mici)")
    ap.add_argument("--crop_margin", type=int, default=40,
                    help="Margine (px) adăugată la bbox pentru crop")
    ap.add_argument("--max_instances", type=int, default=70,
                    help="Număr total de instanțe în set (buget de rulare)")
    ap.add_argument("--max_per_class", type=int, default=8,
                    help="Plafon per categorie coarse (stratificare)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.output_dir)
    (out / "crops").mkdir(parents=True, exist_ok=True)

    # GT per model_id din 3D-FUTURE
    with open(args.model_info, encoding="utf-8") as f:
        info = {m["model_id"]: m for m in json.load(f)}

    candidates = []  # tuple (entry_dict, coarse_cat)
    renders = Path(args.renders_dir)
    view_dirs = [d for d in renders.iterdir() if d.is_dir()]
    print(f"[1/3] {len(view_dirs)} vederi găsite în {renders}")

    for vd in view_dirs:
        try:
            rgb = _load_rgb(vd)
            segmap = np.load(vd / "instance_segmap.npy")
            inst_map = json.loads((vd / "instances.json").read_text())
        except Exception as e:
            print(f"  skip {vd.name}: {e}")
            continue

        H, W = segmap.shape[:2]
        frame_area = H * W

        # Stilul camerei: din toate modelele prezente în vedere
        member_styles = [info.get(mid, {}).get("style", "")
                         for mid in inst_map.values() if mid in info]
        room_style = derive_room_style(member_styles)

        for inst_id_str, model_id in inst_map.items():
            if model_id not in info:
                continue
            meta = info[model_id]
            cat_coarse = coarse_category(meta.get("category", ""))
            sty_canon = normalize_style(meta.get("style", ""))
            if cat_coarse in (OTHER,) or sty_canon in (AMBIGUOUS,):
                continue  # excludem nemapabilele din GT

            try:
                inst_id = int(inst_id_str)
            except ValueError:
                continue
            bb = _instance_bbox_centroid(segmap, inst_id)
            if bb is None:
                continue
            x0, y0, x1, y1, cx, cy, area = bb
            if area / frame_area < args.min_area_frac:
                continue

            candidates.append((vd, rgb, model_id, cat_coarse, sty_canon,
                               room_style, (x0, y0, x1, y1, cx, cy)))

    print(f"[2/3] {len(candidates)} instanțe eligibile (după filtre arie/mapare)")

    # Eșantionare stratificată pe categoria coarse
    by_class = defaultdict(list)
    for c in candidates:
        by_class[c[3]].append(c)
    for k in by_class:
        random.shuffle(by_class[k])

    selected = []
    # round-robin pe clase până la max_instances / max_per_class
    per_class_taken = defaultdict(int)
    classes = list(by_class.keys())
    random.shuffle(classes)
    progress = True
    while len(selected) < args.max_instances and progress:
        progress = False
        for k in classes:
            if len(selected) >= args.max_instances:
                break
            if per_class_taken[k] >= args.max_per_class:
                continue
            if per_class_taken[k] < len(by_class[k]):
                selected.append(by_class[k][per_class_taken[k]])
                per_class_taken[k] += 1
                progress = True

    # Scriere manifest + crops
    manifest_path = out / "eval_manifest.jsonl"
    n = 0
    with open(manifest_path, "w", encoding="utf-8") as fo:
        for (vd, rgb, model_id, cat_coarse, sty_canon, room_style, geom) in selected:
            x0, y0, x1, y1, cx, cy = geom
            W, H = rgb.size
            cx0 = max(0, x0 - args.crop_margin); cy0 = max(0, y0 - args.crop_margin)
            cx1 = min(W, x1 + args.crop_margin); cy1 = min(H, y1 + args.crop_margin)
            inst_uid = f"{vd.name}_inst{n}"
            crop = rgb.crop((cx0, cy0, cx1, cy1))
            crop_path = out / "crops" / f"{inst_uid}.jpg"
            crop.save(crop_path, quality=90)
            frame_path = str(next((vd / nm) for nm in ("rgb.jpg", "rgb.png")
                                  if (vd / nm).exists()))
            fo.write(json.dumps({
                "instance_id": inst_uid,
                "frame_path": frame_path,
                "crop_path": str(crop_path),
                "click_x": cx, "click_y": cy,
                "gt_model_id": model_id,
                "gt_category_coarse": cat_coarse,
                "gt_style_canonical": sty_canon,
                "gt_room_style": room_style,
            }, ensure_ascii=False) + "\n")
            n += 1

    # Distribuție
    dist_cat = defaultdict(int); dist_sty = defaultdict(int)
    for c in selected:
        dist_cat[c[3]] += 1; dist_sty[c[5]] += 1
    print(f"[3/3] {n} instanțe scrise în {manifest_path}")
    print("  Categorii:", dict(dist_cat))
    print("  Stil cameră:", dict(dist_sty))


if __name__ == "__main__":
    main()
