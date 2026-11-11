"""
build_eval_set_photos.py
Variantă a builder-ului de set de evaluare bazată pe fotografii reale: în loc de randări
3D-FRONT (BlenderProc + datele de scenă), construiește manifestul dintr-un folder de
fotografii de cameră și un mic CSV de etichete. Validitatea
rămâne rezonabilă (fotografii reale, ground-truth uman ușor), dar diferă de Tier A
sintetic: nu există univers de obiecte partajat cu indexul, deci din D4 se calculează
`category/style precision@k` ale setului regăsit (avem categoria/stilul obiectului
click-uit din etichetă), dar nu rangul exemplarului GT (nu există un UID țintă în index).

FORMAT INTRARE
--------------
--photos_dir : folder cu imagini (.jpg/.png) de camere mobilate.
--labels     : CSV cu antet:
    filename,room_style,click_x,click_y,gt_category[,gt_style]
  unde:
    room_style    ∈ {Modern,Minimalist,Nordic,Industrial,Art Deco,European Traditional,
                     Southeast Asian,Chinese Traditional,Japanese,Luxury,Kids,Mixed}
    gt_category   ∈ cele 14 coarse (Sofa,Chair,Table,Bed,Cabinet,Nightstand,Shelf,
                     Desk,TV Stand,Floor Lamp,Ceiling Lamp,Pendant Lamp,Wardrobe,Bookcase)
    click_x,click_y : pixel pe obiectul de înlocuit
    gt_style      : (opțional) stilul obiectului, dacă diferă de stilul camerei

Etichetarea a ~20–40 de imagini durează puțin și e suficientă pentru o evaluare
ușoară (contribuție secundară).

Rulare:
    python build_eval_set_photos.py \
        --photos_dir ./photos --labels ./photos/labels.csv \
        --output_dir ./eval_set_photos --crop_margin 90
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from taxonomy import ROOM_STYLE_ENUM, COARSE_CATEGORIES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos_dir", required=True)
    ap.add_argument("--labels", required=True, help="CSV de etichete (vezi docstring)")
    ap.add_argument("--output_dir", default="./eval_set_photos")
    ap.add_argument("--crop_margin", type=int, default=90,
                    help="Jumătate-latura crop-ului în jurul click-ului (px)")
    args = ap.parse_args()

    photos = Path(args.photos_dir)
    out = Path(args.output_dir)
    (out / "crops").mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.labels, encoding="utf-8")))
    print(f"{len(rows)} etichete în {args.labels}")

    n, warnings = 0, 0
    with open(out / "eval_manifest.jsonl", "w", encoding="utf-8") as fo:
        for r in rows:
            fn = r["filename"].strip()
            fpath = photos / fn
            if not fpath.exists():
                print(f"  [!] lipsește {fpath}"); warnings += 1; continue
            try:
                cx, cy = int(r["click_x"]), int(r["click_y"])
            except Exception:
                print(f"  [!] click invalid pentru {fn}"); warnings += 1; continue

            room_style = r.get("room_style", "").strip()
            gt_cat = r.get("gt_category", "").strip()
            if room_style not in ROOM_STYLE_ENUM:
                print(f"  [!] room_style necunoscut '{room_style}' ({fn})"); warnings += 1
            if gt_cat not in COARSE_CATEGORIES:
                print(f"  [!] gt_category necunoscut '{gt_cat}' ({fn})"); warnings += 1

            img = Image.open(fpath).convert("RGB")
            W, H = img.size
            m = args.crop_margin
            crop = img.crop((max(0, cx - m), max(0, cy - m),
                             min(W, cx + m), min(H, cy + m)))
            inst_uid = f"{Path(fn).stem}"
            crop_path = out / "crops" / f"{inst_uid}.jpg"
            crop.save(crop_path, quality=90)

            fo.write(json.dumps({
                "instance_id": inst_uid,
                "frame_path": str(fpath),
                "crop_path": str(crop_path),
                "click_x": cx, "click_y": cy,
                "gt_model_id": None,                       # fără UID țintă în index
                "gt_category_coarse": gt_cat,
                "gt_style_canonical": r.get("gt_style", "").strip() or room_style,
                "gt_room_style": room_style,
            }, ensure_ascii=False) + "\n")
            n += 1

    print(f"{n} instanțe scrise în {out/'eval_manifest.jsonl'} ({warnings} avertismente)")
    print("  NOTĂ: rulează run_eval.py normal; D4 va da precision@k (fără gt_rank).")


if __name__ == "__main__":
    main()
