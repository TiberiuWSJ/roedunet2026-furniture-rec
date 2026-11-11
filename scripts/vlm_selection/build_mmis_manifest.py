"""
build_mmis_manifest.py
Construiește un manifest de evaluare din datasetul MMIS (camere reale etichetate
cu stil). Sarcină: clasificare de stil din imagine (atât).

Detectează automat lista de stiluri din structura de foldere (nu hardcodăm nume),
ca să tolereze variațiile din taxonomia MMIS. Eșantionează echilibrat un subset mic,
suficient pentru a alege un VLM (contribuție de suport, nu benchmark masiv).

Structura MMIS așteptat (oricare din variante):
    <mmis_root>/<Style>/<Room>/*.jpg
    <mmis_root>/<Style>/*.jpg

Ieșire:
    mmis_eval/manifest.jsonl   # {"image_path":..., "gt_style":...}
    mmis_eval/styles.json      # lista de stiluri (label set, pentru prompt închis)

Rulare:
    python build_mmis_manifest.py --mmis_root /cale/MMIS --out ./mmis_eval \
        --per_style 15 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(style_dir: Path) -> list[Path]:
    return [p for p in style_dir.rglob("*") if p.suffix.lower() in IMG_EXT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmis_root", required=True, help="Rădăcina datasetului MMIS")
    ap.add_argument("--out", default="./mmis_eval")
    ap.add_argument("--per_style", type=int, default=15,
                    help="Imagini eșantionate per stil (echilibrat)")
    ap.add_argument("--min_per_style", type=int, default=5,
                    help="Ignoră stilurile cu mai puține imagini decât atât")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.mmis_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    style_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not style_dirs:
        raise SystemExit(f"Niciun subfolder de stil în {root}")

    manifest, styles = [], []
    for sd in style_dirs:
        imgs = list_images(sd)
        if len(imgs) < args.min_per_style:
            print(f"  skip '{sd.name}' ({len(imgs)} imagini)")
            continue
        random.shuffle(imgs)
        chosen = imgs[:args.per_style]
        styles.append(sd.name)
        for p in chosen:
            manifest.append({"image_path": str(p), "gt_style": sd.name})

    random.shuffle(manifest)
    (out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n",
        encoding="utf-8")
    (out / "styles.json").write_text(
        json.dumps(sorted(styles), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{len(manifest)} imagini, {len(styles)} stiluri → {out}/manifest.jsonl")
    print(f"  Stiluri: {sorted(styles)}")


if __name__ == "__main__":
    main()
