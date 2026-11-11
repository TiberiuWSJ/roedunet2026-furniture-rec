"""
01_filter_catalog.py
Parsează model_info.json din 3D-FUTURE și filtrează obiectele relevante
pentru un scenariu de living room / dormitor.

Structura așteptată a model_info.json:
[
  {
    "model_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "super-category": "Chair",
    "category": "Lounge Chair / Chaise Longue",
    "style": "Modern",
    "theme": "",
    "material": "Fabric"
  },
  ...
]

Super-categoriile reale prezente în 3D-FUTURE (model_info.json), cu efective:
    Cabinet/Shelf/Desk    5725
    Sofa                  2701
    Lighting              1921
    Chair                 1775
    Others                1740
    Bed                   1124
    Table                 1090
    Pier/Stool             487

Pentru scenariul living room / dormitor reținem șase super-categorii relevante
(Sofa, Chair, Table, Bed, Cabinet/Shelf/Desk, Lighting). Atenție: numele
super-categoriilor sunt compuse exact ca în dataset (ex. "Cabinet/Shelf/Desk",
nu "Cabinet" / "Shelf" / "Desk" separat).

Rulare:
    python 01_filter_catalog.py \
        --model_info /path/to/3D-FUTURE-model/model_info.json \
        --models_dir /path/to/3D-FUTURE-model \
        --output_dir ./catalog \
        --max_per_category 400
"""

import json
import argparse
import os
from pathlib import Path
from collections import defaultdict

# Super-categorii relevante pentru living room / dormitor.
# Numele trebuie să fie identice cu valorile câmpului "super-category" din
# model_info.json (3D-FUTURE folosește nume compuse cu "/").
RELEVANT_SUPER_CATEGORIES = {
    "Sofa",
    "Chair",
    "Table",
    "Bed",
    "Cabinet/Shelf/Desk",
    "Lighting",
}

# Subcategorii (câmpul "category") excluse explicit din super-categoriile reținute.
# Numele trebuie să fie identice cu valorile din dataset.
EXCLUDED_CATEGORIES = {
    "Children Cabinet",   # mobilier de copii, în afara scenariului
    "Wine Cooler",        # electrocasnic, nu mobilier
    # Stool-uri / taburete (din Pier/Stool, dar și apariții în alte super-cat.)
    "Footstool / Sofastool / Bed End Stool / Stool",
}

# Mapare stil → etichetă canonică (normalizare variante din dataset)
STYLE_NORMALIZATION = {
    # Modern
    "Modern":                  "Modern",
    "Modernist":               "Modern",
    "Contemporary":            "Modern",
    # Minimalist
    "Minimalist":              "Minimalist",
    "Simple":                  "Minimalist",
    # Nordic / Scandinavian
    "Nordic":                  "Nordic",
    "Scandinavian":            "Nordic",
    "Korean":                  "Nordic",       # Korean minimal → Nordic
    # Industrial
    "Industrial":              "Industrial",
    "Loft":                    "Industrial",
    "Vintage/Retro":           "Industrial",
    "Retro":                   "Industrial",
    # Art Deco
    "Art Deco":                "Art Deco",
    # European Traditional
    "European Traditional":    "European Traditional",
    "Classic":                 "European Traditional",
    "European Classic":        "European Traditional",
    "Neoclassical":            "European Traditional",
    "Neo-Classical":           "European Traditional",
    "Mediterranean":           "European Traditional",
    "American Country":        "European Traditional",
    "French":                  "European Traditional",
    "European":                "European Traditional",
    # Southeast Asian
    "Southeast Asian":         "Southeast Asian",
    "Southeast Asia":          "Southeast Asian",
    "Tropical":                "Southeast Asian",
    # Chinese Traditional
    "Chinese Traditional":     "Chinese Traditional",
    "New Chinese":             "Chinese Traditional",
    "Ming Qing":               "Chinese Traditional",
    "Chinese Modern":          "Chinese Traditional",
    "Chinoiserie":             "Chinese Traditional",
    # Japanese
    "Japanese":                "Japanese",
    "Japanese Modern":         "Japanese",
    "Wabi-Sabi":               "Japanese",
    # Luxury
    "Luxury":                  "Luxury",
    "Light Luxury":            "Luxury",
    "Glamour":                 "Luxury",
    # Kids
    "Kids":                    "Kids",
    "Children":                "Kids",
    # Fallback
    "Others":                  "Unspecified",
    "Other":                   "Unspecified",
    "":                        "Unspecified",
    "None":                    "Unspecified",
}


def normalize_style(raw_style: str) -> str:
    raw_style = raw_style.strip()
    return STYLE_NORMALIZATION.get(raw_style, raw_style)


def model_path_exists(models_dir: Path, model_id: str) -> bool:
    """Verifică dacă directorul modelului există și conține un fișier OBJ."""
    model_dir = models_dir / model_id
    if not model_dir.is_dir():
        return False
    obj_files = list(model_dir.glob("*.obj"))
    return len(obj_files) > 0


def find_obj_file(models_dir: Path, model_id: str) -> str | None:
    """Returnează calea relativă la fișierul OBJ normalizat (preferabil)."""
    model_dir = models_dir / model_id
    # Preferință: normalized_model.obj > raw_model.obj > orice .obj
    # (3D-FUTURE folosește exact "normalized_model.obj")
    for preferred in ["normalized_model.obj", "raw_model.obj"]:
        candidate = model_dir / preferred
        if candidate.exists():
            return str(candidate)
    objs = list(model_dir.glob("*.obj"))
    return str(objs[0]) if objs else None


def main():
    parser = argparse.ArgumentParser(description="Filtrare catalog 3D-FUTURE")
    parser.add_argument("--model_info", required=True,
                        help="Calea la model_info.json")
    parser.add_argument("--models_dir", required=True,
                        help="Directorul rădăcină 3D-FUTURE-model/")
    parser.add_argument("--output_dir", default="./catalog",
                        help="Director output pentru catalog.jsonl și statistici")
    parser.add_argument("--max_per_category", type=int, default=400,
                        help="Număr maxim de obiecte per super-categorie (0 = fără limită)")
    parser.add_argument("--verify_files", action="store_true",
                        help="Verifică existența fișierelor OBJ pe disc (lent)")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Citire {args.model_info} ...")
    with open(args.model_info, "r", encoding="utf-8") as f:
        all_models = json.load(f)
    print(f"      Total modele în dataset: {len(all_models)}")

    print("[2/3] Filtrare categorii relevante ...")
    accepted = []
    rejected_category = 0
    rejected_file = 0
    per_category_count = defaultdict(int)

    for entry in all_models:
        model_id  = entry.get("model_id") or ""
        super_cat = (entry.get("super-category") or "").strip()
        category  = (entry.get("category") or "").strip()
        style     = normalize_style(entry.get("style") or "")
        material  = (entry.get("material") or "").strip()
        theme     = (entry.get("theme") or "").strip()

        # Filtru categorie
        if super_cat not in RELEVANT_SUPER_CATEGORIES:
            rejected_category += 1
            continue
        if category in EXCLUDED_CATEGORIES:
            rejected_category += 1
            continue

        # Limită per categorie
        if args.max_per_category > 0:
            if per_category_count[super_cat] >= args.max_per_category:
                continue

        # Verificare fișiere (opțional, lent)
        obj_path = None
        if args.verify_files:
            if not model_path_exists(models_dir, model_id):
                rejected_file += 1
                continue
            obj_path = find_obj_file(models_dir, model_id)
        else:
            # Calculăm calea așteptată fără verificare
            obj_path = str(models_dir / model_id / "normalized_model.obj")

        per_category_count[super_cat] += 1

        record = {
            "uid":          model_id,
            "super_category": super_cat,
            "category":     category,
            "style":        style,
            "material":     material,
            "theme":        theme,
            "obj_path":     obj_path,
            # Câmpuri completate în pasul 03 (VLM):
            "caption":      None,
            "thumbnail_path": str(models_dir / model_id / "thumbnail.png"),
        }
        accepted.append(record)

    print(f"      Acceptate: {len(accepted)}")
    print(f"      Respinse (categorie): {rejected_category}")
    if args.verify_files:
        print(f"      Respinse (fișier lipsă): {rejected_file}")

    # Statistici per categorie / stil
    print("\n[3/3] Salvare catalog și statistici ...")

    catalog_path = output_dir / "catalog.jsonl"
    with open(catalog_path, "w", encoding="utf-8") as f:
        for rec in accepted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"      Catalog salvat: {catalog_path} ({len(accepted)} înregistrări)")

    # Statistici
    stats_by_cat = defaultdict(int)
    stats_by_style = defaultdict(int)
    for rec in accepted:
        stats_by_cat[rec["super_category"]] += 1
        stats_by_style[rec["style"]] += 1

    stats = {
        "total_accepted": len(accepted),
        "by_super_category": dict(sorted(stats_by_cat.items(), key=lambda x: -x[1])),
        "by_style": dict(sorted(stats_by_style.items(), key=lambda x: -x[1])),
    }
    stats_path = output_dir / "catalog_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"      Statistici salvate: {stats_path}")

    print("\nDistribuție per super-categorie:")
    for cat, cnt in stats["by_super_category"].items():
        print(f"  {cat:<25} {cnt:>5}")
    print("\nDistribuție per stil:")
    for sty, cnt in stats["by_style"].items():
        print(f"  {sty:<25} {cnt:>5}")


if __name__ == "__main__":
    main()
