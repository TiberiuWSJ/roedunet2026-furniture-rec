"""
taxonomy.py
Mapări de taxonomie pentru studiul de evaluare VLM.

Taxonomia din 3D-FUTURE (model_info.json, 16.563 obiecte) diferă de enum-urile din prompt
(`05_recommend.py`). Acest modul reconciliază cele două:

  * Categorii: 34 categorii fine reale → 14 categorii "coarse" din prompt.
  * Stiluri:   ~19 stiluri reale → 11 stiluri canonice din prompt.

Observații:
  - "Art Deco" (enum prompt) nu apare deloc în dataset.
  - "Light Luxury" (real) → "Luxury" (canonic).
  - Multe stiluri europene reale (American Country, Neoclassical, European Classic,
    Mediterranean, Vintage/Retro, European) colapsează în "European Traditional".
  - Stiluri chineze (New Chinese, Ming Qing, Chinoiserie) → "Chinese Traditional".
  - "Korean" și "Others" nu au corespondent canonic curat → marcat AMBIGU și
    exclus din numitorul acurateții de stil (raportat separat).

Toate ieșirile mapate la `OTHER`/`AMBIGUOUS` sunt excluse din numitorul metricilor
de clasificare și raportate separat, pentru a nu contamina acuratețea.
"""

from __future__ import annotations

# Enum-uri din prompt (sursă: SCENE_ANALYSIS_PROMPT, 05_recommend.py)

CANONICAL_STYLES = [
    "Modern", "Minimalist", "Nordic", "Industrial", "Art Deco",
    "European Traditional", "Southeast Asian", "Chinese Traditional",
    "Japanese", "Luxury", "Kids",
]
# Valoare suplimentară permisă în ieșirea VLM (cameră fără stil dominant)
ROOM_STYLE_ENUM = CANONICAL_STYLES + ["Mixed"]

COARSE_CATEGORIES = [
    "Sofa", "Chair", "Table", "Bed", "Cabinet", "Nightstand", "Shelf",
    "Desk", "TV Stand", "Floor Lamp", "Ceiling Lamp", "Pendant Lamp",
    "Wardrobe", "Bookcase",
]

# Sentinele pentru valori nemapabile (excluse din numitorul metricilor)
OTHER = "Other"
AMBIGUOUS = "Ambiguous"

# Stiluri oferite în prompt dar ABSENTE din ground-truth-ul 3D-FUTURE.
# Funcționează ca opțiuni-momeală (decoy): orice predicție pe ele nu poate fi
# niciodată corectă, deci frecvența lor este un PROXY de halucinație /
# supra-încredere (D-halucinație). Atenție: semnal moale — un model poate
# confunda legitim o cameră fastuoasă cu "Art Deco", deci rata amestecă
# halucinație adevărată cu confuzie de stil; se raportează ca proxy, nu dovadă.
PHANTOM_STYLES = {"Art Deco"}

# Stiluri: ~19 reale → 11 canonice
# Cheile sunt valorile brute exacte din model_info.json (case-sensitive după strip).

STYLE_TO_CANONICAL = {
    "Modern":            "Modern",
    "Minimalist":        "Minimalist",
    "Nordic":            "Nordic",
    "Industrial":        "Industrial",
    "Japanese":          "Japanese",
    "Light Luxury":      "Luxury",
    "Southeast Asia":    "Southeast Asian",
    # Chineze → Chinese Traditional
    "New Chinese":       "Chinese Traditional",
    "Ming Qing":         "Chinese Traditional",
    "Chinoiserie":       "Chinese Traditional",
    # Europene clasice → European Traditional
    "American Country":  "European Traditional",
    "Neoclassical":      "European Traditional",
    "European Classic":  "European Traditional",
    "European":          "European Traditional",
    "Mediterranean":     "European Traditional",
    "Vintage/Retro":     "European Traditional",
    "Kids":              "Kids",
    # Fără corespondent canonic curat → excluse din numitor (raportate separat)
    "Korean":            AMBIGUOUS,
    "Others":            AMBIGUOUS,
    "":                  AMBIGUOUS,
    "None":              AMBIGUOUS,
}


def normalize_style(raw_style: str) -> str:
    """Normalizează un stil brut 3D-FUTURE la stilul canonic (sau AMBIGUOUS)."""
    if raw_style is None:
        return AMBIGUOUS
    return STYLE_TO_CANONICAL.get(raw_style.strip(), AMBIGUOUS)


# Categorii: 34 fine reale → 14 coarse
# Cheile sunt valorile exacte din câmpul "category" al model_info.json.

CATEGORY_TO_COARSE = {
    # Sofa
    "Three-Seat / Multi-seat Sofa":        "Sofa",
    "Three-Seat / Multi-person sofa":      "Sofa",
    "Loveseat Sofa":                       "Sofa",
    "L-shaped Sofa":                       "Sofa",
    "U-shaped Sofa":                       "Sofa",
    "Two-seat Sofa":                       "Sofa",
    "Lazy Sofa":                           "Sofa",
    "Chaise Longue Sofa":                  "Sofa",
    "Couch Bed":                           "Sofa",
    # Chair
    "armchair":                            "Chair",
    "Dining Chair":                        "Chair",
    "Lounge Chair / Cafe Chair / Office Chair":     "Chair",
    "Lounge Chair / Book-chair / Computer Chair":   "Chair",
    "Dressing Chair":                      "Chair",
    "Classic Chinese Chair":               "Chair",
    "Folding chair":                       "Chair",
    "Hanging Chair":                       "Chair",
    "Barstool":                            "Chair",
    # Table
    "Coffee Table":                        "Table",
    "Corner/Side Table":                   "Table",
    "Dining Table":                        "Table",
    "Tea Table":                           "Table",
    "Round End Table":                     "Table",
    "Dressing Table":                      "Table",
    # Bed
    "King-size Bed":                       "Bed",
    "Double Bed":                          "Bed",
    "Single bed":                          "Bed",
    "Bed Frame":                           "Bed",
    "Kids Bed":                            "Bed",
    "Bunk Bed":                            "Bed",
    # Cabinet (include console/sideboard/wine/shoe/bar)
    "Drawer Chest / Corner cabinet":               "Cabinet",
    "Sideboard / Side Cabinet / Console Table":    "Cabinet",
    "Sideboard / Side Cabinet / Console":          "Cabinet",
    "Children Cabinet":                    "Cabinet",
    "Wine Cabinet":                        "Cabinet",
    "Wine Cooler":                         "Cabinet",
    "Shoe Cabinet":                        "Cabinet",
    "Bar":                                 "Cabinet",
    # Nightstand
    "Nightstand":                          "Nightstand",
    # Shelf
    "Shelf":                               "Shelf",
    # Desk
    "Desk":                                "Desk",
    # TV Stand
    "TV Stand":                            "TV Stand",
    # Wardrobe
    "Wardrobe":                            "Wardrobe",
    # Bookcase
    "Bookcase / jewelry Armoire":          "Bookcase",
    # Lighting
    "Pendant Lamp":                        "Pendant Lamp",
    "Ceiling Lamp":                        "Ceiling Lamp",
    "Floor Lamp":                          "Floor Lamp",
    # Nemapabile la cele 14 coarse (excluse din numitor)
    "Wall Lamp":                           OTHER,
    "Footstool / Sofastool / Bed End Stool / Stool": OTHER,
    "None":                                OTHER,
}


def coarse_category(raw_category: str) -> str:
    """Mapează o categorie fină 3D-FUTURE la categoria coarse (sau OTHER)."""
    if raw_category is None:
        return OTHER
    return CATEGORY_TO_COARSE.get(raw_category.strip(), OTHER)


# Utilitar: derivarea stilului dominant al unei camere din obiectele membre

def derive_room_style(member_styles_raw: list[str], mixed_threshold: float = 0.5) -> str:
    """
    Derivă stilul camerei din stilurile (brute) ale obiectelor de mobilier membre.

    Reguli:
      - se normalizează la stiluri canonice; AMBIGUOUS se ignoră.
      - dacă fracția stilului majoritar < `mixed_threshold` → "Mixed".
      - dacă nu rămâne niciun stil valid → "Mixed".
    """
    from collections import Counter
    canon = [normalize_style(s) for s in member_styles_raw]
    canon = [s for s in canon if s not in (AMBIGUOUS,)]
    if not canon:
        return "Mixed"
    counts = Counter(canon)
    ranked = counts.most_common()
    top_style, top_n = ranked[0]
    # egalitate la vârf (ex. 2-2) → fără stil dominant
    if len(ranked) > 1 and ranked[1][1] == top_n:
        return "Mixed"
    if top_n / len(canon) < mixed_threshold:
        return "Mixed"
    return top_style


if __name__ == "__main__":
    # Sanity check rapid + acoperire pe valorile reale din model_info.json
    print("Canonical styles:", CANONICAL_STYLES)
    print("Coarse categories:", COARSE_CATEGORIES)
    print("normalize 'Light Luxury' →", normalize_style("Light Luxury"))
    print("normalize 'Korean' →", normalize_style("Korean"))
    print("coarse 'L-shaped Sofa' →", coarse_category("L-shaped Sofa"))
    print("coarse 'Wall Lamp' →", coarse_category("Wall Lamp"))
    print("room style (3x Modern, 1x Nordic) →",
          derive_room_style(["Modern", "Modern", "Modern", "Nordic"]))
    print("room style (2x Modern, 2x Nordic) →",
          derive_room_style(["Modern", "Modern", "Nordic", "Nordic"]))
