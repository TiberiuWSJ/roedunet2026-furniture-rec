"""
metrics.py
Funcții pentru metricile D1–D6.

Toate funcțiile operează pe liste de înregistrări per-instanță și sunt pure
(fără efecte secundare), pentru a fi ușor de testat și agregat.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

import numpy as np

from prompts import REQUIRED_KEYS
from taxonomy import ROOM_STYLE_ENUM, COARSE_CATEGORIES, OTHER, AMBIGUOUS, PHANTOM_STYLES


# D1: Fiabilitate structurală (poartă)

def d1_json_valid_rate(records: list[dict]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r.get("json_valid")) / len(records)


def d1_schema_conformance(records: list[dict]) -> dict:
    """Rata de prezență a tuturor cheilor + rata de apartenență la enum-uri."""
    n = len(records) or 1
    all_keys = 0
    style_in_enum = 0
    cat_in_enum = 0
    for r in records:
        p = r.get("parsed") or {}
        if all(k in p for k in REQUIRED_KEYS):
            all_keys += 1
        if p.get("room_style") in ROOM_STYLE_ENUM:
            style_in_enum += 1
        if p.get("selected_category") in COARSE_CATEGORIES:
            cat_in_enum += 1
    return {
        "all_keys_present_rate": all_keys / n,
        "room_style_in_enum_rate": style_in_enum / n,
        "selected_category_in_enum_rate": cat_in_enum / n,
    }


# D2 / D3: Clasificare (stil cameră, categorie obiect)

def _accuracy(pairs: list[tuple[str, str]]) -> float:
    """pairs = [(pred, gt)]; ignoră perechile cu gt nemapabil."""
    valid = [(p, g) for p, g in pairs if g not in (OTHER, AMBIGUOUS, None, "")]
    if not valid:
        return float("nan")
    return sum(1 for p, g in valid if p == g) / len(valid)


def _macro_f1(pairs: list[tuple[str, str]], labels: list[str]) -> float:
    valid = [(p, g) for p, g in pairs if g not in (OTHER, AMBIGUOUS, None, "")]
    if not valid:
        return float("nan")
    f1s = []
    for lab in labels:
        tp = sum(1 for p, g in valid if p == lab and g == lab)
        fp = sum(1 for p, g in valid if p == lab and g != lab)
        fn = sum(1 for p, g in valid if p != lab and g == lab)
        if tp == 0 and (fp == 0 or fn == 0):
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else float("nan")


def d2_room_style(records: list[dict]) -> dict:
    pairs = [((r.get("parsed") or {}).get("room_style"), r.get("gt_room_style"))
             for r in records if r.get("json_valid")]
    return {
        "accuracy": _accuracy(pairs),
        "macro_f1": _macro_f1(pairs, ROOM_STYLE_ENUM),
        "n": len([1 for _, g in pairs if g not in (OTHER, AMBIGUOUS, None, "")]),
    }


def d3_category(records: list[dict]) -> dict:
    pairs = [((r.get("parsed") or {}).get("selected_category"),
              r.get("gt_category_coarse"))
             for r in records if r.get("json_valid")]
    return {
        "top1_accuracy": _accuracy(pairs),
        "n": len([1 for _, g in pairs if g not in (OTHER, AMBIGUOUS, None, "")]),
    }


def confusion_matrix(records: list[dict], field_pred: str, field_gt: str,
                     labels: list[str]) -> dict:
    """Matrice de confuzie ca dict imbricat {gt: {pred: count}}."""
    cm = defaultdict(lambda: defaultdict(int))
    for r in records:
        if not r.get("json_valid"):
            continue
        g = r.get(field_gt)
        p = (r.get("parsed") or {}).get(field_pred)
        if g in (OTHER, AMBIGUOUS, None, ""):
            continue
        cm[g][p] += 1
    return {g: dict(d) for g, d in cm.items()}


# D4: Calitatea replacement_query prin regăsire

def d4_retrieval(records: list[dict], k: int = 5) -> dict:
    """
    Agregă metricile de regăsire pre-calculate per-instanță în câmpul
    r["retrieval"] = {
        "topk_categories": [...], "topk_styles": [...],
        "gt_rank": int|None, "cos_to_gt": float|None
    }
    """
    cat_prec, sty_prec, recall_at_k, ranks, cosines = [], [], [], [], []
    for r in records:
        ret = r.get("retrieval")
        if not ret:
            continue
        gt_cat = r.get("gt_category_coarse")
        gt_sty = r.get("gt_room_style")
        topk_cat = ret.get("topk_categories", [])[:k]
        topk_sty = ret.get("topk_styles", [])[:k]
        if topk_cat and gt_cat not in (OTHER, None, ""):
            cat_prec.append(sum(1 for c in topk_cat if c == gt_cat) / len(topk_cat))
        if topk_sty and gt_sty not in (AMBIGUOUS, None, ""):
            sty_prec.append(sum(1 for s in topk_sty if s == gt_sty) / len(topk_sty))
        gr = ret.get("gt_rank")
        if gr is not None:
            ranks.append(gr)
            recall_at_k.append(1.0 if gr <= k else 0.0)
        cg = ret.get("cos_to_gt")
        if cg is not None:
            cosines.append(cg)

    def _m(x):
        return float(np.mean(x)) if x else float("nan")

    return {
        f"category_precision@{k}": _m(cat_prec),
        f"style_precision@{k}": _m(sty_prec),
        f"gt_recall@{k}": _m(recall_at_k),
        "gt_mean_rank": _m(ranks),
        "query_to_gt_cosine": _m(cosines),
        "n": len(cat_prec),
    }


def query_specificity(query: str) -> int:
    """
    Scor 0–4 de specificitate lingvistică a replacement_query (D4):
    +1 termen de stil, +1 material, +1 formă/proporție, +1 context de cameră.
    Bazat pe cuvinte-cheie (proxy ușor, nu NLP complet).
    """
    if not query:
        return 0
    q = query.lower()
    style_kw = ["modern", "minimalist", "nordic", "scandinav", "industrial",
                "art deco", "traditional", "asian", "chinese", "japanese",
                "luxury", "vintage", "classic", "contemporary"]
    material_kw = ["wood", "oak", "walnut", "linen", "fabric", "leather", "metal",
                   "velvet", "glass", "marble", "rattan", "steel", "cotton"]
    shape_kw = ["slim", "rounded", "curved", "low", "tall", "compact", "wide",
                "rectangular", "round", "clean lines", "legs", "sleek", "boxy"]
    context_kw = ["living room", "bedroom", "dining", "studio", "apartment",
                  "scandinavian", "loft", "office", "room"]
    score = 0
    score += any(w in q for w in style_kw)
    score += any(w in q for w in material_kw)
    score += any(w in q for w in shape_kw)
    score += any(w in q for w in context_kw)
    return int(score)


def d4_specificity(records: list[dict]) -> dict:
    scores = [query_specificity((r.get("parsed") or {}).get("replacement_query", ""))
              for r in records if r.get("json_valid")]
    return {"mean_specificity": float(np.mean(scores)) if scores else float("nan"),
            "n": len(scores)}


# D-halucinație: rata de etichete-fantomă (proxy)

def d_phantom_style_rate(records: list[dict]) -> dict:
    """
    Fracția de predicții (din JSON valid) care aleg un stil-momeală absent din
    ground-truth (PHANTOM_STYLES, ex. "Art Deco"). Proxy de halucinație: opțiune
    care nu poate fi niciodată corectă. Semnal moale (vezi notă în taxonomy.py).
    """
    preds = [(r.get("parsed") or {}).get("room_style")
             for r in records if r.get("json_valid")]
    if not preds:
        return {"phantom_style_rate": float("nan"), "n": 0,
                "phantom_styles": sorted(PHANTOM_STYLES)}
    n_phantom = sum(1 for p in preds if p in PHANTOM_STYLES)
    return {"phantom_style_rate": n_phantom / len(preds),
            "n": len(preds), "n_phantom": n_phantom,
            "phantom_styles": sorted(PHANTOM_STYLES)}


# D5: Eficiență

def d5_efficiency(records: list[dict]) -> dict:
    lat = [r["latency_s"] for r in records if r.get("latency_s") is not None]
    tps = [r["tokens_per_s"] for r in records if r.get("tokens_per_s")]
    return {
        "latency_median_s": float(np.median(lat)) if lat else float("nan"),
        "latency_p90_s": float(np.percentile(lat, 90)) if lat else float("nan"),
        "tokens_per_s_mean": float(np.mean(tps)) if tps else float("nan"),
        "n": len(lat),
    }


# D6: Robustețe / determinism

def d6_determinism(repeat_groups: list[list[dict]]) -> dict:
    """
    repeat_groups: listă de grupuri; fiecare grup = R rulări ale ACELEIAȘI instanțe.
    Măsoară fracția de grupuri cu ieșire identică pe câmpurile-cheie (determinism).
    """
    if not repeat_groups:
        return {"identical_rate": float("nan"), "n": 0}
    identical = 0
    for grp in repeat_groups:
        keys = set()
        for r in grp:
            p = r.get("parsed") or {}
            keys.add((p.get("room_style"), p.get("selected_category"),
                      p.get("replacement_query")))
        if len(keys) == 1:
            identical += 1
    return {"identical_rate": identical / len(repeat_groups), "n": len(repeat_groups)}


# Agregare per model

def aggregate_model(records: list[dict], k: int = 5) -> dict:
    return {
        "n_instances": len(records),
        "D1_json_valid_rate": d1_json_valid_rate(records),
        "D1_schema": d1_schema_conformance(records),
        "D2_room_style": d2_room_style(records),
        "D3_category": d3_category(records),
        "D4_retrieval": d4_retrieval(records, k=k),
        "D4_specificity": d4_specificity(records),
        "D5_efficiency": d5_efficiency(records),
        "D_hallucination": d_phantom_style_rate(records),
    }
