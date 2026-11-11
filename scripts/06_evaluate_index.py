"""
06_evaluate_index.py
Evaluare cantitativă a indexului de retrieval CLIP, fără adnotare manuală
și fără LM Studio. Lucrează direct pe embeddings-urile deja construite de
`04_build_index.py` (catalog/index/), deci rularea durează secunde.

Constrângere de design: runtime total ≤ 30 minute (în practică < 5 min, de
regulă câteva secunde) — nu se reconstruiește niciun embedding, doar se fac
produse scalare pe matricele [N, D] deja salvate.

Trei evaluări:

  1. Self-consistency (coerență index) — PROXY auto-generat
     Pentru fiecare obiect i, query-ul "corect" este chiar embedding-ul propriu
     (text și/sau imagine). Un index coerent trebuie să returneze obiectul i la
     rank 1 când queryezi cu propriile embeddings.
     Aceasta nu este o măsură de relevanță pentru utilizator. Este o
     metrică de COERENȚĂ / sanitate a indexului (un lower bound trivial): dacă
     un retriever nu-și găsește propriul obiect la rank 1, indexul e
     dezaliniat sau conține duplicate. Se raportează ca atare.

  2. Category Precision@K (cross-object, în interiorul aceleiași categorii)
     Pentru N obiecte sample per super-categorie, se folosește caption-ul
     (embedding text) ca query și se verifică ce fracțiune din top-K rezultate
     (excluzând obiectul-query însuși) aparțin aceleiași super-categorii.
     Măsoară cât de bine separă spațiul de embeddings categoriile de mobilier.

  3. Ablație α (pondere canal imagine)
     Pe același set de query-uri (sample per categorie), se calculează scorul
     combinat s = (1-α)·text + α·image pentru α ∈ {0, 0.25, 0.5, 0.75, 1.0}
     și se raportează Category Precision@5 per α, pentru a justifica alegerea
     lui α în pipeline-ul de recomandare.

Rezultatele se scriu în catalog/index/eval_results.json (lângă index) și
se printează un tabel.

Rulare:
    python 06_evaluate_index.py --index_dir ./catalog/index
    python 06_evaluate_index.py --index_dir ./catalog/index \\
        --top_k 5 --sample_per_cat 50 --seed 0
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime


# Încărcare index

def load_index(index_dir: Path):
    with open(index_dir / "catalog_index.jsonl") as f:
        catalog = [json.loads(line) for line in f]

    text_embs = np.load(index_dir / "text_embeddings.npy").astype(np.float32)

    img_path = index_dir / "image_embeddings.npy"
    image_embs = None
    if img_path.exists():
        image_embs = np.load(img_path).astype(np.float32)

    with open(index_dir / "index_meta.json") as f:
        meta = json.load(f)

    return catalog, text_embs, image_embs, meta


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Re-normalizează L2 pe rânduri (embeddings ar trebui deja normalizate,
    dar thumbnails lipsă au vector zero → normă 0; le lăsăm la zero)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    safe = np.where(norms > 1e-8, norms, 1.0)
    return mat / safe


# 1. Self-consistency (coerență index — proxy)

def eval_self_consistency(embs: np.ndarray, label: str, top_k: int) -> dict:
    """
    Query = embedding propriu. Verifică dacă obiectul i apare la rank 1 (și în top-K)
    în propria listă de rezultate. Metrică de COERENȚĂ index, nu de relevanță.

    Obiectele cu embedding zero (ex. thumbnail lipsă pe canalul imagine) sunt
    excluse din numitor.
    """
    n = embs.shape[0]
    valid = np.linalg.norm(embs, axis=1) > 1e-8
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {"channel": label, "n_valid": 0, "recall@1": None, f"recall@{top_k}": None}

    sim = embs @ embs.T  # [N, N], cosine (embeddings normalizate)

    rank1_hits = 0
    topk_hits = 0
    for i in range(n):
        if not valid[i]:
            continue
        order = np.argsort(sim[i])[::-1]
        if order[0] == i:
            rank1_hits += 1
        if i in order[:top_k]:
            topk_hits += 1

    return {
        "channel":           label,
        "n_valid":           n_valid,
        "recall@1":          rank1_hits / n_valid,
        f"recall@{top_k}":   topk_hits / n_valid,
    }


# 2 & 3. Category Precision@K + ablație α

def sample_query_indices(catalog: list[dict], sample_per_cat: int, rng: np.random.Generator) -> list[int]:
    """Eșantionează până la `sample_per_cat` indici per super_category."""
    by_cat: dict[str, list[int]] = {}
    for i, e in enumerate(catalog):
        cat = (e.get("super_category") or "").lower()
        by_cat.setdefault(cat, []).append(i)

    selected = []
    for cat, idxs in by_cat.items():
        if not cat:
            continue
        idxs = np.array(idxs)
        if len(idxs) > sample_per_cat:
            idxs = rng.choice(idxs, size=sample_per_cat, replace=False)
        selected.extend(int(i) for i in idxs)
    return selected


def category_precision_at_k(
    query_idx: list[int],
    combined: np.ndarray,        # matrice de scoruri [N, N] sau funcție de scor
    cat_labels: np.ndarray,      # [N] super_category (lowercase) per obiect
    top_k: int,
) -> float:
    """
    Pentru fiecare query i, ia top-K rezultate (excluzând i însuși) și calculează
    fracțiunea care are aceeași super_category ca i. Media pe toate query-urile.
    """
    precisions = []
    for i in query_idx:
        scores = combined[i].copy()
        scores[i] = -np.inf  # exclude obiectul-query
        topk = np.argsort(scores)[::-1][:top_k]
        same = (cat_labels[topk] == cat_labels[i]).sum()
        precisions.append(same / top_k)
    return float(np.mean(precisions)) if precisions else 0.0


def build_combined_sim(text_embs, image_embs, alpha: float) -> np.ndarray:
    """Matrice de similaritate combinată [N, N]: (1-α)·text·textᵀ + α·img·imgᵀ.

    Pe canalul text query-ul e caption-ul obiectului; pe canalul imagine e
    thumbnail-ul. Pentru obiectele fără imagine (vector zero), contribuția
    canalului imagine e 0, deci scorul cade pe canalul text — consistent cu
    fallback-ul din 05_recommend.py."""
    text_sim = text_embs @ text_embs.T
    if image_embs is None or alpha == 0.0:
        return text_sim
    img_sim = image_embs @ image_embs.T
    return (1.0 - alpha) * text_sim + alpha * img_sim


# Main

def main():
    parser = argparse.ArgumentParser(description="Evaluare cantitativă index CLIP (fără LM Studio)")
    parser.add_argument("--index_dir",      default="./catalog/index")
    parser.add_argument("--top_k",          type=int, default=5)
    parser.add_argument("--sample_per_cat", type=int, default=50,
                        help="Nr. max query-uri eșantionate per super_category")
    parser.add_argument("--seed",           type=int, default=0)
    parser.add_argument("--output_json",    default=None,
                        help="Implicit: <index_dir>/eval_results.json")
    parser.add_argument("--alphas",         default="0,0.25,0.5,0.75,1.0",
                        help="Valori α pentru ablație, separate prin virgulă")
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    rng = np.random.default_rng(args.seed)

    print(f"[Index] Încărcare din {index_dir} ...")
    catalog, text_embs, image_embs, meta = load_index(index_dir)
    text_embs = l2_normalize(text_embs)
    if image_embs is not None:
        image_embs = l2_normalize(image_embs)
    n = len(catalog)
    print(f"  N={n}, D={meta.get('embedding_dim')}, "
          f"image_embs={'DA' if image_embs is not None else 'NU'}")

    cat_labels = np.array([(e.get("super_category") or "").lower() for e in catalog])

    # 1. Self-consistency (coerență index)
    print("\n[1/3] Self-consistency (PROXY de coerență index, nu relevanță) ...")
    sc_text = eval_self_consistency(text_embs, "text", args.top_k)
    sc_results = [sc_text]
    print(f"  text : recall@1={sc_text['recall@1']:.4f}  "
          f"recall@{args.top_k}={sc_text[f'recall@{args.top_k}']:.4f}  "
          f"(n_valid={sc_text['n_valid']})")
    if image_embs is not None:
        sc_img = eval_self_consistency(image_embs, "image", args.top_k)
        sc_results.append(sc_img)
        r1 = sc_img['recall@1']
        rk = sc_img[f'recall@{args.top_k}']
        print(f"  image: recall@1={r1:.4f}  recall@{args.top_k}={rk:.4f}  "
              f"(n_valid={sc_img['n_valid']})")

    # 2. Category Precision@K (canal text)
    print(f"\n[2/3] Category Precision@{args.top_k} "
          f"(sample ≤ {args.sample_per_cat}/categorie, query=caption) ...")
    query_idx = sample_query_indices(catalog, args.sample_per_cat, rng)
    print(f"  Query-uri eșantionate: {len(query_idx)}")

    text_sim = text_embs @ text_embs.T
    cat_p_text = category_precision_at_k(query_idx, text_sim, cat_labels, args.top_k)
    print(f"  Category Precision@{args.top_k} (text-only): {cat_p_text:.4f}")

    # Per-categorie (text-only) pentru detaliu
    per_cat = {}
    for cat in sorted(set(cat_labels)):
        if not cat:
            continue
        q = [i for i in query_idx if cat_labels[i] == cat]
        if q:
            per_cat[cat] = category_precision_at_k(q, text_sim, cat_labels, args.top_k)

    # 3. Ablație α
    print(f"\n[3/3] Ablație α → Category Precision@{args.top_k} ...")
    alphas = [float(a) for a in args.alphas.split(",")]
    ablation = []
    for alpha in alphas:
        if image_embs is None and alpha > 0.0:
            # Fără embeddings de imagine, α>0 nu schimbă nimic; marcăm N/A.
            ablation.append({"alpha": alpha, "category_precision@k": None,
                             "note": "no image embeddings"})
            print(f"  α={alpha:<4}  Cat-P@{args.top_k}=  N/A (fără image embeddings)")
            continue
        combined = build_combined_sim(text_embs, image_embs, alpha)
        cp = category_precision_at_k(query_idx, combined, cat_labels, args.top_k)
        ablation.append({"alpha": alpha, "category_precision@k": cp})
        print(f"  α={alpha:<4}  Cat-P@{args.top_k}={cp:.4f}")

    # Tabel rezumat
    print("\n" + "=" * 60)
    print("REZUMAT EVALUARE INDEX")
    print("=" * 60)
    print(f"  Obiecte index (N):              {n}")
    print(f"  Self-consistency text  R@1:     {sc_text['recall@1']:.4f}")
    if image_embs is not None:
        print(f"  Self-consistency image R@1:     {sc_results[1]['recall@1']:.4f}")
    print(f"  Category Precision@{args.top_k} (text):   {cat_p_text:.4f}")
    best = max((a for a in ablation if a['category_precision@k'] is not None),
               key=lambda a: a['category_precision@k'], default=None)
    if best:
        print(f"  Cel mai bun α (Cat-P@{args.top_k}):       "
              f"α={best['alpha']} → {best['category_precision@k']:.4f}")
    print("=" * 60)

    # Salvare
    output_path = Path(args.output_json) if args.output_json else (index_dir / "eval_results.json")
    results = {
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "index_meta": meta,
        "config": {
            "top_k":          args.top_k,
            "sample_per_cat": args.sample_per_cat,
            "seed":           args.seed,
            "n_query":        len(query_idx),
            "alphas":         alphas,
        },
        "self_consistency": {
            "_note": ("PROXY de coerență index (obiect găsit la rank 1 din "
                      "propriul embedding); nu este evaluare de relevanță pentru "
                      "utilizator, ci lower bound de sanitate a indexului."),
            "channels": sc_results,
        },
        "category_precision": {
            f"text_only@{args.top_k}": cat_p_text,
            "per_category":           per_cat,
        },
        "alpha_ablation": {
            f"category_precision@{args.top_k}": ablation,
        },
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nRezultate salvate: {output_path}")


if __name__ == "__main__":
    main()
