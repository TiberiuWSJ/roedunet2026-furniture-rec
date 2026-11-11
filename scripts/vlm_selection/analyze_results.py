"""
analyze_results.py
Agregă rezultatele brute (results/raw_<model>.jsonl), calculează metricile D1–D6,
aplică testele statistice și produce:
  - results/summary.json        (toate metricile, per model)
  - results/summary_table.csv   (tabel comparativ)
  - results/pareto.png          (front Pareto: calitate regăsire × latență)
  - teste pereche (McNemar pe D2/D3, Wilcoxon pe D4) + Spearman specificitate↔regăsire
  - aplicarea regulii de decizie (poartă D1 + buget latență → recomandare)

Rulare:
    python analyze_results.py
"""

from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import config
import metrics as M
from taxonomy import OTHER, AMBIGUOUS


# Încărcare

def load_all(results_dir: str) -> dict[str, list[dict]]:
    by_model = {}
    for p in glob.glob(str(Path(results_dir) / "raw_*.jsonl")):
        label = Path(p).stem[len("raw_"):]
        with open(p, encoding="utf-8") as f:
            by_model[label] = [json.loads(l) for l in f if l.strip()]
    return by_model


# Statistică

def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05):
    vals = np.array([v for v in values if v == v])  # drop NaN
    if len(vals) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(42)
    boots = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    return (float(vals.mean()),
            float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


def mcnemar(a_correct: list[bool], b_correct: list[bool]) -> dict:
    """Test McNemar pereche (aceleași instanțe). Returnează b, c și p (binomial exact)."""
    from math import comb
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return {"b": b, "c": c, "p_value": min(1.0, p)}


def wilcoxon(x: list[float], y: list[float]) -> dict:
    """Wilcoxon signed-rank pereche (aproximare normală)."""
    pairs = [(a, b) for a, b in zip(x, y) if a == a and b == b]
    diffs = [a - b for a, b in pairs if (a - b) != 0]
    n = len(diffs)
    if n < 1:
        return {"n": 0, "p_value": float("nan")}
    ranks = np.argsort(np.argsort([abs(d) for d in diffs])) + 1
    w_plus = sum(r for r, d in zip(ranks, diffs) if d > 0)
    mu = n * (n + 1) / 4
    sigma = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    if sigma == 0:
        return {"n": n, "p_value": float("nan")}
    z = (w_plus - mu) / sigma
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return {"n": n, "z": float(z), "p_value": float(min(1.0, p))}


def spearman(x: list[float], y: list[float]) -> float:
    pairs = [(a, b) for a, b in zip(x, y) if a == a and b == b]
    if len(pairs) < 3:
        return float("nan")
    xs = np.array([a for a, _ in pairs]); ys = np.array([b for _, b in pairs])
    rx = np.argsort(np.argsort(xs)); ry = np.argsort(np.argsort(ys))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = (np.sqrt((rx**2).sum()) * np.sqrt((ry**2).sum()))
    return float((rx * ry).sum() / denom) if denom else float("nan")


# Pareto

def pareto_front(points: list[tuple[str, float, float]]):
    """points = [(label, quality↑, latency↓)] → subset ne-dominat."""
    front = []
    for lbl, q, lat in points:
        dominated = any((q2 >= q and lat2 <= lat and (q2 > q or lat2 < lat))
                        for l2, q2, lat2 in points if l2 != lbl)
        if not dominated:
            front.append(lbl)
    return front


# Main

def main():
    by_model = load_all(config.RESULTS_DIR)
    if not by_model:
        print(f"Niciun rezultat în {config.RESULTS_DIR}. Rulează întâi run_eval.py.")
        return

    k = config.RETRIEVAL_K
    summary = {}
    for label, recs in by_model.items():
        agg = M.aggregate_model(recs, k=k)
        # CI bootstrap pe D4 category_precision (exemplu de incertitudine)
        catp = []
        for r in recs:
            ret = r.get("retrieval")
            if ret and ret.get("topk_categories") and r.get("gt_category_coarse") not in (OTHER, None, ""):
                tk = ret["topk_categories"][:k]
                catp.append(sum(1 for c in tk if c == r["gt_category_coarse"]) / len(tk))
        agg["D4_category_precision_CI"] = bootstrap_ci(catp)
        summary[label] = agg

    # tabel CSV
    out = Path(config.RESULTS_DIR)
    csv_path = out / "summary_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "n", "D1_json_valid", "D2_style_acc", "D2_macroF1",
                    "D3_cat_acc", f"D4_cat_prec@{k}", f"D4_style_prec@{k}",
                    f"D4_recall@{k}", "D4_specificity", "phantom_style_rate",
                    "lat_median_s", "lat_p90_s", "tok_per_s"])
        for label, a in summary.items():
            w.writerow([
                label, a["n_instances"], f"{a['D1_json_valid_rate']:.3f}",
                f"{a['D2_room_style']['accuracy']:.3f}", f"{a['D2_room_style']['macro_f1']:.3f}",
                f"{a['D3_category']['top1_accuracy']:.3f}",
                f"{a['D4_retrieval'][f'category_precision@{k}']:.3f}",
                f"{a['D4_retrieval'][f'style_precision@{k}']:.3f}",
                f"{a['D4_retrieval'][f'gt_recall@{k}']:.3f}",
                f"{a['D4_specificity']['mean_specificity']:.2f}",
                f"{a['D_hallucination']['phantom_style_rate']:.3f}",
                f"{a['D5_efficiency']['latency_median_s']:.1f}",
                f"{a['D5_efficiency']['latency_p90_s']:.1f}",
                f"{a['D5_efficiency']['tokens_per_s_mean']:.1f}",
            ])
    print(f"tabel: {csv_path}")

    # teste pereche (D2 accuracy, McNemar) pe modele cu aceleași instanțe
    labels = list(by_model.keys())
    pair_tests = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb = labels[i], labels[j]
            # aliniere pe instance_id
            da = {r["instance_id"]: r for r in by_model[la]}
            db = {r["instance_id"]: r for r in by_model[lb]}
            common = [iid for iid in da if iid in db]
            a_corr, b_corr, qa, qb = [], [], [], []
            for iid in common:
                ra, rb = da[iid], db[iid]
                ga = ra.get("gt_room_style")
                if ga in (AMBIGUOUS, OTHER, None, ""):
                    continue
                a_corr.append((ra.get("parsed") or {}).get("room_style") == ga)
                b_corr.append((rb.get("parsed") or {}).get("room_style") == ga)
                # D4 category precision pereche
                for src, dst in ((ra, qa), (rb, qb)):
                    ret = src.get("retrieval")
                    if ret and ret.get("topk_categories"):
                        tk = ret["topk_categories"][:k]
                        gc = src.get("gt_category_coarse")
                        dst.append(sum(1 for c in tk if c == gc) / len(tk) if gc not in (OTHER, None, "") else float("nan"))
                    else:
                        dst.append(float("nan"))
            pair_tests[f"{la}_vs_{lb}"] = {
                "D2_mcnemar": mcnemar(a_corr, b_corr),
                "D4_catprec_wilcoxon": wilcoxon(qa, qb),
                "n_common": len(common),
            }

    # Spearman: specificitate ↔ category precision (pe toate instanțele)
    spec_all, prec_all = [], []
    for recs in by_model.values():
        for r in recs:
            if not r.get("json_valid"):
                continue
            spec_all.append(M.query_specificity((r.get("parsed") or {}).get("replacement_query", "")))
            ret = r.get("retrieval")
            gc = r.get("gt_category_coarse")
            if ret and ret.get("topk_categories") and gc not in (OTHER, None, ""):
                tk = ret["topk_categories"][:k]
                prec_all.append(sum(1 for c in tk if c == gc) / len(tk))
            else:
                prec_all.append(float("nan"))
    spearman_spec_prec = spearman(spec_all, prec_all)

    # Regula de decizie
    eligible = {l: a for l, a in summary.items()
                if a["D1_json_valid_rate"] >= config.D1_GATE_MIN
                and a["D5_efficiency"]["latency_p90_s"] <= config.LATENCY_BUDGET_P90_S}
    recommendation = None
    if eligible:
        recommendation = max(
            eligible.items(),
            key=lambda kv: (kv[1]["D4_retrieval"][f"category_precision@{k}"]
                            if kv[1]["D4_retrieval"][f"category_precision@{k}"] == kv[1]["D4_retrieval"][f"category_precision@{k}"]
                            else -1))[0]

    # Pareto plot
    pts = [(l, a["D4_retrieval"][f"category_precision@{k}"], a["D5_efficiency"]["latency_median_s"])
           for l, a in summary.items()]
    front = pareto_front([(l, q if q == q else -1, lat if lat == lat else 1e9) for l, q, lat in pts])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for l, q, lat in pts:
            on = l in front
            ax.scatter(lat, q, s=90, color="#c0392b" if on else "#7f8c8d", zorder=3)
            ax.annotate(l, (lat, q), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("Latență mediană / inferență (s) ↓")
        ax.set_ylabel(f"Category precision@{k} ↑")
        ax.set_title("Front Pareto: calitate regăsire × latență")
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "pareto.png", dpi=150)
        print(f"Pareto: {out / 'pareto.png'}")
    except Exception as e:
        print(f"  [!] plot Pareto sărit: {e}")

    full = {
        "summary": summary,
        "pairwise_tests": pair_tests,
        "spearman_specificity_vs_category_precision": spearman_spec_prec,
        "pareto_front": front,
        "decision": {
            "D1_gate_min": config.D1_GATE_MIN,
            "latency_budget_p90_s": config.LATENCY_BUDGET_P90_S,
            "eligible_models": list(eligible.keys()),
            "recommended_model": recommendation,
        },
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, ensure_ascii=False)
    print(f"summary: {out / 'summary.json'}")
    print(f"\nFront Pareto: {front}")
    print(f"Spearman(specificitate, category_precision) = {spearman_spec_prec:.3f}")
    print(f"RECOMANDARE (poartă D1 + buget latență → cel mai bun la regăsire): {recommendation}")


if __name__ == "__main__":
    main()
