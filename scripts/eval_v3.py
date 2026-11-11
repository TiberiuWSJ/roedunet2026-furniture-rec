"""
eval_v3.py — selecție pe DEV și evaluare unică pe TEST (numpy only).

  python scripts/eval_v3.py select   # alege prompt + fuziune + α + re-rank pe dev (nDCG@10)
  python scripts/eval_v3.py test     # evaluează configurația aleasă pe cele 300 de interogări test

Metrici per interogare (obiectul-query exclus din ranking):
  cat_p5 / style_p5 / joint_p5 — aceeași super-categorie / același stil / ambele
  fine_p5  — aceeași categorie fină 3D-FUTURE (32 clase)
  ndcg10   — câștig graduat = 1[super-cat] + 1[cat. fină] + 1[stil]  (0..3)
Variante „nodup": se exclud și obiectele din catalog cu cos(imagine) ≥ 0.97 față de obiectul-query
(duplicate / aproape-duplicate vizuale).
Fuziune: raw  = (1-α)·cos_T + α·cos_I ;  z = (1-α)·z(cos_T) + α·z(cos_I), z calculat per interogare
pe catalog. Re-rank: min-max, filtru dur pe categoria VLM, penalizare 0.1 pentru stil ≠ stilul VLM.
"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CAT = ROOT / os.environ.get("PAPER_CATALOG", "catalog")
OUT = ROOT / "results"
STYLES = ["Modern", "Minimalist", "Nordic", "Industrial", "Art Deco", "European Traditional",
          "Southeast Asian", "Chinese Traditional", "Japanese", "Luxury", "Kids"]
SUPERCATS = ["Sofa", "Chair", "Table", "Bed", "Cabinet/Shelf/Desk", "Lighting"]
ALPHAS = [round(a, 1) for a in np.linspace(0, 1, 11)]
K, KN, DUP = 5, 10, 0.97

cat = [json.loads(l) for l in open(CAT / "index/catalog_index.jsonl")]
T = np.load(CAT / "index/text_embeddings.npy").astype(np.float32)
I = np.load(CAT / "index/image_embeddings.npy").astype(np.float32)
sc = np.array([e["super_category"] for e in cat]); st = np.array([e["style"] for e in cat])
fn = np.array([e["category"] for e in cat]); N = len(cat)
img_ok = np.linalg.norm(I, axis=1) > 1e-8
SIMI = I @ I.T


def norm_label(x, vocab):
    return next((v for v in vocab if isinstance(x, str) and x.strip().lower() == v.lower()), None)


def load(split, prompt):
    tag = "" if (split == "test" and prompt == "P0") else f"_{split}_{prompt}"
    f = OUT / f"queries_vlm{tag}.jsonl"
    if not f.exists():
        return None
    rows = {json.loads(l)["idx"]: json.loads(l) for l in open(f)}
    z = np.load(OUT / f"queries_emb{tag}.npz")
    idx = z["idx"]; R = [rows[int(i)] for i in idx]
    pc = [norm_label((r["parsed"] or {}).get("category"), SUPERCATS) for r in R]
    ps = [norm_label((r["parsed"] or {}).get("style"), STYLES) for r in R]
    return {"idx": idx, "QT": z["q_text"], "QI": z["q_img"], "pc": pc, "ps": ps, "rows": R}


def valid_mask(q, nodup):
    m = img_ok.copy(); m[q] = False
    if nodup:
        m &= SIMI[q] < DUP
    return m


def metrics(order, q, m):
    o = order[:K]; o10 = order[:KN]
    gain = lambda idx: (sc[idx] == sc[q]).astype(float) + (fn[idx] == fn[q]) + (st[idx] == st[q])
    disc = 1 / np.log2(np.arange(2, KN + 2))
    dcg = (gain(o10) * disc).sum()
    ideal = np.sort(gain(np.where(m)[0]))[::-1][:KN]; idcg = (ideal * disc[:len(ideal)]).sum()
    return [np.mean(sc[o] == sc[q]), np.mean(st[o] == st[q]), np.mean((sc[o] == sc[q]) & (st[o] == st[q])),
            np.mean(fn[o] == fn[q]), dcg / idcg if idcg > 0 else 0.0]


def zs(v, m):
    mu, sd = v[m].mean(), v[m].std() + 1e-8
    return (v - mu) / sd


def ranking(D, j, alpha, fusion, rerank, nodup):
    q = int(D["idx"][j]); m = valid_mask(q, nodup)
    t = T @ D["QT"][j]; im = I @ D["QI"][j]
    s = (1 - alpha) * t + alpha * im if fusion == "raw" else (1 - alpha) * zs(t, m) + alpha * zs(im, m)
    s = s.astype(np.float64); s[~m] = -np.inf
    if rerank:
        fin = np.isfinite(s); lo, hi = s[fin].min(), s[fin].max(); s[fin] = (s[fin] - lo) / max(hi - lo, 1e-8)
        if D["pc"][j]: s[sc != D["pc"][j]] = -np.inf
        if D["ps"][j]: s[(st != D["ps"][j]) & np.isfinite(s)] -= 0.1
    return np.argsort(-s), q, m


def run(D, alpha, fusion, rerank, nodup=False):
    return np.array([metrics(*ranking(D, j, alpha, fusion, rerank, nodup)) for j in range(len(D["idx"]))])


def metadata_filter(D, nodup=False, draws=100):
    rng = np.random.default_rng(0); out = []
    for j, q in enumerate(D["idx"]):
        q = int(q); m = valid_mask(q, nodup)
        c1 = np.where(m & (sc == D["pc"][j]) & (st == D["ps"][j]))[0] if D["pc"][j] and D["ps"][j] else np.array([], int)
        c2 = np.setdiff1d(np.where(m & (sc == D["pc"][j]))[0], c1) if D["pc"][j] else np.array([], int)
        c3 = np.setdiff1d(np.where(m)[0], np.concatenate([c1, c2]))
        acc = [metrics(np.concatenate([rng.permutation(c1), rng.permutation(c2), rng.permutation(c3)]).astype(int), q, m)
               for _ in range(draws)]
        out.append(np.mean(acc, 0))
    return np.array(out)


def random_rank(D, nodup=False, draws=100):
    rng = np.random.default_rng(0); out = []
    for q in D["idx"]:
        q = int(q); m = valid_mask(q, nodup); pool = np.where(m)[0]
        out.append(np.mean([metrics(rng.permutation(pool), q, m) for _ in range(draws)], 0))
    return np.array(out)


def knn_heads_run(D, nodup=False, k=15, lam=0.5, alpha=1.0, fusion="raw"):
    """Image-only + re-rank cu etichete din capete kNN pe catalog (k, λ aleși prin CV pe train în
    learned_heads.py). Train = catalog fără interogările de test (și fără cele de dev, dacă există)."""
    STY = sorted(set(st) - {"Unspecified"}); CATS = sorted(set(sc))
    excl = set(int(i) for i in D["idx"])
    tr = np.array([i for i in range(N) if i not in excl and img_ok[i]]); trs = tr[st[tr] != "Unspecified"]
    out = []
    for j, q in enumerate(D["idx"]):
        q = int(q); m = valid_mask(q, nodup)
        oc = np.argsort(-(I[tr] @ D["QI"][j]))[:k]; os_ = np.argsort(-(I[trs] @ D["QI"][j]))[:k]
        pc = CATS[np.bincount([CATS.index(x) for x in sc[tr][oc]], minlength=len(CATS)).argmax()]
        ps = np.bincount([STY.index(x) for x in st[trs][os_]], minlength=len(STY)) / k
        t = T @ D["QT"][j]; im = I @ D["QI"][j]
        s = (1 - alpha) * t + alpha * im if fusion == "raw" else (1 - alpha) * zs(t, m) + alpha * zs(im, m)
        s = s.astype(np.float64); s[~m] = -np.inf
        fin = np.isfinite(s); lo, hi = s[fin].min(), s[fin].max(); s[fin] = (s[fin] - lo) / max(hi - lo, 1e-8)
        s[sc != pc] = -np.inf
        pst = np.array([ps[STY.index(x)] if x in STY else 0.0 for x in st])
        s = s - lam * (1 - pst)
        out.append(metrics(np.argsort(-s), q, m))
    return np.array(out)


def boot(v, n=2000):
    m = np.random.default_rng(0).choice(v, (n, len(v))).mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


NAMES = ["cat_p5", "style_p5", "joint_p5", "fine_p5", "ndcg10"]
summ = lambda A: {n: boot(A[:, c]) for c, n in enumerate(NAMES)}


def cmd_select():
    grid = []
    for p in ["P0", "P1", "P2"]:
        D = load("dev", p)
        if D is None:
            print(f"[select] lipsește dev {p}"); continue
        acc = np.mean([a == b for a, b in zip(D["pc"], [sc[int(i)] for i in D["idx"]])])
        for fusion in ["raw", "z"]:
            for rr in [False, True]:
                for a in ALPHAS:
                    if fusion == "z" and a in (0.0, 1.0):
                        pass
                    A = run(D, a, fusion, rr)
                    grid.append({"prompt": p, "fusion": fusion, "rerank": rr, "alpha": a,
                                 **{n: float(A[:, c].mean()) for c, n in enumerate(NAMES)}})
        print(f"[select] {p}: dev cat_acc(VLM)={acc:.3f}; best nDCG@10 text-only="
              f"{max(g['ndcg10'] for g in grid if g['prompt']==p and g['alpha']==0.0):.3f}")
    best = max(grid, key=lambda g: g["ndcg10"])
    print("[select] ALES:", best)
    for p in ["P0", "P1", "P2"]:
        rows = [g for g in grid if g["prompt"] == p]
        if rows:
            b = max(rows, key=lambda g: g["ndcg10"])
            print(f"   best {p}: {b['fusion']} rr={b['rerank']} α={b['alpha']} nDCG={b['ndcg10']:.3f} Cat={b['cat_p5']:.3f} Fine={b['fine_p5']:.3f}")
    json.dump({"chosen": best, "grid": grid, "criterion": "dev mean nDCG@10 (argmax)"},
              open(OUT / "selection_dev.json", "w"), indent=2)
    print(best["prompt"])


def cmd_test():
    sel = json.load(open(OUT / "selection_dev.json"))["chosen"]
    p, fusion, rr, a = sel["prompt"], sel["fusion"], sel["rerank"], sel["alpha"]
    D = load("test", p); D0 = load("test", "P0")
    res, perq = {}, {}
    for nodup in [False, True]:
        sfx = "_nodup" if nodup else ""
        cfg = {
            "random": random_rank(D, nodup),
            "metadata_filter": metadata_filter(D, nodup),
            "text_only_P0": run(D0, 0.0, "raw", False, nodup),
            f"text_only_{p}": run(D, 0.0, fusion, False, nodup),
            "image_only": run(D, 1.0, "raw", False, nodup),
            "image_only_rerank": run(D, 1.0, "raw", True, nodup),
            "image_knn_heads": knn_heads_run(D, nodup),
            # FINAL: configurația aleasă pe DEV (nDCG@10) + re-rank cu capete kNN (selecție a doua pe dev)
            "final": knn_heads_run(D, nodup, alpha=a, fusion=fusion),
            "selected": run(D, a, fusion, rr, nodup),
            "selected_no_rerank" if rr else "selected_rerank": run(D, a, fusion, not rr, nodup),
        }
        for k, A in cfg.items():
            res[k + sfx] = summ(A); perq[k + sfx] = A
    def paired(x, y):
        return {n: boot(perq[x][:, c] - perq[y][:, c]) for c, n in enumerate(NAMES)}
    diffs = {}
    for sfx in ["", "_nodup"]:
        for x, y in [("selected", "image_only"), ("selected", "metadata_filter"), ("selected", "text_only_P0"),
                     (f"text_only_{p}", "text_only_P0"), ("image_only", "metadata_filter"),
                     ("image_knn_heads", "image_only"), ("image_knn_heads", "metadata_filter"),
                     ("image_knn_heads", "selected"), ("final", "image_only"), ("final", "metadata_filter"),
                     ("final", "image_knn_heads"), ("final", "selected"), ("final", "text_only_P0")]:
            diffs[f"{x}{sfx} - {y}{sfx}"] = paired(x + sfx, y + sfx)
    vlm = {"cat_acc": boot(np.array([a_ == sc[int(i)] for a_, i in zip(D["pc"], D["idx"])], float)),
           "style_acc": boot(np.array([a_ == st[int(i)] for a_, i in zip(D["ps"], D["idx"])], float)),
           "modern_share": float(np.mean([x == "Modern" for x in D["ps"]])),
           "latency_median_s": float(np.median([r["latency_s"] for r in D["rows"]]))}
    ndup = [int(((SIMI[int(q)] >= DUP) & img_ok).sum() - 1) for q in D["idx"]]
    out = {"selection": sel, "vlm_test": vlm, "retrieval": res, "paired": diffs,
           "dups_per_query_mean": float(np.mean(ndup)), "queries_with_dups": int(np.sum(np.array(ndup) > 0))}
    json.dump(out, open(OUT / "metrics_v3.json", "w"), indent=2)
    np.savez(OUT / "per_query_v3.npz", **perq)
    print(json.dumps(sel)); print(json.dumps(vlm))
    print(f"duplicate/query (cos≥{DUP}): medie {out['dups_per_query_mean']:.2f}, interogări cu duplicate: {out['queries_with_dups']}")
    f = lambda v: f"{v[0]:.3f}"
    print(f"{'method':32s} " + " ".join(f"{n:>9s}" for n in NAMES))
    for k, v in res.items():
        print(f"{k:32s} " + " ".join(f"{f(v[n]):>9s}" for n in NAMES))
    for k, v in diffs.items():
        print(f"{k:45s} " + " ".join(f"{v[n][0]:+.3f}[{v[n][1]:+.2f},{v[n][2]:+.2f}]" for n in ["cat_p5", "style_p5", "fine_p5", "ndcg10"]))


if __name__ == "__main__":
    {"select": cmd_select, "test": cmd_test}[sys.argv[1]]()
