"""
learned_heads.py — capete kNN (vot ponderat pe vecinii cosinus) pe embedding-urile CLIP de imagine pentru
categorie și stil, antrenate doar pe catalog (fără cele 300 de obiecte-query), folosite
la re-ranking în locul etichetelor prezise de VLM. numpy only.

Protocol (fără contaminarea setului de test):
  * train = catalog \ {query objects}  (2 100 obiecte, embedding view_0)
  * hiperparametri (L2 pentru capete; λ pentru penalizarea de stil) aleși prin
    validare încrucișată 5-fold pe train: interogarea = view_0 al obiectului ținut deoparte,
    indexul = restul catalogului, capete antrenate pe celelalte fold-uri.
  * evaluare o singură dată pe cele 300 de interogări view_1 (aceleași ca în eval_v3.py).
Scor re-rank:  s_i = minmax(cos(q_I, e_I_i)) − λ·(1 − P̂(stil_i | q))  ; filtru dur pe
categoria argmax P̂(cat | q).
"""
import json, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CAT = ROOT / os.environ.get("PAPER_CATALOG", "catalog")
OUT = ROOT / "results"
K = 5
rng0 = np.random.default_rng(0)


def knn_fit(X, y, n_cls, k):
    return (X, y, n_cls, k)


def knn_pred(model, Q):
    X, y, n_cls, k = model
    S = Q @ X.T; o = np.argsort(-S, 1)[:, :k]
    P = np.zeros((len(Q), n_cls))
    for r in range(len(Q)):
        np.add.at(P[r], y[o[r]], 1.0)
    return P / k


def softmax_fit(X, y, n_cls, l2, iters=300, lr=0.5):
    """Regresie logistică multinomială, gradient descent complet (X normalizat L2)."""
    n, d = X.shape
    W = np.zeros((d, n_cls)); b = np.zeros(n_cls)
    Y = np.eye(n_cls)[y]
    Xs = X * 20.0   # scalare: cosinusuri CLIP au varianță mică
    for _ in range(iters):
        Z = Xs @ W + b; Z -= Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = (P - Y) / n
        W -= lr * (Xs.T @ G + l2 * W); b -= lr * G.sum(0)
    return W, b


def softmax_pred(model, X):
    W, b = model; Z = (X * 20.0) @ W + b; Z -= Z.max(1, keepdims=True); P = np.exp(Z)
    return P / P.sum(1, keepdims=True)


def prec(order, q, sc, st):
    o = order[:K]
    return (np.mean(sc[o] == sc[q]), np.mean(st[o] == st[q]), np.mean((sc[o] == sc[q]) & (st[o] == st[q])))


def rank(qvec, I, pc, ps, cats, styles, st_idx, lam, exclude):
    s = I @ qvec
    fin = np.ones(len(s), bool); fin[exclude] = False
    s = (s - s[fin].min()) / max(s[fin].max() - s[fin].min(), 1e-8)
    s[~fin] = -np.inf
    s[cats != pc] = -np.inf
    s = s - lam * (1 - ps[st_idx])
    return np.argsort(-s)


def boot(v, n=2000):
    v = np.asarray(v, float); m = np.random.default_rng(0).choice(v, (n, len(v))).mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    cat = [json.loads(l) for l in open(CAT / "index/catalog_index.jsonl")]
    I = np.load(CAT / "index/image_embeddings.npy").astype(np.float32)
    sc = np.array([e["super_category"] for e in cat]); st = np.array([e["style"] for e in cat])
    CATS = sorted(set(sc)); STY = sorted(set(st) - {"Unspecified"})
    ci = np.array([CATS.index(x) for x in sc]); si = np.array([STY.index(x) if x in STY else -1 for x in st])
    z = np.load(OUT / "queries_emb.npz"); qidx = z["idx"]; QI = z["q_img"]
    ok = np.linalg.norm(I, axis=1) > 1e-8
    train = np.array([i for i in range(len(cat)) if i not in set(qidx.tolist()) and ok[i]])
    tr_st = train[si[train] >= 0]

    # CV 5-fold pe train
    folds = np.array_split(rng0.permutation(train), 5)
    L2S = [5, 15, 30, 50]; LAMS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]   # aici: k pentru kNN
    cv = {}
    for l2 in L2S:
        accs_c, accs_s, J = [], [], {lam: [] for lam in LAMS}
        for f in folds:
            tr = np.setdiff1d(train, f); trs = tr[si[tr] >= 0]
            mc = knn_fit(I[tr], ci[tr], len(CATS), l2); ms = knn_fit(I[trs], si[trs], len(STY), l2)
            Pc = knn_pred(mc, I[f]); Ps = knn_pred(ms, I[f])
            accs_c.append(np.mean(Pc.argmax(1) == ci[f]))
            fs = f[si[f] >= 0]; accs_s.append(np.mean(knn_pred(ms, I[fs]).argmax(1) == si[fs]))
            sub = f[:80]   # retrieval CV pe un subeșantion per fold (cost)
            for j, q in enumerate(sub):
                pc = CATS[Pc[j].argmax()]
                for lam in LAMS:
                    o = rank(I[q], I, pc, Ps[j], sc, STY, np.clip(si, 0, None), lam, [q] + [i for i in range(len(cat)) if not ok[i]])
                    J[lam].append(prec(o, q, sc, st)[2])
        cv[l2] = {"cat_acc": float(np.mean(accs_c)), "style_acc": float(np.mean(accs_s)),
                  "joint_by_lambda": {str(k): float(np.mean(v)) for k, v in J.items()}}
        print(f"k={l2:g}: CV cat_acc={cv[l2]['cat_acc']:.3f} style_acc={cv[l2]['style_acc']:.3f} "
              + " ".join(f"λ{k}={v:.3f}" for k, v in cv[l2]["joint_by_lambda"].items()))
    best_l2 = max(L2S, key=lambda l: cv[l]["style_acc"] + cv[l]["cat_acc"])
    best_lam = float(max(cv[best_l2]["joint_by_lambda"].items(), key=lambda kv: kv[1])[0])
    print(f"→ ales pe CV: k={best_l2}, λ={best_lam}")

    # antrenare finală pe tot train, evaluare O DATĂ pe test
    mc = knn_fit(I[train], ci[train], len(CATS), best_l2)
    ms = knn_fit(I[tr_st], si[tr_st], len(STY), best_l2)
    Pc = knn_pred(mc, QI); Ps = knn_pred(ms, QI)
    pc = [CATS[k] for k in Pc.argmax(1)]; ps_lab = [STY[k] for k in Ps.argmax(1)]
    cat_acc = boot([a == b for a, b in zip(pc, sc[qidx])]); sty_acc = boot([a == b for a, b in zip(ps_lab, st[qidx])])
    bad = [i for i in range(len(cat)) if not ok[i]]
    res = {}
    for lam in sorted({0.0, best_lam}):
        trip = [prec(rank(QI[j], I, pc[j], Ps[j], sc, STY, np.clip(si, 0, None), lam, [int(q)] + bad), int(q), sc, st)
                for j, q in enumerate(qidx)]
        a = np.array(trip)
        res[f"image_heads_lambda{lam}"] = {"cat_p5": boot(a[:, 0]), "style_p5": boot(a[:, 1]), "joint_p5": boot(a[:, 2])}
        np.save(OUT / f"perq_knnheads_lambda{lam}.npy", a)
    out = {"method": "kNN heads", "cv": {str(k): v for k, v in cv.items()}, "chosen": {"k": best_l2, "lambda": best_lam},
           "test_head_cat_acc": cat_acc, "test_head_style_acc": sty_acc,
           "test_pred_style_distribution": {s: int(sum(p == s for p in ps_lab)) for s in STY},
           "retrieval": res}
    json.dump(out, open(OUT / "knn_heads.json", "w"), indent=2)
    print("TEST cat_acc", cat_acc, "style_acc", sty_acc)
    for k, v in res.items():
        print(k, {m: [round(x, 3) for x in v[m]] for m in v})


if __name__ == "__main__":
    main()
