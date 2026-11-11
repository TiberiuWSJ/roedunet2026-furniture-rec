"""
extra_controls.py — controls reported in the paper that are not produced by eval_v3.py:
  * metadata filter with VLM labels from prompt P0 and with catalogue kNN labels (Table III);
  * self-query setting: caption->caption and thumbnail->thumbnail (Table III, lower block);
  * paired differences of Dual + kNN re-rank against the kNN-label filter;
  * style-label noise audit on near-identical catalogue pairs (Section V-B).

  python scripts/extra_controls.py
"""
import json, math
import numpy as np
import eval_v3 as E

def knn_labels(D, k=15):
    STY = sorted(set(E.st) - {"Unspecified"}); CATS = sorted(set(E.sc))
    excl = set(int(i) for i in D["idx"])
    tr = np.array([i for i in range(E.N) if i not in excl and E.img_ok[i]]); trs = tr[E.st[tr] != "Unspecified"]
    pc, ps = [], []
    for j in range(len(D["idx"])):
        oc = np.argsort(-(E.I[tr] @ D["QI"][j]))[:k]; os_ = np.argsort(-(E.I[trs] @ D["QI"][j]))[:k]
        pc.append(CATS[np.bincount([CATS.index(x) for x in E.sc[tr][oc]], minlength=len(CATS)).argmax()])
        ps.append(STY[np.bincount([STY.index(x) for x in E.st[trs][os_]], minlength=len(STY)).argmax()])
    return pc, ps

def self_query(D, emb):
    out = []
    for q in D["idx"]:
        q = int(q); m = E.valid_mask(q, False); s = (emb @ emb[q]).astype(np.float64); s[~m] = -np.inf
        out.append(E.metrics(np.argsort(-s), q, m))
    return np.array(out)

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d

NAMES = ["cat_p5", "style_p5", "joint_p5", "fine_p5", "ndcg10"]
P0 = E.load("test", "P0"); P2 = E.load("test", "P2")
res, perq = {}, {}
perq["metadata_filter_vlm_P0"] = E.metadata_filter(P0)
pc, ps = knn_labels(P2); Dk = dict(P2, pc=pc, ps=ps)
perq["metadata_filter_knn"] = E.metadata_filter(Dk)
perq["self_caption_to_caption"] = self_query(P2, E.T)
perq["self_thumbnail_to_thumbnail"] = self_query(P2, E.I)
perq["final"] = E.knn_heads_run(P2, k=15, lam=0.5, alpha=0.8, fusion="z")
for k, v in perq.items():
    res[k] = {n: E.boot(v[:, c]) for c, n in enumerate(NAMES)}
def paired(a, b):
    return {n: E.boot(perq[a][:, c] - perq[b][:, c]) for c, n in enumerate(NAMES)}
res["paired"] = {"final - metadata_filter_knn": paired("final", "metadata_filter_knn"),
                 "final - metadata_filter_vlm_P0": paired("final", "metadata_filter_vlm_P0")}
# style-label noise audit over all catalogue pairs
iu = np.triu_indices(E.N, 1); ok = E.img_ok[iu[0]] & E.img_ok[iu[1]]; sim = E.SIMI[iu][ok]; a, b = iu[0][ok], iu[1][ok]
audit = {}
for th in [0.99, 0.97]:
    sel = sim >= th; agree = int((E.st[a][sel] == E.st[b][sel]).sum()); n = int(sel.sum())
    audit[f">={th}"] = {"pairs": n, "style_agree": agree, "rate": agree / n, "wilson95": wilson(agree, n),
                        "supercat_agree": float((E.sc[a][sel] == E.sc[b][sel]).mean())}
rng = np.random.default_rng(0); ra, rb = rng.integers(0, E.N, 100000), rng.integers(0, E.N, 100000)
audit["random_pairs_style_agree"] = float((E.st[ra] == E.st[rb])[ra != rb].mean())
res["label_noise_audit"] = audit
json.dump(res, open(E.OUT / "extra_controls.json", "w"), indent=1)
for k in ["metadata_filter_vlm_P0", "metadata_filter_knn", "self_caption_to_caption", "self_thumbnail_to_thumbnail", "final"]:
    print(f"{k:30s}", " ".join(f"{n}={res[k][n][0]:.3f}" for n in NAMES))
for k, v in res["paired"].items():
    print(k, " ".join(f"{n}={v[n][0]:+.3f}[{v[n][1]:+.2f},{v[n][2]:+.2f}]" for n in NAMES))
print(json.dumps(audit, indent=1))
