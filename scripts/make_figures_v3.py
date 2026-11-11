"""make_figures_v3.py — figurile finale (configurația aleasă pe dev). numpy + matplotlib."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_v3 as E

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})
OUT = E.OUT
sel = json.load(open(OUT / "selection_dev.json"))["chosen"]
P, FUS, A_SEL = sel["prompt"], sel["fusion"], sel["alpha"]
ABBR = {"European Traditional": "Eur. Trad.", "Chinese Traditional": "Chin. Trad.", "Southeast Asian": "SE Asian"}
short = lambda x: ABBR.get(x, x)


def fig_alpha():
    D = E.load("test", P); mf = E.metadata_filter(D).mean(0)
    rows = {"plain": [], "knn": []}
    for a in E.ALPHAS:
        rows["plain"].append(E.run(D, a, FUS, False).mean(0))
        rows["knn"].append(E.knn_heads_run(D, alpha=a, fusion=FUS).mean(0))
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    for col, lbl, c in [(4, "nDCG@10", "#1f4e79"), (0, "Cat-P@5", "#6a3d9a"), (1, "Style-P@5", "#b85c00")]:
        ax.plot(E.ALPHAS, [r[col] for r in rows["plain"]], "-o", ms=2.5, lw=1, color=c, label=lbl)
        ax.plot(E.ALPHAS, [r[col] for r in rows["knn"]], "--s", ms=2.5, lw=1, color=c, label=lbl + " + kNN re-rank")
        ax.axhline(mf[col], color=c, lw=0.6, ls=":")
    ax.axvline(A_SEL, color="gray", lw=0.6, ls="-.")
    ax.set_xlabel(r"visual-channel weight $\alpha$ (z-score fusion)"); ax.set_ylabel("score")
    ax.set_ylim(0.15, 0.9); ax.grid(alpha=0.25, lw=0.4)
    ax.legend(fontsize=5.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False, columnspacing=0.8, handlelength=2.2)
    fig.savefig(OUT / "fig_alpha_v3.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)


def fig_qual():
    Q = json.load(open(OUT / f"qualitative_vlm_{P}.json")); z = np.load(OUT / f"qualitative_emb_{P}.npz")
    STY = sorted(set(E.st) - {"Unspecified"}); CATS = sorted(set(E.sc))
    tr = np.where(E.img_ok)[0]; trs = tr[E.st[tr] != "Unspecified"]
    fig, axes = plt.subplots(len(Q), 6, figsize=(7.1, 1.3 * len(Q)), gridspec_kw={"wspace": 0.04, "hspace": 0.42})
    log = []
    for r, q in enumerate(Q):
        p = q["parsed"] or {}; qt, qi = z["q_text"][r], z["q_img"][r]
        m = E.img_ok.copy()
        oc = np.argsort(-(E.I[tr] @ qi))[:15]; os_ = np.argsort(-(E.I[trs] @ qi))[:15]
        pc = CATS[np.bincount([CATS.index(x) for x in E.sc[tr][oc]], minlength=6).argmax()]
        ps = np.bincount([STY.index(x) for x in E.st[trs][os_]], minlength=len(STY)) / 15
        s = (1 - A_SEL) * E.zs(E.T @ qt, m) + A_SEL * E.zs(E.I @ qi, m) if FUS == "z" else (1 - A_SEL) * (E.T @ qt) + A_SEL * (E.I @ qi)
        s = s.astype(float); s[~m] = -np.inf; fin = np.isfinite(s)
        s[fin] = (s[fin] - s[fin].min()) / (s[fin].max() - s[fin].min()); s[E.sc != pc] = -np.inf
        s = s - 0.5 * (1 - np.array([ps[STY.index(x)] if x in STY else 0 for x in E.st]))
        idx = np.argsort(-s)[:5]
        fr = Image.open(E.ROOT / "scripts" / "vlm_selection" / q["path"]).convert("RGB")
        ax = axes[r, 0]; ax.imshow(fr); x, y = q["click"]; mg = q["crop_margin"]
        ax.add_patch(plt.Rectangle((x - mg, y - mg), 2 * mg, 2 * mg, fill=False, ec="#00e5ff", lw=0.8))
        ax.plot(x, y, "o", ms=4, mfc="none", mec="red", mew=1.2)
        ax.set_title(f"VLM: {short(p.get('room_style','?'))}\nkNN: {pc}", fontsize=5.8, pad=2, linespacing=0.9)
        for c, i in enumerate(idx):
            e = E.cat[i]; axes[r, c + 1].imshow(Image.open(E.ROOT / e["thumbnail_path"]).convert("RGB"))
            axes[r, c + 1].set_title(short(e["style"]), fontsize=6, pad=2)
        log.append({"scene": q["path"], "vlm": p, "knn_cat": pc, "top5": [(E.cat[i]["style"], E.cat[i]["category"]) for i in idx]})
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); [sp.set_linewidth(0.3) for sp in ax.spines.values()]
    fig.savefig(OUT / "fig_qualitative_v3.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    json.dump(log, open(OUT / "qualitative_top5_v3.json", "w"), indent=2)
    for l in log: print(l["vlm"].get("room_style"), l["vlm"].get("selected_category"), l["knn_cat"], l["top5"])


fig_alpha(); print("ok")
