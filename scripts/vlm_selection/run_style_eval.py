"""
run_style_eval.py
Evaluare simplă: care VLM clasifică cel mai bine stilul unei camere dintr-o imagine.
Substrat: manifest MMIS (build_mmis_manifest.py). Fără click, fără CLIP, fără randare.

Pentru fiecare model × imagine:
  - trimite doar imaginea + un prompt închis (lista de stiluri MMIS);
  - primește {"style": "..."}, normalizează la cel mai apropiat label;
  - compară cu gt_style; măsoară latența.

Rezultate:
  results_style/raw_<model>.jsonl
  results_style/summary.csv         (acuratețe, macro-F1, latență/model)
  results_style/confusion_<model>.json

Rulare:
    # 1) LM Studio pornit (server :1234), modele încărcate, ID-uri reale în config.MODELS
    python run_style_eval.py --manifest ./mmis_eval/manifest.jsonl \
        --styles ./mmis_eval/styles.json
    python run_style_eval.py --dry_run     # doar estimare timp
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import config
from lmstudio_client import LMStudioClient, _extract_json


def build_style_prompt(styles: list[str]) -> str:
    lst = ", ".join(styles)
    return (
        "You are shown a photograph of an interior room. "
        "Classify its INTERIOR-DESIGN STYLE.\n"
        f"Choose exactly one style from this list: {lst}.\n"
        'Respond in JSON only: {"style": "<one style from the list>"}'
    )


def normalize_pred(pred: str, styles: list[str]) -> str | None:
    """Potrivește predicția la lista de stiluri (exact → case-insensitive → substring)."""
    if not pred:
        return None
    p = pred.strip()
    by_lower = {s.lower(): s for s in styles}
    if p in styles:
        return p
    if p.lower() in by_lower:
        return by_lower[p.lower()]
    # substring în ambele sensuri (ex. "Zen" ~ "Japanese Zen")
    cand = [s for s in styles if s.lower() in p.lower() or p.lower() in s.lower()]
    return cand[0] if len(cand) == 1 else (cand[0] if cand else None)


def macro_f1(pairs, labels):
    f1s = []
    for lab in labels:
        tp = sum(1 for pr, gt in pairs if pr == lab and gt == lab)
        fp = sum(1 for pr, gt in pairs if pr == lab and gt != lab)
        fn = sum(1 for pr, gt in pairs if pr != lab and gt == lab)
        if tp == 0 and (fp == 0 and fn == 0):
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def build_summary(out: Path, styles: list[str], params_by_label: dict) -> list[dict]:
    """Construiește rândurile de summary din TOATE fișierele raw_*.jsonl din `out`."""
    import glob
    rows = []
    for rp in glob.glob(str(out / "raw_*.jsonl")):
        recs = [json.loads(l) for l in open(rp, encoding="utf-8") if l.strip()]
        if not recs:
            continue
        label = Path(rp).stem[len("raw_"):]
        pairs = [(r.get("pred"), r.get("gt_style")) for r in recs]
        valid = [(p, g) for p, g in pairs if p is not None]
        lats = [r["latency_s"] for r in recs if r.get("latency_s") is not None]
        tps = [r["tokens_per_s"] for r in recs if r.get("tokens_per_s")]
        acc = sum(1 for p, g in valid if p == g) / len(valid) if valid else float("nan")
        rows.append({
            "model": label, "params": params_by_label.get(label, ""), "n": len(pairs),
            "parse_rate": round(len(valid) / len(pairs), 3) if pairs else 0.0,
            "accuracy": round(acc, 3), "macro_f1": round(macro_f1(valid, styles), 3),
            "lat_mean_s": round(float(np.mean(lats)), 2) if lats else "",
            "lat_median_s": round(float(np.median(lats)), 2) if lats else "",
            "lat_p90_s": round(float(np.percentile(lats, 90)), 2) if lats else "",
            "tok_per_s_median": round(float(np.median(tps)), 1) if tps else "",
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="./mmis_eval/manifest.jsonl")
    ap.add_argument("--styles", default="./mmis_eval/styles.json")
    ap.add_argument("--models", nargs="*", default=None, help="Etichete din config.MODELS")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=40,
                    help="Plafon tokeni generați (crește pentru modele care nu închid JSON-ul)")
    ap.add_argument("--base_url", default=config.LM_STUDIO_BASE_URL,
                    help="Endpoint API (ex. https://openrouter.ai/api/v1 pentru cloud)")
    ap.add_argument("--api_key", default="lm-studio",
                    help="Cheie API (pentru OpenRouter: cheie de forma sk-or-...)")
    ap.add_argument("--override_id", default="",
                    help="Înlocuiește ID-ul modelului (util când rulezi UN model pe alt provider)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Apeluri paralele (folosește 8-16 doar pe cloud/OpenRouter; "
                         "pe LM Studio local ține 1, altfel latențele devin nereale)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    manifest = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    if args.limit:
        manifest = manifest[:args.limit]
    styles = json.load(open(args.styles, encoding="utf-8"))
    prompt = build_style_prompt(styles)

    models = config.MODELS
    if args.models:
        models = [m for m in models if m["label"] in args.models]
    if args.override_id:
        if len(models) != 1:
            raise SystemExit("--override_id necesită exact un model în --models")
        models = [{**models[0], "id": args.override_id}]
        print(f"  ID suprascris → {args.override_id}")

    total = len(manifest) * len(models)
    print(f"Imagini: {len(manifest)} | Stiluri: {len(styles)} | Modele: {len(models)}")
    print(f"Total inferențe: {total} | @8s ≈ {total*8/60:.0f} min | @12s ≈ {total*12/60:.0f} min")
    if args.dry_run:
        return

    client = LMStudioClient(base_url=args.base_url, api_key=args.api_key)
    try:
        avail = set(client.list_models())
        for m in models:
            if m["id"] not in avail:
                print(f"  [!] '{m['id']}' nu e încărcat în LM Studio")
    except Exception as e:
        print(f"  [!] nu pot lista modelele: {e}")

    out = Path("./results_style"); out.mkdir(exist_ok=True)
    params_by_label = {m["label"]: m.get("params", "") for m in config.MODELS}

    for m in models:
        print(f"\n=== {m['label']} ({m['id']}) ===")
        pairs, lats, tps = [], [], []
        raw_path = out / f"raw_{m['label']}.jsonl"

        # warm-up: declanșează încărcarea modelului (JIT) fără a cronometra,
        # ca latențele măsurate să reflecte inferența în regim stabil, nu load-ul.
        try:
            warm = Image.open(manifest[0]["image_path"]).convert("RGB")
            print("  warm-up (încărcare model, necronometrat) ...")
            client.analyze_scene(model=m["id"], prompt=prompt, full_frame=warm,
                                 crop=None, max_tokens=args.max_tokens, temperature=0.0)
        except Exception as e:
            print(f"  [!] warm-up eșuat: {e}")

        def process(inst):
            img = Image.open(inst["image_path"]).convert("RGB")
            resp = client.analyze_scene(model=m["id"], prompt=prompt,
                                        full_frame=img, crop=None,
                                        max_tokens=args.max_tokens, temperature=0.0)
            parsed = resp.parsed or _extract_json(resp.raw_text) or {}
            pred_raw = parsed.get("style") if isinstance(parsed, dict) else None
            pred = normalize_pred(pred_raw or resp.raw_text, styles)
            return inst, resp, pred_raw, pred

        results = []
        done = 0
        errors = 0
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = [ex.submit(process, inst) for inst in manifest]
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    with lock:
                        errors += 1
                    print(f"  [!] eroare la o imagine: {e}")
                with lock:
                    done += 1
                    if done % 20 == 0:
                        print(f"  {done}/{len(manifest)}")
        if errors:
            print(f"  [!] {errors} imagini eșuate (omise)")

        with open(raw_path, "w", encoding="utf-8") as fo:
            for inst, resp, pred_raw, pred in results:
                pairs.append((pred, inst["gt_style"]))
                lats.append(resp.latency_s)
                if resp.tokens_per_s:
                    tps.append(resp.tokens_per_s)
                fo.write(json.dumps({**inst, "model": m["label"],
                                     "pred_raw": pred_raw, "pred": pred,
                                     "json_valid": resp.json_valid,
                                     "latency_s": resp.latency_s,
                                     "completion_tokens": resp.completion_tokens,
                                     "tokens_per_s": resp.tokens_per_s},
                                    ensure_ascii=False) + "\n")

        valid = [(p, g) for p, g in pairs if p is not None]
        acc = sum(1 for p, g in valid if p == g) / len(valid) if valid else float("nan")
        f1 = macro_f1(valid, styles)

        # confuzii
        cm = defaultdict(lambda: defaultdict(int))
        for p, g in pairs:
            cm[g][p or "PARSE_FAIL"] += 1
        json.dump({g: dict(d) for g, d in cm.items()},
                  open(out / f"confusion_{m['label']}.json", "w"),
                  ensure_ascii=False, indent=2)
        print(f"  acc={acc:.3f}  macroF1={f1:.3f}  "
              f"lat_med={np.median(lats):.2f}s  parse={len(valid)/len(pairs):.2f}")

    # summary CSV — se reconstruiește din TOATE fișierele raw_*.jsonl existente,
    # ca rerularea unui subset de modele să nu piardă rezultatele celorlalte.
    summary_rows = build_summary(out, styles, params_by_label)
    summary_rows.sort(key=lambda r: (r["accuracy"] if r["accuracy"] == r["accuracy"] else -1),
                      reverse=True)
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f"\n{out}/summary.csv")
    print("Recomandare (acc brută):", summary_rows[0]["model"],
          f"(acc={summary_rows[0]['accuracy']}, lat_med={summary_rows[0]['lat_median_s']}s)")


if __name__ == "__main__":
    main()
