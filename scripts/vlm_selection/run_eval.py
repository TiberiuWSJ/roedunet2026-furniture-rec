"""
run_eval.py
Runner principal al studiului VLM. Pentru fiecare model × instanță:
  1. apelează VLM-ul (LM Studio) cu cadru + crop + prompt;
  2. parsează JSON-ul (D1);
  3. encodează `replacement_query` cu CLIP și rulează regăsirea (D4);
  4. salvează rezultatele brute per-instanță.

Rezultatele se scriu în RESULTS_DIR/raw_<model_label>.jsonl și pot fi agregate
ulterior cu analyze_results.py.

Estimare buget: înainte de rulare, scriptul afișează numărul total de inferențe
și o estimare a timpului total de rulare.

Rulare:
    python run_eval.py                      # toate modelele din config
    python run_eval.py --models Qwen2.5-VL-7B Gemma3-12B
    python run_eval.py --dry_run            # doar estimare buget, fără apeluri
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

import config
from prompts import SCENE_ANALYSIS_PROMPT, PROMPT_VERSION
from lmstudio_client import LMStudioClient
from clip_index import ClipIndex
from taxonomy import coarse_category, normalize_style


def load_manifest(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_inputs(inst: dict, input_mode: str):
    frame = Image.open(inst["frame_path"]).convert("RGB")
    crop = Image.open(inst["crop_path"]).convert("RGB") if inst.get("crop_path") else None
    if input_mode == "crop_only":
        return crop, None  # trimitem crop-ul ca "full", fără al doilea
    if input_mode == "full_only":
        return frame, None
    return frame, crop  # full+crop


def run_retrieval(idx: ClipIndex, query: str, gt_model_id: str, k: int) -> dict:
    top = idx.retrieve_text(query, top_k=k)
    topk_categories = [coarse_category(e.get("category", "")) for e in top]
    topk_styles = [normalize_style(e.get("style", "")) for e in top]
    return {
        "topk_uids": [e.get("uid") for e in top],
        "topk_categories": topk_categories,
        "topk_styles": topk_styles,
        "gt_rank": idx.rank_of_uid(query, gt_model_id),
        "cos_to_gt": idx.cosine_to_image_emb(query, gt_model_id),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="Etichete de model (din config.MODELS); implicit toate")
    ap.add_argument("--limit", type=int, default=0, help="Limitează nr. de instanțe (debug)")
    ap.add_argument("--dry_run", action="store_true", help="Doar estimare buget")
    ap.add_argument("--no_retrieval", action="store_true",
                    help="Sari peste CLIP (dacă indexul nu e gata)")
    args = ap.parse_args()

    manifest = load_manifest(config.EVAL_MANIFEST)
    if args.limit:
        manifest = manifest[:args.limit]

    models = config.MODELS
    if args.models:
        models = [m for m in models if m["label"] in args.models]

    n_inst = len(manifest)
    n_models = len(models)
    total_calls = n_inst * n_models + config.VARIANCE_CHECK_N * (config.VARIANCE_CHECK_REPEATS - 1)
    print(f"Instanțe: {n_inst} | Modele: {n_models} | Prompt: {PROMPT_VERSION} | mod intrare: {config.INPUT_MODE}")
    print(f"Total inferențe (aprox): {total_calls}")
    print(f"  @10s/inferență ≈ {total_calls*10/60:.0f} min | @15s ≈ {total_calls*15/60:.0f} min")
    if args.dry_run:
        return

    client = LMStudioClient(base_url=config.LM_STUDIO_BASE_URL)
    try:
        available = set(client.list_models())
        for m in models:
            if m["id"] not in available:
                print(f"  [!] model '{m['id']}' nu e încărcat în LM Studio "
                      f"(disponibile: {sorted(available)})")
    except Exception as e:
        print(f"  [!] nu pot lista modelele LM Studio: {e}")

    idx = None
    if not args.no_retrieval:
        try:
            idx = ClipIndex(config.CLIP_INDEX_DIR)
            print(f"  CLIP index: N={len(idx.catalog)}")
        except Exception as e:
            print(f"  [!] index CLIP indisponibil ({e}); rulez fără D4 (--no_retrieval implicit)")
            idx = None

    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    for m in models:
        out_path = results_dir / f"raw_{m['label']}.jsonl"
        print(f"\n=== {m['label']} ({m['id']}) → {out_path} ===")
        t_model = time.perf_counter()
        with open(out_path, "w", encoding="utf-8") as fo:
            for i, inst in enumerate(manifest):
                full, crop = build_inputs(inst, config.INPUT_MODE)
                resp = client.analyze_scene(
                    model=m["id"], prompt=SCENE_ANALYSIS_PROMPT,
                    full_frame=full, crop=crop,
                    max_tokens=config.MAX_TOKENS, temperature=config.TEMPERATURE)

                rec = {**inst,
                       "model_id": m["id"], "model_label": m["label"],
                       "raw_text": resp.raw_text, "parsed": resp.parsed,
                       "json_valid": resp.json_valid, "latency_s": resp.latency_s,
                       "completion_tokens": resp.completion_tokens,
                       "tokens_per_s": resp.tokens_per_s, "error": resp.error,
                       "input_mode": config.INPUT_MODE, "prompt_version": PROMPT_VERSION}

                if idx is not None and resp.json_valid:
                    q = (resp.parsed or {}).get("replacement_query", "")
                    if q:
                        try:
                            rec["retrieval"] = run_retrieval(idx, q, inst["gt_model_id"],
                                                             config.RETRIEVAL_K)
                        except Exception as e:
                            rec["retrieval_error"] = str(e)

                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{n_inst}  (last {resp.latency_s:.1f}s, "
                          f"json_valid={resp.json_valid})")
        print(f"  gata în {(time.perf_counter()-t_model)/60:.1f} min")

    print("\nRulare terminată. Rulează: python analyze_results.py")


if __name__ == "__main__":
    main()
