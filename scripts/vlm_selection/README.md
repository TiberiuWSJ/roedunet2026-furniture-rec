# VLM selection study (Table II)

Closed-set classification of interior style from a single photograph on a balanced subset of MMIS (220 images, 20 per style, 11 styles), used to choose the VLM for the online query.

| File | Role |
|---|---|
| `download_mmis_subset.py` | Downloads the first archive of each MMIS style and extracts a few images per style. |
| `build_mmis_manifest.py` | Builds `mmis_eval/manifest.jsonl` (image, ground-truth style) and `mmis_eval/styles.json`. |
| `config.py` | Model list (LM Studio identifiers) and run parameters. |
| `lmstudio_client.py` | OpenAI-compatible client; measures wall-clock latency per request. |
| `run_style_eval.py` | Runs every model on the manifest; writes `results_style/`. |
| `results_style/` | Raw answers, confusion matrices and `summary.csv` (accuracy, macro-F1, latency). |

```bash
pip install -r requirements.txt
python run_style_eval.py --manifest ./mmis_eval/manifest.jsonl --styles ./mmis_eval/styles.json
```

Local models were served by LM Studio (4-bit) on an Apple M1 Pro with 32 GB; Qwen3.5-9B and Qwen3-VL-8B-Thinking were queried through OpenRouter with reasoning enabled, so their latencies are not comparable with the local runs. Llama 3.2 11B Vision failed at the backend level and is not in `config.py`.

The remaining files (`build_eval_set.py`, `build_eval_set_photos.py`, `render_3dfront.py`, `run_eval.py`, `metrics.py`, `analyze_results.py`, `clip_index.py`, `build_index_fast.py`, `prompts.py`, `taxonomy.py`) belong to a broader evaluation design based on 3D-FRONT renders that was not used for the results in the paper.

In these files, Tier A denotes the 3D-FRONT render set and Tier B qualitative checks on public 3DGS scenes; D1–D6 are the metrics of that design (D1 JSON validity, D2 room-style accuracy, D3 category accuracy, D4 retrieval quality through the CLIP index, D5 latency, D6 determinism).
