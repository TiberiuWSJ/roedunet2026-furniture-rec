# Style-aware furniture recommendation for AR replacement (RoEduNet 2026)

Code and data for the paper *Catalogue-Scale, Style-Aware Furniture Recommendation for AR Replacement Using Vision-Language Models and Dual-Channel CLIP Retrieval* (D. Toderici, T.-G. Rodanciuc, G.-A. Micu, L. Dunai, RoEduNet 2026).

## Contents

| Path | Content |
|---|---|
| `data/catalogue.jsonl` | The 2,400 catalogue items: 3D-FUTURE model ID (`uid`), super-category, fine category, style, material and the caption used in the text index. |
| `data/splits.json` | Model IDs of the 300 test queries (50 per super-category, seed 0) and of the 102 development queries (17 per super-category, seed 1, disjoint from test). |
| `catalog/index/` | CLIP ViT-L/14 (OpenCLIP, `openai` weights) caption and thumbnail embeddings of the catalogue. |
| `results/queries_vlm_*.jsonl`, `results/queries_emb_*.npz` | VLM outputs and CLIP query embeddings for every query (test: prompts P0 and P2; development: P0, P1, P2). Files without a suffix are test/P0. |
| `results/selection_dev.json` | Development-set sweep used to fix the configuration. |
| `results/knn_heads.json` | Cross-validation of the kNN heads (k, lambda). |
| `results/metrics_v3.json`, `results/per_query_v3.npz` | Test metrics (Table III), paired differences and per-query scores. |
| `results/extra_controls.json` | Metadata-filter baselines (VLM labels from prompt P0, kNN labels), self-query rows of Table III and the style-label audit (Section V-B). |
| `results/qualitative_top5_v3.json` | Top-5 results shown in Fig. 3. |
| `scripts/01_*.py` to `06_*.py` | Catalogue filtering, rendering, captioning, index construction, inference and index checks. |
| `scripts/run_paper_eval.py`, `eval_v3.py`, `learned_heads.py`, `extra_controls.py`, `run_v3.sh` | Evaluation protocol. |
| `scripts/fix_thumbnail_roll.py` | In-plane rotation correction of renders made with a +Z up-vector (the meshes are Y-up). |
| `scripts/vlm_selection/` | VLM style-classification study on MMIS (Table II). |

## Reproducing the tables

Table III and the controls can be recomputed from the files in this repository, without the 3D-FUTURE models:

```bash
pip install numpy
python scripts/eval_v3.py test        # results/metrics_v3.json
python scripts/extra_controls.py      # results/extra_controls.json
```

Rebuilding the catalogue and re-running the VLM queries requires 3D-FUTURE, the packages in `scripts/requirements.txt` and LM Studio:

```bash
python scripts/01_filter_catalog.py --help    # then 02, 03, 04 in order
bash scripts/run_v3.sh
```

## Not included

- 3D-FUTURE models and renders: obtain the dataset from its official source; the model IDs above identify the catalogue. Category, style and material labels come from the dataset's `model_info.json`.
- MMIS images: obtain MMIS from its authors; `scripts/vlm_selection/mmis_eval/` lists the images used.
- API keys: captioning used an OpenAI-compatible API; pass your own key on the command line.
