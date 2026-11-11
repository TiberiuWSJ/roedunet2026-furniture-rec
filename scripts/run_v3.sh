#!/bin/bash
# Development-set selection (prompt, fusion, alpha, re-ranking), then a single run on the test set.
# Run from the repository root. Requires LM Studio serving qwen/qwen2.5-vl-7b on localhost:1234
# and a catalogue built with scripts 01-04 under ./catalog.
set -euo pipefail
cd "$(dirname "$0")/.."
T0=$(date +%s); stamp() { echo "[$(($(date +%s)-T0)) s] $*"; }

for P in P0 P1 P2; do
  stamp "dev queries, prompt $P (102 objects)"
  python scripts/run_paper_eval.py queries --split dev --prompt $P
done

stamp "selection on the development set (mean nDCG@10)"
python scripts/eval_v3.py select | tee results/select.txt
CHOSEN=$(tail -1 results/select.txt)
stamp "selected prompt: $CHOSEN"

if [ "$CHOSEN" != "P0" ]; then
  stamp "test queries, prompt $CHOSEN (300 objects)"
  python scripts/run_paper_eval.py queries --split test --prompt $CHOSEN
fi
stamp "qualitative examples"
python scripts/run_paper_eval.py qualitative --prompt $CHOSEN

stamp "test evaluation"
python scripts/eval_v3.py test
python scripts/extra_controls.py
stamp "done: results/metrics_v3.json, results/extra_controls.json"
