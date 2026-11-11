"""
config.py
=========
Configurația studiului de evaluare VLM. Editează aici lista de modele și căile.

Identificatorii de model trebuie să corespundă exact cu ID-urile încărcate
în LM Studio (vezi `LMStudioClient.list_models()`). Cei de mai jos sunt nume
tipice; ajustează-i după ce încarci modelele local.
"""

# Endpoint LM Studio (OpenAI-compatibil)
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

# Modele evaluate (rularea principală sub bugetul de ≤1–2h)
# Qwen2.5-VL-32B este exclus din rularea principală (prea mare/lent pe M1 Pro 32GB);
# poate fi re-adăugat ca rulare separată "peste buget".
MODELS = [
    # id_LM_Studio (exact din list_models())   # etichetă raport          # ~param  # familie
    {"id": "qwen/qwen2.5-vl-7b",                "label": "Qwen2.5-VL-7B",  "params": "7B",  "family": "Qwen"},
    {"id": "qwen/qwen3.5-9b",                   "label": "Qwen3.5-9B",     "params": "9B",  "family": "Qwen"},
    {"id": "qwen/qwen3-vl-8b-thinking",         "label": "Qwen3-VL-8B-Think", "params": "8B", "family": "Qwen"},
    {"id": "minicpm-v-4_5",                     "label": "MiniCPM-V-4.5",  "params": "8B",  "family": "MiniCPM"},
    {"id": "google/gemma-3-12b",                "label": "Gemma3-12B",     "params": "12B", "family": "Gemma"},
    {"id": "google/gemma-4-e4b",                "label": "Gemma-E4B",      "params": "~4B", "family": "Gemma"},
]
# Llama3.2-11B-V exclus: nesuportat corect ca vision în llama.cpp/LM Studio (a dat eroare la fiecare imagine).

# Se poate re-adăuga la MODELS ca punct de capacitate (rulează pe un subset):
#   {"id": "qwen2.5-vl-32b-instruct", "label": "Qwen2.5-VL-32B", "params": "32B", "family": "Qwen"},

# Căi (relative la acest folder sau absolute)
EVAL_MANIFEST = "./eval_set_tierA/eval_manifest.jsonl"   # din build_eval_set.py
CLIP_INDEX_DIR = "../catalog/index"                       # din 04_build_index.py
RESULTS_DIR = "./results"

# Parametri experiment
MAX_TOKENS = 300
TEMPERATURE = 0.0          # decodare deterministă
RETRIEVAL_K = 5
INPUT_MODE = "full+crop"   # "full+crop" | "crop_only" | "full_only"  (D6)

# Verificare de varianță (D6): câte instanțe se repetă și de câte ori
VARIANCE_CHECK_N = 10
VARIANCE_CHECK_REPEATS = 3

# Poartă D1 și buget latență pentru regula de decizie
D1_GATE_MIN = 0.98         # rata minimă de JSON valid pentru a rămâne în competiție
LATENCY_BUDGET_P90_S = 25  # buget p90 per inferență (interactivitate AR)
