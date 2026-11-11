"""
03_annotate_captions.py
Generează caption-uri text bogate per obiect folosind un VLM (vision-language model).

Caption-ul este folosit pentru CLIP text embedding în pasul 04.
Structura unui caption:
  "A [style] [category] made of [material]. [Forma/caracteristici vizuale].
   Suitable for [context]. Style: [style_tags]."

Suportă două backend-uri (OpenAI-compatible):
  1. OpenRouter (recomandat): cloud, rapid, ~$0.10-0.20 pentru 2400 obiecte
       python 03_annotate_captions.py \\
           --catalog ./catalog/catalog_rendered.jsonl \\
           --output  ./catalog/catalog_annotated.jsonl \\
           --api_key sk-or-v1-... \\
           --base_url https://openrouter.ai/api/v1 \\
           --model qwen/qwen3.5-9b \\
           --resume

  2. LM Studio (local, offline):
       python 03_annotate_captions.py \\
           --catalog ./catalog/catalog_rendered.jsonl \\
           --output  ./catalog/catalog_annotated.jsonl \\
           --base_url http://localhost:1234/v1 \\
           --resume

  3. Fallback fără VLM (caption determinist din metadata):
       python 03_annotate_captions.py \\
           --catalog ./catalog/catalog_rendered.jsonl \\
           --output  ./catalog/catalog_annotated.jsonl \\
           --no_vlm
"""

import json
import base64
import argparse
import threading
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm


# Prompt VLM
SYSTEM_PROMPT = """You are a furniture description assistant for an interior design recommendation system.
Given an image of a furniture item and its metadata, generate a concise but rich English description
that captures style, shape, materials, color, and functional context.
Return ONLY the description string, no JSON, no extra formatting."""

USER_PROMPT_TEMPLATE = """Furniture metadata:
- Category: {category}
- Style: {style}
- Material: {material}

Describe this furniture item in 2-3 sentences focusing on:
1. Overall aesthetic and style
2. Key visual characteristics (shape, color, proportions)
3. Suitable interior context (e.g., "ideal for a minimalist living room")

Keep it under 80 words."""


# Utilitare

def image_to_base64(img: Image.Image, max_size: int = 512) -> str:
    """Redimensionează și codifică imaginea ca data URL base64 PNG."""
    img = img.convert("RGB")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def fallback_caption(entry: dict) -> str:
    """Caption generat din metadata fără VLM (mod offline, determinist)."""
    style    = entry.get("style", "Modern")
    cat      = entry.get("category", entry.get("super_category", "furniture"))
    material = entry.get("material", "")
    mat_str  = f" crafted in {material}" if material else ""

    style_context = {
        "Nordic":               "a Scandinavian-inspired living space",
        "Modern":               "a contemporary interior",
        "Minimalist":           "a clean, uncluttered room",
        "Industrial":           "an industrial-style loft",
        "Art Deco":             "an elegant Art Deco setting",
        "European Traditional": "a classic European interior",
        "Southeast Asian":      "a warm Southeast Asian-inspired space",
        "Chinese Traditional":  "a traditional Chinese interior",
        "Japanese":             "a serene Japanese-style room",
        "Luxury":               "a high-end luxury interior",
        "Kids":                 "a children's room",
    }.get(style, "a well-designed interior")

    return (
        f"A {style.lower()} style {cat.lower()}{mat_str}. "
        f"Designed with clean proportions and careful attention to detail. "
        f"Well-suited for {style_context}."
    )


# VLM client (OpenAI-compatible: LM Studio sau OpenRouter)

def build_vlm_client(base_url: str, api_key: str = "lm-studio"):
    """
    Construiește un client OpenAI-compatible.
    - LM Studio local: base_url=http://localhost:1234/v1, api_key='lm-studio'
    - OpenRouter:      base_url=https://openrouter.ai/api/v1, api_key='sk-or-v1-...'
    """
    from openai import OpenAI
    extra_headers = {}
    if "openrouter" in base_url:
        extra_headers = {
            "HTTP-Referer": "https://github.com/TiberiuWSJ/roedunet2026-furniture-rec",
            "X-Title": "Furniture Recommendation Pipeline",
        }
    return OpenAI(base_url=base_url, api_key=api_key,
                  default_headers=extra_headers if extra_headers else None)


def generate_caption_lmstudio(
    client,
    model: str,
    entry: dict,
    max_tokens: int = 150,
) -> str:
    """
    Apelează VLM API (OpenRouter sau LM Studio) pentru a genera un caption.
    Trimite thumbnail-ul (dacă modelul suportă imagini) + metadata.

    Note:
    - Qwen3 (qwen3-*): text-only reasoning model. Thinking mode dezactivat explicit.
    - Qwen2.5-VL (qwen2.5-vl-*): suportă imagini.
    - LM Studio: suportă imagini dacă modelul e multimodal.
    """
    category  = entry.get("category", entry.get("super_category", "furniture"))
    style     = entry.get("style", "")
    material  = entry.get("material", "")
    thumbnail = entry.get("thumbnail_path", "")

    user_text = USER_PROMPT_TEMPLATE.format(
        category=category, style=style, material=material
    )

    # Modele VL (vision-language) care acceptă imagini
    VL_MODEL_PATTERNS = ("qwen2.5-vl", "qwen2-vl", "qwen3-vl", "llava", "idefics",
                         "pixtral", "minicpm-v", "internvl", "cogvlm")
    is_vl_model = any(p in model.lower() for p in VL_MODEL_PATTERNS)

    # Construim content-ul mesajului cu sau fără imagine
    if is_vl_model and thumbnail and Path(thumbnail).exists():
        try:
            img = Image.open(thumbnail).convert("RGB")
            data_url = image_to_base64(img)
            user_content = [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text",      "text": user_text},
            ]
        except Exception:
            user_content = user_text
    else:
        user_content = user_text

    # Extra body: dezactivează thinking mode pentru task-uri simple de captioning
    # (Qwen3-VL-8B-Thinking returnează content=None dacă thinking e activ)
    extra_body: dict = {}
    if "qwen3" in model.lower():
        extra_body["enable_thinking"] = False  # OpenRouter Qwen3 flag

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        temperature=0,
        max_tokens=max_tokens,
        extra_body=extra_body if extra_body else None,
    )

    # Qwen3 thinking mode poate returna content=None cu reasoning_content populated
    content = response.choices[0].message.content
    if not content:
        content = getattr(response.choices[0].message, "reasoning_content", None) or ""
    return content.strip()


# Main

def main():
    parser = argparse.ArgumentParser(description="Generare caption-uri VLM pentru catalog mobilier")
    parser.add_argument("--catalog",      required=True, help="catalog_rendered.jsonl")
    parser.add_argument("--output",       required=True, help="catalog_annotated.jsonl (output)")
    # Backend VLM — OpenRouter sau LM Studio (OpenAI-compatible)
    parser.add_argument("--base_url",     default="http://localhost:1234/v1",
                        help="URL API VLM (LM Studio sau https://openrouter.ai/api/v1)")
    parser.add_argument("--api_key",      default="lm-studio",
                        help="API key (LM Studio: 'lm-studio'; OpenRouter: 'sk-or-v1-...')")
    parser.add_argument("--model",        default="qwen/qwen3.5-9b",
                        help="Model VLM (ex: qwen/qwen3.5-9b, qwen/qwen2.5-vl-7b)")
    # Alias-uri pentru compatibilitate înapoi
    parser.add_argument("--lmstudio_url",   default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lmstudio_model", default=None, help=argparse.SUPPRESS)
    # Alți parametri
    parser.add_argument("--max_tokens",   type=int, default=150,
                        help="Număr maxim de tokeni generați per caption")
    parser.add_argument("--no_vlm",       action="store_true",
                        help="Mod fallback: caption-uri din metadata fără VLM")
    parser.add_argument("--resume",       action="store_true",
                        help="Skip obiecte care au deja caption != null în fișierul de output")
    parser.add_argument("--limit",        type=int, default=0,
                        help="Procesează doar primele N obiecte (pentru testare)")
    parser.add_argument("--workers",      type=int, default=16,
                        help="Număr de thread-uri paralele pentru apeluri API (default: 16)")
    args = parser.parse_args()

    # Compatibilitate înapoi cu --lmstudio_url / --lmstudio_model
    if args.lmstudio_url:
        args.base_url = args.lmstudio_url
    if args.lmstudio_model:
        args.model = args.lmstudio_model

    # Citire catalog
    with open(args.catalog, "r") as f:
        entries = [json.loads(line) for line in f]

    if args.limit > 0:
        entries = entries[: args.limit]

    # Resume: încarcă caption-urile deja generate
    existing = {}
    output_path = Path(args.output)
    if args.resume and output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                rec = json.loads(line)
                # Caption-urile fallback (din metadata, ex. după eroare API / credit epuizat)
                # nu sunt păstrate la resume: obiectele respective se reprocesează cu VLM.
                cap = rec.get("caption")
                if cap and "Designed with clean proportions and careful attention to detail." not in cap:
                    existing[rec["uid"]] = cap
        print(f"[Resume] {len(existing)} caption-uri existente vor fi păstrate.")

    # Inițializare client VLM (dacă nu e mod fallback)
    client = None
    if not args.no_vlm:
        client = build_vlm_client(args.base_url, args.api_key)
        backend = "OpenRouter" if "openrouter" in args.base_url else "LM Studio"
        print(f"[VLM] {backend} → {args.base_url}  model={args.model}")
        # Warm-up / verificare conexiune
        try:
            client.models.list()
            print(f"[VLM] Conexiune {backend} OK.")
        except Exception as e:
            print(f"[AVERTISMENT] Nu s-a putut contacta {backend}: {e}")
            print("              Continuăm cu modul --no_vlm (fallback).")
            args.no_vlm = True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_vlm      = 0
    n_fallback = 0
    n_skipped  = 0
    write_lock = threading.Lock()

    # Separă obiectele care necesită procesare de cele deja existente
    to_process = []
    skipped_entries = []
    for entry in entries:
        if args.resume and entry["uid"] in existing:
            entry["caption"] = existing[entry["uid"]]
            skipped_entries.append(entry)
            n_skipped += 1
        else:
            to_process.append(entry)

    def process_entry(entry):
        """Procesează un singur obiect (rulat în thread pool)."""
        if args.no_vlm:
            return entry, fallback_caption(entry), "fallback"
        try:
            caption = generate_caption_lmstudio(
                client, args.model, entry, args.max_tokens
            )
            return entry, caption, "vlm"
        except Exception as e:
            tqdm.write(f"  [EROARE VLM] {entry['uid'][:8]}: {e} → fallback")
            return entry, fallback_caption(entry), "fallback"

    # Scriem în ordine: mai întâi skip-urile (în ordinea originală din catalog),
    # apoi rezultatele în ordinea originală (results_map indexat după uid).
    with open(output_path, "w") as f_out:
        # Scriem obiectele skip-uite imediat (sunt deja sortate)
        for entry in skipped_entries:
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Procesăm în paralel, dar scriem în ordinea ORIGINALĂ a catalogului
        results_map: dict[str, dict] = {}

        workers = min(args.workers, len(to_process)) if to_process else 1
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_entry, e): e["uid"] for e in to_process}
            pbar = tqdm(total=len(to_process), desc="Caption-uri VLM")
            for future in as_completed(futures):
                entry, caption, kind = future.result()
                entry["caption"] = caption
                results_map[entry["uid"]] = entry
                if kind == "vlm":
                    n_vlm += 1
                else:
                    n_fallback += 1
                pbar.update(1)
            pbar.close()

        # Scriem în ordinea originală a catalogului
        for entry in to_process:
            f_out.write(json.dumps(results_map[entry["uid"]], ensure_ascii=False) + "\n")

    total = len(entries)
    print(f"\nCatalog adnotat salvat: {args.output}")
    print(f"  Total:    {total}")
    print(f"  VLM:      {n_vlm}")
    print(f"  Fallback: {n_fallback}")
    print(f"  Skipped:  {n_skipped}")


if __name__ == "__main__":
    main()
