"""
run_paper_eval.py — experimente suplimentare pentru articolul RoEduNet 2026
Rulare locală (necesită LM Studio cu Qwen2.5-VL-7B încărcat + open_clip/torch).
Nu modifică indexul; scrie în results/.

Etapa 1  `queries`     — evaluare fără scurgere de etichete (leakage-free):
    Pentru 300 de obiecte-query (50 / super-categorie, seed 0) se folosește
    vederea laterală `view_1.png` (nu vederea indexată `view_0`). VLM-ul online
    primește doar imaginea (fără metadate) și returnează stil, categorie și
    `replacement_query`. Se salvează ieșirile VLM și embedding-urile CLIP
    (text al interogării, imagine a vederii laterale).
Etapa 2  `qualitative` — 4 scene reale MMIS + click → analiză VLM online
    (cadru complet + crop), embedding-uri CLIP pentru interogare și crop.

Metricile se calculează separat, doar cu numpy (eval_v3.py).

Rulare (din rădăcina repo-ului):
    python scripts/run_paper_eval.py queries --split test --prompt P2
    python scripts/run_paper_eval.py qualitative --prompt P2
Opțiuni: --model qwen/qwen2.5-vl-7b  --base_url http://localhost:1234/v1
"""
import argparse, base64, json, random, sys, time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch, open_clip
from PIL import Image
from openai import OpenAI

import os
ROOT = Path(__file__).resolve().parent.parent
CAT = ROOT / os.environ.get("PAPER_CATALOG", "catalog")
OUT = ROOT / "results"
INDEX = CAT / "index"

STYLES = ["Modern", "Minimalist", "Nordic", "Industrial", "Art Deco",
          "European Traditional", "Southeast Asian", "Chinese Traditional",
          "Japanese", "Luxury", "Kids"]
SUPERCATS = ["Sofa", "Chair", "Table", "Bed", "Cabinet/Shelf/Desk", "Lighting"]

OBJECT_PROMPT = f"""This image shows a single furniture item rendered on a plain background.
A user wants to REPLACE this item with a different catalogue item that has the same function and fits the same interior style.
Respond in JSON only:
{{
  "style": "<one of: {', '.join(STYLES)}>",
  "category": "<one of: {', '.join(SUPERCATS)}>",
  "replacement_query": "<a 2-sentence description of the IDEAL replacement piece: style, shape, material, colour>"
}}"""

DESCRIBE_FOCUS = """focusing on: 1. overall aesthetic and style; 2. key visual characteristics (shape, colour, proportions, materials); 3. suitable interior context (e.g. 'ideal for a minimalist living room'). Under 80 words."""
DESCRIBE_BODY = "2-3 sentences describing THIS item, " + DESCRIBE_FOCUS

# P1: descrie obiectul în forma caption-urilor din catalog (fără metadate)
OBJECT_PROMPT_P1 = f"""This image shows a single furniture item rendered on a plain background.
Respond in JSON only:
{{
  "style": "<one of: {', '.join(STYLES)}>",
  "category": "<one of: {', '.join(SUPERCATS)}>",
  "replacement_query": "<{DESCRIBE_BODY}>"
}}"""

# P2: ca P1, dar descrierea începe cu stilul și tipul obiectului (ca în caption-uri)
OBJECT_PROMPT_P2 = f"""This image shows a single furniture item rendered on a plain background.
Respond in JSON only:
{{
  "style": "<one of: {', '.join(STYLES)}>",
  "category": "<one of: {', '.join(SUPERCATS)}>",
  "replacement_query": "<{DESCRIBE_BODY} Start with 'This <style> <specific furniture type>', e.g. 'This Nordic dining chair ...'.>"
}}"""

OBJECT_PROMPTS = {"P0": OBJECT_PROMPT, "P1": OBJECT_PROMPT_P1, "P2": OBJECT_PROMPT_P2}

ROOM_PROMPT = f"""Analyze this room scene. A user has clicked on a furniture item shown in the second image.
Respond in JSON format only:
{{
  "room_style": "<one of: {', '.join(STYLES)}, Mixed>",
  "selected_object": "<brief description of the clicked object>",
  "selected_category": "<one of: {', '.join(SUPERCATS)}>",
  "replacement_query": "<a 2-sentence description of the IDEAL replacement furniture piece that would fit this room's style and the user's selection>",
  "reasoning": "<why this style and category were inferred>"
}}"""

ROOM_PROMPT_P1 = f"""Analyze this room scene. A user has clicked on a furniture item shown in the second image.
Respond in JSON format only:
{{
  "room_style": "<one of: {', '.join(STYLES)}, Mixed>",
  "selected_category": "<one of: {', '.join(SUPERCATS)}>",
  "replacement_query": "<2-3 sentences describing a replacement for the selected item that fits this room, {DESCRIBE_FOCUS}>"
}}"""

ROOM_PROMPT_P2 = f"""Analyze this room scene. A user has clicked on a furniture item shown in the second image.
Respond in JSON format only:
{{
  "room_style": "<one of: {', '.join(STYLES)}, Mixed>",
  "selected_category": "<one of: {', '.join(SUPERCATS)}>",
  "replacement_query": "<2-3 sentences describing a replacement for the selected item that fits this room, {DESCRIBE_FOCUS} Start with 'This <room style> <specific furniture type>'.>"
}}"""

ROOM_PROMPTS = {"P0": ROOM_PROMPT, "P1": ROOM_PROMPT_P1, "P2": ROOM_PROMPT_P2}

# Scene MMIS pentru figura calitativă: (cale relativă la scripts/vlm_selection/, click_x, click_y)
SCENES = [
    ("mmis_eval/raw/Mid-Century Modern/02_29664.jpg", 190, 140, "sofa"),
    ("mmis_eval/raw/Eclectic/03_16204.jpg",          150, 170, "bed"),
    ("mmis_eval/raw/Eclectic/02_17335.jpg",          150,  95, "table"),
    ("mmis_eval/raw/Bohemian/00_5098.jpg",           125,  72, "pendant lamp"),
]
CROP_MARGIN = 60   # imaginile MMIS au 256x256 px


def data_url(img, max_size=512):
    img = img.convert("RGB")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ask_vlm(client, model, content, max_tokens=300):
    t0 = time.time()
    r = client.chat.completions.create(model=model, temperature=0, max_tokens=max_tokens,
                                       messages=[{"role": "user", "content": content}])
    raw = (r.choices[0].message.content or "").strip()
    lat = time.time() - t0
    try:
        parsed = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except Exception:
        parsed = None
    return raw, parsed, lat


class Clip:
    def __init__(self):
        dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dev = dev
        self.m, _, self.pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=dev)
        self.m.eval(); self.tok = open_clip.get_tokenizer("ViT-L-14")

    @torch.no_grad()
    def text(self, s):
        e = self.m.encode_text(self.tok([s]).to(self.dev)); return (e / e.norm(dim=-1, keepdim=True)).cpu().float().numpy()[0]

    @torch.no_grad()
    def image(self, img):
        e = self.m.encode_image(self.pre(img.convert("RGB")).unsqueeze(0).to(self.dev))
        return (e / e.norm(dim=-1, keepdim=True)).cpu().float().numpy()[0]


def load_catalog():
    return [json.loads(l) for l in open(INDEX / "catalog_index.jsonl")]


def sample_queries(catalog, per_cat=50, seed=0):
    rng = random.Random(seed)
    idx = []
    for sc in SUPERCATS:
        pool = [i for i, e in enumerate(catalog) if e["super_category"] == sc
                and (CAT / "thumbnails" / e["uid"] / "view_1.png").exists()]
        idx += sorted(rng.sample(pool, per_cat))
    return idx


def sample_dev(catalog, test_idx, per_cat=17, seed=1):
    """Set de dezvoltare disjunct de test (102 obiecte, 17 / super-categorie, seed 1)."""
    rng = random.Random(seed); test = set(int(i) for i in test_idx); idx = []
    for sc in SUPERCATS:
        pool = [i for i, e in enumerate(catalog) if e["super_category"] == sc and i not in test
                and (CAT / "thumbnails" / e["uid"] / "view_1.png").exists()]
        idx += sorted(rng.sample(pool, per_cat))
    return idx


def cmd_queries(args):
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog()
    test_idx = sample_queries(catalog)
    qidx = test_idx if args.split == "test" else sample_dev(catalog, test_idx)
    tag = "" if (args.split == "test" and args.prompt == "P0") else f"_{args.split}_{args.prompt}"
    out_jsonl = OUT / f"queries_vlm{tag}.jsonl"
    prompt = OBJECT_PROMPTS[args.prompt]
    done = {}
    if out_jsonl.exists():
        for l in open(out_jsonl):
            d = json.loads(l); done[d["idx"]] = d
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    print(f"[queries] {len(qidx)} interogări, {len(done)} deja făcute; model={args.model}")
    with open(out_jsonl, "a") as f:
        for n, i in enumerate(qidx):
            if i in done:
                continue
            e = catalog[i]
            img = Image.open(CAT / "thumbnails" / e["uid"] / "view_1.png")
            raw, parsed, lat = ask_vlm(client, args.model, [
                {"type": "image_url", "image_url": {"url": data_url(img)}},
                {"type": "text", "text": prompt}])
            d = {"idx": i, "split": args.split, "prompt": args.prompt, "uid": e["uid"], "gold_style": e["style"], "gold_cat": e["super_category"],
                 "raw": raw, "parsed": parsed, "latency_s": round(lat, 3), "model": args.model}
            f.write(json.dumps(d) + "\n"); f.flush(); done[i] = d
            if n % 25 == 0:
                print(f"  {n+1}/{len(qidx)}  lat={lat:.2f}s  parsed={parsed is not None}")
    print("[queries] CLIP embeddings ...")
    clip = Clip()
    rows = [done[i] for i in qidx]
    q_text = np.stack([clip.text((r["parsed"] or {}).get("replacement_query") or "a furniture item") for r in rows])
    q_img = np.stack([clip.image(Image.open(CAT / "thumbnails" / r["uid"] / "view_1.png")) for r in rows])
    np.savez(OUT / f"queries_emb{tag}.npz", idx=np.array(qidx), q_text=q_text, q_img=q_img)
    print(f"[queries] gata → {OUT}")


def cmd_qualitative(args):
    OUT.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    clip = Clip()
    res, qt, qi = [], [], []
    for path, x, y, label in SCENES:
        frame = Image.open(ROOT / "scripts" / "vlm_selection" / path).convert("RGB")
        W, H = frame.size
        crop = frame.crop((max(0, x - CROP_MARGIN), max(0, y - CROP_MARGIN), min(W, x + CROP_MARGIN), min(H, y + CROP_MARGIN)))
        raw, parsed, lat = ask_vlm(client, args.model, [
            {"type": "text", "text": "Full room view:"},
            {"type": "image_url", "image_url": {"url": data_url(frame)}},
            {"type": "text", "text": "\nSelected object:"},
            {"type": "image_url", "image_url": {"url": data_url(crop)}},
            {"type": "text", "text": "\n" + ROOM_PROMPTS[args.prompt]}], max_tokens=400)
        p = parsed or {}
        print(f"  {label}: {p.get('room_style')} / {p.get('selected_category')} ({lat:.1f}s)\n    {p.get('replacement_query')}")
        res.append({"path": path, "click": [x, y], "label": label, "crop_margin": CROP_MARGIN,
                    "raw": raw, "parsed": parsed, "latency_s": round(lat, 3)})
        qt.append(clip.text(p.get("replacement_query") or "a furniture item")); qi.append(clip.image(crop))
    tag = "" if args.prompt == "P0" else f"_{args.prompt}"
    json.dump(res, open(OUT / f"qualitative_vlm{tag}.json", "w"), indent=2)
    np.savez(OUT / f"qualitative_emb{tag}.npz", q_text=np.stack(qt), q_img=np.stack(qi))
    print(f"[qualitative] gata → {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["queries", "qualitative"])
    ap.add_argument("--model", default="qwen/qwen2.5-vl-7b")
    ap.add_argument("--base_url", default="http://localhost:1234/v1")
    ap.add_argument("--api_key", default="lm-studio")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--prompt", default="P0", choices=["P0", "P1", "P2"])
    a = ap.parse_args()
    {"queries": cmd_queries, "qualitative": cmd_qualitative}[a.cmd](a)
