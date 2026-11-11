"""
build_index_fast.py
Construiește rapid indexul CLIP necesar metricii D4, fără randare (02) și fără VLM (03).

Trucuri pentru viteză:
  - thumbnail = `image.jpg` care există DEJA în fiecare folder 3D-FUTURE (nu se randează);
  - caption text = generat din metadata (fallback), nu din VLM;
  - filtrare la categoriile relevante + plafon per categorie pentru a ține N mic.

Pentru studiul VLM indexul e doar ținta de regăsire (aceeași pentru toate modelele),
deci caption-urile fallback sunt suficiente. Indexul "de producție" (cu caption-uri
VLM) se poate reconstrui ulterior cu 03/04.

Rulare:
    python build_index_fast.py \
        --models_dir ../../3D-FUTURE-model \
        --out ../catalog/index \
        --max_per_coarse 200

Ieșire (compatibilă cu clip_index.py):
    index/catalog_index.jsonl, text_embeddings.npy, image_embeddings.npy, index_meta.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from taxonomy import coarse_category, normalize_style, OTHER


def fallback_caption(meta: dict) -> str:
    cat = meta.get("category") or meta.get("super-category") or "furniture"
    style = meta.get("style") or "Modern"
    mat = meta.get("material") or ""
    mat_str = f" made of {mat.lower()}" if mat else ""
    return (f"A {style.lower()} {cat.lower()}{mat_str}, "
            f"suitable for interior decoration.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--out", default="../catalog/index")
    ap.add_argument("--max_per_coarse", type=int, default=200,
                    help="Plafon obiecte per categorie coarse (viteză)")
    ap.add_argument("--clip_model", default="ViT-L-14")
    ap.add_argument("--clip_pretrained", default="openai")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    info = json.load(open(models_dir / "model_info.json", encoding="utf-8"))

    # Filtrare + plafon pe categoria coarse
    per_coarse = defaultdict(int)
    entries = []
    for m in info:
        cc = coarse_category(m.get("category", ""))
        if cc == OTHER:
            continue
        if per_coarse[cc] >= args.max_per_coarse:
            continue
        uid = m["model_id"]
        thumb = models_dir / uid / "image.jpg"
        if not thumb.exists():
            continue
        per_coarse[cc] += 1
        entries.append({
            "uid": uid,
            "super_category": m.get("super-category", ""),
            "category": m.get("category", ""),
            "coarse_category": cc,
            "style": m.get("style", ""),
            "style_canonical": normalize_style(m.get("style", "")),
            "material": m.get("material", ""),
            "caption": fallback_caption(m),
            "thumbnail_path": str(thumb),
            "obj_path": str(models_dir / uid / "normalized_model.obj"),
        })
    print(f"Catalog filtrat: {len(entries)} obiecte. Distribuție coarse:",
          dict(per_coarse))

    # CLIP
    import torch
    import open_clip
    dev = ("mps" if torch.backends.mps.is_available()
           else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"CLIP {args.clip_model} pe {dev} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained)
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    model = model.to(dev).eval()

    # Text embeddings
    caps = [e["caption"] for e in entries]
    txt = []
    for i in range(0, len(caps), args.batch_size):
        toks = tokenizer(caps[i:i + args.batch_size]).to(dev)
        with torch.no_grad():
            emb = model.encode_text(toks)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        txt.append(emb.cpu().float().numpy())
        print(f"  text {min(i+args.batch_size,len(caps))}/{len(caps)}", end="\r")
    text_embs = np.vstack(txt)
    print(f"\n  text_embeddings: {text_embs.shape}")

    # Image embeddings
    img = []
    for i in range(0, len(entries), args.batch_size):
        batch = entries[i:i + args.batch_size]
        tensors = []
        for e in batch:
            try:
                tensors.append(preprocess(Image.open(e["thumbnail_path"]).convert("RGB")))
            except Exception:
                tensors.append(torch.zeros(3, 224, 224))
        t = torch.stack(tensors).to(dev)
        with torch.no_grad():
            emb = model.encode_image(t)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        img.append(emb.cpu().float().numpy())
        print(f"  image {min(i+args.batch_size,len(entries))}/{len(entries)}", end="\r")
    image_embs = np.vstack(img)
    print(f"\n  image_embeddings: {image_embs.shape}")

    # Scriere
    np.save(out / "text_embeddings.npy", text_embs)
    np.save(out / "image_embeddings.npy", image_embs)
    with open(out / "catalog_index.jsonl", "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    json.dump({"model": args.clip_model, "pretrained": args.clip_pretrained,
               "embedding_dim": int(text_embs.shape[1]), "n_objects": len(entries),
               "has_image_embeddings": True, "fast_build": True,
               "built_at": datetime.utcnow().isoformat() + "Z"},
              open(out / "index_meta.json", "w"), indent=2)
    print(f"Index în {out} (N={len(entries)}, D={text_embs.shape[1]})")


if __name__ == "__main__":
    main()
