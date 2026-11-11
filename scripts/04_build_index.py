"""
04_build_index.py
Construiește indexul de embeddings CLIP pentru retrieval.

Două canale:
  - TEXT:  CLIP.encode_text(caption)      → text_embeddings.npy
  - IMAGE: CLIP.encode_image(thumbnail)   → image_embeddings.npy

Ambele normalizate L2. La retrieval, se face dot product (= cosine similarity).

Instalare:
    pip install open_clip_torch Pillow numpy torch

Rulare:
    python 04_build_index.py \
        --catalog ./catalog/catalog_annotated.jsonl \
        --output_dir ./catalog/index \
        --model ViT-L-14 \
        --pretrained openai

Output:
    index/text_embeddings.npy     float32 [N, D]
    index/image_embeddings.npy    float32 [N, D]
    index/catalog_index.jsonl     metadata pentru fiecare din cele N obiecte
    index/index_meta.json         info model, D, N, timestamp
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
import open_clip


def load_catalog(catalog_path: str) -> list[dict]:
    with open(catalog_path, "r") as f:
        return [json.loads(line) for line in f]


def build_text_embeddings(entries: list[dict], model, tokenizer, device, batch_size: int) -> np.ndarray:
    """Embed caption-urile text cu CLIP."""
    all_embs = []
    captions = [e.get("caption") or fallback_caption_text(e) for e in entries]

    for i in range(0, len(captions), batch_size):
        batch = captions[i:i+batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embs.append(emb.cpu().float().numpy())
        if (i // batch_size) % 10 == 0:
            print(f"  Text embeddings: {min(i+batch_size, len(captions))}/{len(captions)}")

    return np.vstack(all_embs)


def build_image_embeddings(entries: list[dict], model, preprocess, device, batch_size: int,
                           emb_dim: int | None = None) -> np.ndarray:
    """Embed thumbnails cu CLIP. Obiectele fără thumbnail primesc vector zero.

    emb_dim: dimensiunea embedding-ului CLIP (din text_embeddings) folosită pentru
    a aloca corect placeholder-ele atunci când un întreg batch nu are thumbnails.
    """
    all_embs = []
    # Reține pentru fiecare batch fără nicio imagine validă câte rânduri zero trebuie
    # generate ulterior (None placeholder -> (poziție, n_rânduri)).
    none_batch_sizes = []
    dim = emb_dim

    for i in range(0, len(entries), batch_size):
        batch_entries = entries[i:i+batch_size]
        imgs = []           # tensorii preprocesați, doar pentru thumbnails valide
        valid_idx = []      # indecșii (relativi la batch) pentru care avem imagine

        for j, e in enumerate(batch_entries):
            thumb = e.get("thumbnail_path")
            if thumb and Path(thumb).exists():
                try:
                    img = Image.open(thumb).convert("RGB")
                    imgs.append(preprocess(img))
                    valid_idx.append(j)
                except Exception:
                    # Imagine coruptă / necitibilă → tratată ca lipsă
                    continue

        if not imgs:
            # Niciun thumbnail valid în acest batch → placeholder; reținem câte
            # rânduri zero trebuie generate (= numărul de obiecte din batch).
            all_embs.append(None)
            none_batch_sizes.append(len(batch_entries))
            continue

        img_tensor = torch.stack(imgs).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
            emb = model.encode_image(img_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        dim = emb.shape[-1]
        batch_result = np.zeros((len(batch_entries), dim), dtype=np.float32)
        # valid_idx și rândurile lui emb sunt în aceeași ordine (ambele construite
        # iterând batch_entries în ordine) → maparea este corectă.
        batch_result[valid_idx] = emb.cpu().float().numpy()
        all_embs.append(batch_result)

        if (i // batch_size) % 10 == 0:
            print(f"  Image embeddings: {min(i+batch_size, len(entries))}/{len(entries)}")

    # Rezolvăm placeholder-urile None (batch-uri fără nicio imagine validă).
    # dim poate fi cunoscut din emb_dim (text) sau dedus din primul batch valid.
    if dim is None:
        dim = 768  # fallback ultim (ViT-L/14)
    result = []
    none_iter = iter(none_batch_sizes)
    for emb in all_embs:
        if emb is None:
            # Alocăm exact câte rânduri avea batch-ul respectiv (nu doar 1!),
            # altfel embeddings-urile se dezaliniază de catalog.
            n_rows = next(none_iter)
            result.append(np.zeros((n_rows, dim), dtype=np.float32))
        else:
            result.append(emb)

    return np.vstack(result)


def fallback_caption_text(entry: dict) -> str:
    cat = entry.get("category", entry.get("super_category", "furniture"))
    style = entry.get("style", "Modern")
    return f"A {style.lower()} {cat.lower()} for interior decoration."


def main():
    parser = argparse.ArgumentParser(description="Construiește index CLIP pentru catalog 3D-FUTURE")
    parser.add_argument("--catalog",    required=True, help="catalog_annotated.jsonl")
    parser.add_argument("--output_dir", default="./catalog/index")
    parser.add_argument("--model",      default="ViT-L-14",
                        help="Arhitectura CLIP (ex: ViT-L-14, ViT-B-32)")
    parser.add_argument("--pretrained", default="openai",
                        help="Weights pretrained (openai, laion2b_s32b_b82k, etc.)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--skip_images", action="store_true",
                        help="Skip image embedding (dacă thumbnails nu sunt disponibile)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_catalog(args.catalog)
    print(f"[1/4] Catalog: {len(entries)} obiecte")

    # Încărcare model CLIP
    print(f"[2/4] Încărcare CLIP {args.model} ({args.pretrained}) ...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"      Device: {device}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()
    dim = model.text_projection.shape[-1] if hasattr(model, "text_projection") else 512

    # Text embeddings
    print(f"[3/4] Text embeddings ...")
    text_embs = build_text_embeddings(entries, model, tokenizer, device, args.batch_size)
    np.save(output_dir / "text_embeddings.npy", text_embs)
    print(f"      Salvat: text_embeddings.npy  shape={text_embs.shape}")

    # Image embeddings
    if not args.skip_images:
        print(f"[4/4] Image embeddings ...")
        img_embs = build_image_embeddings(
            entries, model, preprocess, device, args.batch_size,
            emb_dim=text_embs.shape[1],
        )
        np.save(output_dir / "image_embeddings.npy", img_embs)
        print(f"      Salvat: image_embeddings.npy  shape={img_embs.shape}")
    else:
        print("[4/4] Skip image embeddings (--skip_images)")

    # Catalog index (metadata stripped)
    index_entries = []
    for e in entries:
        index_entries.append({
            "uid":            e["uid"],
            "super_category": e.get("super_category", ""),
            "category":       e.get("category", ""),
            "style":          e.get("style", ""),
            "material":       e.get("material", ""),
            "caption":        e.get("caption", ""),
            "obj_path":       e.get("obj_path", ""),
            "thumbnail_path": e.get("thumbnail_path", ""),
        })

    index_catalog_path = output_dir / "catalog_index.jsonl"
    with open(index_catalog_path, "w") as f:
        for rec in index_entries:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Meta
    meta = {
        "model": args.model,
        "pretrained": args.pretrained,
        "embedding_dim": text_embs.shape[1],
        "n_objects": len(entries),
        "has_image_embeddings": not args.skip_images,
        "built_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(output_dir / "index_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nIndex complet:")
    print(f"  {output_dir}/text_embeddings.npy")
    print(f"  {output_dir}/image_embeddings.npy")
    print(f"  {output_dir}/catalog_index.jsonl")
    print(f"  {output_dir}/index_meta.json")
    print(f"  N={meta['n_objects']}, D={meta['embedding_dim']}")


if __name__ == "__main__":
    main()
