"""
clip_index.py
Wrapper subțire peste indexul CLIP construit de `04_build_index.py`, pentru a
calcula metricile de regăsire D4. Reutilizează aceeași logică de encodare și
produs scalar ca `05_recommend.py`, ca rezultatele să fie comparabile cu pipeline-ul.

Indexul așteptat în `index_dir`:
  - catalog_index.jsonl      (o linie/obiect: uid, category, super_category, style, ...)
  - text_embeddings.npy      (N × D, normalizate)
  - image_embeddings.npy     (N × D, opțional)
  - index_meta.json          ({"embedding_dim": D, ...})
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


class ClipIndex:
    def __init__(
        self,
        index_dir: str,
        clip_model: str = "ViT-L-14",
        clip_pretrained: str = "openai",
        device: Optional[str] = None,
    ):
        self.index_dir = Path(index_dir)
        self._load_index()
        self._device = device
        self._clip = None
        self._tokenizer = None
        self._clip_name = clip_model
        self._clip_pretrained = clip_pretrained

    # Index
    def _load_index(self):
        with open(self.index_dir / "catalog_index.jsonl") as f:
            self.catalog = [json.loads(line) for line in f]
        self.text_embs = np.load(self.index_dir / "text_embeddings.npy").astype(np.float32)
        img_path = self.index_dir / "image_embeddings.npy"
        self.image_embs = (np.load(img_path).astype(np.float32)
                           if img_path.exists() else None)
        with open(self.index_dir / "index_meta.json") as f:
            self.meta = json.load(f)

    # CLIP (lazy)
    def _ensure_clip(self):
        if self._clip is not None:
            return
        import torch
        import open_clip
        dev = self._device or ("mps" if torch.backends.mps.is_available()
                               else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._torch = torch
        self._device = dev
        self._clip, _, self._preprocess = open_clip.create_model_and_transforms(
            self._clip_name, pretrained=self._clip_pretrained)
        self._tokenizer = open_clip.get_tokenizer(self._clip_name)
        self._clip = self._clip.to(dev).eval()

    def embed_text(self, text: str) -> np.ndarray:
        self._ensure_clip()
        toks = self._tokenizer([text]).to(self._device)
        with self._torch.no_grad():
            e = self._clip.encode_text(toks)
            e = e / e.norm(dim=-1, keepdim=True)
        return e.cpu().float().numpy()[0]

    def embed_image(self, img) -> np.ndarray:
        self._ensure_clip()
        t = self._preprocess(img).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            e = self._clip.encode_image(t)
            e = e / e.norm(dim=-1, keepdim=True)
        return e.cpu().float().numpy()[0]

    # Regăsire
    def retrieve_text(self, query: str, top_k: int = 10) -> list[dict]:
        """Regăsire pe canalul text. Returnează intrările catalog + scor + index global."""
        q = self.embed_text(query)
        scores = self.text_embs @ q
        idx = np.argsort(scores)[::-1][:top_k]
        return [{**self.catalog[i], "_score": float(scores[i]), "_idx": int(i)}
                for i in idx]

    def rank_of_uid(self, query: str, target_uid: str) -> Optional[int]:
        """Rangul (1-based) al obiectului GT în lista completă ordonată după scor text."""
        q = self.embed_text(query)
        scores = self.text_embs @ q
        order = np.argsort(scores)[::-1]
        for rank, i in enumerate(order, start=1):
            if self.catalog[i].get("uid") == target_uid:
                return rank
        return None

    def cosine_to_image_emb(self, query: str, target_uid: str) -> Optional[float]:
        """Cosinus între embedding-ul interogării și embedding-ul de imagine al obiectului GT."""
        if self.image_embs is None:
            return None
        uid_to_idx = {e.get("uid"): i for i, e in enumerate(self.catalog)}
        i = uid_to_idx.get(target_uid)
        if i is None:
            return None
        q = self.embed_text(query)
        return float(self.image_embs[i] @ q)
