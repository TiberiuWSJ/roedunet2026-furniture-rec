"""
05_recommend.py
Pipeline de inferență: click utilizator → recomandări de mobilier 3D.

Flow:
  1. Input: imagine cadru RGB + coordonate click (x, y)
  2. Crop în jurul click-ului (300×300 px)
  3. VLM (Qwen2.5-VL-7B via LM Studio) analizează stilul scenei și obiectul selectat
  4. Retrieval dual-canal CLIP:
       s_i = (1 - α) · (q_T · e_T^i) + α · (q_I · e_I^i)
       cu α = 0.4 (implicit)
  5. Re-rank (min-max + filtru de categorie + penalizare de stil):
       - normalizare min-max a scorurilor combinate în [0,1] pe tot catalogul
       - hard filter pe categorie (excludere completă a categoriei greșite)
       - penalizare aditivă fixă pentru stil incompatibil (scădere, nu înmulțire)
  6. Returnare top-K cu rationale

Rulare demo (CLI):
    python 05_recommend.py \\
        --index_dir  ./catalog/index \\
        --frame      /path/to/frame.jpg \\
        --click_x    640 \\
        --click_y    360 \\
        --top_k      5

Import ca modul:
    import importlib; FurnitureRecommender = importlib.import_module("05_recommend").FurnitureRecommender
    rec = FurnitureRecommender("./catalog/index")
    analysis, results = rec.recommend(frame_img, click_x=640, click_y=360)

Cerințe:
    pip install open_clip_torch torch pillow numpy openai tqdm
    # LM Studio rulând local cu Qwen2.5-VL-7B-Instruct încărcat
"""

import json
import base64
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from io import BytesIO

import torch
from PIL import Image
import open_clip
from openai import OpenAI


# Structuri de date

@dataclass
class RecommendationResult:
    rank:        int
    uid:         str
    category:    str
    style:       str
    material:    str
    caption:     str
    obj_path:    str
    thumbnail:   str
    score:       float
    score_text:  float
    score_image: float
    rationale:   str


@dataclass
class SceneAnalysis:
    room_style:         str    # stilul dominant al camerei
    selected_object:    str    # ce obiect a fost selectat
    query_caption:      str    # caption pentru CLIP text retrieval
    preferred_category: str    # categoria inferată


# Prompt VLM

SCENE_ANALYSIS_PROMPT = """Analyze this room scene. A user has clicked on a furniture item at approximately the marked position.

Respond in JSON format only:
{
  "room_style": "<one of: Modern, Minimalist, Nordic, Industrial, Art Deco, European Traditional, Southeast Asian, Chinese Traditional, Japanese, Luxury, Kids, Mixed>",
  "selected_object": "<brief description of the clicked object, e.g. 'a beige fabric sofa'>",
  "selected_category": "<one of: Sofa, Chair, Table, Bed, Cabinet/Shelf/Desk, Lighting>",
  "replacement_query": "<a 2-sentence description of the IDEAL replacement furniture piece that would fit this room's style and the user's selection>",
  "reasoning": "<why this style and category were inferred>"
}"""


# Utilitare imagine

def image_to_data_url(img: Image.Image, max_size: int = 768) -> str:
    """Redimensionează și codifică imaginea ca data URL base64 PNG."""
    img = img.convert("RGB")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# Clasa principală

class FurnitureRecommender:
    def __init__(
        self,
        index_dir: str,
        clip_model: str = "ViT-L-14",
        clip_pretrained: str = "openai",
        lmstudio_url: str = "http://localhost:1234/v1",
        lmstudio_model: str = "qwen/qwen2.5-vl-7b",
        device: Optional[str] = None,
        alpha: float = 0.4,    # pondere canal imagine (0=doar text, 1=doar imagine)
    ):
        self.index_dir      = Path(index_dir)
        self.alpha          = alpha
        self.lmstudio_model = lmstudio_model
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._load_index()
        self._load_clip(clip_model, clip_pretrained)
        self._init_lmstudio(lmstudio_url)

    # Încărcare index

    def _load_index(self):
        print(f"[Index] Încărcare din {self.index_dir} ...")
        with open(self.index_dir / "catalog_index.jsonl") as f:
            self.catalog = [json.loads(line) for line in f]

        self.text_embs = np.load(
            self.index_dir / "text_embeddings.npy"
        ).astype(np.float32)

        img_path = self.index_dir / "image_embeddings.npy"
        self.image_embs = None
        if img_path.exists():
            self.image_embs = np.load(img_path).astype(np.float32)

        with open(self.index_dir / "index_meta.json") as f:
            self.meta = json.load(f)

        print(
            f"  N={len(self.catalog)}, D={self.meta['embedding_dim']}, "
            f"image_embs={'DA' if self.image_embs is not None else 'NU'}"
        )

    # Încărcare CLIP

    def _load_clip(self, model_name: str, pretrained: str):
        print(f"[CLIP] {model_name} ({pretrained}) → {self.device}")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.clip_tokenizer = open_clip.get_tokenizer(model_name)
        self.clip_model = self.clip_model.to(self.device).eval()

    # Inițializare LM Studio

    def _init_lmstudio(self, base_url: str):
        print(f"[VLM] LM Studio → {base_url}  model={self.lmstudio_model}")
        self.lm_client = OpenAI(base_url=base_url, api_key="lm-studio")

    # Analiză scenă VLM

    def analyze_scene(
        self,
        frame: Image.Image,
        click_x: int,
        click_y: int,
        crop_margin: int = 150,
    ) -> SceneAnalysis:
        """
        Trimite cadrul complet + crop-ul obiectului selectat la LM Studio
        și parsează răspunsul JSON pentru stilul camerei și query-ul CLIP.
        """
        W, H = frame.size
        x0 = max(0, click_x - crop_margin)
        y0 = max(0, click_y - crop_margin)
        x1 = min(W, click_x + crop_margin)
        y1 = min(H, click_y + crop_margin)
        crop = frame.crop((x0, y0, x1, y1))

        # Codificare imagini
        frame_url = image_to_data_url(frame.resize((640, 480)))
        crop_url  = image_to_data_url(crop, max_size=512)

        response = self.lm_client.chat.completions.create(
            model=self.lmstudio_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text",      "text": "Full room view:"},
                        {"type": "image_url", "image_url": {"url": frame_url}},
                        {"type": "text",      "text": f"\nSelected object (user clicked at ~{click_x},{click_y}):"},
                        {"type": "image_url", "image_url": {"url": crop_url}},
                        {"type": "text",      "text": "\n" + SCENE_ANALYSIS_PROMPT},
                    ],
                }
            ],
            temperature=0,
            max_tokens=300,
        )

        raw_output = response.choices[0].message.content.strip()

        # Parsare JSON (poate fi înconjurat de text explicativ)
        try:
            start  = raw_output.find("{")
            end    = raw_output.rfind("}") + 1
            parsed = json.loads(raw_output[start:end])
        except Exception:
            parsed = {}

        return SceneAnalysis(
            room_style          = parsed.get("room_style", "Modern"),
            selected_object     = parsed.get("selected_object", "furniture item"),
            query_caption       = parsed.get(
                "replacement_query",
                f"A {parsed.get('room_style', 'modern')} style furniture piece.",
            ),
            preferred_category  = parsed.get("selected_category", ""),
        )

    # CLIP embed

    def _embed_text(self, text: str) -> np.ndarray:
        tokens = self.clip_tokenizer([text]).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().float().numpy()[0]

    def _embed_image(self, img: Image.Image) -> np.ndarray:
        tensor = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().float().numpy()[0]

    # Retrieval dual-canal

    def retrieve(
        self,
        query_text: str,
        query_image: Optional[Image.Image] = None,
        category_filter: Optional[str] = None,
        style_filter: Optional[str] = None,
        top_k: int = 10,
        style_penalty: float = 0.1,
    ) -> list[dict]:
        """
        Dual-canal CLIP retrieval cu re-rank (filtru de categorie + penalizare de stil).
          s_i = (1-α)·(q_T·e_T^i) + α·(q_I·e_I^i)

        Re-rank:
          1. Normalizare min-max a scorurilor combinate în [0,1] pe tot catalogul.
          2. Hard filter pe categorie: obiectele cu super_category greșit sunt
             excluse complet (scor → -inf), nu doar penalizate.
          3. Penalizare aditivă pentru stil incompatibil: se scade o valoare fixă
             (style_penalty) din scorul normalizat, în loc de înmulțire.

        Returnează top_k intrări sortate descrescător după scorul final.
        """
        # Text scores
        q_text      = self._embed_text(query_text)
        text_scores = self.text_embs @ q_text  # [N]

        # Image scores
        if query_image is not None and self.image_embs is not None:
            q_img      = self._embed_image(query_image)
            img_scores = self.image_embs @ q_img  # [N]
        else:
            img_scores = text_scores  # fallback text-only

        # Combinare
        combined = (1 - self.alpha) * text_scores + self.alpha * img_scores

        # Re-rank
        # 1. Normalizare min-max în [0,1] pe tot catalogul
        c_min, c_max = float(combined.min()), float(combined.max())
        if c_max - c_min > 1e-8:
            scores = (combined - c_min) / (c_max - c_min)
        else:
            scores = np.zeros_like(combined)

        scores = scores.astype(np.float32)

        # 2. Hard filter categorie + 3. penalizare aditivă stil
        for i, entry in enumerate(self.catalog):
            if (
                category_filter
                and entry["super_category"].lower() != category_filter.lower()
            ):
                scores[i] = -np.inf   # hard filter: excludere completă
                continue
            if (
                style_filter
                and entry["style"].lower() != style_filter.lower()
            ):
                scores[i] -= style_penalty  # penalizare aditivă stil

        top_idx = np.argsort(scores)[::-1][:top_k]
        # Elimină intrările hard-filtrate (dacă toate au fost excluse, lasă cele rămase)
        top_idx = [int(i) for i in top_idx if np.isfinite(scores[i])]

        results = []
        for rank, idx in enumerate(top_idx):
            e = self.catalog[idx]
            results.append({
                "rank":           rank + 1,
                "uid":            e["uid"],
                "category":       e["category"],
                "super_category": e["super_category"],
                "style":          e["style"],
                "material":       e["material"],
                "caption":        e["caption"],
                "obj_path":       e["obj_path"],
                "thumbnail":      e.get("thumbnail_path", ""),
                "score":          float(scores[idx]),       # scor final normalizat [0,1] după re-rank
                "score_combined": float(combined[idx]),     # scor combinat brut (pre-normalizare)
                "score_text":     float(text_scores[idx]),
                "score_image":    float(img_scores[idx]),
            })

        return results

    # Pipeline principal

    def recommend(
        self,
        frame: Image.Image,
        click_x: int,
        click_y: int,
        category_hint: Optional[str] = None,
        top_k: int = 5,
        crop_margin: int = 150,
    ) -> tuple[SceneAnalysis, list[RecommendationResult]]:
        """
        Pipeline end-to-end: cadru + click → top-K recomandări.

        Args:
            frame:         Imagine RGB a camerei (PIL.Image)
            click_x/y:     Coordonate pixel click utilizator
            category_hint: Categorie forțată (opțional, ex: "Sofa")
            top_k:         Număr de recomandări returnate
            crop_margin:   Raza crop-ului în jurul click-ului (pixeli)

        Returns:
            (SceneAnalysis, [RecommendationResult × top_k])
        """
        # 1. Analiză scenă prin VLM
        print(f"[Rec] Analiză VLM (click {click_x},{click_y}) ...")
        analysis = self.analyze_scene(frame, click_x, click_y, crop_margin)
        print(f"      Stil cameră:     {analysis.room_style}")
        print(f"      Obiect selectat: {analysis.selected_object}")
        print(f"      Categorie:       {analysis.preferred_category}")
        print(f"      Query CLIP:      {analysis.query_caption[:70]}...")

        # 2. Crop pentru canalul imagine
        W, H = frame.size
        x0   = max(0, click_x - crop_margin)
        y0   = max(0, click_y - crop_margin)
        x1   = min(W, click_x + crop_margin)
        y1   = min(H, click_y + crop_margin)
        crop = frame.crop((x0, y0, x1, y1))

        # 3. Retrieval dual-canal
        category = category_hint or analysis.preferred_category or None
        raw_results = self.retrieve(
            query_text      = analysis.query_caption,
            query_image     = crop,
            category_filter = category,
            style_filter    = analysis.room_style,
            top_k           = top_k,
        )

        # 4. Construire RecommendationResult cu rationale
        results = []
        for r in raw_results:
            rationale = (
                f"Stil compatibil cu camera ({analysis.room_style}): "
                f"obiect {r['style']} {r['category']} "
                f"cu similaritate cosinus {r['score']:.3f}. "
                f"{r['caption'][:100]}..."
            )
            results.append(RecommendationResult(
                rank        = r["rank"],
                uid         = r["uid"],
                category    = r["category"],
                style       = r["style"],
                material    = r["material"],
                caption     = r["caption"],
                obj_path    = r["obj_path"],
                thumbnail   = r["thumbnail"],
                score       = r["score"],
                score_text  = r["score_text"],
                score_image = r["score_image"],
                rationale   = rationale,
            ))

        return analysis, results


# CLI demo

def main():
    parser = argparse.ArgumentParser(description="Furniture recommendation demo (LM Studio)")
    parser.add_argument("--index_dir",      required=True, help="Director index CLIP")
    parser.add_argument("--frame",          required=True, help="Imagine cameră (JPG/PNG)")
    parser.add_argument("--click_x",        type=int, required=True)
    parser.add_argument("--click_y",        type=int, required=True)
    parser.add_argument("--category",       default=None, help="Forțează categorie (ex: Sofa)")
    parser.add_argument("--top_k",          type=int, default=5)
    parser.add_argument("--alpha",          type=float, default=0.4,
                        help="Pondere canal imagine CLIP (0=text only, 1=image only)")
    parser.add_argument("--lmstudio_url",   default="http://localhost:1234/v1")
    parser.add_argument("--lmstudio_model", default="qwen/qwen2.5-vl-7b")
    parser.add_argument("--output_json",    default=None, help="Salvează rezultate JSON")
    args = parser.parse_args()

    frame = Image.open(args.frame).convert("RGB")

    recommender = FurnitureRecommender(
        index_dir      = args.index_dir,
        lmstudio_url   = args.lmstudio_url,
        lmstudio_model = args.lmstudio_model,
        alpha          = args.alpha,
    )

    analysis, results = recommender.recommend(
        frame         = frame,
        click_x       = args.click_x,
        click_y       = args.click_y,
        category_hint = args.category,
        top_k         = args.top_k,
    )

    print("\n" + "=" * 60)
    print("ANALIZĂ SCENĂ")
    print(f"  Stil cameră:   {analysis.room_style}")
    print(f"  Obiect ales:   {analysis.selected_object}")
    print(f"  Categorie:     {analysis.preferred_category}")
    print(f"  Query CLIP:    {analysis.query_caption}")
    print("=" * 60)
    print(f"\nTOP {args.top_k} RECOMANDĂRI:\n")

    output_data = {
        "scene_analysis": {
            "room_style":         analysis.room_style,
            "selected_object":    analysis.selected_object,
            "query_caption":      analysis.query_caption,
            "preferred_category": analysis.preferred_category,
        },
        "recommendations": [],
    }

    for r in results:
        print(f"  #{r.rank}  [{r.style}] {r.category}")
        print(f"      Score: {r.score:.4f}  (text={r.score_text:.4f}, img={r.score_image:.4f})")
        print(f"      Caption: {r.caption[:80]}...")
        print(f"      OBJ: {r.obj_path}")
        print(f"      Rationale: {r.rationale[:100]}...")
        print()

        output_data["recommendations"].append({
            "rank":      r.rank,
            "uid":       r.uid,
            "category":  r.category,
            "style":     r.style,
            "material":  r.material,
            "caption":   r.caption,
            "obj_path":  r.obj_path,
            "thumbnail": r.thumbnail,
            "score":     r.score,
            "rationale": r.rationale,
        })

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Rezultate salvate: {args.output_json}")


if __name__ == "__main__":
    main()
