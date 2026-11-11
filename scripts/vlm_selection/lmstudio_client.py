"""
lmstudio_client.py
Client pentru API-ul LM Studio (endpoint OpenAI-compatibil, implicit
http://localhost:1234/v1). Înlocuiește încărcarea HuggingFace locală din
`03_annotate_captions.py` și `05_recommend.py`.

Funcționalități:
  - trimite cadru complet + crop al obiectului click-uit + prompt;
  - măsoară latența end-to-end și throughput-ul (din usage.completion_tokens);
  - decodare deterministă (temperature=0) pentru reproducibilitate;
  - parser strict de JSON (D1) cu extragerea blocului {...} din răspuns.

Depinde doar de `requests` (fără SDK suplimentar), pentru a păstra harness-ul minimal.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests
from PIL import Image


# Structuri de date

@dataclass
class VLMResponse:
    raw_text: str                       # textul brut returnat de model
    parsed: Optional[dict]              # JSON parsat (None dacă a eșuat)
    json_valid: bool                    # s-a parsat un obiect JSON?
    latency_s: float                    # latență end-to-end (s)
    completion_tokens: Optional[int]    # din usage (dacă disponibil)
    tokens_per_s: Optional[float]       # throughput decodare
    error: Optional[str] = None


# Utilitare imagine

def _img_to_data_url(img: Image.Image, max_side: int = 768, fmt: str = "JPEG") -> str:
    """Convertește o imagine PIL într-un data URL base64 (redimensionată)."""
    img = img.convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def _strip_reasoning(text: str) -> str:
    """Elimină blocurile de raționament ale modelelor 'thinking' (<think>...</think>
    și variante), păstrând doar răspunsul final."""
    if not text:
        return text
    # elimină <think>...</think> (și alte tag-uri uzuale de reasoning)
    text = re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", " ", text,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<\s*reasoning\s*>.*?<\s*/\s*reasoning\s*>", " ", text,
                  flags=re.DOTALL | re.IGNORECASE)
    # dacă a rămas doar tag-ul de deschidere (răspuns truncat), îl tăiem
    text = re.sub(r"<\s*think\s*>.*$", " ", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _extract_json(text: str) -> Optional[dict]:
    """Parser strict: aruncă raționamentul, apoi extrage ultimul bloc {...}."""
    if not text:
        return None
    text = _strip_reasoning(text)
    # încercare directă
    try:
        return json.loads(text)
    except Exception:
        pass
    # extragere bloc {...} (ultimul obiect = răspunsul final după reasoning)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


# Client

class LMStudioClient:
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        timeout_s: int = 300,
        api_key: str = "lm-studio",
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    # Listare modele disponibile (pentru validarea configului)
    def list_models(self) -> list[str]:
        r = requests.get(f"{self.base_url}/models", headers=self.headers,
                         timeout=self.timeout_s)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    # Analiză scenă (sarcina online)
    def analyze_scene(
        self,
        model: str,
        prompt: str,
        full_frame: Image.Image,
        crop: Optional[Image.Image],
        max_tokens: int = 300,
        temperature: float = 0.0,
    ) -> VLMResponse:
        """
        Trimite cadrul complet (+ crop dacă există) și promptul, returnează VLMResponse.
        Configurația de intrare (full+crop vs. doar crop) e controlată de apelant (D6).
        """
        content = [{"type": "text", "text": "Full room view:"},
                   {"type": "image_url",
                    "image_url": {"url": _img_to_data_url(full_frame)}}]
        if crop is not None:
            content += [{"type": "text", "text": "\nSelected object (clicked region):"},
                        {"type": "image_url",
                         "image_url": {"url": _img_to_data_url(crop)}}]
        content.append({"type": "text", "text": "\n" + prompt})

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        t0 = time.perf_counter()
        try:
            r = requests.post(f"{self.base_url}/chat/completions",
                              headers=self.headers, json=payload,
                              timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            return VLMResponse(raw_text="", parsed=None, json_valid=False,
                               latency_s=time.perf_counter() - t0,
                               completion_tokens=None, tokens_per_s=None,
                               error=str(e))
        latency = time.perf_counter() - t0

        # Pe modelele cu reasoning (ex. OpenRouter), `content`
        # poate fi None, iar răspunsul/raționamentul stă pe `reasoning`.
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        raw = msg.get("content")
        if not raw:
            raw = msg.get("reasoning") or msg.get("reasoning_content") or ""
        raw = (raw or "").strip()
        parsed = _extract_json(raw)
        usage = data.get("usage", {}) or {}
        ctoks = usage.get("completion_tokens")
        tps = (ctoks / latency) if (ctoks and latency > 0) else None

        return VLMResponse(
            raw_text=raw,
            parsed=parsed,
            json_valid=parsed is not None,
            latency_s=latency,
            completion_tokens=ctoks,
            tokens_per_s=tps,
        )
