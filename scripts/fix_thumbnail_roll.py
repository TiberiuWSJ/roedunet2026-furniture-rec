"""
fix_thumbnail_roll.py — corectează orientarea thumbnails-urilor v1 fără re-randare.

02_render_thumbnails.py (v1) a folosit vectorul „up" al camerei (0,0,1), deși mesh-urile
3D-FUTURE sunt Y-up. Vectorul up al camerei influențează doar rotația în planul imaginii
(roll), nu poziția camerei; deci imaginea corectă se obține exact printr-o rotație 2D cu
unghiul la care axa Y a lumii se proiectează în imagine. Poziția camerei rămâne cea din v1,
interpretată în sistemul Y-up: vederea 0 → elevație ≈ 37.8° (de sus), vederea 1 ≈ 39.9°
(de sus), vederea 2 ≈ −41.1° (de dedesubt; neutilizată).

Rulare (din rădăcina repo-ului):  python scripts/fix_thumbnail_roll.py
Scrie catalog/thumbnails_fixed/<uid>/view_{0,1,2}.png și catalog/catalog_rendered_fixed.jsonl.
"""
import json, math
from pathlib import Path
import numpy as np
from PIL import Image

VIEWS = [(45, 30), (135, 25), (225, 20)]   # (azimut, elevație) din v1


def roll_and_elevation(az, el):
    a, e = np.radians([az, el])
    c = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])   # poziția camerei (v1)
    f = -c / np.linalg.norm(c)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    Y = np.array([0, 1.0, 0])
    roll = 90 - math.degrees(math.atan2(Y @ u, Y @ r))      # rotație CCW care aduce Y în sus
    elev = math.degrees(math.asin(np.clip(c @ Y / np.linalg.norm(c), -1, 1)))
    return roll, elev


def fix(img, roll, size=512):
    img = img.convert("RGB").rotate(roll, resample=Image.BICUBIC, expand=True, fillcolor="white")
    s = max(img.size); canvas = Image.new("RGB", (s, s), "white")
    canvas.paste(img, ((s - img.size[0]) // 2, (s - img.size[1]) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


def main():
    root = Path(__file__).resolve().parent.parent
    params = [roll_and_elevation(*v) for v in VIEWS]
    for i, (r, e) in enumerate(params):
        print(f"view_{i}: roll {r:.1f}°, elevație reală {e:.1f}°")
    out_root = root / "catalog" / "thumbnails_fixed"
    rows, n = [], 0
    for l in open(root / "catalog" / "catalog_rendered.jsonl"):
        e = json.loads(l); paths = []
        for i, (roll, _) in enumerate(params):
            src = root / "catalog" / "thumbnails" / e["uid"] / f"view_{i}.png"
            dst = out_root / e["uid"] / f"view_{i}.png"
            if src.exists():
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    fix(Image.open(src), roll).save(dst); n += 1
                paths.append(str(dst.relative_to(root)))
        e["thumbnail_paths"] = paths
        e["thumbnail_path"] = paths[0] if paths else None
        rows.append(e)
    with open(root / "catalog" / "catalog_rendered_fixed.jsonl", "w") as f:
        for e in rows:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"{n} imagini noi; {sum(1 for e in rows if e['thumbnail_path'])}/{len(rows)} obiecte cu thumbnail")


if __name__ == "__main__":
    main()
