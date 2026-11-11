"""
download_mmis_subset.py
Descarcă un subset mic din MMIS de pe Google Drive. Se rulează local.

Structura MMIS: imaginile sunt în arhive .zip per stil
(fiecare stil are 2 părți: -001.zip, -002.zip), plus câteva fișiere libere
(json/pkl) care se ignoră.

Strategie (evită descărcarea întregului set de ~50 GB):
  1. enumeră folderul Drive fără descărcare (gdown skip_download=True);
  2. pentru fiecare stil, descarcă doar primul zip (-001);
  3. extrage primele N imagini din el în <out>/<Style>/;
  4. șterge zip-ul imediat (peak disk = un singur zip la un moment dat).

Necesită:  pip install gdown

Rulare:
    python download_mmis_subset.py \
        --url "https://drive.google.com/drive/folders/1FO_sNVZi757I_QBdwibPX--14O24ff0m" \
        --out ./mmis_eval/raw --per_style 8

    python download_mmis_subset.py --url "..." --list_only      # doar structura
    python download_mmis_subset.py --url "..." --max_styles 6   # mai puține stiluri
"""

from __future__ import annotations

import argparse
import random
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import gdown

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def dir_parts(path: str) -> list[str]:
    return [p for p in path.replace("\\", "/").split("/")[:-1] if p]


def common_prefix_len(list_of_parts):
    lst = [p for p in list_of_parts if p]
    if not lst:
        return 0
    n = min(len(p) for p in lst)
    pref = 0
    for i in range(n):
        if len({p[i] for p in lst}) == 1:
            pref += 1
        else:
            break
    return pref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", default="./mmis_eval/raw")
    ap.add_argument("--per_style", type=int, default=8)
    ap.add_argument("--max_styles", type=int, default=0)
    ap.add_argument("--only_styles", default="",
                    help="Listă separată prin virgulă: descarcă doar aceste stiluri (retry)")
    ap.add_argument("--style_index", type=int, default=-1)
    ap.add_argument("--list_only", action="store_true")
    ap.add_argument("--keep_zip", action="store_true", help="Nu șterge zip-urile")
    ap.add_argument("--seed", type=int, default=42, help="Seed pentru eșantionarea aleatoare")
    args = ap.parse_args()
    random.seed(args.seed)

    print("[1/3] Enumerare folder Drive (fără descărcare) ...")
    files = gdown.download_folder(url=args.url, skip_download=True,
                                  quiet=True, use_cookies=False)
    if not files:
        raise SystemExit("Nu am putut enumera folderul.")

    zips = [f for f in files if Path(f.path).suffix.lower() == ".zip"
            and len(dir_parts(f.path)) > 0]  # zip-uri aflate într-un subfolder (stil)
    print(f"  {len(files)} fișiere totale, {len(zips)} arhive de stil")

    parts = [dir_parts(f.path) for f in zips]
    style_idx = args.style_index if args.style_index >= 0 else common_prefix_len(parts)

    def style_of(f):
        p = dir_parts(f.path)
        return p[style_idx] if style_idx < len(p) else "UNKNOWN"

    by_style = defaultdict(list)
    for f in zips:
        by_style[style_of(f)].append(f)

    styles = sorted(by_style.keys())
    print(f"  nivel stil: index {style_idx} | {len(styles)} stiluri:")
    for s in styles:
        print(f"    {s:<24} {len(by_style[s])} arhive")

    if args.list_only:
        print("\n(--list_only) Dacă numele de stiluri arată corect, rulează fără --list_only.")
        return

    if args.only_styles:
        want = {s.strip().lower() for s in args.only_styles.split(",")}
        styles = [s for s in styles if s.lower() in want]
        print(f"  filtrat la --only_styles: {styles}")
    if args.max_styles > 0:
        styles = styles[:args.max_styles]

    out = Path(args.out)
    total = 0
    for s in styles:
        sdir = out / s
        sdir.mkdir(parents=True, exist_ok=True)
        # alege prima arhivă (preferă cea cu '001')
        arcs = sorted(by_style[s], key=lambda f: ("001" not in f.path, f.path))
        arc = arcs[0]
        print(f"\n[stil] {s}: descarc {Path(arc.path).name} ...")
        with tempfile.TemporaryDirectory() as td:
            zpath = Path(td) / "part.zip"
            try:
                gdown.download(id=arc.id, output=str(zpath), quiet=False, use_cookies=False)
            except Exception as e:
                print(f"  [!] eșec descărcare: {e}")
                continue
            try:
                with zipfile.ZipFile(zpath) as zf:
                    members = [m for m in zf.namelist()
                               if Path(m).suffix.lower() in IMG_EXT and not m.endswith("/")]
                    members.sort()
                    random.shuffle(members)  # eșantion aleator din arhivă
                    taken = 0
                    for m in members:
                        if taken >= args.per_style:
                            break
                        data = zf.read(m)
                        dest = sdir / f"{taken:02d}_{Path(m).name}"
                        dest.write_bytes(data)
                        taken += 1
                    total += taken
                    print(f"  extras {taken} imagini din {len(members)} disponibile")
            except zipfile.BadZipFile:
                print(f"  [!] arhivă coruptă/incompletă pentru {s}")
            if args.keep_zip:
                import shutil
                shutil.copy(zpath, out / f"{s}.zip")
        # zip-ul temporar se șterge automat aici

    print(f"\n{total} imagini în {out}")
    print(f"Continuă: python build_mmis_manifest.py --mmis_root {out} --out ./mmis_eval --per_style {args.per_style}")


if __name__ == "__main__":
    main()
