from fontTools.ttLib import TTFont
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path
import argparse
import os
import pickle

parser = argparse.ArgumentParser('arguments for the code')
parser.add_argument('--font_path', type=str, default="/tmp/fonts-filtered", help='Font root directory containing subdirectories of font families') 

args = parser.parse_args()

def extract_supported_characters(font_path):
    """
    Worker function — must be top-level for multiprocessing.
    Returns (font_path, supported_set)
    """
    try:
        font = TTFont(str(font_path), lazy=True)
    except Exception:
        return font_path, set()

    try:
        cmap = font["cmap"]
        supported = set()
        for table in cmap.tables:
            supported.update(table.cmap.keys())
        return font_path, supported
    except Exception:
        return font_path, set()
    finally:
        try:
            font.close()
        except Exception:
            pass


print("Font path:", args.font_path)
print("Exists:", os.path.exists(args.font_path))

fontdir_paths = [Path(x) for x in glob(f"{args.font_path}/*/", recursive=True)]
fontfile_paths = (
    [Path(x) for x in glob(f"{args.font_path}/*/*.ttf", recursive=True)] +
    [Path(x) for x in glob(f"{args.font_path}/*/*.otf", recursive=True)]
)

print(f"Found {len(fontfile_paths)} font files.")
print("Preprocessing fonts...")

font_to_supported_character = {}

max_workers = 32

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = [
        executor.submit(extract_supported_characters, font_path)
        for font_path in fontfile_paths
    ]

    for future in as_completed(futures):
        font_path, supported = future.result()
        font_to_supported_character[str(font_path)] = supported


with open("/tmp/font_cmap.pkl", "wb") as f:
    pickle.dump(font_to_supported_character, f)