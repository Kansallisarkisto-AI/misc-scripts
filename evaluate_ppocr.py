#!/usr/bin/env python3
"""
Evaluate a PP-OCRv6 recognition model (as a plain ONNX export) on a tab-separated
test file of the form:

    <relative/path/to/image.webp>\t<ground truth transcription>

Computes CER and WER using the HuggingFace `evaluate` library.

This uses the recognition-only building blocks (RecPreProcess + CTCLabelDecode)
from https://github.com/AIwork4me/ppocrv6_onnx directly, bypassing that
project's detection step entirely -- appropriate here since the test set is
already made of pre-cropped text-line images, not full pages.

Setup
-----
1. Get the custom model as ONNX (e.g. via `paddle2onnx` from your PaddleOCR
   training output -- this preserves the exact trained architecture, image
   size, and head size, unlike re-implementing it in another framework).
2. Get `ppocrv6_onnx.py` importable. Either:
     pip install git+https://github.com/AIwork4me/ppocrv6_onnx.git
   or clone it and point --ppocrv6-onnx-path at the directory containing
   ppocrv6_onnx.py:
     git clone https://github.com/AIwork4me/ppocrv6_onnx.git
3. Install the rest: pip install onnxruntime opencv-python numpy pyclipper evaluate jiwer

Example
-------
    python evaluate_ppocr.py \
        --rec-model /4tb_01/models/ppocrv6_multicentury/supermalli_onnx_opset21/inference.onnx \
        --char-dict /4tb_01/models/ppocrv6_multicentury/ppocrv6_dict.txt \
        --test-file /4tb_01/koodit/ppocrv6_onnx/supermalli_downloaded/supermalli_test_downloaded.txt \
        --img-height 96 --img-width-min 1536 \
        --ppocrv6-onnx-path ./ppocrv6_onnx \
        --batch-size 96 --prefer-accelerator --predictions-out supermalli_test_preds_minwidth1536.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a PP-OCRv6 ONNX recognition model (CER/WER) on a "
        "tab-separated test file, using ppocrv6_onnx's recognition pipeline."
    )
    p.add_argument(
        "--rec-model",
        required=True,
        help="Path to the recognition ONNX model (e.g. exported with paddle2onnx).",
    )
    p.add_argument(
        "--char-dict",
        required=True,
        help="Path to the character dictionary text file used to train this model "
        "(one character per line; matches PaddleOCR's character_dict_path).",
    )
    p.add_argument(
        "--test-file",
        required=True,
        help="Path to the tab-separated test file (image_path<TAB>transcription per line).",
    )
    p.add_argument(
        "--images-dir",
        default=None,
        help="Base directory the image paths in --test-file are relative to. "
        "Defaults to the directory containing --test-file.",
    )
    p.add_argument(
        "--img-height",
        type=int,
        default=48,
        help="Fixed recognition input height, must match training (e.g. 96 for a "
        "model trained with RecResizeImg image_shape [3, 96, 1280]). Default: 48 "
        "(the stock PP-OCRv6 recognizer height).",
    )
    p.add_argument(
        "--img-width-min",
        type=int,
        default=320,
        help="Minimum/base recognition width used when computing the per-batch "
        "target width from each image's aspect ratio (RecPreProcess's rec_image_shape "
        "width component). Default: 320.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of samples to evaluate (for quick smoke tests).",
    )
    p.add_argument(
        "--predictions-out",
        default=None,
        help="Optional path to write a TSV of image_path\\tprediction\\treference for inspection.",
    )
    p.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase both predictions and references before scoring.",
    )
    p.add_argument(
        "--prefer-accelerator",
        action="store_true",
        help="Allow ONNX Runtime to use CUDA/CoreML if available (default is CPU-only, "
        "for bit-exact reproducibility as recommended by ppocrv6_onnx).",
    )
    p.add_argument(
        "--use-space-char",
        dest="use_space_char",
        action="store_true",
        default=True,
        help="Append a trailing space character to the decoder vocabulary, matching "
        "PaddleOCR's Global.use_space_char=true behavior (dict file does not need to "
        "contain a space line itself -- PaddleOCR appends it automatically after "
        "blank + dict entries). Default: True.",
    )
    p.add_argument(
        "--no-use-space-char",
        dest="use_space_char",
        action="store_false",
        help="Disable the automatic trailing space class (use only if your model was "
        "trained with Global.use_space_char=false, or your dict file already ends "
        "with an explicit space line).",
    )
    p.add_argument(
        "--ppocrv6-onnx-path",
        default=None,
        help="Directory containing ppocrv6_onnx.py, if it's not pip-installed and not "
        "already on PYTHONPATH (e.g. a local `git clone` of the repo).",
    )
    return p.parse_args()


def load_test_pairs(test_file: str, images_dir: str, max_samples=None):
    """Read the TSV test file and resolve image paths against images_dir."""
    pairs = []
    test_file = os.path.expanduser(test_file)
    with open(test_file, "r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                print(
                    f"[warn] line {lineno} in {test_file} does not have a tab-separated "
                    f"image_path/transcription pair, skipping: {line!r}",
                    file=sys.stderr,
                )
                continue
            rel_path, transcription = parts
            image_path = os.path.join(images_dir, rel_path)
            pairs.append((image_path, transcription))
            if max_samples is not None and len(pairs) >= max_samples:
                break
    return pairs


def batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    args = parse_args()

    test_file = os.path.expanduser(args.test_file)
    images_dir = (
        os.path.expanduser(args.images_dir)
        if args.images_dir
        else str(Path(test_file).resolve().parent)
    )
    rec_model_path = os.path.expanduser(args.rec_model)
    char_dict_path = os.path.expanduser(args.char_dict)

    if not os.path.isfile(test_file):
        print(f"[error] test file not found: {test_file}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(images_dir):
        print(f"[error] images directory not found: {images_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(rec_model_path):
        print(f"[error] recognition ONNX model not found: {rec_model_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(char_dict_path):
        print(f"[error] character dictionary not found: {char_dict_path}", file=sys.stderr)
        sys.exit(1)

    if args.ppocrv6_onnx_path:
        sys.path.insert(0, os.path.expanduser(args.ppocrv6_onnx_path))

    try:
        from ppocrv6_onnx import RecPreProcess, CTCLabelDecode, _resolve_ort_providers
    except ImportError as e:
        print(
            "[error] could not import ppocrv6_onnx. Install it "
            "(pip install git+https://github.com/AIwork4me/ppocrv6_onnx.git) or pass "
            f"--ppocrv6-onnx-path pointing at a local clone. Original error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    import cv2
    import numpy as np
    import onnxruntime as ort
    import evaluate

    print(f"[info] test file:   {test_file}")
    print(f"[info] images dir:  {images_dir}")
    print(f"[info] rec model:   {rec_model_path}")
    print(f"[info] char dict:   {char_dict_path}")
    print(f"[info] rec shape:   (3, {args.img_height}, {args.img_width_min})")

    pairs = load_test_pairs(test_file, images_dir, args.max_samples)
    print(f"[info] loaded {len(pairs)} test samples")
    if not pairs:
        print("[error] no valid samples loaded, aborting.", file=sys.stderr)
        sys.exit(1)

    providers = _resolve_ort_providers(prefer_cpu=not args.prefer_accelerator)
    session = ort.InferenceSession(rec_model_path, providers=providers)
    input_name = session.get_inputs()[0].name

    rec_pre = RecPreProcess(rec_image_shape=(3, args.img_height, args.img_width_min))
    rec_post = CTCLabelDecode(char_dict_path)

    if args.use_space_char:
        # PaddleOCR's CTCLabelDecode builds its vocabulary as
        # blank + <dict file lines> + " " (space) when Global.use_space_char is
        # true -- the space is appended automatically, not stored in the dict
        # file. ppocrv6_onnx's CTCLabelDecode doesn't do this append, so we do
        # it here to match training. See BaseRecLabelDecode in PaddleOCR's
        # ppocr/postprocess/rec_postprocess.py for the reference behavior.
        if not rec_post._chars or rec_post._chars[-1] != " ":
            rec_post._chars = rec_post._chars + (" ",)

    print(f"[info] decoder vocab size (incl. blank{'+ space' if args.use_space_char else ''}): {rec_post.vocab_size}")

    model_num_classes = None
    for out in session.get_outputs():
        if out.shape and len(out.shape) == 3:
            model_num_classes = out.shape[-1]
            break
    if isinstance(model_num_classes, int) and model_num_classes != rec_post.vocab_size:
        print(
            f"[warn] ONNX model output classes ({model_num_classes}) != decoder vocab "
            f"size ({rec_post.vocab_size}). Decoding will likely crash or produce "
            f"garbage. Double-check --char-dict and --use-space-char/--no-use-space-char "
            f"against how this model was actually trained.",
            file=sys.stderr,
        )

    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    predictions = []
    references = []
    resolved_paths = []

    try:
        from tqdm import tqdm
        progress = tqdm(
            batched(pairs, args.batch_size),
            total=(len(pairs) + args.batch_size - 1) // args.batch_size,
            desc="Running inference",
        )
    except ImportError:
        progress = batched(pairs, args.batch_size)

    for batch in progress:
        images = []
        refs = []
        paths = []
        for image_path, transcription in batch:
            if not os.path.isfile(image_path):
                print(f"[warn] missing image, skipping: {image_path}", file=sys.stderr)
                continue
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR, as ppocrv6_onnx expects
            if img is None:
                print(f"[warn] failed to load {image_path}, skipping", file=sys.stderr)
                continue
            images.append(img)
            refs.append(transcription)
            paths.append(image_path)

        if not images:
            continue

        x = rec_pre(images)
        outputs = session.run(None, {input_name: x})
        texts, _scores = rec_post(outputs[0])

        for pred_text, ref, path in zip(texts, refs, paths):
            predictions.append(pred_text)
            references.append(ref)
            resolved_paths.append(path)

    if not predictions:
        print("[error] no predictions were produced, aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"[info] scoring {len(predictions)} samples")
    cer = cer_metric.compute(predictions=predictions, references=references)
    wer = wer_metric.compute(predictions=predictions, references=references)

    print("\n===== Results =====")
    print(f"Samples evaluated: {len(predictions)}")
    print(f"CER: {cer:.4f}")
    print(f"WER: {wer:.4f}")

    if args.predictions_out:
        out_path = os.path.expanduser(args.predictions_out)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["image_path", "prediction", "reference"])
            for path, pred, ref in zip(resolved_paths, predictions, references):
                writer.writerow([path, pred, ref])
        print(f"[info] wrote predictions to: {out_path}")


if __name__ == "__main__":
    main()