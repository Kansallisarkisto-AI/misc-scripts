#!/usr/bin/env python3
"""Build and access a random-access NIST SD19 character store.

The store consists of two files:

    <output-prefix>.pkl   Small metadata/index dictionary
    <output-prefix>.bin   Flat uint8 image records

The index has the form:

    index[writer_id][character] -> list[image_index]

Every image has the same shape and occupies exactly ``width * height`` bytes.
Image ``i`` therefore starts at byte offset ``i * width * height``. The reader
uses ordinary Python ``seek`` and ``read`` calls; it does not load the image
array into RAM and does not use mmap/memmap.

Second-edition PNGs are streamed from ``by_write.zip``. Character labels are
read from an extracted first-edition ``by_write`` directory containing the
matching ``.cls`` files.
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import BinaryIO, DefaultDict, IO, TypeAlias
from zipfile import ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

CharacterRecord: TypeAlias = int
CharacterIndex: TypeAlias = dict[str, dict[str, list[CharacterRecord]]]

FORMAT_NAME = "nist-sd19-seek-character-store"
FORMAT_VERSION = 1
FIELD_RE = re.compile(r"^(?P<kind>[dulc])(?P<stem>.+)$", re.IGNORECASE)
IMAGE_RE = re.compile(
    r"^(?P<field>[dulc].+?)_(?P<index>\d+)\.png$", re.IGNORECASE
)
ASCII_ALNUM = frozenset(
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)

try:
    RESAMPLING = Image.Resampling
except AttributeError:  # pragma: no cover
    RESAMPLING = Image


def find_by_write_root(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.name == "by_write" and path.is_dir():
        return path
    if (path / "by_write").is_dir():
        return path / "by_write"
    if path.is_dir() and any(path.glob("hsf_*")):
        return path
    raise FileNotFoundError(f"Could not find by_write/hsf_* below {path}")


def _label_tokens(cls_path: Path) -> list[str]:
    lines = [
        line.strip()
        for line in cls_path.read_text("ascii", errors="strict").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError(f"Empty label file: {cls_path}")

    expected: int | None = None
    start = 0
    first = lines[0].split()[0]
    if first.isdecimal():
        expected = int(first)
        start = 1

    labels: list[str] = []
    for line_no, line in enumerate(lines[start:], start=start + 1):
        token = line.split()[0]
        try:
            labels.append(chr(int(token, 16)))
        except (ValueError, OverflowError) as exc:
            raise ValueError(
                f"Invalid hexadecimal label {token!r} in {cls_path}:{line_no}"
            ) from exc

    if expected is not None and len(labels) != expected:
        raise ValueError(
            f"{cls_path}: header says {expected} labels, found {len(labels)}"
        )
    return labels


def _find_cls(
    labels_root: Path,
    partition: str,
    writer_id: str,
    field_name: str,
) -> Path | None:
    candidates = (
        labels_root / partition / writer_id / f"{field_name}.cls",
        labels_root / "by_write" / partition / writer_id / f"{field_name}.cls",
    )
    return next((path for path in candidates if path.is_file()), None)


import cv2
import numpy as np
from PIL import Image, ImageOps

def normalize_character(
    source: BinaryIO,
    size: tuple[int, int] = (28, 28),
    content_size: int = 20,
) -> np.ndarray:
    """Decode one PNG and return a contiguous black-on-white uint8 image."""
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if content_size <= 0 or content_size > min(size):
        raise ValueError("content_size must be in 1..min(width, height)")

    with Image.open(source) as opened:
        opened.load()
        image = opened.convert("L").copy()

    array = np.asarray(image, dtype=np.uint8)
    border = np.concatenate(
        (array[0], array[-1], array[:, 0], array[:, -1])
    )

    if float(border.mean()) < 127.5:
        image = ImageOps.invert(image)

    bbox = ImageOps.invert(image).getbbox()
    if bbox is None:
        return np.full((height, width), 255, dtype=np.uint8)

    cropped = np.asarray(image.crop(bbox), dtype=np.uint8)

    cropped_height, cropped_width = cropped.shape
    scale = min(
        content_size / cropped_width,
        content_size / cropped_height,
    )
    resized_width = max(1, round(cropped_width * scale))
    resized_height = max(1, round(cropped_height * scale))

    resized = cv2.resize(
        cropped,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full((height, width), 255, dtype=np.uint8)

    left = (width - resized_width) // 2
    top = (height - resized_height) // 2

    canvas[
        top : top + resized_height,
        left : left + resized_width,
    ] = resized

    # gaussian blur
    canvas = cv2.GaussianBlur(
        canvas,
        ksize=(11, 11),
        sigmaX=0,
    )

    return np.ascontiguousarray(canvas)


def _parse_png_member(info: ZipInfo) -> tuple[str, str, str, int] | None:
    if info.is_dir():
        return None

    parts = PurePosixPath(info.filename).parts
    try:
        partition_pos = next(
            position
            for position, part in enumerate(parts)
            if part.startswith("hsf_")
        )
    except StopIteration:
        return None

    tail = parts[partition_pos:]
    if len(tail) != 4:
        return None

    partition, writer_id, field_name, filename = tail
    if FIELD_RE.fullmatch(field_name) is None:
        return None

    match = IMAGE_RE.fullmatch(filename)
    if match is None or match.group("field").lower() != field_name.lower():
        return None

    return partition, writer_id, field_name, int(match.group("index"))


def _store_paths(output_prefix: str | Path) -> tuple[Path, Path]:
    prefix = Path(output_prefix).expanduser().resolve()
    if prefix.suffix in {".pkl", ".bin"}:
        prefix = prefix.with_suffix("")
    return prefix.with_suffix(".pkl"), prefix.with_suffix(".bin")


def build_character_store(
    by_write_zip: str | Path,
    labels_root: str | Path,
    output_prefix: str | Path,
    *,
    size: tuple[int, int] = (28, 28),
    content_size: int = 20,
    allowed_characters: set[str] | None = None,
    strict: bool = True,
    overwrite: bool = False,
) -> dict:
    """Stream SD19 images into a flat binary file and return its small index."""
    zip_path = Path(by_write_zip).expanduser().resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"SD19 by_write ZIP does not exist: {zip_path}")

    labels = find_by_write_root(labels_root)
    index_path, binary_path = _store_paths(output_prefix)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        existing = [path for path in (index_path, binary_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "Output already exists; use --overwrite: "
                + ", ".join(str(path) for path in existing)
            )

    allowed = ASCII_ALNUM if allowed_characters is None else frozenset(allowed_characters)
    mutable: DefaultDict[str, DefaultDict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )

    label_cache: dict[tuple[str, str, str], list[str]] = {}
    seen_indices: DefaultDict[tuple[str, str, str], set[int]] = defaultdict(set)
    image_count = 0
    width, height = size
    record_size = width * height

    temporary_binary = binary_path.with_suffix(binary_path.suffix + ".tmp")
    try:
        with ZipFile(zip_path, "r") as archive, temporary_binary.open("wb") as image_file:
            members = [
                info for info in archive.infolist()
                if _parse_png_member(info) is not None
            ]
            members.sort(key=lambda info: info.header_offset)

            for info in tqdm(members, desc="SD19 PNGs", unit="image"):
                parsed = _parse_png_member(info)
                assert parsed is not None
                partition, writer_id, field_name, source_index = parsed
                field_key = (partition, writer_id, field_name)

                field_labels = label_cache.get(field_key)
                if field_labels is None:
                    cls_path = _find_cls(labels, partition, writer_id, field_name)
                    if cls_path is None:
                        if strict:
                            raise FileNotFoundError(
                                f"Missing {field_name}.cls for {partition}/{writer_id}"
                            )
                        label_cache[field_key] = []
                        continue
                    field_labels = _label_tokens(cls_path)
                    label_cache[field_key] = field_labels

                if source_index in seen_indices[field_key]:
                    if strict:
                        raise ValueError(
                            f"Duplicate PNG index {source_index} for "
                            f"{partition}/{writer_id}/{field_name}"
                        )
                    continue
                seen_indices[field_key].add(source_index)

                if source_index >= len(field_labels):
                    if strict:
                        raise ValueError(
                            f"PNG index {source_index} exceeds {len(field_labels)} "
                            f"labels for {partition}/{writer_id}/{field_name}"
                        )
                    continue

                character = field_labels[source_index]
                if character not in allowed:
                    continue

                with archive.open(info, "r") as member:
                    image = normalize_character(
                        member,
                        size=size,
                        content_size=content_size,
                    )

                payload = image.tobytes(order="C")
                if len(payload) != record_size:
                    raise RuntimeError(
                        f"Internal record-size error: {len(payload)} != {record_size}"
                    )
                image_file.write(payload)
                mutable[writer_id][character].append(image_count)
                image_count += 1

            image_file.flush()
            os.fsync(image_file.fileno())

        if strict:
            for field_key, field_labels in label_cache.items():
                expected = set(range(len(field_labels)))
                actual = seen_indices.get(field_key, set())
                if actual != expected:
                    missing = sorted(expected - actual)
                    extra = sorted(actual - expected)
                    partition, writer_id, field_name = field_key
                    raise ValueError(
                        f"Image/label mismatch for {partition}/{writer_id}/{field_name}: "
                        f"{len(actual)} PNGs, {len(field_labels)} labels; "
                        f"missing={missing[:10]}, extra={extra[:10]}"
                    )

        index: CharacterIndex = {
            writer: {
                character: records
                for character, records in sorted(characters.items())
            }
            for writer, characters in sorted(mutable.items())
        }
        metadata = {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "image_size": size,
            "dtype": "uint8",
            "record_size": record_size,
            "image_count": image_count,
            "binary_file": binary_path.name,
            "index": index,
            "source_zip": str(zip_path),
        }

        temporary_binary.replace(binary_path)
        temporary_index = index_path.with_suffix(index_path.suffix + ".tmp")
        with temporary_index.open("wb") as handle:
            pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_index.replace(index_path)
        return metadata
    except Exception:
        temporary_binary.unlink(missing_ok=True)
        raise


class SD19CharacterStore:
    """Random-access reader backed by ordinary file seeking."""

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path).expanduser().resolve()
        with self.index_path.open("rb") as handle:
            metadata = pickle.load(handle)

        if metadata.get("format") != FORMAT_NAME:
            raise ValueError(f"Not an SD19 seek store: {self.index_path}")
        if metadata.get("dtype") != "uint8":
            raise ValueError(f"Unsupported dtype: {metadata.get('dtype')!r}")

        self.metadata = metadata
        self.index: CharacterIndex = metadata["index"]
        self.width, self.height = map(int, metadata["image_size"])
        self.record_size = int(metadata["record_size"])
        self.image_count = int(metadata["image_count"])
        self.binary_path = self.index_path.parent / metadata["binary_file"]
        self._handle: IO[bytes] | None = None

        expected_size = self.image_count * self.record_size
        actual_size = self.binary_path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"Binary size mismatch: expected {expected_size}, found {actual_size}"
            )

    def _file(self) -> IO[bytes]:
        if self._handle is None or self._handle.closed:
            self._handle = self.binary_path.open("rb", buffering=0)
        return self._handle

    def read_array(self, image_index: int) -> np.ndarray:
        if not 0 <= image_index < self.image_count:
            raise IndexError(image_index)
        handle = self._file()
        handle.seek(image_index * self.record_size, os.SEEK_SET)
        payload = handle.read(self.record_size)
        if len(payload) != self.record_size:
            raise EOFError(
                f"Short read for image {image_index}: "
                f"expected {self.record_size}, got {len(payload)}"
            )
        return np.frombuffer(payload, dtype=np.uint8).reshape(
            self.height, self.width
        )

    def image(self, image_index: int) -> Image.Image:
        # frombuffer is backed by an immutable bytes object; copy creates a
        # fully detached Pillow image safe for later mutation.
        return Image.fromarray(self.read_array(image_index).copy(), mode="L")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "SD19CharacterStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state


def load_character_store(path: str | Path) -> SD19CharacterStore:
    return SD19CharacterStore(path)


def available_writers(store: SD19CharacterStore) -> tuple[str, ...]:
    return tuple(store.index)


def choose_character_records(
    text: str,
    writer: str,
    character_index: CharacterIndex,
    rng,
) -> dict[int, CharacterRecord] | None:
    positions = [
        (position, character)
        for position, character in enumerate(text)
        if character in ASCII_ALNUM
    ]

    required: DefaultDict[str, int] = defaultdict(int)
    for _position, character in positions:
        required[character] += 1

    writer_characters = character_index.get(str(writer), {})
    if any(not writer_characters.get(character) for character in required):
        return None

    pools: dict[str, list[int]] = {}
    cursors: dict[str, int] = {}
    for character in required:
        pools[character] = list(writer_characters[character])
        rng.shuffle(pools[character])
        cursors[character] = 0

    chosen: dict[int, int] = {}
    for text_position, character in positions:
        pool = pools[character]
        cursor = cursors[character]
        if cursor >= len(pool):
            previous = pool[-1]
            rng.shuffle(pool)
            if len(pool) > 1 and pool[0] == previous:
                pool[0], pool[1] = pool[1], pool[0]
            cursor = 0
        chosen[text_position] = pool[cursor]
        cursors[character] = cursor + 1
    return chosen


def character_image(
    store: SD19CharacterStore,
    image_index: int,
) -> Image.Image:
    return store.image(image_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("by_write_zip", type=Path)
    parser.add_argument("labels_root", type=Path)
    parser.add_argument(
        "output_prefix",
        type=Path,
        help="Output prefix; writes <prefix>.pkl and <prefix>.bin",
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument(
        "--content-size",
        type=int,
        default=92,
        help="Maximum character bounding-box dimension inside the canvas",
    )
    parser.add_argument(
        "--characters",
        default="".join(sorted(ASCII_ALNUM)),
        help="Exact characters to retain",
    )
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = build_character_store(
        args.by_write_zip,
        args.labels_root,
        args.output_prefix,
        size=(args.width, args.height),
        content_size=args.content_size,
        allowed_characters=set(args.characters),
        strict=not args.non_strict,
        overwrite=args.overwrite,
    )
    index_path, binary_path = _store_paths(args.output_prefix)
    print(
        f"Saved {store['image_count']:,} images from "
        f"{len(store['index']):,} writers\n"
        f"Index:  {index_path}\n"
        f"Images: {binary_path}"
    )


if __name__ == "__main__":
    main()
