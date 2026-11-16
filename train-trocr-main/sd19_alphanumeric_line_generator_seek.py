#!/usr/bin/env python3
"""Generate same-writer NIST SD19 alphanumeric text-line images with variable overlap.

For every generated sample this script:
  1. Chooses a random background image.
  2. Chooses a text string and an SD19 writer who has every required alphanumeric character.
  3. Randomly chooses an SD19 variant for each alphanumeric character, without reusing the
     same variant for repeated copies of a digit within the same line unless
     the writer has too few variants.
  4. Combines black-on-white digit images with a random amount of overlap
     using multiply blending.
  5. Resizes the digit line to the background height while preserving the
     line aspect ratio, resizes (stretches/squeezes) the background to exactly
     that resulting width, and blends the digit line on top.
  6. Saves lossless WebP images and TrOCR CSV metadata for train/val/test.
"""

from __future__ import annotations

import glob
import argparse
import csv
import multiprocessing as mp
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Apple Accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageEnhance, ImageOps
from tqdm import tqdm

import colorsys
from pathlib import Path

import cv2

# OpenCV's own parallel operations
cv2.setNumThreads(1)

from bisect import bisect_left, bisect_right

from numeric_samplers import sample_numeral
from sd19_character_index_seek import (
    CharacterIndex,
    character_image,
    choose_character_records,
    load_character_store,
)


IMAGE_EXTENSIONS = {
    ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"
}
SPLITS = ("train", "val", "test")

# Pillow 9/10 compatibility.
try:
    RESAMPLING = Image.Resampling
except AttributeError:  # pragma: no cover - for older Pillow versions
    RESAMPLING = Image


CharacterRecord = int  # image index in the preprocessed SD19 store
SymbolIndex = Dict[str, List[Path]]
CHARACTER_SIZE = (128, 128)

# Per-process state, initialized once by _init_worker.
_WORKER_SD19_STORE: object | None = None
_WORKER_CHARACTER_INDEX: CharacterIndex | None = None
_WORKER_SYMBOL_INDEX: SymbolIndex = {}
_WORKER_BACKGROUNDS: Sequence[Path] = ()
_WORKER_CONFIG: dict = {}

# Reusable single-process state for generate_sample_image().
_SAMPLE_SD19_STORE_CACHE: dict[str, object] = {}
_SAMPLE_SYMBOL_INDEX_CACHE: dict[str, SymbolIndex] = {}
_SAMPLE_BACKGROUNDS_CACHE: dict[str, tuple[Path, ...]] = {}
_SAMPLE_WRITERS_CACHE = None

_DIDA_INDEX_CACHE = None

DidaDict = Dict[str, List]  # digit label, list of records

class DidaIndex:
    '''Index to store DIDA characters and metadata'''
    def __init__(self, filenames):
        dida_dict = {}
        for filename in filenames:
            fields = os.path.basename(filename).split("_")
            label = fields[0]
            index = int(fields[1])
            scale = float(fields[2])
            confidence = float(fields[3].split(".")[0])

            if label not in dida_dict:
                dida_dict[label] = []
            else:
                dida_dict[label].append({"label": label, "index": index, "scale": scale,
                                         "confidence": confidence, "filename": filename})

        for key in dida_dict.keys():
            dida_dict[key] = sorted(dida_dict[key], key=lambda x: x["scale"])  # sort by scale

        self.dida_dict = dida_dict

    def get_random_scale(self, rng, digit):
        return rng.choice(self.dida_dict[digit])["scale"]

    def get_digit(self, rng, digit, scale, scale_slop=0.15):
        """Sample randomly from a given digit type, limited between scale*(1-scale_slop) and scale*(1+scale_slop)"""
        scale_low = scale * (1-scale_slop)
        scale_high = scale * (1+scale_slop)
        left_index = bisect_left(self.dida_dict[digit], scale_low, key=lambda x: x["scale"])
        right_index = bisect_right(self.dida_dict[digit], scale_high, key=lambda x: x["scale"])

        values = self.dida_dict[digit][left_index:right_index]
        if not values:
            values = self.dida_dict[digit]

        # return black on white image
        inverted = ImageOps.invert(Image.open(rng.choice(values)["filename"]))

        # Pillow image -> NumPy array
        composed_array = np.asarray(inverted)

        # Gaussian blur
        blurred_array = cv2.GaussianBlur(
            composed_array,
            ksize=(5, 5),
            sigmaX=0,
        )

        # NumPy array -> Pillow image
        composed = Image.fromarray(blurred_array)
        return composed


def load_writers(path: Path) -> List[str]:
    """Load SD19 writer directory IDs from the first CSV column."""
    writers: List[str] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        first = next(reader, None)
        if first is None:
            return writers
        if first and first[0].strip().lower() not in {"writer", "writer_id"}:
            writers.append(first[0].strip())
        for row in reader:
            if row and row[0].strip():
                writers.append(row[0].strip())
    if not writers:
        raise ValueError(f"No writer IDs found in {path}")
    return writers


def load_first_column(path: Path) -> List[str]:
    """Load values from the first column, tolerating an optional header."""
    values: List[str] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        first = next(reader, None)
        if first is None:
            return values

        if first and first[0].strip():
            token = first[0].strip()
            # Known resource headers are non-numeric; data values begin numeric.
            if token[0].isdigit() or token[0] in "+-.":
                values.append(token)

        for row in reader:
            if row and row[0].strip():
                values.append(row[0].strip())

    if not values:
        raise ValueError(f"No data values found in {path}")
    return values


def find_backgrounds(directory: Path) -> List[Path]:
    """Recursively find supported background images."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Background directory does not exist: {directory}")
    paths = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported background images found in {directory}")
    return paths


def build_symbol_index(curated_dir: Path) -> SymbolIndex:
    """Index curated symbols stored under ``curated/<decimal ASCII code>/``.

    Directory names are interpreted as decimal code points. This makes the
    loader forward compatible with any ASCII character that appears in a
    generated string, not only comma and period.
    """
    if not curated_dir.is_dir():
        raise FileNotFoundError(f"Curated symbol directory does not exist: {curated_dir}")

    index: SymbolIndex = {}
    for child in curated_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            code_point = int(child.name, 10)
            character = chr(code_point)
        except (ValueError, OverflowError):
            continue
        if not character.isascii():
            continue
        images = sorted(
            path for path in child.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            index[character] = images
    return index


def curated_symbol_as_black_on_white(
    path: Path,
    crop: bool = False,
    erode=False
) -> Image.Image:
    """Load a curated symbol and scale it to the character height."""
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composed = Image.alpha_composite(white, rgba).convert("L")

    array = np.asarray(composed, dtype=np.uint8)
    border = np.concatenate(
        (array[0], array[-1], array[:, 0], array[:, -1])
    )

    # if float(border.mean()) < 127.5:
    composed = Image.fromarray(255 - array, mode="L")

    if erode:
        composed = composed #.filter(ImageFilter.MinFilter(1))
    else:
        # MinFilter expands dark regions.
        composed = composed.filter(ImageFilter.MinFilter(5))

    # add 10px padding on both left and right sides
    composed = ImageOps.expand(composed, border=(10, 0, 10, 0), fill=255)

    if crop:
        # Keep this fraction of the original empty side margins.
        empty_space_keep = 0.25

        ink_mask = ImageOps.invert(composed)
        bbox = ink_mask.getbbox()

        if bbox is not None:
            ink_left, _top, ink_right, _bottom = bbox

            left_empty = ink_left
            right_empty = composed.width - ink_right

            crop_left = round(left_empty * (1.0 - empty_space_keep))
            crop_right = composed.width - round(
                right_empty * (1.0 - empty_space_keep)
            )

            composed = composed.crop(
                (crop_left, 0, crop_right, composed.height)
            )

    # Scale to the desired height while preserving aspect ratio.
    target_height = CHARACTER_SIZE[1]
    target_width = max(
        1,
        round(composed.width * (target_height / composed.height)),
    )

    composed = composed.resize(
        (target_width, target_height),
        RESAMPLING.BICUBIC,
    )

    # Pillow image -> NumPy array
    composed_array = np.asarray(composed)

    # Gaussian blur
    if erode:
        blurred_array = cv2.GaussianBlur(
            composed_array,
            ksize=(5, 5),
            sigmaX=0,
        )
    else:
        blurred_array = cv2.GaussianBlur(
            composed_array,
            ksize=(11, 11),
            sigmaX=0,
        )

    # NumPy array -> Pillow image
    composed = Image.fromarray(blurred_array)

    return composed

EMPTY_IMAGE = Image.new("L", CHARACTER_SIZE, 255)

def render_text_characters(
    text: str,
    character_records: Mapping[int, CharacterRecord],
    sd19_store: object,
    symbol_index: SymbolIndex,
    dida_index: DidaIndex,
    rng: random.Random,
    use_dida_digits=False,
) -> List[Image.Image]:
    """Render letters/digits from one SD19 writer and punctuation from curated PNGs."""
    images: List[Image.Image] = []

    dida_scale = None
    
    for position, character in enumerate(text):
        if character.isnumeric() and use_dida_digits:
            if dida_scale is None:
                dida_scale = dida_index.get_random_scale(rng, character)

            images.append(dida_index.get_digit(rng, character, dida_scale))
            continue
        elif character.isascii() and character.isalnum():
            image_index = character_records[position]
            images.append(character_image(sd19_store, image_index))
            continue

        if character == " ":
            images.append(EMPTY_IMAGE.copy())
            continue

        paths = symbol_index.get(character)
        if not paths:
            code = ord(character)
            raise ValueError(
                f"No curated images found for character {character!r} "
                f"(decimal code {code}) in curated/{code}/"
            )
        crop = 39 <= ord(character) <= 47 and character not in "-+"
        images.append(curated_symbol_as_black_on_white(rng.choice(paths), crop=crop, erode=use_dida_digits))
    return images


def make_overlapped_line(
    character_images: Sequence[Image.Image],
    min_overlap: float,
    max_overlap: float,
    rng: random.Random,
    max_rotation_degrees: float = 10.0,
    max_line_slope: float = 0.15,
    max_curve_amplitude: float = 6.0,
    curve_probability: float = 0.5,
) -> Tuple[Image.Image, List[float]]:
    """Combine characters with correlated overlap, sloped/curved placement,
    and rotation.

    Most lines follow a nearly straight, rotated baseline. Sometimes, a
    parabolic component is mixed into that baseline (curve_probability). 
    Vertical placement is calculated before character rotation.
    """
    if not character_images:
        raise ValueError("At least one character image is required")

    heights = {image.height for image in character_images}
    if len(heights) != 1:
        raise ValueError("All character images must have the same height")

    if not 0.0 <= curve_probability <= 1.0:
        raise ValueError("curve_probability must be between 0 and 1")

    # Each line has its own typical overlap and character rotation.
    line_overlap_mode = rng.uniform(min_overlap, max_overlap)
    line_rotation_mode = rng.uniform(
        -max_rotation_degrees,
        max_rotation_degrees,
    )

    # --------------------------------------------------------------
    # 1. Calculate horizontal positions using the original images.
    # --------------------------------------------------------------
    x_positions = [0]
    overlaps: List[float] = []

    for previous in character_images[:-1]:
        overlap_fraction = rng.triangular(
            min_overlap,
            max_overlap,
            line_overlap_mode,
        )

        overlap_pixels = int(round(previous.width * overlap_fraction))
        advance = max(1, previous.width - overlap_pixels)

        x_positions.append(x_positions[-1] + advance)
        overlaps.append(overlap_fraction)

    line_width = x_positions[-1] + character_images[-1].width

    # --------------------------------------------------------------
    # 2. Select the baseline shape.
    #
    # Every line has a linear slope. Most lines have no parabola.
    # When enabled, the parabola is added to the linear baseline.
    # --------------------------------------------------------------
    line_slope = rng.uniform(-max_line_slope, max_line_slope)

    if rng.random() < curve_probability:
        curve_amplitude = rng.uniform(
            -max_curve_amplitude,
            max_curve_amplitude,
        )
    else:
        curve_amplitude = 0.0

    original_height = character_images[0].height
    base_center_y = original_height / 2.0
    line_center_x = line_width / 2.0

    character_centers: List[Tuple[float, float]] = []

    for image, x in zip(character_images, x_positions):
        center_x = x + image.width / 2.0

        # Linear component: produces a mostly straight, sloped baseline.
        linear_offset = line_slope * (center_x - line_center_x)

        # Existing parabolic component:
        # zero at both ends and strongest at the center.
        if line_width > 1:
            t = center_x / (line_width - 1)
        else:
            t = 0.5

        u = 2.0 * t - 1.0
        parabolic_offset = curve_amplitude * (u * u - 1.0)

        center_y = (
            base_center_y
            + linear_offset
            + parabolic_offset
        )

        character_centers.append((center_x, center_y))

    # --------------------------------------------------------------
    # 3. Rotate after calculating the vertical placement.
    # --------------------------------------------------------------
    rotated_images: List[Image.Image] = []

    for image in character_images:
        angle = rng.triangular(
            -max_rotation_degrees,
            max_rotation_degrees,
            line_rotation_mode,
        )

        rotated = image.rotate(
            angle,
            resample=RESAMPLING.BICUBIC,
            expand=True,
            fillcolor=255,
        )
        rotated_images.append(rotated)

    # --------------------------------------------------------------
    # 4. Calculate a canvas that contains every rotated character.
    # --------------------------------------------------------------
    min_x = min(
        center_x - image.width / 2.0
        for image, (center_x, center_y)
        in zip(rotated_images, character_centers)
    )
    min_y = min(
        center_y - image.height / 2.0
        for image, (center_x, center_y)
        in zip(rotated_images, character_centers)
    )
    max_x = max(
        center_x + image.width / 2.0
        for image, (center_x, center_y)
        in zip(rotated_images, character_centers)
    )
    max_y = max(
        center_y + image.height / 2.0
        for image, (center_x, center_y)
        in zip(rotated_images, character_centers)
    )

    width = max(1, int(np.ceil(max_x - min_x)))
    height = max(1, int(np.ceil(max_y - min_y)))

    line = Image.new("L", (width, height), 255)

    # --------------------------------------------------------------
    # 5. Paste each rotated image around its precomputed center.
    # --------------------------------------------------------------
    for rotated, (center_x, center_y) in zip(
        rotated_images,
        character_centers,
    ):
        paste_x = int(round(
            center_x - rotated.width / 2.0 - min_x
        ))
        paste_y = int(round(
            center_y - rotated.height / 2.0 - min_y
        ))

        layer = Image.new("L", line.size, 255)
        layer.paste(rotated, (paste_x, paste_y))
        line = ImageChops.multiply(line, layer)

    return line, overlaps

def make_overlapped_line_old(
    character_images: Sequence[Image.Image],
    min_overlap: float,
    max_overlap: float,
    rng: random.Random,
) -> Tuple[Image.Image, List[float]]:
    """Combine characters using line-correlated triangular overlap."""
    if not character_images:
        raise ValueError("At least one character image is required")

    heights = {image.height for image in character_images}
    if len(heights) != 1:
        raise ValueError("All character images must have the same height")

    # Selected once per line, so different lines have different typical spacing.
    line_overlap_mode = rng.uniform(min_overlap, max_overlap)

    x_positions = [0]
    overlaps: List[float] = []

    for previous in character_images[:-1]:
        # Values within this line cluster around line_overlap_mode.
        overlap_fraction = rng.triangular(
            min_overlap,
            max_overlap,
            line_overlap_mode,
        )

        overlap_pixels = int(round(previous.width * overlap_fraction))
        advance = max(1, previous.width - overlap_pixels)

        x_positions.append(x_positions[-1] + advance)
        overlaps.append(overlap_fraction)

    width = x_positions[-1] + character_images[-1].width
    height = character_images[0].height
    line = Image.new("L", (width, height), 255)

    for character_image, x in zip(character_images, x_positions):
        layer = Image.new("L", line.size, 255)
        layer.paste(character_image, (x, 0))
        line = ImageChops.multiply(line, layer)

    return line, overlaps

def random_erode_or_dilate_line(
    image: Image.Image,
    probability: float = 0.3,
    rng: random.Random | None = None,
    using_dida=False
) -> Image.Image:
    """
    Randomly alter dark linework.

    The probability controls whether morphology is applied. Once activated,
    erosion or dilation is selected with equal probability.
    """
    rng = rng or random

    if rng.random() >= probability:
        return image

    if rng.random() < 0.5:
        if using_dida:  # dida is already thinner, don't erode too much
            radius = rng.randint(1, 2)
        else:
            radius = rng.randint(1, 4)

        # Pillow filter sizes must be odd: radius 1 -> 3x3, radius 2 -> 5x5.
        kernel_size = radius * 2 + 1
        # Erode dark regions, thinning the linework.
        return image.filter(ImageFilter.MaxFilter(kernel_size))

    # Dilate dark regions, thickening the linework.
    radius = rng.randint(1, 2)

    # Pillow filter sizes must be odd: radius 1 -> 3x3, radius 2 -> 5x5.
    kernel_size = radius * 2 + 1
    return image.filter(ImageFilter.MinFilter(kernel_size))

def random_pen_color(rng: random.Random) -> tuple[int, int, int]:
    """
    Generate a mostly gray pen color with an intensity mode of 96.

    The base gray intensity follows a triangular distribution over 0...255,
    peaking at 96. A small amount of saturation adds subtle color variation.
    """
    # Values span 0...255, with 96 being the most likely intensity.
    value = rng.triangular(0, 255, 96) / 255

    hue = rng.random()

    # Strongly favor low saturation while permitting occasional visible color.
    saturation = rng.random() ** 2.5 * 0.30

    red, green, blue = colorsys.hsv_to_rgb(
        hue,
        saturation,
        value,
    )

    return (
        round(red * 255),
        round(green * 255),
        round(blue * 255),
    )


def blend_line_on_background(
    line: Image.Image,
    background_path: Path,
    min_background_scale: float,
    max_background_scale: float,
    rng: random.Random,
    using_dida=False
) -> Image.Image:
    """
    Center a line on a scaled background with a bounds-safe random offset.

    The grayscale line is inverted into an ink mask. A continuously sampled pen
    color is multiplied into the background so the underlying paper texture
    remains visible.
    """
    if line.width <= 0 or line.height <= 0:
        raise ValueError("The line image must have positive dimensions.")

    if min_background_scale <= 0 or max_background_scale <= 0:
        raise ValueError("Background scale values must be positive.")

    if min_background_scale > max_background_scale:
        raise ValueError(
            "min_background_scale cannot exceed max_background_scale."
        )

    # erode or dilate line (in original resolution)
    line = random_erode_or_dilate_line(
        line,
        probability=0.5,
        rng=rng,
        using_dida=using_dida
    ).convert("L")

    with Image.open(background_path) as source:
        background = source.convert("RGB")
        transposes = (  None,
                        Image.Transpose.ROTATE_180,
                        Image.Transpose.FLIP_LEFT_RIGHT,
                        Image.Transpose.FLIP_TOP_BOTTOM,
                        Image.Transpose.TRANSPOSE,
                        Image.Transpose.TRANSVERSE)
        choice = rng.choice(transposes)
        if choice:
            background = background.transpose(choice)

    target_height = background.height
    target_width = max(
        1,
        round(line.width * target_height / line.height),
    )

    resized_line = line.convert("L").resize(
        (target_width, target_height),
        RESAMPLING.BICUBIC,
    )

    scale = rng.uniform(
        min_background_scale,
        max_background_scale,
    )

    # The background must be at least as large as the line so every generated
    # offset can keep the complete line within the output image.
    '''background_width = max(
        target_width,
        round(target_width * scale),
    )
    background_height = max(
        target_height,
        round(target_height * scale),
    )'''
    background_width = round(target_width * scale)
    background_height = round(target_height * scale)

    # random stretch
    if rng.random() < 0.5:
        current_ratio = background_width / background_height
        desired_ratio = rng.uniform(current_ratio, max(current_ratio, 5.0))

        width_factor = max(
            1.05,
            desired_ratio / current_ratio,
        )

        background_width = max(
            target_width,
            round(background_width * width_factor),
        )

    fitted_background = background.resize(
        (background_width, background_height),
        RESAMPLING.BICUBIC,
    )

    centered_x = (background_width - target_width) // 2
    centered_y = (background_height - target_height) // 2

    offset_x = rng.randint(
        min(
            -centered_x,
            background_width - target_width - centered_x,
        ),
        max(
            -centered_x,
            background_width - target_width - centered_x,
        ),
    )

    offset_y = rng.randint(
        min(
            -centered_y,
            background_height - target_height - centered_y,
        ),
        max(
            -centered_y,
            background_height - target_height - centered_y,
        ),
    )

    line_x = centered_x + offset_x
    line_y = centered_y + offset_y

    result = fitted_background.copy()

    region_box = (
        line_x,
        line_y,
        line_x + target_width,
        line_y + target_height,
    )
    background_region = result.crop(region_box)

    # The source is assumed to contain dark ink on a light background.
    # Convert it into a mask where:
    #   white = pen
    #   black = transparent
    ink_mask = ImageOps.invert(resized_line.convert("L"))

    # Optional nonlinear strengthening.
    ink_mask = ink_mask.point(
        lambda value: round(255 * (value / 255) ** 0.5)
    )

    # Alpha controls the overall pen strength.
    alpha = max(0.0, min(1.0, rng.triangular(0.95, 1.0, 0.99)))

    ink_mask = ink_mask.point(
        lambda value: round(value * alpha)
    )

    pen_color = random_pen_color(rng)

    # Create a solid pen-colored image and use the ink mask as its alpha channel.
    pen_layer = Image.new(
        "RGBA",
        resized_line.size,
        (*pen_color, 255),
    )
    pen_layer.putalpha(ink_mask)

    # Alpha-blend the colored pen directly over the background.
    blended_region = Image.alpha_composite(
        background_region.convert("RGBA"),
        pen_layer,
    ).convert(background_region.mode)

    result.paste(
        blended_region,
        (line_x, line_y),
    )

    return result


def make_numeral_source(
    domain: str,
    zip_code_file: Path,
    check_amount_file: Path,
) -> Sequence[str] | None:
    """Load the reusable numeral source for file-backed domains."""
    if domain == "zip_code":
        return [str(value).zfill(5) for value in load_first_column(zip_code_file)]
    if domain == "check_amount":
        return load_first_column(check_amount_file)
    return None


def sample_numeral_old(
    domain: str,
    source: Sequence[str] | None,
    rng: random.Random,
    min_digits: int,
    max_digits: int,
    min_decimal_places: int,
    max_decimal_places: int,
) -> str:
    """Sample a numeral matching the behavior of the original generator."""
    if domain == "zip_code":
        assert source is not None
        return str(rng.choice(source)).zfill(5)

    if domain == "check_amount":
        assert source is not None
        amount = float(rng.choice(source))
        rendered = f"{amount:.2f}"
        return rendered.replace(".", "")

    if domain == "clock_time":
        minutes = rng.randrange(0, 1440)
        return f"{minutes // 60:d}{minutes % 60:02d}"

    if domain == "decimal":
        integer_length = rng.randint(min_digits, max_digits)
        decimal_places = rng.randint(min_decimal_places, max_decimal_places)
        if integer_length == 1:
            integer_part = str(rng.randrange(10))
        else:
            integer_part = str(rng.randrange(1, 10)) + "".join(
                str(rng.randrange(10)) for _ in range(integer_length - 1)
            )
        fractional_part = "".join(
            str(rng.randrange(10)) for _ in range(decimal_places)
        )
        separator = rng.choice((".", ","))
        return f"{integer_part}{separator}{fractional_part}"

    length = rng.randint(min_digits, max_digits)
    return "".join(str(rng.randrange(10)) for _ in range(length))



def sample_text(
    rng: random.Random,
    *,
    domain: str = "alphanumeric",
    alphabet: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    min_length: int = 3,
    max_length: int = 7,
) -> str:
    """Sample text; preserve the existing numeric sampler for numeric domains."""
    if domain != "alphanumeric":
        return sample_numeral(rng)
    if not alphabet:
        raise ValueError("alphabet must not be empty")
    length = rng.randint(min_length, max_length)
    return "".join(rng.choice(alphabet) for _ in range(length))

def relative_output_path(path: Path) -> str:
    """Return a portable path relative to the current working directory."""
    return Path(os.path.relpath(path.resolve(), Path.cwd().resolve())).as_posix()



def generate_sample_image(
    *,
    seed: int | None = None,
    sd19_index_file: Path = Path("sd19.pkl"),
    backgrounds_dir: Path = Path("backgrounds"),
    curated_dir: Path = Path("curated"),
    dida_dir: Path = Path("/home/makelajo/dida_segmented"),
    domain: str = "numeric",
    alphabet: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    min_length: int = 3,
    max_length: int = 7,
    min_overlap: float = 0.0,
    max_overlap: float = 0.6,
    min_background_scale: float = 1-(8.0/28),
    max_background_scale: float = 2.0,
    max_attempts: int = 100,
    dida_use_probability = 0.5
) -> tuple[Image.Image, str]:
    """Generate one sample image and its text without using CLI arguments.

    Expensive, read-only resources are cached at module level after the first
    call.  The returned image is independent of the cache and may be modified
    freely by the caller.

    Args:
        seed: Optional deterministic seed. ``None`` uses fresh randomness.
        sd19_index_file: Seek-based SD19 index from sd19_character_index_seek.py.
        backgrounds_dir: Directory recursively containing background images.
        curated_dir: Directory containing ASCII symbols under decimal-code
            subdirectories, for example ``curated/46`` for ``.``.
        writers_file: CSV whose first column contains eligible SD19 writers.
        domain: "alphanumeric" or a domain supported by numeric_samplers.
        alphabet: Characters sampled by the alphanumeric domain.
        min_length: Minimum alphanumeric string length.
        max_length: Maximum alphanumeric string length.
        min_overlap: Minimum adjacent-character overlap fraction.
        max_overlap: Maximum adjacent-character overlap fraction.
        min_background_scale: Minimum background scale.
        max_background_scale: Maximum background scale.
        max_attempts: Maximum numeral/writer combinations to try.

    Returns:
        A ``(PIL.Image.Image, text)`` tuple.
    """
    if not 0.0 <= min_overlap <= max_overlap < 1.0:
        raise ValueError("Require 0 <= min_overlap <= max_overlap < 1")
    if min_background_scale <= 0 or max_background_scale < min_background_scale:
        raise ValueError(
            "Require 0 < min_background_scale <= max_background_scale"
        )
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    global _DIDA_INDEX_CACHE
    if _DIDA_INDEX_CACHE is None:
        filenames = glob.glob(str(dida_dir) + "/*/*.png")
        _DIDA_INDEX_CACHE = DidaIndex(filenames)

    dida_index = _DIDA_INDEX_CACHE

    sd19_index_file = sd19_index_file.expanduser().resolve()
    backgrounds_dir = backgrounds_dir.expanduser().resolve()
    curated_dir = curated_dir.expanduser().resolve()

    store_key = str(sd19_index_file)
    sd19_store = _SAMPLE_SD19_STORE_CACHE.get(store_key)
    if sd19_store is None:
        sd19_store = load_character_store(sd19_index_file)
        _SAMPLE_SD19_STORE_CACHE[store_key] = sd19_store
        print(
            f"Opened {sd19_store.image_count} SD19 characters from "
            f"{len(sd19_store.index)} writers"
        )
    character_index: CharacterIndex = sd19_store.index

    backgrounds_key = str(backgrounds_dir)
    backgrounds = _SAMPLE_BACKGROUNDS_CACHE.get(backgrounds_key)
    if backgrounds is None:
        backgrounds = tuple(find_backgrounds(backgrounds_dir))
        _SAMPLE_BACKGROUNDS_CACHE[backgrounds_key] = backgrounds
        print(f"Indexed {len(backgrounds)} backgrounds")

    symbols_key = str(curated_dir)
    symbol_index = _SAMPLE_SYMBOL_INDEX_CACHE.get(symbols_key)
    if symbol_index is None:
        symbol_index = build_symbol_index(curated_dir)
        _SAMPLE_SYMBOL_INDEX_CACHE[symbols_key] = symbol_index

        print(f"Indexed {len(symbol_index)} curated ASCII symbols")

    global _SAMPLE_WRITERS_CACHE
    writers = _SAMPLE_WRITERS_CACHE
    if writers is None:
        writers = tuple(character_index.keys())
        _SAMPLE_WRITERS_CACHE = writers

    rng = random.Random(seed)
    for _attempt in range(max_attempts):
        numeral = sample_text(
            rng, domain=domain, alphabet=alphabet,
            min_length=min_length, max_length=max_length,
        )
        writer_id = rng.choice(writers)
        records = choose_character_records(numeral, writer_id, character_index, rng)
        if records is None:
            continue

        use_dida = rng.random() < dida_use_probability
        
        try:
            character_images = render_text_characters(
                numeral,
                records,
                sd19_store,
                symbol_index,
                dida_index,
                rng,
                use_dida_digits=use_dida
            )
        except ValueError as error:
            raise RuntimeError(
                f"Cannot render generated text {numeral!r}: {error}"
            ) from error

        line, _overlaps = make_overlapped_line(
            character_images,
            min_overlap,
            max_overlap,
            rng,
        )
        image = blend_line_on_background(
            line,
            rng.choice(backgrounds),
            min_background_scale,
            max_background_scale,
            rng,
            using_dida=use_dida
        )
        return image, numeral

    raise RuntimeError(
        f"Could not generate a sample after {max_attempts} attempts. "
        "Check writer coverage and curated symbol availability."
    )


def _init_worker(
    sd19_index_file: str,
    symbol_index: SymbolIndex,
    backgrounds: Sequence[Path],
    config: dict,
) -> None:
    """Load the read-only preprocessed SD19 store once per worker."""
    global _WORKER_SD19_STORE, _WORKER_CHARACTER_INDEX
    global _WORKER_SYMBOL_INDEX, _WORKER_BACKGROUNDS, _WORKER_CONFIG

    _WORKER_SD19_STORE = load_character_store(sd19_index_file)
    _WORKER_CHARACTER_INDEX = _WORKER_SD19_STORE.index
    _WORKER_SYMBOL_INDEX = symbol_index
    _WORKER_BACKGROUNDS = backgrounds
    _WORKER_CONFIG = config


def _generate_sample(task: Tuple[int, int, str, Sequence[str], Path]) -> Tuple[int, str, str]:
    """Generate and save one sample; return metadata to the parent process."""
    sample_number, seed, split, writers, lines_dir = task
    rng = random.Random(seed)

    if _WORKER_SD19_STORE is None or _WORKER_CHARACTER_INDEX is None:
        raise RuntimeError("Worker was not initialized")

    config = _WORKER_CONFIG
    max_attempts = config["max_attempts_per_sample"]
    for _attempt in range(max_attempts):
        numeral = sample_text(
            rng,
            domain=config["domain"],
            alphabet=config["alphabet"],
            min_length=config["min_digits"],
            max_length=config["max_digits"],
        )
        '''numeral = sample_numeral(
            config["domain"],
            config["numeral_source"],
            rng,
            config["min_digits"],
            config["max_digits"],
            config["min_decimal_places"],
            config["max_decimal_places"],
        )'''
        writer_id = rng.choice(writers)
        records = choose_character_records(
            numeral, writer_id, _WORKER_CHARACTER_INDEX, rng
        )
        if records is None:
            continue

        try:
            character_images = render_text_characters(
                numeral, records, _WORKER_SD19_STORE, _WORKER_SYMBOL_INDEX, rng
            )
        except ValueError as error:
            raise RuntimeError(
                f"Cannot render generated text {numeral!r}: {error}"
            ) from error

        line, _overlaps = make_overlapped_line(
            character_images,
            config["min_overlap"],
            config["max_overlap"],
            rng,
        )
        background_path = rng.choice(_WORKER_BACKGROUNDS)
        composed = blend_line_on_background(
            line,
            background_path,
            config["min_background_scale"],
            config["max_background_scale"],
            rng,
        )

        image_path = lines_dir / f"{sample_number}.webp"
        composed.save(image_path, format="WEBP", lossless=True, method=4)
        return sample_number, relative_output_path(image_path), numeral

    raise RuntimeError(
        f"Could not generate sample {sample_number} for {split} after "
        f"{max_attempts} attempts. Check writer coverage for the chosen domain."
    )


def generate_split(
    split: str,
    count: int,
    writers: Sequence[str],
    backgrounds: Sequence[Path],
    output_dir: Path,
    sd19_index_file: Path,
    symbol_index: SymbolIndex,
    domain: str,
    alphabet: str,
    numeral_source: Sequence[str] | None,
    min_digits: int,
    max_digits: int,
    min_decimal_places: int,
    max_decimal_places: int,
    min_overlap: float,
    max_overlap: float,
    min_background_scale: float,
    max_background_scale: float,
    seed: int,
    workers: int,
    chunksize: int,
) -> None:
    """Generate one split in parallel and write ordered TrOCR metadata."""
    lines_dir = output_dir / f"{split}-lines"
    trocr_dir = output_dir / f"{split}-trocr"
    lines_dir.mkdir(parents=True, exist_ok=True)
    trocr_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = trocr_dir / "data.csv"
    config = {
        "domain": domain,
        "alphabet": alphabet,
        "numeral_source": numeral_source,
        "min_digits": min_digits,
        "max_digits": max_digits,
        "min_decimal_places": min_decimal_places,
        "max_decimal_places": max_decimal_places,
        "min_overlap": min_overlap,
        "max_overlap": max_overlap,
        "min_background_scale": min_background_scale,
        "max_background_scale": max_background_scale,
        "max_attempts_per_sample": 100,
    }

    # A separate deterministic seed per output index makes results independent
    # of worker scheduling and chunksize.
    seed_rng = random.Random(f"{seed}:{split}")
    tasks = [
        (sample_number, seed_rng.getrandbits(64), split, writers, lines_dir)
        for sample_number in range(1, count + 1)
    ]

    rows: List[Tuple[int, str, str]] = []
    if count:
        context = mp.get_context("spawn")
        with context.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(
                str(sd19_index_file),
                symbol_index,
                backgrounds,
                config,
            ),
        ) as pool:
            iterator = pool.imap_unordered(_generate_sample, tasks, chunksize=chunksize)
            rows = list(tqdm(iterator, total=count, desc=split, unit="image"))

    rows.sort(key=lambda row: row[0])
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=["file_name", "text"])
        writer_csv.writeheader()
        for _sample_number, file_name, text in rows:
            writer_csv.writerow({"file_name": file_name, "text": text})

    print(f"Wrote {metadata_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate overlapped same-writer SD19 alphanumeric line datasets."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory containing *-lines and *-trocr directories.",
    )
    parser.add_argument(
        "--backgrounds-dir",
        type=Path,
        default=Path("backgrounds"),
        help="Directory of background images (default: %(default)s).",
    )
    parser.add_argument(
        "--curated-dir",
        type=Path,
        default=Path("curated"),
        help=(
            "Directory containing symbol images in decimal-code subdirectories "
            "such as curated/44 for comma and curated/46 for period."
        ),
    )
    parser.add_argument(
        "--sd19-index",
        type=Path,
        default=Path("data/sd19-alphanumeric.pkl.gz"),
        help="Preprocessed SD19 binary store (default: %(default)s).",
    )
    parser.add_argument(
        "--domain",
        choices=["alphanumeric", "zip_code", "check_amount", "clock_time", "decimal", "random"],
        default="alphanumeric",
        help="Text source (default: %(default)s).",
    )
    parser.add_argument(
        "--alphabet",
        default="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        help="Alphabet used by the alphanumeric domain.",
    )
    parser.add_argument(
        "--zip-code-file",
        type=Path,
        default=Path("resources/us_zip_codes_unique.csv"),
        help="ZIP-code CSV used by the zip_code domain.",
    )
    parser.add_argument(
        "--check-amount-file",
        type=Path,
        default=Path("resources/check_amounts_benford.csv"),
        help="Amount CSV used by the check_amount domain.",
    )
    parser.add_argument(
        "--train-writers-file",
        type=Path,
        default=Path("resources/writers-all-train.csv"),
    )
    parser.add_argument(
        "--val-writers-file",
        type=Path,
        default=Path("resources/writers-all-train.csv"),
    )
    parser.add_argument(
        "--test-writers-file",
        type=Path,
        default=Path("resources/writers-all-test.csv"),
    )
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--val-samples", type=int, default=100)
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.0,
        help="Minimum adjacent overlap as a fraction of digit width.",
    )
    parser.add_argument(
        "--max-overlap",
        type=float,
        default=0.6,
        help="Maximum adjacent overlap as a fraction of digit width.",
    )
    parser.add_argument(
        "--min-digits",
        type=int,
        default=3,
        help="Minimum length for the random domain.",
    )
    parser.add_argument(
        "--max-digits",
        type=int,
        default=7,
        help="Maximum length for the random domain.",
    )
    parser.add_argument(
        "--min-decimal-places",
        type=int,
        default=1,
        help="Minimum fractional digits for the decimal domain.",
    )
    parser.add_argument(
        "--max-decimal-places",
        type=int,
        default=4,
        help="Maximum fractional digits for the decimal domain.",
    )
    parser.add_argument(
        "--min-background-scale",
        type=float,
        default=1.0,
        help="Minimum background scale (>=1).",
    )
    parser.add_argument(
        "--max-background-scale",
        type=float,
        default=2.0,
        help="Maximum background scale (>=1).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Worker processes (default: CPU count minus one).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=8,
        help="Tasks submitted to each worker at once (default: %(default)s).",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.min_overlap <= args.max_overlap < 1.0:
        raise ValueError("Require 0 <= min_overlap <= max_overlap < 1")
    if args.min_digits < 1 or args.max_digits < args.min_digits:
        raise ValueError("Require 0 <= min_digits <= max_digits")
    if (
        args.min_decimal_places < 0
        or args.max_decimal_places < args.min_decimal_places
    ):
        raise ValueError(
            "Require 0 <= min_decimal_places <= max_decimal_places"
        )
    for name in ("train_samples", "val_samples", "test_samples"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    if args.chunksize < 1:
        raise ValueError("chunksize must be at least 1")


def main() -> None:
    args = parse_args()
    validate_args(args)
    rng = random.Random(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Create all requested directories even for zero-sample splits.
    for split in SPLITS:
        (args.output_dir / f"{split}-lines").mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"{split}-trocr").mkdir(parents=True, exist_ok=True)

    backgrounds = find_backgrounds(args.backgrounds_dir)
    print(f"Found {len(backgrounds)} backgrounds in {args.backgrounds_dir}")

    print("Loading preprocessed SD19 store...")
    sd19_store = load_character_store(args.sd19_index)
    available_writer_ids = set(sd19_store.index)
    print(
        f"Opened {sd19_store.image_count} characters from "
        f"{len(available_writer_ids)} writers"
    )

    symbol_index: SymbolIndex = {}
    if args.domain == "decimal":
        symbol_index = build_symbol_index(args.curated_dir)
        for required_character in (",", "."):
            if required_character not in symbol_index:
                code = ord(required_character)
                raise ValueError(
                    f"Decimal domain requires symbol images in "
                    f"{args.curated_dir / str(code)}"
                )
        print(
            f"Indexed {len(symbol_index)} curated ASCII symbols in "
            f"{args.curated_dir}"
        )

    writers_by_split: Mapping[str, List[str]] = {
        "train": load_writers(args.train_writers_file),
        "val": load_writers(args.val_writers_file),
        "test": load_writers(args.test_writers_file),
    }

    for split, writers in writers_by_split.items():
        missing = [writer for writer in writers if writer not in available_writer_ids]
        if missing:
            raise ValueError(
                f"{split} writer CSV contains IDs absent from the SD19 store: "
                f"{missing[:10]}"
            )
    samples_by_split = {
        "train": args.train_samples,
        "val": args.val_samples,
        "test": args.test_samples,
    }

    numeral_source = make_numeral_source(
        args.domain, args.zip_code_file, args.check_amount_file
    )

    for split in SPLITS:
        generate_split(
            split=split,
            count=samples_by_split[split],
            writers=writers_by_split[split],
            backgrounds=backgrounds,
            output_dir=args.output_dir,
            sd19_index_file=args.sd19_index,
            symbol_index=symbol_index,
            domain=args.domain,
            alphabet=args.alphabet,
            numeral_source=numeral_source,
            min_digits=args.min_digits,
            max_digits=args.max_digits,
            min_decimal_places=args.min_decimal_places,
            max_decimal_places=args.max_decimal_places,
            min_overlap=args.min_overlap,
            max_overlap=args.max_overlap,
            min_background_scale=args.min_background_scale,
            max_background_scale=args.max_background_scale,
            seed=args.seed,
            workers=args.workers,
            chunksize=args.chunksize,
        )


if __name__ == "__main__":
    main()
