#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand‑written line generator – core (UI‑free) implementation.
The code is a refactored version of the original monolithic script.
It is deliberately split into small, test‑able functions and a single
public class `HandwrittenGenerator`) that the Gradio UI imports.
Key new feature:
    • `generate(..., num_images=…)` lets the caller request an exact
      number of image/GT pairs. If the requested amount is smaller than
      the number of distinct lines in the corpus a random subset is used;
      if larger, lines are re‑used (with replacement) so that the final
      count matches the request.
"""
from __future__ import annotations
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import csv  # Import the CSV module
import json  # Import JSON module for JSON Lines support
# ----------------------------------------------------------------------
# Configuration data class (lightweight – you can extend it later)
# ----------------------------------------------------------------------
class Config:
    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    CORPUS_DIR: str = "corpus"          # folder containing *.txt files
    FONT_DIR: str = "fonts"            # folder containing .ttf/.otf files
    OUTPUT_DIR: str = "output"          # where the PNG/GT files will be written
    # ------------------------------------------------------------------
    # Image layout
    # ------------------------------------------------------------------
    SCALE_FACTOR: float = 4
    IMG_HEIGHT: int = 64                # final fixed height (pixels)
    MARGIN_X: int = 10                  # horizontal margin (px)
    MARGIN_Y: int = 1                   # vertical margin (px)
    MAX_LINE_GAP: int = 16               # max vertical and horizontal jitter (px), 8
    # ------------------------------------------------------------------
    # Text rendering
    # ------------------------------------------------------------------
    FONT_SIZE_RANGE: Tuple[int, int] = (15, 90)   # points 30, 48
    SLANT_RANGE: Tuple[int, int] = (-12, 12)     # degrees
    CHARACTER_SPACING: int = 0                   # extra px between glyphs
    # ------------------------------------------------------------------
    # Augmentation
    # -------------------------------------------------------------------
    NOISE_STDDEV: float = 0.0                   # 0 → no noise
    # ------------------------------------------------------------------
    # Deterioration effects
    # ------------------------------------------------------------------
    # Each effect has a _probability_ (0‑1) and a _severity_ (0‑1) that
    # controls how strong the effect looks.
    MOLD_PROB: float = 0.0
    MOLD_SEVERITY: float = 0.5
    STAIN_PROB: float = 0.0
    STAIN_SEVERITY: float = 0.5
    FAINT_PROB: float = 0.0          # whole‑image fading
    FAINT_STRENGTH: float = 0.5      # 0 = unchanged, 1 = completely white
    PARTIAL_FAINT_PROB: float = 0.0
    PARTIAL_FAINT_STRENGTH: float = 0.5
    # ---------- Uudet asetukset ----------
    # Paksu‑efektin todennäköisyys ja maksimi‑vahvuus (0‑1)
    BOLD_PROB: float = 0.0
    BOLD_STRENGTH: float = 0.0          # 0 = ei muutosta, 1 = maksimi paksuus
    # Katko‑efektin todennäköisyys ja maksimi‑vahvuus (0‑1)
    BREAK_PROB: float = 0.0
    BREAK_STRENGTH: float = 0.0         # 0 = ei katkoja, 1 = erittäin katkonainen
    # --------------------------------------
    # ------------------------------------------------------------------
    # Paper Wave / curvature effect
    # ------------------------------------------------------------------
    # Probability that the line will be warped.
    WAVE_PROB: float = 0.0
    # Amplitude of the sinusoidal wave in pixels (max vertical shift).
    WAVE_AMPLITUDE: Tuple[float, float] = (5, 60)
    # Frequency of the wave: number of complete periods across the image width.
    # Small values → long gentle wave, larger → many short wiggles.
    WAVE_FREQUENCY: Tuple[float, float] = (1.0, 5.0)
    # If True, each line gets its own random phase offset so the wave is
    # shifted left/right (or even upside‑down) per line.
    WAVE_RANDOM_PHASE: bool = True
    WAVE_NOISE_STRENGTH = 0.5
    WAVE_NATURAL_VARIATION = 1.0
    WAVE_SMOOTHNESS = 1.0
    WAVE_HORIZONTAL_SKEW = 0.2
    # ------------------------------------------------------------------
    # Paper texture effects
    # ------------------------------------------------------------------
    PAPER_TEXTURE_PROB: float = 0.0
    PAPER_TEXTURE_STRENGTH: float = 0.3
    PAPER_TEXTURE_TYPE: str = "ruled"  # "ruled", "graph", or "none"
    RULED_LINE_SPACING: int = 25  # Pixels between lines
    RULED_LINE_COLOR: tuple = (200, 200, 255)  # Light blue-gray
    RULED_LINE_THICKNESS: int = 1
    GRAPH_GRID_SPACING: int = 20  # Pixels between grid lines
    GRAPH_LINE_COLOR: tuple = (220, 220, 240)  # Very light blue
    GRAPH_LINE_THICKNESS: int = 1
    GRAPH_MAJOR_LINE_EVERY: int = 5  # Every 5th line is darker
    GRAPH_MAJOR_LINE_COLOR: tuple = (180, 180, 220)  # Darker blue
    # ------------------------------------------------------------------
    # Randomness
    # ------------------------------------------------------------------
    SEED: int = 42
    # ------------------------------------------------------------------
    # Background color
    # ------------------------------------------------------------------
    BACKGROUND_COLOR: Tuple[int, int, int] = (255, 255, 255)  # Default white
    BACKGROUND_COLOR_VARIATION: int = 0       # 0 = no variation, 5 = max variation

    FONT_COLOR: Tuple[int, int, int] = (0, 0, 0)
    FONT_COLOR_VARIATION: int = 0

    def apply_scale(self):
        s = self.SCALE_FACTOR
        self.FONT_SIZE_RANGE = tuple(int(x*s) for x in self.FONT_SIZE_RANGE)
        self.IMG_HEIGHT = int(s*self.IMG_HEIGHT)
        self.MARGIN_X = int(s*self.MARGIN_X)
        self.MARGIN_Y = int(s*self.MARGIN_Y)
        self.MAX_LINE_GAP = int(s*self.MAX_LINE_GAP)
    
    def __init__(self):
        self.apply_scale()
# ----------------------------------------------------------------------
# Helper utilities – unchanged from the original script
# ----------------------------------------------------------------------
def _process_one(args: Tuple[int, str, List[Path], Config, str]) -> Tuple[int, bool, str, str]:
    """
    Wrapper executed inside the ProcessPool.
    Returns (idx, success_flag, error_message_if_any, base_name).
    """
    idx, line, fonts, cfg, source_name = args
    try:
        img, font_name = render_line(line, fonts, cfg)          # now also returns font name
        base_name = save_image_and_gt(img, line, cfg.OUTPUT_DIR, idx, source_name, font_name)
        return idx, True, "", base_name
    except Exception as exc:                    # pragma: no‑cover
        return idx, False, f"{type(exc).__name__}: {exc}", ""
def load_fonts(font_dir: str) -> List[Path]:
    """Return a list of pathlib.Path objects for all TTF/OTF files."""
    p = Path(font_dir)
    fonts = list(p.glob("*.ttf")) + list(p.glob("*.otf"))
    if not fonts:
        raise FileNotFoundError(f"No .ttf/.otf fonts found in {font_dir}")
    return fonts
def load_corpus_dir(corpus_dir: str) -> List[Tuple[str, str]]:
    """
    Walk through ``corpus_dir`` and collect every non‑empty line from each
    ``*.txt`` file.
    Returns a list of (source_name, line) tuples where ``source_name`` is
    the stem of the *.txt file (e.g. ``mycorpus``) and ``line`` is the stripped text.
    """
    root = Path(corpus_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus directory '{corpus_dir}' does not exist or is not a folder.")
    lines: List[Tuple[str, str]] = []
    for txt_file in sorted(root.glob("*.txt")):
        source_name = txt_file.stem
        with txt_file.open(encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if line:
                    lines.append((source_name, line))
    if not lines:
        raise ValueError(f"No usable text lines found in directory '{corpus_dir}'.")
    return lines
def _apply_local_bold(img: np.ndarray, strength: float) -> np.ndarray:
    """
    Thicken (bold) **random parts** of the handwriting strokes.
    The algorithm works on a BGR uint8 image that contains black text on a
    white background.
    Steps
    -----
    1️⃣  Build a binary mask of the foreground (black) pixels.
    2️⃣  Create a random mask that indicates _where_ we want to apply bolding
        (a handful of ellipses – their number and size depend on ``strength``).
    3️⃣  Intersect the two masks → we only thicken text that lies inside the
        random region.
    4️⃣  Dilate that intersected mask; the kernel size is also driven by
        ``strength``.
    5️⃣  Paint the dilated mask back as black on the original image.
    ``strength`` ∈ [0,1] controls both the size of the random region and the
    dilation kernel size (0 = no change, 1 = fairly strong bolding).
    Returns
    -------
    np.ndarray
        The image with locally‑bolded text (uint8, BGR).
    """
    if strength <= 0:
        return img
    # ------------------------------------------------------------------
    # 1️⃣  Foreground mask (text = 1, background = 0)
    # ------------------------------------------------------------------
    # Convert to gray if needed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img.copy()
    # Text is dark → use an inverted binary threshold
    # 250 is a safe cutoff for “almost white” background.
    _, fg_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # ------------------------------------------------------------------
    # 2️⃣  Random region mask – where we _might_ thicken the text
    # ------------------------------------------------------------------
    h, w = img.shape[:2]
    region_mask = np.zeros((h, w), dtype=np.uint8)
    # Number of random blobs grows with strength (at least 1)
    n_blobs = random.randint(1,5)
    for i in range(n_blobs):
        # Random ellipse size – larger when strength is larger
        radius = random.randint(
            int(5 + 10 * strength),          # min radius
            int(20 + 30 * strength)          # max radius
        )
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        cv2.ellipse(
            region_mask,
            (cx, cy),
            (radius, radius // 2),
            angle=random.uniform(0, 360),
            startAngle=0,
            endAngle=360,
            color=255,
            thickness=-1,
        )
    # ------------------------------------------------------------------
    # 3️⃣  Intersect: we only want to thicken _text_ that lies inside the
    #             random region.
    # ------------------------------------------------------------------
    target_mask = cv2.bitwise_and(fg_mask, region_mask)
    # ------------------------------------------------------------------
    # 4️⃣  Dilate the intersected mask
    # ------------------------------------------------------------------
    # Kernel size proportional to strength (minimum = 1)
    ksize = max(1, int(1 + 4 * strength))   # 1‑5 pixels
    kernel = np.ones((ksize, ksize), np.uint8)
    dilated = cv2.dilate(target_mask, kernel, iterations=1)
    # ------------------------------------------------------------------
    # 5️⃣  Paint dilated mask back as black (0,0,0) on the original image
    # ------------------------------------------------------------------
    # Convert the single‑channel mask to 3‑channels for indexing.
    dilated_3ch = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    # Work on a copy to avoid side‑effects.
    out = img.copy()
    out[dilated_3ch == 255] = 0   # black colour for the bolded pixels
    return out
def _apply_wave(
    img: np.ndarray,
    amplitude: int,
    frequency: float,
    config: Config,
) -> np.ndarray:
    """
    Apply a more realistic sinusoidal vertical displacement to make lines appear
    naturally wavy like real handwriting.
    """
    if amplitude <= 0 or frequency <= 0:
        return img
    h, w = img.shape[:2]
    # Create coordinate grids
    map_x, map_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    # Create more natural, non-uniform wave patterns
    total_shift = np.zeros_like(map_y)
    # 1. Base wave with random phase
    phase1 = random.uniform(0, 2 * np.pi) if config.WAVE_RANDOM_PHASE else 0.0
    shift1 = amplitude * 0.6 * np.sin(2.0 * np.pi * frequency * map_x / w + phase1)
    total_shift += shift1
    # 2. Secondary wave with different frequency and phase
    phase2 = random.uniform(0, 2 * np.pi) if config.WAVE_RANDOM_PHASE else 0.0
    shift2 = amplitude * 0.3 * np.sin(2.0 * np.pi * frequency * 1.5 * map_x / w + phase2)
    total_shift += shift2
    # 3. Add some random variation across the image
    if config.WAVE_NOISE_STRENGTH > 0:
        # Create noise that matches the image dimensions
        noise = np.random.normal(0, config.WAVE_NOISE_STRENGTH, (h, w))
        total_shift += noise
    # 4. Apply Gaussian smoothing to make transitions smoother
    if config.WAVE_SMOOTHNESS > 0:
        from scipy import ndimage
        total_shift = ndimage.gaussian_filter(total_shift, sigma=config.WAVE_SMOOTHNESS)
    # 5. Add slight horizontal skew for more realism
    if config.WAVE_HORIZONTAL_SKEW > 0:
        # Create horizontal skew that varies with y position
        skew_amount = config.WAVE_HORIZONTAL_SKEW * np.sin(2.0 * np.pi * 0.01 * map_y / h)
        map_x = map_x + skew_amount
    # Ensure the coordinate arrays have the same shape as the original image
    # Apply the transformation
    map_y = map_y + total_shift
    # Make sure we're not going out of bounds
    map_x = np.clip(map_x, 0, w-1)
    map_y = np.clip(map_y, 0, h-1)
    # Remap with bicubic interpolation for better quality
    warped = cv2.remap(
        img,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=config.BACKGROUND_COLOR,
    )
    return warped
def _apply_faint(img: np.ndarray, strength: float) -> np.ndarray:
    """
    Fade the whole image towards white.
    strength ∈ [0,1] (0 = unchanged, 1 = completely white).
    """
    img = img.astype(np.float32)
    img = img * (1 - strength) + 255 * strength
    return np.clip(img, 0, 255).astype(np.uint8)

def _apply_partial_faint(img: np.ndarray, strength: float) -> np.ndarray:
    """
    Randomly mask out a rectangular region and fade it.
    """
    h, w = img.shape[:2]
    # pick a random rectangle (size controlled loosely by strength)
    max_frac = 0.2 + 0.6 * strength  # up to 80% of the image for strong faint
    rect_w = int(w * random.uniform(0.1, max_frac))
    rect_h = int(h * random.uniform(0.1, max_frac))
    x0 = random.randint(0, w - rect_w)
    y0 = random.randint(0, h - rect_h)
    region = img[y0:y0 + rect_h, x0:x0 + rect_w].astype(np.float32)
    region = region * (1 - strength) + 255 * strength
    img[y0:y0 + rect_h, x0:x0 + rect_w] = np.clip(region, 0, 255).astype(np.uint8)
    return img
# ----------------------------------------------------------------------
# Deterioration helpers
# ----------------------------------------------------------------------
def _apply_mold(img: np.ndarray, severity: float) -> np.ndarray:
    """
    Add brownish “mold” blobs.
    severity ∈ [0,1] controls the number & size of blobs.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    # number of spots grows with severity
    spots = int(5 + 30 * severity)
    for _ in range(spots):
        # random radius (bigger for higher severity)
        r = int(random.uniform(5, 30) * (0.5 + severity))
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        cv2.circle(mask, (cx, cy), r, 255, -1)
    # blur the mask a little to get soft edges
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    # brownish colour (light brown)
    mold_color = np.array([200, 180, 150], dtype=np.uint8)
    # blend
    alpha = (mask.astype(np.float32) / 255.0) * severity
    img = img.astype(np.float32)
    img = img * (1 - alpha[..., None]) + mold_color * alpha[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)

def _apply_stain(img: np.ndarray, severity: float) -> np.ndarray:
    """
    Add darker irregular stains.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    spots = int(3 + 20 * severity)
    for _ in range(spots):
        r = int(random.uniform(8, 40) * (0.5 + severity))
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        cv2.circle(mask, (cx, cy), r, 255, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    stain_color = np.array([80, 70, 60], dtype=np.uint8)  # dark brown/gray
    alpha = (mask.astype(np.float32) / 255.0) * severity
    img = img.astype(np.float32)
    img = img * (1 - alpha[..., None]) + stain_color * alpha[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)

def _apply_ruled_paper(image: Image.Image, config: Config) -> Image.Image:
    """Apply ruled/lined paper texture to the background."""
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    # Create ruled lines
    line_spacing = config.RULED_LINE_SPACING
    line_color = config.RULED_LINE_COLOR
    line_thickness = config.RULED_LINE_THICKNESS
    # Apply lines horizontally across the image
    for y in range(line_spacing, height, line_spacing):
        y_start = max(0, y - line_thickness // 2)
        y_end = min(height, y + line_thickness // 2 + 1)
        img_array[y_start:y_end, :] = line_color
    return Image.fromarray(img_array)
def _apply_graph_paper(image: Image.Image, config: Config) -> Image.Image:
    """Apply graph paper texture to the background."""
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    grid_spacing = config.GRAPH_GRID_SPACING
    line_color = config.GRAPH_LINE_COLOR
    major_line_color = config.GRAPH_MAJOR_LINE_COLOR
    line_thickness = config.GRAPH_LINE_THICKNESS
    major_every = config.GRAPH_MAJOR_LINE_EVERY
    # Horizontal lines
    for i, y in enumerate(range(grid_spacing, height, grid_spacing)):
        is_major = (i + 1) % major_every == 0
        color = major_line_color if is_major else line_color
        y_start = max(0, y - line_thickness // 2)
        y_end = min(height, y + line_thickness // 2 + 1)
        img_array[y_start:y_end, :] = color
    # Vertical lines
    for i, x in enumerate(range(grid_spacing, width, grid_spacing)):
        is_major = (i + 1) % major_every == 0
        color = major_line_color if is_major else line_color
        x_start = max(0, x - line_thickness // 2)
        x_end = min(width, x + line_thickness // 2 + 1)
        img_array[:, x_start:x_end] = color
    return Image.fromarray(img_array)
def _apply_paper_texture(image: Image.Image, config: Config) -> Image.Image:
    """Apply paper texture based on configuration."""
    if config.PAPER_TEXTURE_PROB <= 0 or np.random.random() > config.PAPER_TEXTURE_PROB:
        return image
    # Apply the texture based on type
    if config.PAPER_TEXTURE_TYPE == "ruled":
        textured = _apply_ruled_paper(image, config)
    elif config.PAPER_TEXTURE_TYPE == "graph":
        textured = _apply_graph_paper(image, config)
    else:
        return image
    # Blend the textured background with original based on strength
    strength = config.PAPER_TEXTURE_STRENGTH
    if strength <= 0:
        return image
    elif strength >= 1:
        return textured
    else:
        # Blend the two images
        img_array = np.array(image)
        textured_array = np.array(textured)
        blended = (1 - strength) * img_array + strength * textured_array
        return Image.fromarray(blended.astype(np.uint8))
def _apply_breaks(img: np.ndarray, strength: float, background_color:Tuple[int,int,int] = (255,255,255)) -> np.ndarray:
    """
    cuts random parts into image
    strength ∈ [0,1] kontrolloi katkojen määrää ja leveyttä.
    """
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    # 1️⃣ Piirrä satunnaisia viivoja tai act‑shapes maskiin
    n_lines = int(2 + 8 * strength)          # 2‑10 viivaa
    for i in range(n_lines):
        x0 = random.randint(0, w - 1)
        y0 = random.randint(0, h - 1)
        # satunnainen suunta ja pituus
        angle = random.uniform(0, np.pi)
        length = random.randint(int(0.2 * w), int(0.6 * w))
        x1 = int(x0 + length * np.cos(angle))
        y1 = int(y0 + length * np.sin(angle))
        thickness = int(1 + 1 * strength)    # 1-2
        cv2.line(mask, (x0, y0), (x1, y1), 255, thickness)
    # 2️⃣ “Puhdistus” – poista osia alkuperäisestä kuvasta maskissa
    #   (asetetaan maskin alueet valkoiseksi)
    img = img.copy()
     # Use the background color instead of hardcoded 255 (white)
    if len(background_color) == 3:
        # For color images, set all channels to background color
        img[mask == 255] = background_color
    else:
        # For grayscale, set to single value
        img[mask == 255] = background_color[0]
    return img
def apply_augmentations(img: np.ndarray, config: Config) -> np.ndarray:
     # ----- NEW wave / curvature effect -----
    if random.random() < config.WAVE_PROB:
        img = _apply_wave(
            img,
            amplitude=random.uniform(config.WAVE_AMPLITUDE[0], config.WAVE_AMPLITUDE[1]),
            frequency=random.uniform(config.WAVE_FREQUENCY[0], config.WAVE_FREQUENCY[1]),
            config=config,
        )
    if random.random() < config.BOLD_PROB:
        img = _apply_local_bold(img, config.BOLD_STRENGTH)
    if random.random() < config.BREAK_PROB:
        img = _apply_breaks(img, config.BREAK_STRENGTH, config.BACKGROUND_COLOR)
    if random.random() < config.FAINT_PROB:
        img = _apply_faint(img, config.FAINT_STRENGTH)
    if random.random() < config.PARTIAL_FAINT_PROB:
        img = _apply_partial_faint(img, config.PARTIAL_FAINT_STRENGTH)
    # ---- Noise -------------------------------------------------------
    if config.NOISE_STDDEV > 0:
        noise = np.random.normal(0, config.NOISE_STDDEV, img.shape).astype(np.float32)
        img = img.astype(np.float32) + noise
        img = np.clip(img, 0, 255)
    # ---- Deterioration ------------------------------------------------
    # Order: mold → stain → faint → partial faint (feel free to re‑order)
    if random.random() < config.MOLD_PROB:
        img = _apply_mold(img, config.MOLD_SEVERITY)
    if random.random() < config.STAIN_PROB:
        img = _apply_stain(img, config.STAIN_SEVERITY)
    return img.astype(np.uint8)
def _get_background_color(config: Config) -> Tuple[int, int, int]:
    """
    Get a background color with optional variation.
    If background color variation is enabled, randomly vary the RGB components
    within the specified range (up to 5 units in either direction).
    """
    base_color = config.BACKGROUND_COLOR
    variation = config.BACKGROUND_COLOR_VARIATION
    # If no variation is requested, return the base color
    if variation <= 0:
        return base_color
    # Generate variation for each RGB component
    r = max(0, min(255, base_color[0] + random.randint(-variation, variation)))
    g = max(0, min(255, base_color[1] + random.randint(-variation, variation)))
    b = max(0, min(255, base_color[2] + random.randint(-variation, variation)))
    return (r, g, b)

def _get_text_color(config: Config) -> Tuple[int, int, int]:
    """
    Get a text color with optional variation.
    If text color variation is enabled, randomly vary the RGB components
    within the specified range (up to 5 units in either direction).
    """
    base_color = config.FONT_COLOR
    variation = config.FONT_COLOR_VARIATION
    # If no variation is requested, return the base color
    if variation <= 0:
        return base_color
    # Generate variation for each RGB component
    r = max(0, min(255, base_color[0] + random.randint(-variation, variation)))
    g = max(0, min(255, base_color[1] + random.randint(-variation, variation)))
    b = max(0, min(255, base_color[2] + random.randint(-variation, variation)))
    return (r, g, b)

def add_slant(img: np.ndarray, angle_deg: float, config: Config) -> np.ndarray:
    """Shear the image horizontally to simulate handwriting slant."""
    if abs(angle_deg) < 0.1:
        return img
    h, w = img.shape[:2]
    angle_rad = np.deg2rad(angle_deg)
    shear = np.tan(angle_rad)
    M = np.array([[1, shear, -shear * h / 2],
                  [0,   1,               0]],
                 dtype=np.float32)
    new_w = int(w + abs(shear) * h)
    out = cv2.warpAffine(
        img, M, (new_w, h),
        borderValue=config.BACKGROUND_COLOR,
        flags=cv2.INTER_LINEAR,
    )
    return out

def find_suitable_font(text, fontdir_paths, font_to_supported_character):
    """
    Pick a font file path from a randomly selected font directory.

    Rules:
    - Randomly select a directory from fontdir_paths.
    - Collect all .ttf and .otf files inside that directory.
    - If one or more fonts contain "regular" (case-insensitive):
        - 70% probability: pick the first alphabetically among those.
        - 30% probability: pick randomly from the other fonts in the directory.
    - If no "regular" fonts exist:
        - Pick randomly from all fonts in the directory.
    - Optionally ensure the chosen font supports all characters in `text`
      (if font_to_supported_character is provided).
    """

    if not fontdir_paths:
        raise ValueError("fontdir_paths is empty")

    # Keep trying until we find a usable directory
    while True:
        font_dir = random.choice(fontdir_paths)

        # Collect .ttf and .otf files directly inside this directory
        font_paths = sorted(list(font_dir.glob("*.ttf")) + list(font_dir.glob("*.otf")))

        if not font_paths:
            continue  # try another directory

        # Optional: filter by character support
        if text and font_to_supported_character:
            text_codepoints = set(ord(c) for c in text)
            filtered = []
            for fp in font_paths:
                supported = font_to_supported_character.get(str(fp))
                if supported and text_codepoints.issubset(supported):
                    filtered.append(fp)
            if filtered:
                font_paths = filtered
            else:
                continue  # no supporting font in this dir, try another

        # Split regular vs non-regular fonts
        regular_fonts = sorted(
            [fp for fp in font_paths if "regular" in fp.name.lower()],
            key=lambda p: p.name.lower()
        )
        non_regular_fonts = [fp for fp in font_paths if fp not in regular_fonts]

        if regular_fonts:
            if random.random() < 0.7:
                # 70% → first alphabetically among regular
                return regular_fonts[0]
            else:
                # 30% → random non-regular (if available)
                if non_regular_fonts:
                    return random.choice(non_regular_fonts)
                else:
                    # Only regular fonts exist
                    return regular_fonts[0]
        else:
            # No regular fonts → pick any randomly
            return random.choice(font_paths)


def render_line(text: str,
                fontdir_paths: List[Path],
                font_to_supported_character,
                config: Config, random_seed: int) -> Tuple[Image.Image, str]:
    """
    Render a single line of text into a Pillow Image (RGB).
    Returns (final_image, font_name) where ``font_name`` is the stem of the
    font file that was used.
    """
    # seed
    random.seed(random_seed)
    np.random.seed(random_seed % (2**32))

    # --------------------------------------------------------------
    # 1️⃣ Choose a random font size and a concrete font file
    # --------------------------------------------------------------
    size_pt = random.randint(*config.FONT_SIZE_RANGE)
    #font_path = random.choice(fonts)
    font_path = find_suitable_font(text, fontdir_paths, font_to_supported_character)
    #print(f"Found font {font_path}")

    if hasattr(ImageFont, "LAYOUT_BASIC"):
        font = ImageFont.truetype(str(font_path), size=size_pt, layout_engine=ImageFont.LAYOUT_BASIC)
    else:
        font = ImageFont.truetype(str(font_path), size=size_pt)
    font_name = font_path.stem
    # --------------------------------------------------------------
    # 2️⃣ Estimate rendered size (width, height)
    # --------------------------------------------------------------
    dummy_img = Image.new("RGB", (1, 1), config.BACKGROUND_COLOR)
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font, spacing=config.CHARACTER_SPACING)
    txt_w = bbox[2] - bbox[0]
    txt_h = bbox[3] - bbox[1]
    # --------------------------------------------------------------
    # 3️⃣ Create canvas + jitter
    # --------------------------------------------------------------
    jitter_y = random.randint(-config.MAX_LINE_GAP, config.MAX_LINE_GAP)
    jitter_x = random.randint(-config.MAX_LINE_GAP, config.MAX_LINE_GAP)
    canvas_w = txt_w + 2 * config.MARGIN_X + config.MAX_LINE_GAP  # add space for jitter
    canvas_h = max(txt_h, config.IMG_HEIGHT) + 2 * config.MARGIN_Y + 2 * config.MAX_LINE_GAP  # add space for jitter
    # Get the actual background color with possible variation
    actual_bg_color = _get_background_color(config)
    img = Image.new("RGB", (canvas_w, canvas_h), actual_bg_color)
    img = _apply_paper_texture(img, config)
    draw = ImageDraw.Draw(img)
    # --------------------------------------------------------------
    # 4️⃣ Render the text (black)
    # --------------------------------------------------------------
    baseline_x = config.MARGIN_X + jitter_x + config.MAX_LINE_GAP
    #baseline_y = config.MARGIN_Y + jitter_y
    center_y = (canvas_h - txt_h) // 2
    #baseline_y = center_y + jitter_y
    baseline_y = center_y - bbox[1] + jitter_y

    draw.text((baseline_x, baseline_y),
              text,
              font=font,
              fill=_get_text_color(config),  # use random text color
              spacing=config.CHARACTER_SPACING)
    # --------------------------------------------------------------
    # 5️⃣ Convert to OpenCV, apply slant + augmentations, back to Pillow
    # --------------------------------------------------------------
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    slant_angle = random.uniform(*config.SLANT_RANGE)
    cv_img = add_slant(cv_img, slant_angle, config)
    cv_img = apply_augmentations(cv_img, config)
    final_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    # --------------------------------------------------------------
    # 6️⃣ Resize to a fixed height (preserve aspect)
    # --------------------------------------------------------------
    if config.IMG_HEIGHT:
        w, h = final_img.size
        new_h = config.IMG_HEIGHT
        new_w = int(w * new_h / h)
        final_img = final_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return final_img, font_name
def save_image_and_gt(img: Image.Image,
                      gt_text: str,
                      output_dir: str,
                      idx: int,
                      source_name: Optional[str] = None,
                      font_name: Optional[str] = None) -> str:
    """
    Write the image and its matching ground‑truth file.
    Filenames follow the pattern:
        <source>_<font>_<idx:05d>.png
        <source>_<font>_<idx:05d>.gt.txt
    If ``source_name`` is omitted we fall back to the legacy
    ``img_<idx:05d>`` format.
    
    Returns the base name used for the files.
    """
    if source_name:
        parts = [source_name]
        if font_name:
            parts.append(font_name)
        parts.append(f"{idx:05d}")
        base_name = "_".join(parts)
    else:
        base_name = f"img_{idx:05d}"
    img_path = Path(output_dir) / f"{base_name}.png"
    gt_path = Path(output_dir) / f"{base_name}.gt.txt"
    img.save(img_path, format="PNG")
    gt_path.write_text(gt_text, encoding="utf-8")
    return base_name
# ----------------------------------------------------------------------
# Public class – the only API the UI (or any other script) will touch
# ----------------------------------------------------------------------
class HandwrittenGenerator:
    """
    Thin wrapper that orchestrates the whole pipeline.
    All heavy work stays in this class; the Gradio UI (or a notebook)
    simply creates an instance and calls ``generate``.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
    
    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def generate(
        self,
        corpus_dir: Optional[str] = None,
        font_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        num_images: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        training_format: str = 'jsonl'  # 'csv', 'jsonl', or 'simple'
    ) -> str:
        """
        Execute the full generation pipeline.
        Parameters
        ----------
        corpus_dir, font_dir, output_dir : optional overrides for the values
                                          stored in ``self.cfg``.
        num_images : int or None
            Desired total number of image/GT pairs.
             _If ``None`` → use_ *all** lines from the corpus.
            * If ``num_images`` ≤ number of distinct lines → a random
              subset of that size is taken (no repetition).
            * If ``num_images``  > number of distinct lines → lines are
              re‑used (with replacement) so the final count matches the request.
        progress_callback : callable(current:int, total:int) → None
            Called after each line is processed; the UI can use it to update a
            progress bar.
        training_format : str - 'csv', 'jsonl', or 'simple'
            Format for the training data files.
        Returns
        -------
        str
            Absolute path to the folder that now contains the generated PNG
            files and their ``*.gt.txt`` counterparts.
        """
        # --------------------------------------------------------------
        # 0️⃣ Seed & path handling
        # --------------------------------------------------------------
        random.seed(self.cfg.SEED)
        np.random.seed(self.cfg.SEED)
        if corpus_dir:
            self.cfg.CORPUS_DIR = corpus_dir
        if font_dir:
            self.cfg.FONT_DIR = font_dir
        if output_dir:
            self.cfg.OUTPUT_DIR = output_dir
        # --------------------------------------------------------------
        # 1️⃣ Clean (or create) the output directory
        # --------------------------------------------------------------
        out_path = Path(self.cfg.OUTPUT_DIR)
        if out_path.exists():
            shutil.rmtree(out_path)
        out_path.mkdir(parents=True, exist_ok=True)
        # --------------------------------------------------------------
        # 2️⃣ Load resources
        # --------------------------------------------------------------
        fonts = load_fonts(self.cfg.FONT_DIR)          # list of Path objects
        all_lines = load_corpus_dir(self.cfg.CORPUS_DIR)   # List[(source, line)]
        # --------------------------------------------------------------
        # 3️⃣ Decide which lines we will actually render
        # --------------------------------------------------------------
        if num_images is not None:
            total_available = len(all_lines)
            if num_images <= total_available:
                # Random subset, no duplication
                selected = random.sample(all_lines, k=num_images)
            else:
                # Need more than we have → reuse with replacement
                selected = list(all_lines)                     # keep all unique lines
                extra_needed = num_images - total_available
                selected.extend(random.choices(all_lines, k=extra_needed))
                random.shuffle(selected)                       # mix duplicated entries
            print(
                f"🖋️  Selected {len(selected)} lines (requested {num_images}, "
                f"corpus size {total_available})"
            )
        else:
            selected = all_lines
            print(f"🖋️  Using all {len(selected)} lines from the corpus")
        # --------------------------------------------------------------
        # 4️⃣ Build the task list for the ProcessPool
        # --------------------------------------------------------------
        tasks = [
            (i + 1, line, fonts, self.cfg, source_name)
            for i, (source_name, line) in enumerate(selected)
        ]
        # --------------------------------------------------------------
        # 5️⃣ Parallel generation
        # --------------------------------------------------------------
        max_workers = max(os.cpu_count() - 1, 1)
        successes = 0
        failures = 0
        total = len(tasks)
        # Store metadata for training data creation
        training_data = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for idx, ok, err_msg, base_name in tqdm(
                executor.map(_process_one, tasks),
                total=total,
                desc="Generating",
                unit="line",
            ):
                if progress_callback:
                    progress_callback(idx, total)
                if ok:
                    successes += 1
                    # Add entry to training data
                    source_name, line = selected[idx-1]  # idx is 1-based
                    
                    # Calculate absolute paths
                    img_path = str((out_path / f"{base_name}.png").resolve())
                    gt_path = str((out_path / f"{base_name}.gt.txt").resolve())
                    
                    # Add to training data
                    training_data.append({
                        'image_path': img_path,
                        'text': line
                    })
                else:
                    failures += 1
                    print(f"\n❌  Line {idx:05d} failed → {err_msg}")
        
        # --------------------------------------------------------------
        # 6️⃣ Create training data files in requested format
        # --------------------------------------------------------------
        if training_format == 'jsonl':
            # JSON Lines format - most robust for text with commas/quotes
            jsonl_path = out_path / "training_data.jsonl"
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for item in training_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"JSON Lines training data created at: {jsonl_path}")
            
        elif training_format == 'csv':
            # CSV format with proper quoting
            csv_path = out_path / "training_data.csv"
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['image_path', 'text']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for item in training_data:
                    writer.writerow(item)
            print(f"CSV training data created at: {csv_path}")
            
        elif training_format == 'simple':
            # Simple format with tab delimiter (safer than comma)
            simple_path = out_path / "training_data.tsv"
            with open(simple_path, 'w', newline='', encoding='utf-8') as f:
                f.write('image_path\ttext\n')
                for item in training_data:
                    # Escape tabs in text and write safely
                    escaped_text = item['text'].replace('\t', '\\t').replace('\n', '\\n')
                    f.write(f"{item['image_path']}\t{escaped_text}\n")
            print(f"Simple training data (TSV) created at: {simple_path}")
        
        # --------------------------------------------------------------
        # 7️⃣ Summary & return folder path
        # --------------------------------------------------------------
        print("\n✅  Finished!")
        print(f"   • Images written : {successes}")
        print(f"   • Failed lines   : {failures}")
        if training_format == 'jsonl':
            print(f"   • JSON Lines file created at: {out_path / 'training_data.jsonl'}")
        elif training_format == 'csv':
            print(f"   • CSV file created at: {out_path / 'training_data.csv'}")
        elif training_format == 'simple':
            print(f"   • Simple TSV file created at: {out_path / 'training_data.tsv'}")
        return str(out_path.resolve())