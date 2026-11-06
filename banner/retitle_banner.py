# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "opencv-python-headless", "pillow"]
# ///
"""Replace a line of text on a raster banner, matching the original font.

Built for the GitHub profile banner (``profile.png``): white text on a blue
patterned background, set in Red Hat Display. The routine:

1. Finds the old text's pixels inside a horizontal band of the image.
2. Fits the font's weight, size and position to those pixels by least squares.
3. Inpaints the old text away so the background pattern carries on through.
4. Draws the new text with the fitted settings, centred where the old text
   was centred and on the same baseline.

Only the text area changes; the rest of the image, including the transparent
rounded corners, is left byte-for-byte as it was.

Examples:
    uv run banner/retitle_banner.py profile.png profile.png \\
        --font RedHatDisplay.ttf --old "Data Scientist" --new "Data Engineer"
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Text pixels are near-white; background lines and dots stay well below this.
TEXT_RED_THRESHOLD = 150
# Pixels darker than this (red channel) count as plain background.
BACKGROUND_RED_MAX = 90
# Observed coverage below this is treated as background lines, not text.
MIN_TEXT_COVERAGE = 0.25
# Margin, in pixels, kept around the text when measuring and erasing it.
PAD = 6


@dataclass(frozen=True)
class FontFit:
    """Font settings that best reproduce the original text.

    Attributes:
        weight: Variable-font weight axis value (e.g. 350).
        size: Font size in pixels.
        x: Left draw origin of the text in image coordinates.
        y: Top draw origin of the text in image coordinates.
        error: Mean squared error between rendered and observed coverage.
    """

    weight: int
    size: float
    x: int
    y: int
    error: float


def load_font(font_path: Path, size: float, weight: int) -> ImageFont.FreeTypeFont:
    """Load a variable font at the given size and weight.

    Args:
        font_path: Path to a variable TrueType font with a weight axis.
        size: Font size in pixels.
        weight: Weight axis value.

    Returns:
        The configured font.
    """
    font = ImageFont.truetype(str(font_path), size)
    font.set_variation_by_axes([weight])
    return font


def render_coverage(
    text: str,
    font: ImageFont.FreeTypeFont,
    width: int,
    height: int,
    origin: tuple[int, int],
) -> np.ndarray:
    """Render text as a coverage (alpha) map between 0 and 1.

    Args:
        text: Text to draw.
        font: Font to draw it in.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        origin: Draw origin (x, y) on the canvas.

    Returns:
        Float array of shape (height, width) holding text coverage.
    """
    layer = Image.new("L", (width, height), 0)
    ImageDraw.Draw(layer).text(origin, text, font=font, fill=255)
    return np.asarray(layer, dtype=np.float32) / 255.0


def find_text_bbox(rgb: np.ndarray, band: tuple[int, int]) -> tuple[int, int, int, int]:
    """Find the bounding box of bright text pixels within a band of rows.

    Args:
        rgb: Image as a float array of shape (height, width, 3).
        band: First and last row (exclusive) to search.

    Returns:
        Bounding box as (x0, x1, y0, y1), inclusive.

    Raises:
        ValueError: If no text pixels are found in the band.
    """
    top, bottom = band
    bright = np.zeros(rgb.shape[:2], dtype=bool)
    bright[top:bottom] = rgb[top:bottom, :, 0] > TEXT_RED_THRESHOLD
    ys, xs = np.nonzero(bright)
    if xs.size == 0:
        raise ValueError(f"No text found between rows {top} and {bottom}")
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def fit_font(
    rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    text: str,
    font_path: Path,
) -> FontFit:
    """Fit font weight, size and position to the text already in the image.

    Args:
        rgb: Image as a float array of shape (height, width, 3).
        bbox: Bounding box of the existing text as (x0, x1, y0, y1).
        text: The existing text, exactly as it appears.
        font_path: Variable font to fit.

    Returns:
        The best-fitting font settings, in image coordinates.
    """
    x0, x1, y0, y1 = bbox
    window = rgb[y0 - PAD : y1 + PAD + 1, x0 - PAD : x1 + PAD + 1]
    background = np.median(window[window[..., 0] < BACKGROUND_RED_MAX], axis=0)
    logger.info(f"Background colour around the text: {background.round(1).tolist()}")

    observed = np.clip((window[..., 0] - background[0]) / (255.0 - background[0]), 0, 1)
    observed[observed < MIN_TEXT_COVERAGE] = 0
    height, width = observed.shape

    best: FontFit | None = None
    for weight in range(300, 501, 25):
        for size in np.arange(14.0, 24.01, 0.5):
            font = load_font(font_path, float(size), weight)
            left, top, _, _ = font.getbbox(text)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    origin = (PAD - int(left) + dx, PAD - int(top) + dy)
                    rendered = render_coverage(text, font, width, height, origin)
                    error = float(np.mean((rendered - observed) ** 2))
                    if best is None or error < best.error:
                        best = FontFit(
                            weight=weight,
                            size=float(size),
                            x=x0 - PAD + origin[0],
                            y=y0 - PAD + origin[1],
                            error=error,
                        )
    logger.info(
        f"Best fit: weight {best.weight}, size {best.size}px, "
        f"origin ({best.x}, {best.y}), error {best.error:.5f}"
    )
    return best


def erase_text(
    rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fit: FontFit,
) -> np.ndarray:
    """Inpaint the existing text away, keeping the background pattern.

    Args:
        rgb: Image as a float array of shape (height, width, 3).
        bbox: Bounding box of the existing text as (x0, x1, y0, y1).
        text: The existing text.
        font: Font fitted to the existing text.
        fit: Fitted position of the existing text.

    Returns:
        Float RGB array with the text removed.
    """
    height, width = rgb.shape[:2]
    x0, x1, y0, y1 = bbox
    rendered = render_coverage(text, font, width, height, (fit.x, fit.y))

    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    near_text = (
        (np.abs(rows - (y0 + y1) / 2) < (y1 - y0) / 2 + 3)
        & (cols > x0 - 3)
        & (cols < x1 + 3)
    )
    mask = ((rendered > 0.02) | ((rgb[..., 0] > 110) & near_text)).astype(
        np.uint8
    ) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cleaned = cv2.inpaint(bgr, mask, 4, cv2.INPAINT_TELEA)
    return cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB).astype(np.float32)


def retitle(
    source: Path,
    destination: Path,
    font_path: Path,
    old_text: str,
    new_text: str,
    band: tuple[int, int],
) -> None:
    """Replace ``old_text`` with ``new_text`` on the banner.

    Args:
        source: Banner PNG to read.
        destination: Where to write the result (may equal ``source``).
        font_path: Variable font matching the banner's text.
        old_text: Text currently on the banner, exactly as it appears.
        new_text: Replacement text.
        band: First and last row (exclusive) that contain ``old_text``.
    """
    rgba = np.asarray(Image.open(source).convert("RGBA"), dtype=np.float32)
    rgb, alpha = rgba[..., :3], rgba[..., 3:]
    logger.info(f"Read {source} ({rgba.shape[1]}x{rgba.shape[0]})")

    bbox = find_text_bbox(rgb, band)
    logger.info(
        f"Old text '{old_text}' found at x={bbox[0]}..{bbox[1]}, y={bbox[2]}..{bbox[3]}"
    )

    fit = fit_font(rgb, bbox, old_text, font_path)
    font = load_font(font_path, fit.size, fit.weight)
    centre_x = fit.x + font.getlength(old_text) / 2

    cleaned = erase_text(rgb, bbox, old_text, font, fit)

    new_x = round(centre_x - font.getlength(new_text) / 2)
    height, width = rgb.shape[:2]
    coverage = render_coverage(new_text, font, width, height, (new_x, fit.y))[..., None]
    result = cleaned * (1 - coverage) + 255.0 * coverage

    out = np.concatenate([result, alpha], axis=-1).clip(0, 255).astype(np.uint8)
    Image.fromarray(out).save(destination, optimize=True)
    logger.info(f"Wrote '{new_text}' centred at x={centre_x:.1f} to {destination}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="Banner PNG to read")
    parser.add_argument("destination", type=Path, help="Where to write the result")
    parser.add_argument("--font", type=Path, required=True, help="Variable font (.ttf)")
    parser.add_argument("--old", required=True, help="Text currently on the banner")
    parser.add_argument("--new", required=True, help="Replacement text")
    parser.add_argument(
        "--band",
        type=int,
        nargs=2,
        default=(120, 170),
        metavar=("TOP", "BOTTOM"),
        help="Rows containing the old text (default: 120 170, the subtitle)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the banner retitle from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    retitle(
        args.source, args.destination, args.font, args.old, args.new, tuple(args.band)
    )


if __name__ == "__main__":
    main()
