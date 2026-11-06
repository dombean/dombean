# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow"]
# ///
"""Draw the GitHub profile banner at any resolution.

Redraws the banner from `pattern.json` (the network pattern, colours and
font settings recovered from the 2022 original by `extract_pattern.py`), so
it stays sharp on high-DPI screens. Everything is drawn oversized and then
averaged down, which gives clean anti-aliased lines, dots, corners and text.

Examples:
    uv run banner/make_banner.py profile.png --font /tmp/RedHatDisplay.ttf
    uv run banner/make_banner.py profile.png --font /tmp/RedHatDisplay.ttf \\
        --subtitle "Senior Data Engineer"
"""

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

PATTERN = Path(__file__).with_name("pattern.json")


def coverage(
    size: tuple[int, int],
    draw_fn: Callable[[ImageDraw.ImageDraw], None],
    final: tuple[int, int],
) -> np.ndarray:
    """Draw a white-on-black mask oversized and average it down to `final`.

    Args:
        size: Oversized canvas (width, height) to draw on.
        draw_fn: Draws in white on an `ImageDraw.ImageDraw`.
        final: Output (width, height).

    Returns:
        Coverage between 0 and 1 as a float array of shape (height, width).
    """
    mask = Image.new("L", size, 0)
    draw_fn(ImageDraw.Draw(mask))
    small = mask.resize(final, Image.Resampling.BOX)
    return np.asarray(small, dtype=np.float32) / 255.0


def load_font(font_path: Path, size: float, weight: int) -> ImageFont.FreeTypeFont:
    """Load a variable font at the given size and weight.

    Args:
        font_path: Variable TrueType font with a weight axis.
        size: Font size in pixels.
        weight: Weight axis value.

    Returns:
        The configured font.
    """
    font = ImageFont.truetype(str(font_path), size)
    font.set_variation_by_axes([weight])
    return font


def render(
    design: dict,
    title: str,
    subtitle: str,
    font_path: Path,
    scale: float,
    supersample: int,
) -> Image.Image:
    """Render the banner.

    Args:
        design: Contents of `pattern.json`.
        title: First line of text.
        subtitle: Second line of text.
        font_path: Red Hat Display variable font.
        scale: Output size relative to the 958x209 original.
        supersample: Oversampling factor used for anti-aliasing.

    Returns:
        The banner as an RGBA image.
    """
    final = (round(design["width"] * scale), round(design["height"] * scale))
    k = scale * supersample
    big = (round(design["width"] * k), round(design["height"] * k))

    def draw_lines(draw: ImageDraw.ImageDraw) -> None:
        width = max(1, round(design["line_width"] * k))
        for (x0, y0), (x1, y1) in design["lines"]:
            draw.line([(x0 * k, y0 * k), (x1 * k, y1 * k)], fill=255, width=width)

    def draw_dots(draw: ImageDraw.ImageDraw) -> None:
        r = design["dot_radius"] * k
        for x, y in design["dots"]:
            draw.ellipse([x * k - r, y * k - r, x * k + r, y * k + r], fill=255)

    def draw_shape(draw: ImageDraw.ImageDraw) -> None:
        radius = design["corner_radius"] * k
        draw.rounded_rectangle([0, 0, big[0] - 1, big[1] - 1], radius=radius, fill=255)

    def draw_text(draw: ImageDraw.ImageDraw) -> None:
        for text, spec in ((title, design["title"]), (subtitle, design["subtitle"])):
            font = load_font(font_path, spec["size"] * k, spec["weight"])
            position = (spec["centre_x"] * k, spec["baseline"] * k)
            draw.text(position, text, font=font, fill=255, anchor="ms")

    background = np.array(design["background"], dtype=np.float32)
    white = np.array(design["text_colour"], dtype=np.float32)
    rgb = np.broadcast_to(background, (final[1], final[0], 3)).copy()

    layers = (
        (coverage(big, draw_lines, final) * design["line_alpha"], "lines"),
        (coverage(big, draw_dots, final) * design["dot_alpha"], "dots"),
        (coverage(big, draw_text, final), "text"),
    )
    for alpha, name in layers:
        rgb = rgb * (1 - alpha[..., None]) + white * alpha[..., None]
        logger.info(f"Drew {name}")

    shape = coverage(big, draw_shape, final)
    rgba = np.dstack([rgb, shape * 255]).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgba)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("destination", type=Path, help="Where to write the PNG")
    parser.add_argument("--font", type=Path, required=True, help="Red Hat Display .ttf")
    parser.add_argument("--title", default="Dominic Bean", help="First line of text")
    parser.add_argument("--subtitle", default="Data Engineer", help="Second line")
    parser.add_argument(
        "--scale", type=float, default=3, help="Size relative to 958x209 (default 3)"
    )
    parser.add_argument(
        "--supersample", type=int, default=4, help="Anti-aliasing factor (default 4)"
    )
    parser.add_argument("--pattern", type=Path, default=PATTERN, help="Design JSON")
    return parser.parse_args()


def main() -> None:
    """Render the banner from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    design = json.loads(args.pattern.read_text())
    image = render(
        design, args.title, args.subtitle, args.font, args.scale, args.supersample
    )
    image.save(args.destination, optimize=True)
    logger.info(f"Wrote {args.destination} ({image.width}x{image.height})")


if __name__ == "__main__":
    main()
