# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "opencv-python-headless", "pillow"]
# ///
"""Recover the 2022 banner's design as geometry, so it can be redrawn sharply.

One-off. The original banner came from an online generator as a 958x209 PNG
with no source file. This reads that PNG and writes `pattern.json`: the
background colour, corner radius, every dot and line of the network pattern
(including dots hidden behind the text), their calibrated width and opacity,
and sub-pixel font settings for both lines of text. `make_banner.py` redraws
the banner from that file at any resolution.

Coordinates in `pattern.json` are pixel-edge based: (0, 0) is the top-left
corner of the image, so pixel (i, j) has its centre at (i + 0.5, j + 0.5).

Examples:
    git show 15348b6:profile.png > /tmp/original.png
    uv run banner/extract_pattern.py /tmp/original.png banner/pattern.json \\
        --font /tmp/RedHatDisplay.ttf
"""

import argparse
import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Excess brightness over the background (0 = background, 1 = white).
LINE_MIN = 0.015  # faintest anti-aliased line pixel
DOT_MIN = 0.12  # dots are much brighter than lines
TEXT_MIN = 0.6  # text cores are near-white
SUPPORT_MIN = 0.9  # share of a dot-to-dot segment that must sit on line pixels
OFF_CANVAS = 400.0  # how far past the edge to extend lines that leave the image
SUPERSAMPLE = 8  # anti-aliasing factor for calibration renders
TEXT_LINES = {
    # key: (text on the 2022 banner, rows containing it)
    "title": ("Dominic Bean", (50, 120)),
    "subtitle": ("Data Scientist", (120, 170)),
}

Point = tuple[float, float]


@dataclass
class Banner:
    """The original banner, measured.

    Attributes:
        excess: White-over-background coverage per pixel, from 0 to 1.
        text: Pixels covered by text (dilated), ignored for the pattern.
        dot_zone: Pixels on or next to a dot, skipped when tracing lines.
        background: Background RGB colour.
        alpha: Alpha channel.
    """

    excess: np.ndarray
    text: np.ndarray
    dot_zone: np.ndarray
    background: np.ndarray
    alpha: np.ndarray

    @property
    def size(self) -> tuple[int, int]:
        """Return (width, height)."""
        return self.excess.shape[1], self.excess.shape[0]

    def inside(self, x: float, y: float) -> bool:
        """Return whether pixel-centre coordinates round to a pixel in the image."""
        width, height = self.size
        return 0 <= round(x) < width and 0 <= round(y) < height

    def line_hit(self, x: float, y: float) -> bool | None:
        """Check for a line pixel within one pixel of (x, y).

        Returns:
            True or False, or None when the spot is text or a dot (no evidence).
        """
        xi, yi = round(x), round(y)
        if self.text[yi, xi] or self.dot_zone[yi, xi]:
            return None
        window = self.excess[max(yi - 1, 0) : yi + 2, max(xi - 1, 0) : xi + 2]
        return bool(window.max() >= LINE_MIN)


def load(source: Path) -> Banner:
    """Load the original banner and measure its layers.

    Args:
        source: The original banner PNG.

    Returns:
        The measured banner.
    """
    rgba = np.asarray(Image.open(source).convert("RGBA"), dtype=np.float32)
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    opaque = alpha > 250
    background = np.median(rgb[opaque & (rgb[..., 0] < 60)], axis=0)
    excess = np.clip((rgb[..., 0] - background[0]) / (255.0 - background[0]), 0, 1)
    excess[~opaque] = 0
    text = cv2.dilate((excess > TEXT_MIN).astype(np.uint8), np.ones((9, 9), np.uint8))
    dots = cv2.dilate((excess > DOT_MIN).astype(np.uint8), np.ones((5, 5), np.uint8))
    return Banner(excess, text.astype(bool), dots.astype(bool), background, alpha)


def find_dots(banner: Banner) -> list[Point]:
    """Find the visible dots as brightness-weighted blob centres.

    Args:
        banner: The measured banner.

    Returns:
        Dot centres in pixel-centre coordinates.
    """
    bright = ((banner.excess > DOT_MIN) & ~banner.text).astype(np.uint8)
    count, labels = cv2.connectedComponents(bright, connectivity=8)
    dots = []
    for label in range(1, count):
        ys, xs = np.nonzero(labels == label)
        weights = banner.excess[ys, xs]
        dots.append(
            (
                float(np.average(xs, weights=weights)),
                float(np.average(ys, weights=weights)),
            )
        )
    return dots


def segment_support(
    banner: Banner, start: Point, end: Point, trim: float = 4.0
) -> tuple[float, int]:
    """Measure how much of a straight segment lies on line pixels.

    Args:
        banner: The measured banner.
        start: Segment start.
        end: Segment end.
        trim: Length to ignore at each end, in pixels.

    Returns:
        Share of samples on a line, and how many samples gave evidence.
    """
    length = math.dist(start, end)
    hits = considered = 0
    for t in np.arange(trim, length - trim, 1.0):
        x = start[0] + (end[0] - start[0]) * t / length
        y = start[1] + (end[1] - start[1]) * t / length
        if not banner.inside(x, y):
            continue
        hit = banner.line_hit(x, y)
        if hit is not None:
            considered += 1
            hits += hit
    return (hits / considered if considered else 0.0), considered


def refine_direction(
    banner: Banner, origin: Point, angle: float, length: float
) -> float:
    """Refine a line's direction by weighted least squares through `origin`.

    Collects line pixels within 1.5px of the current direction and takes the
    principal axis of their offsets from `origin`, weighted by brightness.

    Args:
        banner: The measured banner.
        origin: The dot the line starts from.
        angle: Current direction estimate, in radians.
        length: How far along the line to collect pixels.

    Returns:
        The refined direction, in radians.
    """
    for _ in range(3):
        ux, uy = math.cos(angle), math.sin(angle)
        seen: dict[tuple[int, int], float] = {}
        for t in np.arange(4.0, length, 0.5):
            for offset in np.arange(-1.5, 1.51, 0.5):
                xi = round(origin[0] + ux * t - uy * offset)
                yi = round(origin[1] + uy * t + ux * offset)
                if (
                    banner.inside(xi, yi)
                    and not banner.text[yi, xi]
                    and not banner.dot_zone[yi, xi]
                    and banner.excess[yi, xi] >= LINE_MIN
                ):
                    seen[(xi, yi)] = float(banner.excess[yi, xi])
        if len(seen) < 6:
            return angle
        vectors = np.array([(x - origin[0], y - origin[1]) for x, y in seen])
        weights = np.array(list(seen.values()))
        moment = (vectors * weights[:, None]).T @ vectors
        _, eigenvectors = np.linalg.eigh(moment)
        direction = eigenvectors[:, -1]
        if direction @ (ux, uy) < 0:
            direction = -direction
        angle = math.atan2(direction[1], direction[0])
    return angle


def walk(
    banner: Banner, origin: Point, angle: float, dots: list[Point], own: int
) -> tuple[str, float, int | None]:
    """Follow a line from a dot until it ends.

    Args:
        banner: The measured banner.
        origin: The dot the line starts from.
        angle: Direction, in radians.
        dots: All visible dots.
        own: Index of the starting dot in `dots`.

    Returns:
        Why it ended ("border", "dot", "text" or "gap"), how far it got, and
        the dot it reached, if any.
    """
    ux, uy = math.cos(angle), math.sin(angle)
    misses = 0
    text_entry: float | None = None
    for t in np.arange(4.0, 3000.0, 1.0):
        x, y = origin[0] + ux * t, origin[1] + uy * t
        if not banner.inside(x, y):
            return "border", t, None
        for index, (dx, dy) in enumerate(dots):
            if index != own and math.hypot(x - dx, y - dy) < 4.0:
                return "dot", t, index
        if banner.text[round(y), round(x)]:
            text_entry = t if text_entry is None else text_entry
            continue
        hit = banner.line_hit(x, y)
        if hit is None:
            continue
        if hit:
            misses, text_entry = 0, None
            continue
        misses += 1
        if misses >= 5:
            if text_entry is not None:
                return "text", text_entry, None
            return "gap", t - misses, None
    return "gap", 3000.0, None


def extend_to(origin: Point, angle: float, distance: float) -> list[float]:
    """Return the point `distance` along a direction, rounded for the JSON."""
    return [
        round(origin[0] + math.cos(angle) * distance, 3),
        round(origin[1] + math.sin(angle) * distance, 3),
    ]


def ray_intersection(a: tuple[Point, float], b: tuple[Point, float]) -> Point | None:
    """Intersect two forward rays given as (origin, angle).

    Returns:
        The intersection point, or None if they are parallel or meet behind.
    """
    (ax, ay), angle_a = a
    (bx, by), angle_b = b
    dax, day = math.cos(angle_a), math.sin(angle_a)
    dbx, dby = math.cos(angle_b), math.sin(angle_b)
    det = dax * (-dby) - day * (-dbx)
    if abs(det) < 1e-6:
        return None
    s = ((bx - ax) * (-dby) - (by - ay) * (-dbx)) / det
    u = (dax * (by - ay) - day * (bx - ax)) / det
    if s <= 0 or u <= 0:
        return None
    return ax + dax * s, ay + day * s


def distance_to_segment(point: Point, start: Point, end: Point) -> float:
    """Return the distance from `point` to the segment from `start` to `end`."""
    (px, py), (x0, y0), (x1, y1) = point, start, end
    dx, dy = x1 - x0, y1 - y0
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def closest_point(rays: list[tuple[Point, float]]) -> Point:
    """Return the point closest, in least squares, to several lines."""
    a = np.zeros((2, 2))
    b = np.zeros(2)
    for (x, y), angle in rays:
        d = np.array([math.cos(angle), math.sin(angle)])
        projector = np.eye(2) - np.outer(d, d)
        a += projector
        b += projector @ np.array([x, y])
    px, py = np.linalg.solve(a, b)
    return float(px), float(py)


def trace_pattern(
    banner: Banner, dots: list[Point]
) -> tuple[list[Point], list[list[list[float]]]]:
    """Trace every line of the pattern and recover dots hidden by text.

    Args:
        banner: The measured banner.
        dots: Visible dots.

    Returns:
        Hidden dots, and lines as pairs of points in pixel-centre coordinates.
    """
    lines: list[list[list[float]]] = []
    linked: dict[int, list[float]] = {i: [] for i in range(len(dots))}
    pairs: dict[tuple[int, int], int] = {}  # dot-to-dot line -> index in `lines`

    def add_link(i: int, j: int) -> None:
        a, b = dots[i], dots[j]
        pairs[(i, j)] = len(lines)
        lines.append(
            [[round(a[0], 3), round(a[1], 3)], [round(b[0], 3), round(b[1], 3)]]
        )
        linked[i].append(math.atan2(b[1] - a[1], b[0] - a[0]))
        linked[j].append(math.atan2(a[1] - b[1], a[0] - b[0]))

    # Dot to dot: test every pair directly. Lines end at dots, so a pair with
    # another dot on the segment between them is two lines, not one.
    for i in range(len(dots)):
        for j in range(i + 1, len(dots)):
            blocked = any(
                k not in (i, j) and distance_to_segment(dots[k], dots[i], dots[j]) < 3.0
                for k in range(len(dots))
            )
            if blocked:
                continue
            share, considered = segment_support(banner, dots[i], dots[j])
            if considered >= 4 and share >= SUPPORT_MIN:
                add_link(i, j)
    logger.info(f"Dot-to-dot lines: {len(lines)}")

    # Every other line leaving a dot: scan directions, refine, then follow.
    dangling: list[tuple[Point, float]] = []
    counts = {"border": 0, "dot": 0, "text": 0, "gap": 0}
    for i, origin in enumerate(dots):
        candidates = []
        for degrees in np.arange(0, 360, 0.5):
            angle = math.radians(degrees)
            if any(
                abs(math.remainder(angle - d, math.tau)) < math.radians(3)
                for d in linked[i]
            ):
                continue
            reach = 30.0
            end = (
                origin[0] + math.cos(angle) * reach,
                origin[1] + math.sin(angle) * reach,
            )
            share, considered = segment_support(banner, origin, end)
            if considered >= 4 and share >= 0.95:
                candidates.append(angle)
        # Group neighbouring directions; each group is one line.
        groups: list[list[float]] = []
        for angle in candidates:
            if groups and abs(
                math.remainder(angle - groups[-1][-1], math.tau)
            ) <= math.radians(1.01):
                groups[-1].append(angle)
            else:
                groups.append([angle])
        if len(groups) > 1 and abs(
            math.remainder(groups[0][0] - groups[-1][-1], math.tau)
        ) <= math.radians(1.01):
            groups[0] = groups.pop() + groups[0]
        for group in groups:
            mean = math.atan2(
                sum(math.sin(a) for a in group), sum(math.cos(a) for a in group)
            )
            angle = refine_direction(banner, origin, mean, 40.0)
            reason, distance, other = walk(banner, origin, angle, dots, i)
            angle = refine_direction(banner, origin, angle, max(distance, 10.0))
            reason, distance, other = walk(banner, origin, angle, dots, i)
            counts[reason] += 1
            if any(
                abs(math.remainder(angle - d, math.tau)) < math.radians(3)
                for d in linked[i]
            ):
                continue  # converged onto a line already found
            if reason == "border":
                lines.append(
                    [
                        [round(origin[0], 3), round(origin[1], 3)],
                        extend_to(origin, angle, distance + OFF_CANVAS),
                    ]
                )
                linked[i].append(angle)
            elif reason == "dot":
                add_link(i, other)
            elif reason == "text":
                dangling.append((origin, angle))
    logger.info(f"Lines followed from dots: {counts}")

    # Dots hidden behind text: where two or more dangling lines meet in text.
    zone = cv2.dilate(banner.text.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(
        bool
    )
    meetings = []
    for a in range(len(dangling)):
        for b in range(a + 1, len(dangling)):
            point = ray_intersection(dangling[a], dangling[b])
            if (
                point
                and banner.inside(*point)
                and zone[round(point[1]), round(point[0])]
            ):
                meetings.append((point, {a, b}))
    clusters: list[tuple[list[Point], set[int]]] = []
    for point, rays in meetings:
        for points, members in clusters:
            if math.dist(point, np.mean(points, axis=0)) < 3.0:
                points.append(point)
                members |= rays
                break
        else:
            clusters.append(([point], set(rays)))

    def add_to_hidden(origin: Point, angle: float, dot: Point) -> None:
        # Keep the measured direction; end where it passes the hidden dot.
        t = (dot[0] - origin[0]) * math.cos(angle) + (dot[1] - origin[1]) * math.sin(
            angle
        )
        lines.append(
            [[round(origin[0], 3), round(origin[1], 3)], extend_to(origin, angle, t)]
        )

    hidden: list[Point] = []
    used: set[int] = set()
    for _, members in sorted(clusters, key=lambda c: -len(c[1])):
        members = members - used
        if len(members) >= 2:
            dot = closest_point([dangling[m] for m in members])
            hidden.append(dot)
            used |= members
            for m in members:
                origin, angle = dangling[m]
                add_to_hidden(origin, angle, dot)
                linked[dots.index(origin)].append(angle)
    logger.info(
        f"Hidden dots: {len(hidden)} "
        f"(from {len(used)} of {len(dangling)} lines into text)"
    )

    # A dot-to-dot line through a hidden dot is really two lines meeting there.
    split = set()
    for (i, j), index in pairs.items():
        for dot in hidden:
            through = distance_to_segment(dot, dots[i], dots[j]) < 3.0
            if through and min(math.dist(dot, dots[i]), math.dist(dot, dots[j])) > 5:
                split.add(index)
                for origin in (dots[i], dots[j]):
                    angle = math.atan2(dot[1] - origin[1], dot[0] - origin[0])
                    angle = refine_direction(
                        banner, origin, angle, math.dist(origin, dot)
                    )
                    add_to_hidden(origin, angle, dot)
                break
    lines[:] = [line for index, line in enumerate(lines) if index not in split]
    logger.info(f"Dot-to-dot lines split at a hidden dot: {len(split)}")

    # Visible dots with a line towards a hidden dot that the scan missed.
    extra = 0
    for dot in hidden:
        for i, origin in enumerate(dots):
            if math.dist(origin, dot) > 200:
                continue
            angle = math.atan2(dot[1] - origin[1], dot[0] - origin[0])
            if any(
                abs(math.remainder(angle - d, math.tau)) < math.radians(3)
                for d in linked[i]
            ):
                continue
            share, considered = segment_support(banner, origin, dot)
            if considered >= 4 and share >= SUPPORT_MIN:
                angle = refine_direction(banner, origin, angle, math.dist(origin, dot))
                add_to_hidden(origin, angle, dot)
                linked[i].append(angle)
                extra += 1
    logger.info(f"Extra lines to hidden dots: {extra}")
    for dot in hidden:
        logger.info(f"  hidden dot at ({dot[0]:.1f}, {dot[1]:.1f})")

    lines += trace_edge_to_edge(banner, lines)
    return hidden, lines


def trace_edge_to_edge(
    banner: Banner, lines: list[list[list[float]]]
) -> list[list[list[float]]]:
    """Find lines that cross the image without touching a visible dot.

    These cut across corners, with both ends off-canvas. Looks for straight
    runs of line pixels not explained by `lines`, fits each by weighted least
    squares and keeps it if it is on line pixels from edge to edge.

    Args:
        banner: The measured banner.
        lines: Lines found so far, in pixel-centre coordinates.

    Returns:
        New lines, with both ends pushed well outside the image.
    """
    width, height = banner.size
    explained = np.zeros((height, width), np.uint8)
    for (x0, y0), (x1, y1) in lines:
        cv2.line(explained, (round(x0), round(y0)), (round(x1), round(y1)), 1, 5)
    residual = (
        (banner.excess >= LINE_MIN)
        & ~banner.text
        & ~banner.dot_zone
        & ~explained.astype(bool)
    )
    segments = cv2.HoughLinesP(
        residual.astype(np.uint8) * 255,
        1,
        np.pi / 720,
        threshold=15,
        minLineLength=20,
        maxLineGap=5,
    )
    found: list[list[list[float]]] = []
    for x0, y0, x1, y1 in [] if segments is None else segments[:, 0]:
        # Weighted least-squares fit to the residual pixels near the segment.
        ys, xs = np.nonzero(residual)
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        distance = np.abs((xs - x0) * dy - (ys - y0) * dx) / length
        along = ((xs - x0) * dx + (ys - y0) * dy) / length
        near = (distance <= 1.5) & (along >= -2) & (along <= length + 2)
        if near.sum() < 10:
            continue
        points = np.stack([xs[near], ys[near]], axis=1).astype(float)
        weights = banner.excess[ys[near], xs[near]]
        centre = np.average(points, axis=0, weights=weights)
        offsets = points - centre
        _, eigenvectors = np.linalg.eigh((offsets * weights[:, None]).T @ offsets)
        angle = math.atan2(eigenvectors[1, -1], eigenvectors[0, -1])
        mid = (float(centre[0]), float(centre[1]))
        ends = []
        for direction in (angle, angle + math.pi):
            t = 0.0
            while banner.inside(
                mid[0] + math.cos(direction) * t, mid[1] + math.sin(direction) * t
            ):
                t += 0.5
            ends.append(
                (mid[0] + math.cos(direction) * t, mid[1] + math.sin(direction) * t)
            )
        share, considered = segment_support(banner, ends[1], ends[0], trim=1.0)
        if considered >= 15 and share >= 0.85:
            found.append(
                [
                    extend_to(
                        mid, angle + math.pi, math.dist(mid, ends[1]) + OFF_CANVAS
                    ),
                    extend_to(mid, angle, math.dist(mid, ends[0]) + OFF_CANVAS),
                ]
            )
            cv2.line(
                explained,
                (round(ends[0][0]), round(ends[0][1])),
                (round(ends[1][0]), round(ends[1][1])),
                1,
                5,
            )
            residual &= ~explained.astype(bool)
    logger.info(f"Edge-to-edge lines: {len(found)}")
    return found


def coverage(
    size: tuple[int, int], draw_fn: Callable[[ImageDraw.ImageDraw], None]
) -> np.ndarray:
    """Draw a mask at `SUPERSAMPLE` times `size` and average it down.

    Args:
        size: Output (width, height).
        draw_fn: Draws in white on an `ImageDraw.ImageDraw` at the large size.

    Returns:
        Coverage between 0 and 1, shape (height, width).
    """
    big = (size[0] * SUPERSAMPLE, size[1] * SUPERSAMPLE)
    mask = Image.new("L", big, 0)
    draw_fn(ImageDraw.Draw(mask))
    return np.asarray(mask.resize(size, Image.Resampling.BOX), dtype=np.float32) / 255.0


def calibrate(
    banner: Banner, dots: list[Point], lines: list[list[list[float]]]
) -> dict[str, float]:
    """Fit line width and opacity, and dot radius and opacity, by rendering.

    Args:
        banner: The measured banner.
        dots: Visible dots, in pixel-edge coordinates.
        lines: Lines, in pixel-edge coordinates.

    Returns:
        Line width, line opacity, dot radius and dot opacity.
    """
    k = SUPERSAMPLE
    size = banner.size
    usable = ~banner.text & (banner.alpha > 250)
    near_dot = banner.dot_zone & usable
    on_line = usable & ~banner.dot_zone

    best_line = None
    for width in np.arange(0.7, 3.01, 0.05):

        def draw_lines(draw: ImageDraw.ImageDraw, width: float = width) -> None:
            for (x0, y0), (x1, y1) in lines:
                draw.line(
                    [(x0 * k, y0 * k), (x1 * k, y1 * k)],
                    fill=255,
                    width=max(1, round(width * k)),
                )

        c = coverage(size, draw_lines)
        target = banner.excess[on_line]
        model = c[on_line]
        opacity = float((model * target).sum() / (model * model).sum())
        error = float(((opacity * model - target) ** 2).sum())
        if best_line is None or error < best_line[0]:
            best_line = (error, float(width), opacity, c)
    _, line_width, line_alpha, line_cov = best_line
    logger.info(f"Lines: width {line_width:.2f}px, opacity {line_alpha:.4f}")

    best_dot = None
    lines_part = line_alpha * line_cov[near_dot]
    for radius in np.arange(1.0, 2.61, 0.05):

        def draw_dots(draw: ImageDraw.ImageDraw, radius: float = radius) -> None:
            r = radius * k
            for x, y in dots:
                draw.ellipse([x * k - r, y * k - r, x * k + r, y * k + r], fill=255)

        d = coverage(size, draw_dots)[near_dot]
        # Composite: excess = lines + dot_alpha * d * (1 - lines).
        basis = d * (1 - lines_part)
        target = banner.excess[near_dot] - lines_part
        opacity = float((basis * target).sum() / (basis * basis).sum())
        error = float(((opacity * basis - target) ** 2).sum())
        if best_dot is None or error < best_dot[0]:
            best_dot = (error, float(radius), opacity)
    _, dot_radius, dot_alpha = best_dot
    logger.info(f"Dots: radius {dot_radius:.2f}px, opacity {dot_alpha:.4f}")
    return {
        "line_width": round(line_width, 3),
        "line_alpha": round(line_alpha, 4),
        "dot_radius": round(dot_radius, 3),
        "dot_alpha": round(dot_alpha, 4),
    }


def fit_text(
    banner: Banner, text: str, rows: tuple[int, int], font_path: Path
) -> dict[str, float]:
    """Fit font weight, size, centre and baseline to one line of text.

    Renders the way `make_banner.py` does (centred on the baseline, drawn
    oversized and averaged down) and refines each setting in turn.

    Args:
        banner: The measured banner.
        text: The text exactly as it appears.
        rows: First and last row containing the text.
        font_path: Red Hat Display variable font.

    Returns:
        Weight, size, horizontal centre and baseline, in pixel-edge coordinates.
    """
    band = np.zeros_like(banner.text)
    band[rows[0] : rows[1]] = banner.excess[rows[0] : rows[1]] > 0.5
    ys, xs = np.nonzero(band)
    pad = 8
    x0, y0 = int(xs.min()) - pad, int(ys.min()) - pad
    x1, y1 = int(xs.max()) + pad + 1, int(ys.max()) + pad + 1
    observed = banner.excess[y0:y1, x0:x1]
    size = (x1 - x0, y1 - y0)
    k = SUPERSAMPLE

    def error(spec: dict[str, float]) -> float:
        font = ImageFont.truetype(str(font_path), spec["size"] * k)
        font.set_variation_by_axes([spec["weight"]])

        def draw_text(draw: ImageDraw.ImageDraw) -> None:
            position = ((spec["centre_x"] - x0) * k, (spec["baseline"] - y0) * k)
            draw.text(position, text, font=font, fill=255, anchor="ms")

        rendered = coverage(size, draw_text)
        considered = (rendered > 0.1) | (observed > 0.15)
        return float(np.mean((rendered[considered] - observed[considered]) ** 2))

    spec = {
        "weight": 400.0,
        "size": float(ys.max() - ys.min()) * 1.35,
        "centre_x": (xs.min() + xs.max() + 1) / 2,
        "baseline": float(ys.max() + 1),
    }
    steps = {"size": 2.0, "weight": 50.0, "centre_x": 1.0, "baseline": 1.0}
    for _ in range(7):
        for key in ("size", "weight", "centre_x", "baseline"):
            best_value, best_error = spec[key], error(spec)
            for factor in np.arange(-4, 5):
                trial = dict(spec, **{key: spec[key] + factor * steps[key] / 2})
                if key == "weight" and not 300 <= trial[key] <= 700:
                    continue
                trial_error = error(trial)
                if trial_error < best_error:
                    best_value, best_error = trial[key], trial_error
            spec[key] = best_value
        steps = {key: value / 2.5 for key, value in steps.items()}
    result = {
        "weight": round(spec["weight"]),
        "size": round(float(spec["size"]), 3),
        "centre_x": round(float(spec["centre_x"]), 3),
        "baseline": round(float(spec["baseline"]), 3),
    }
    logger.info(f"'{text}': {result} (error {error(spec):.5f})")
    return result


def to_edges(points: list[list[float]]) -> list[float]:
    """Convert a pixel-centre point to pixel-edge coordinates."""
    return [round(points[0] + 0.5, 3), round(points[1] + 0.5, 3)]


def extract(source: Path, destination: Path, font_path: Path) -> None:
    """Extract the banner design from `source` into `destination`.

    Args:
        source: The original banner PNG.
        destination: Where to write the JSON design.
        font_path: Red Hat Display variable font.
    """
    banner = load(source)
    dots = find_dots(banner)
    logger.info(f"Visible dots: {len(dots)}")
    hidden, lines = trace_pattern(banner, dots)

    all_dots = [to_edges([x, y]) for x, y in dots + hidden]
    edge_lines = [[to_edges(a), to_edges(b)] for a, b in lines]
    width, height = banner.size
    design = {
        "source": "2022 banner (dombean/dombean@15348b6:profile.png), 958x209",
        "width": width,
        "height": height,
        "corner_radius": int(np.argmax(banner.alpha[0] > 250)) + 1,
        "background": [int(c) for c in banner.background],
        "text_colour": [255, 255, 255],
        **calibrate(banner, all_dots, edge_lines),
        **{
            key: fit_text(banner, text, rows, font_path)
            for key, (text, rows) in TEXT_LINES.items()
        },
        "dots": all_dots,
        "lines": edge_lines,
    }
    destination.write_text(json.dumps(design, indent=1))
    logger.info(f"Wrote {destination}: {len(all_dots)} dots, {len(edge_lines)} lines")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="The original 2022 banner PNG")
    parser.add_argument("destination", type=Path, help="Where to write pattern.json")
    parser.add_argument("--font", type=Path, required=True, help="Red Hat Display .ttf")
    return parser.parse_args()


def main() -> None:
    """Run the extraction from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    extract(args.source, args.destination, args.font)


if __name__ == "__main__":
    main()
