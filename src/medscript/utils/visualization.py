"""Visualization utilities — overlay predictions on prescription images."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Entity type → color mapping
ENTITY_COLORS = {
    "medicine": (79, 195, 247),    # Blue
    "dosage": (0, 230, 118),       # Green
    "frequency": (255, 183, 77),   # Orange
    "duration": (206, 147, 216),   # Purple
    "instruction": (255, 138, 128),  # Red
}

DEFAULT_COLOR = (200, 200, 200)  # Gray


def visualize_entities(
    text: str,
    entities: list[dict[str, Any]],
) -> str:
    """
    Create a colored text visualization of entities (for terminal/notebooks).

    Returns ANSI-colored string for terminal display.
    """
    # ANSI color codes
    ANSI_COLORS = {
        "medicine": "\033[94m",      # Blue
        "dosage": "\033[92m",        # Green
        "frequency": "\033[93m",     # Yellow/Orange
        "duration": "\033[95m",      # Purple
        "instruction": "\033[91m",   # Red
    }
    RESET = "\033[0m"

    output = text
    for entity in sorted(entities, key=lambda e: -len(e.get("value", ""))):
        value = entity.get("value", "")
        etype = entity.get("type", "unknown")
        conf = entity.get("confidence", 0.0)
        color = ANSI_COLORS.get(etype, "")
        label = f"[{etype}:{conf:.0%}]"
        output = output.replace(value, f"{color}{value}{RESET}{label}")

    return output


def draw_confidence_bars(
    words: list[str],
    confidences: list[float],
    width: int = 600,
    bar_height: int = 25,
    padding: int = 5,
) -> np.ndarray:
    """
    Draw confidence bar chart for each word.

    Returns numpy image array.
    """
    num_words = len(words)
    height = num_words * (bar_height + padding) + 40  # Extra space for header

    image = np.full((height, width, 3), 255, dtype=np.uint8)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    # Header
    draw.text((10, 5), "Word Confidence Scores", fill=(50, 50, 50), font=font)

    for i, (word, conf) in enumerate(zip(words, confidences)):
        y = 35 + i * (bar_height + padding)

        # Word label
        label = f"{word[:20]:20s}"
        draw.text((10, y + 3), label, fill=(50, 50, 50), font=font)

        # Bar
        bar_x_start = 180
        bar_width = int((width - bar_x_start - 60) * conf)

        # Color: green (high) → red (low)
        r = int(255 * (1 - conf))
        g = int(255 * conf)
        bar_color = (r, g, 50)

        draw.rectangle(
            [(bar_x_start, y), (bar_x_start + bar_width, y + bar_height)],
            fill=bar_color,
        )

        # Confidence text
        draw.text(
            (bar_x_start + bar_width + 5, y + 3),
            f"{conf:.0%}",
            fill=(50, 50, 50),
            font=font,
        )

    return np.array(pil_image)
