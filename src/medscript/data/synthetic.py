"""
Synthetic prescription data generator.

Generates realistic-looking prescription images with perfect ground truth labels
using handwriting-style fonts and controlled distortions. This is critical since
no custom dataset is available.
"""

from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from medscript.utils.logging import get_logger
from medscript.utils.medical_vocab import (
    DOSAGE_FORMS,
    DOSAGE_UNITS,
    DRUG_NAMES,
    DURATION_TERMS,
    FREQUENCY_TERMS,
    INSTRUCTION_TERMS,
)

logger = get_logger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

# Handwriting-style fonts (Google Fonts that mimic handwriting)
# Users need to download these fonts to a local fonts/ directory
HANDWRITING_FONTS = [
    "Caveat-Regular.ttf",
    "Caveat-Bold.ttf",
    "DancingScript-Regular.ttf",
    "Kalam-Regular.ttf",
    "Kalam-Bold.ttf",
    "IndieFlower-Regular.ttf",
    "PatrickHand-Regular.ttf",
    "Satisfy-Regular.ttf",
    "CoveredByYourGrace-Regular.ttf",
    "HomemadeApple-Regular.ttf",
]

# Fallback to system default if fonts not found
DEFAULT_FONT_SIZE_RANGE = (18, 32)

# Doctor name prefixes
DOCTOR_PREFIXES = ["Dr.", "DR.", "Dr"]
DOCTOR_FIRST_NAMES = [
    "Rajesh", "Amit", "Priya", "Sunita", "Vikram", "Anita", "Sanjay",
    "Meera", "Arun", "Kavita", "Suresh", "Deepa", "Rakesh", "Neha",
    "Mohan", "Pooja", "Ashok", "Ritu", "Dinesh", "Swati",
]
DOCTOR_LAST_NAMES = [
    "Sharma", "Gupta", "Patel", "Singh", "Kumar", "Verma", "Joshi",
    "Agarwal", "Mehta", "Reddy", "Nair", "Iyer", "Rao", "Chatterjee",
    "Banerjee", "Mishra", "Pandey", "Saxena", "Kapoor", "Malhotra",
]
SPECIALIZATIONS = [
    "MBBS, MD", "MBBS", "MD, DM", "MBBS, MS", "BDS",
    "MBBS, DNB", "MD (Med)", "MS (Ortho)", "MD (Paed)",
]
CLINIC_NAMES = [
    "City Clinic", "Health Care Center", "Family Clinic",
    "Apollo Clinic", "Max Hospital", "Fortis Clinic",
    "Life Care Hospital", "Shanti Nursing Home",
    "Sunrise Medical Center", "Saket Hospital",
]

# Patient demographics
PATIENT_NAMES = [
    "Rahul", "Priya", "Amit", "Neha", "Vikram", "Sunita", "Arjun",
    "Kavita", "Rohit", "Anita", "Deepak", "Meera", "Suresh", "Ritu",
]
PATIENT_AGES = list(range(5, 85))
PATIENT_GENDERS = ["M", "F"]


@dataclass
class PrescriptionLine:
    """A single line in a prescription."""
    medicine: str
    dosage_form: str
    dosage: str
    frequency: str
    duration: str
    instruction: str = ""
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "medicine": self.medicine,
            "dosage_form": self.dosage_form,
            "dosage": self.dosage,
            "frequency": self.frequency,
            "duration": self.duration,
            "instruction": self.instruction,
            "raw_text": self.raw_text,
        }


@dataclass
class SyntheticPrescription:
    """A complete synthetic prescription."""
    doctor_name: str = ""
    doctor_qualifications: str = ""
    clinic_name: str = ""
    patient_name: str = ""
    patient_age: int = 0
    patient_gender: str = ""
    date: str = ""
    lines: list[PrescriptionLine] = field(default_factory=list)
    full_text: str = ""
    image_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "doctor_name": self.doctor_name,
            "doctor_qualifications": self.doctor_qualifications,
            "clinic_name": self.clinic_name,
            "patient_name": self.patient_name,
            "patient_age": self.patient_age,
            "patient_gender": self.patient_gender,
            "date": self.date,
            "lines": [line.to_dict() for line in self.lines],
            "full_text": self.full_text,
            "image_path": self.image_path,
        }


class SyntheticPrescriptionGenerator:
    """Generate synthetic prescription images with ground truth labels."""

    def __init__(
        self,
        fonts_dir: str | Path = "data/fonts",
        output_dir: str | Path = "data/synthetic",
        image_width: int = 800,
        image_height: int = 1100,
        seed: int = 42,
    ) -> None:
        self.fonts_dir = Path(fonts_dir)
        self.output_dir = Path(output_dir)
        self.image_width = image_width
        self.image_height = image_height
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Load available fonts
        self._fonts: list[Path] = []
        self._load_fonts()

    def _load_fonts(self) -> None:
        """Load available handwriting fonts."""
        if self.fonts_dir.exists():
            for font_file in self.fonts_dir.glob("*.ttf"):
                self._fonts.append(font_file)
            for font_file in self.fonts_dir.glob("*.otf"):
                self._fonts.append(font_file)

        if not self._fonts:
            logger.warning(
                "no_fonts_found",
                fonts_dir=str(self.fonts_dir),
                message="Will use PIL default font. Download handwriting fonts for better results.",
            )

    def _get_font(self, size: int | None = None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Get a random handwriting font."""
        if size is None:
            size = self.rng.randint(*DEFAULT_FONT_SIZE_RANGE)

        if self._fonts:
            font_path = self.rng.choice(self._fonts)
            try:
                return ImageFont.truetype(str(font_path), size)
            except Exception:
                pass

        # Fallback to default
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()

    def _generate_prescription_data(self) -> SyntheticPrescription:
        """Generate random prescription content."""
        rx = SyntheticPrescription()

        # Doctor info
        first = self.rng.choice(DOCTOR_FIRST_NAMES)
        last = self.rng.choice(DOCTOR_LAST_NAMES)
        prefix = self.rng.choice(DOCTOR_PREFIXES)
        rx.doctor_name = f"{prefix} {first} {last}"
        rx.doctor_qualifications = self.rng.choice(SPECIALIZATIONS)
        rx.clinic_name = self.rng.choice(CLINIC_NAMES)

        # Patient info
        rx.patient_name = self.rng.choice(PATIENT_NAMES)
        rx.patient_age = self.rng.choice(PATIENT_AGES)
        rx.patient_gender = self.rng.choice(PATIENT_GENDERS)

        # Date
        day = self.rng.randint(1, 28)
        month = self.rng.randint(1, 12)
        year = self.rng.randint(2023, 2026)
        rx.date = f"{day:02d}/{month:02d}/{year}"

        # Prescription lines (3-7 medicines)
        num_lines = self.rng.randint(3, 7)
        used_drugs: set[str] = set()

        for _ in range(num_lines):
            # Pick a unique drug
            drug = self.rng.choice(DRUG_NAMES)
            while drug in used_drugs:
                drug = self.rng.choice(DRUG_NAMES)
            used_drugs.add(drug)

            form = self.rng.choice(DOSAGE_FORMS)
            dosage = self.rng.choice(DOSAGE_UNITS)
            freq = self.rng.choice(FREQUENCY_TERMS[:14])  # Prefer abbreviated forms
            dur = self.rng.choice(DURATION_TERMS[6:20])  # Prefer specific durations
            instruction = self.rng.choice(INSTRUCTION_TERMS) if self.rng.random() > 0.6 else ""

            # Build raw text in common prescription style
            raw_parts = [f"{drug} {form} {dosage}"]
            raw_parts.append(freq)
            raw_parts.append(f"x {dur}" if not dur.startswith("for") else dur)
            if instruction:
                raw_parts.append(f"({instruction})")

            raw_text = " ".join(raw_parts)

            line = PrescriptionLine(
                medicine=drug,
                dosage_form=form,
                dosage=dosage,
                frequency=freq,
                duration=dur,
                instruction=instruction,
                raw_text=raw_text,
            )
            rx.lines.append(line)

        # Build full text
        full_lines = [
            rx.doctor_name,
            rx.doctor_qualifications,
            rx.clinic_name,
            "",
            f"Patient: {rx.patient_name}  Age: {rx.patient_age}/{rx.patient_gender}  Date: {rx.date}",
            "",
            "Rx",
        ]
        for i, line in enumerate(rx.lines, 1):
            full_lines.append(f"{i}. {line.raw_text}")

        rx.full_text = "\n".join(full_lines)
        return rx

    def _create_paper_background(self) -> np.ndarray:
        """Create a realistic paper texture background."""
        # Start with off-white
        base_value = self.rng.randint(235, 250)
        bg = np.full(
            (self.image_height, self.image_width, 3),
            base_value,
            dtype=np.uint8,
        )

        # Add paper texture noise
        noise = self.np_rng.normal(0, 3, bg.shape).astype(np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Optionally add ruled lines
        if self.rng.random() > 0.4:
            line_spacing = self.rng.randint(28, 40)
            line_color = (200, 210, 220)  # Light blue ruled lines
            start_y = self.rng.randint(150, 200)
            for y in range(start_y, self.image_height - 50, line_spacing):
                cv2.line(bg, (40, y), (self.image_width - 40, y), line_color, 1)

        # Add a margin line occasionally
        if self.rng.random() > 0.6:
            margin_x = self.rng.randint(60, 90)
            cv2.line(bg, (margin_x, 0), (margin_x, self.image_height), (220, 180, 180), 1)

        return bg

    def _apply_handwriting_effects(self, image: np.ndarray) -> np.ndarray:
        """Apply effects that simulate real handwriting artifacts."""
        # Random slight rotation (tilted writing)
        angle = self.np_rng.uniform(-2, 2)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(
            image, matrix, (w, h),
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Slight Gaussian blur (ink spread)
        if self.rng.random() > 0.5:
            kernel_size = self.rng.choice([1, 3])
            if kernel_size > 1:
                image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0.5)

        # Random brightness/contrast variation
        alpha = self.np_rng.uniform(0.85, 1.15)  # Contrast
        beta = self.np_rng.uniform(-10, 10)       # Brightness
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # Salt-and-pepper noise (paper imperfections)
        if self.rng.random() > 0.6:
            noise_ratio = self.np_rng.uniform(0.001, 0.005)
            num_noise = int(noise_ratio * image.size / 3)
            for _ in range(num_noise):
                y = self.rng.randint(0, h - 1)
                x = self.rng.randint(0, w - 1)
                color = self.rng.choice([0, 255])
                image[y, x] = [color, color, color]

        return image

    def generate_single(self, idx: int) -> SyntheticPrescription:
        """Generate a single synthetic prescription image with annotations."""
        rx = self._generate_prescription_data()

        # Create background
        bg = self._create_paper_background()
        pil_image = Image.fromarray(bg)
        draw = ImageDraw.Draw(pil_image)

        # Font settings
        header_font = self._get_font(self.rng.randint(22, 30))
        subheader_font = self._get_font(self.rng.randint(16, 22))
        body_font = self._get_font(self.rng.randint(18, 26))
        small_font = self._get_font(self.rng.randint(14, 18))

        # Ink color (varies from black to dark blue/gray)
        ink_colors = [
            (10, 10, 30),    # Near-black
            (0, 0, 100),     # Dark blue (pen)
            (20, 20, 60),    # Dark navy
            (30, 30, 30),    # Charcoal
            (0, 50, 120),    # Medium blue
        ]
        ink_color = self.rng.choice(ink_colors)

        # Draw header
        y_pos = self.rng.randint(30, 60)
        x_margin = self.rng.randint(50, 80)

        # Doctor name (larger, sometimes centered)
        draw.text((x_margin, y_pos), rx.doctor_name, fill=ink_color, font=header_font)
        y_pos += self.rng.randint(30, 40)

        # Qualifications
        draw.text((x_margin, y_pos), rx.doctor_qualifications, fill=ink_color, font=small_font)
        y_pos += self.rng.randint(22, 30)

        # Clinic name
        draw.text((x_margin, y_pos), rx.clinic_name, fill=ink_color, font=small_font)
        y_pos += self.rng.randint(30, 50)

        # Horizontal separator line (drawn casually)
        sep_y = y_pos
        draw.line(
            [(x_margin, sep_y), (self.image_width - x_margin, sep_y)],
            fill=ink_color,
            width=1,
        )
        y_pos += self.rng.randint(15, 25)

        # Patient info line
        patient_line = f"Pt: {rx.patient_name}   Age: {rx.patient_age}/{rx.patient_gender}   Date: {rx.date}"
        draw.text((x_margin, y_pos), patient_line, fill=ink_color, font=small_font)
        y_pos += self.rng.randint(30, 45)

        # "Rx" symbol (larger, stylized)
        rx_font = self._get_font(self.rng.randint(28, 38))
        draw.text((x_margin, y_pos), "Rx", fill=ink_color, font=rx_font)
        y_pos += self.rng.randint(35, 50)

        # Prescription lines
        for i, line in enumerate(rx.lines, 1):
            if y_pos > self.image_height - 100:
                break  # Don't overflow

            # Line number
            line_text = f"{i}) {line.raw_text}"

            # Random horizontal jitter (natural hand movement)
            x_jitter = self.rng.randint(-5, 10)
            y_jitter = self.rng.randint(-2, 3)

            draw.text(
                (x_margin + x_jitter, y_pos + y_jitter),
                line_text,
                fill=ink_color,
                font=body_font,
            )
            y_pos += self.rng.randint(32, 48)

        # Signature area (scribble at bottom)
        sig_y = max(y_pos + 40, self.image_height - 150)
        sig_x = self.image_width - 200

        # Draw a wavy signature line
        points = []
        for sx in range(sig_x, sig_x + 120, 3):
            sy = sig_y + int(8 * np.sin(sx * 0.1 + self.rng.random() * 3))
            points.append((sx, sy))
        if len(points) >= 2:
            draw.line(points, fill=ink_color, width=2)

        draw.text((sig_x, sig_y + 15), "Signature", fill=ink_color, font=small_font)

        # Convert back to numpy and apply effects
        image_np = np.array(pil_image)
        image_np = self._apply_handwriting_effects(image_np)

        # Save image
        image_filename = f"synth_{idx:06d}.png"
        image_path = self.images_dir / image_filename
        cv2.imwrite(str(image_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        rx.image_path = f"images/{image_filename}"

        logger.debug("generated_prescription", idx=idx, lines=len(rx.lines))
        return rx

    def generate_batch(
        self,
        count: int = 1000,
        start_idx: int = 0,
    ) -> list[SyntheticPrescription]:
        """
        Generate a batch of synthetic prescriptions.

        Args:
            count: Number of prescriptions to generate
            start_idx: Starting index for filenames

        Returns:
            List of SyntheticPrescription objects
        """
        logger.info("generating_batch", count=count)

        prescriptions: list[SyntheticPrescription] = []

        for i in range(count):
            idx = start_idx + i
            rx = self.generate_single(idx)
            prescriptions.append(rx)

            if (i + 1) % 100 == 0:
                logger.info("generation_progress", completed=i + 1, total=count)

        # Save annotations
        annotations = [rx.to_dict() for rx in prescriptions]
        annotations_path = self.output_dir / "annotations.json"
        with open(annotations_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)

        logger.info(
            "batch_complete",
            total=len(prescriptions),
            output_dir=str(self.output_dir),
        )

        return prescriptions


# ── Convenience Function ─────────────────────────────────────────────────────


def generate_synthetic_dataset(
    count: int = 20000,
    output_dir: str | Path = "data/synthetic",
    fonts_dir: str | Path = "data/fonts",
    seed: int = 42,
) -> Path:
    """
    Generate a complete synthetic prescription dataset.

    Args:
        count: Number of samples to generate
        output_dir: Output directory
        fonts_dir: Directory containing .ttf handwriting fonts
        seed: Random seed for reproducibility

    Returns:
        Path to the output directory
    """
    generator = SyntheticPrescriptionGenerator(
        fonts_dir=fonts_dir,
        output_dir=output_dir,
        seed=seed,
    )
    generator.generate_batch(count=count)
    return Path(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic prescriptions")
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--output", type=str, default="data/synthetic")
    parser.add_argument("--fonts", type=str, default="data/fonts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from medscript.utils.logging import setup_logging
    setup_logging(level="INFO")

    generate_synthetic_dataset(
        count=args.count,
        output_dir=args.output,
        fonts_dir=args.fonts,
        seed=args.seed,
    )
