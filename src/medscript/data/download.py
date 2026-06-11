"""Download open-source medical handwriting datasets from HuggingFace."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from medscript.utils.logging import get_logger

logger = get_logger(__name__)

# ── Dataset Registry ─────────────────────────────────────────────────────────

DATASETS = {
    "avi-kai/Medical_Prescription_Handwritten_Words": {
        "description": "Individual handwritten medical words from prescriptions",
        "split": "train",
        "output_dir": "medical_words",
    },
    "chinmays18/medical-prescription-dataset": {
        "description": "Medical prescription images for Donut fine-tuning",
        "split": "train",
        "output_dir": "medical_prescriptions",
    },
}


def download_hf_dataset(
    dataset_name: str,
    output_dir: Path,
    split: str = "train",
    max_samples: int | None = None,
) -> Path:
    """
    Download a HuggingFace dataset and save images + annotations locally.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "avi-kai/...")
        output_dir: Directory to save downloaded data
        split: Dataset split to download
        max_samples: Optional limit on number of samples

    Returns:
        Path to the output directory
    """
    from datasets import load_dataset

    logger.info(
        "downloading_dataset",
        dataset=dataset_name,
        split=split,
        max_samples=max_samples,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        logger.error("dataset_download_failed", dataset=dataset_name, error=str(e))
        raise

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    annotations: list[dict[str, Any]] = []

    for idx, sample in enumerate(dataset):
        # Handle different dataset schemas
        image = None
        text = ""

        # Try common column names for images
        for img_col in ["image", "pixel_values", "img", "input_image"]:
            if img_col in sample:
                image = sample[img_col]
                break

        # Try common column names for text/labels
        for text_col in ["text", "label", "ground_truth", "transcription", "words"]:
            if text_col in sample:
                text = sample[text_col]
                break

        if image is None:
            logger.warning("no_image_found", idx=idx, columns=list(sample.keys()))
            continue

        # Save image
        image_filename = f"{idx:06d}.png"
        image_path = images_dir / image_filename

        if hasattr(image, "save"):
            # PIL Image
            image.save(str(image_path))
        elif isinstance(image, dict) and "path" in image:
            # Image reference
            shutil.copy2(image["path"], str(image_path))

        # Build annotation
        annotation = {
            "image_id": idx,
            "image_path": f"images/{image_filename}",
            "text": str(text) if text else "",
            "source": dataset_name,
        }

        # Add any extra metadata
        for key in ["medicine", "dosage", "frequency", "duration"]:
            if key in sample:
                annotation[key] = sample[key]

        annotations.append(annotation)

        if (idx + 1) % 100 == 0:
            logger.info("download_progress", processed=idx + 1, total=len(dataset))

    # Save annotations
    annotations_path = output_dir / "annotations.json"
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    logger.info(
        "dataset_downloaded",
        dataset=dataset_name,
        total_samples=len(annotations),
        output_dir=str(output_dir),
    )

    return output_dir


def download_all_datasets(
    base_dir: Path | str = "data/raw",
    max_samples_per_dataset: int | None = None,
) -> dict[str, Path]:
    """
    Download all registered datasets.

    Args:
        base_dir: Base directory for all downloads
        max_samples_per_dataset: Optional limit per dataset

    Returns:
        Dict mapping dataset name to output path
    """
    base_dir = Path(base_dir)
    results: dict[str, Path] = {}

    for name, config in DATASETS.items():
        output_dir = base_dir / config["output_dir"]
        try:
            path = download_hf_dataset(
                dataset_name=name,
                output_dir=output_dir,
                split=config["split"],
                max_samples=max_samples_per_dataset,
            )
            results[name] = path
        except Exception as e:
            logger.error("dataset_failed", dataset=name, error=str(e))
            continue

    return results


# ── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download MedScript datasets")
    parser.add_argument("--output", type=str, default="data/raw")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    from medscript.utils.logging import setup_logging
    setup_logging(level="INFO")

    download_all_datasets(
        base_dir=Path(args.output),
        max_samples_per_dataset=args.max_samples,
    )
