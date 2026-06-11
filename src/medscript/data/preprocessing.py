"""OpenCV preprocessing pipeline for prescription images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from medscript.utils.logging import get_logger

logger = get_logger(__name__)


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as BGR numpy array."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale if needed."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def deskew(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Deskew a document image using Hough Line Transform.

    Detects dominant line angle and rotates to correct skew.
    """
    gray = to_grayscale(image) if len(image.shape) == 3 else image.copy()

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Hough Line Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=gray.shape[1] // 4,
        maxLineGap=20,
    )

    if lines is None:
        return image

    # Calculate dominant angle
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < max_angle:  # Filter extreme angles
            angles.append(angle)

    if not angles:
        return image

    median_angle = np.median(angles)

    # Rotate to correct skew
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    logger.debug("deskew_applied", angle=float(median_angle))
    return rotated


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    """
    if len(image.shape) == 3:
        # Apply CLAHE to L channel of LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        l_enhanced = clahe.apply(l_channel)

        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(image)


def binarize(
    image: np.ndarray,
    method: str = "adaptive",
    block_size: int = 15,
    constant: int = 10,
) -> np.ndarray:
    """
    Binarize image using Otsu's or adaptive thresholding.

    Args:
        image: Input image
        method: "otsu" or "adaptive"
        block_size: Block size for adaptive thresholding (must be odd)
        constant: Constant subtracted from mean (adaptive only)
    """
    gray = to_grayscale(image) if len(image.shape) == 3 else image.copy()

    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            constant,
        )
    else:
        raise ValueError(f"Unknown binarization method: {method}")

    return binary


def denoise(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """Remove noise using Non-Local Means Denoising."""
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)


def remove_borders(image: np.ndarray, margin: int = 10) -> np.ndarray:
    """Remove black borders by finding content region."""
    gray = to_grayscale(image) if len(image.shape) == 3 else image.copy()
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    # Find bounding box of all content
    all_points = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_points)

    # Add margin
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2 * margin)
    h = min(image.shape[0] - y, h + 2 * margin)

    return image[y:y + h, x:x + w]


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_height: int = 960,
    target_width: int = 1280,
    pad_value: int = 255,
) -> np.ndarray:
    """
    Resize image while preserving aspect ratio, padding if needed.

    Args:
        image: Input image
        target_height: Target height
        target_width: Target width
        pad_value: Padding color (255=white, 0=black)
    """
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target size
    if len(image.shape) == 3:
        padded = np.full((target_height, target_width, 3), pad_value, dtype=np.uint8)
    else:
        padded = np.full((target_height, target_width), pad_value, dtype=np.uint8)

    # Center the image
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return padded


def preprocess_image(
    image: np.ndarray,
    target_height: int = 960,
    target_width: int = 1280,
    enhance: bool = True,
    do_deskew: bool = True,
    do_denoise: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a prescription image.

    Steps:
    1. Deskew (optional)
    2. Denoise (optional)
    3. Contrast enhancement (optional)
    4. Resize with padding

    Args:
        image: Input BGR image
        target_height: Target output height
        target_width: Target output width
        enhance: Apply contrast enhancement
        do_deskew: Apply deskewing
        do_denoise: Apply denoising
    """
    result = image.copy()

    if do_deskew:
        result = deskew(result)

    if do_denoise:
        result = denoise(result, strength=6)

    if enhance:
        result = enhance_contrast(result, clip_limit=2.0)

    result = resize_with_aspect_ratio(
        result,
        target_height=target_height,
        target_width=target_width,
    )

    return result


def preprocess_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    target_height: int = 960,
    target_width: int = 1280,
) -> int:
    """
    Preprocess all images in a directory.

    Returns:
        Number of images processed
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    count = 0

    for image_path in sorted(input_dir.rglob("*")):
        if image_path.suffix.lower() not in extensions:
            continue

        try:
            image = load_image(image_path)
            processed = preprocess_image(
                image,
                target_height=target_height,
                target_width=target_width,
            )

            out_path = output_dir / image_path.relative_to(input_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), processed)
            count += 1

            if count % 100 == 0:
                logger.info("preprocessing_progress", processed=count)

        except Exception as e:
            logger.warning("preprocessing_failed", path=str(image_path), error=str(e))

    logger.info("preprocessing_complete", total=count)
    return count
