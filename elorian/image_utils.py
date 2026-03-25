"""Utilities for loading and processing images into DataDesigner seed datasets.

Supports loading from:
- A local directory of image files
- A HuggingFace dataset with an image column
- A list of PIL images
- A list of image URLs
"""

from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path

import pandas as pd
from PIL import Image


def resize_image(image: Image.Image, height: int) -> Image.Image:
    """Resize image while maintaining aspect ratio.

    Args:
        image: PIL Image object.
        height: Target height in pixels.

    Returns:
        Resized PIL Image.
    """
    original_width, original_height = image.size
    width = int(original_width * (height / original_height))
    return image.resize((width, height))


def pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL Image to a base64-encoded string.

    Args:
        image: PIL Image object.
        fmt: Image format for encoding (e.g. "PNG", "JPEG").

    Returns:
        Base64-encoded string of the image.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_images_from_directory(
    directory: str | Path,
    *,
    max_images: int | None = None,
    target_height: int = 512,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".gif"),
) -> pd.DataFrame:
    """Load images from a local directory into a seed DataFrame.

    Args:
        directory: Path to directory containing image files.
        max_images: Maximum number of images to load (None = all).
        target_height: Resize images to this height (preserving aspect ratio).
        extensions: File extensions to include.

    Returns:
        DataFrame with columns: uuid, filename, base64_image.
    """
    directory = Path(directory)
    image_files = sorted(
        f for f in directory.iterdir()
        if f.suffix.lower() in extensions and f.is_file()
    )
    if max_images is not None:
        image_files = image_files[:max_images]

    records = []
    for img_path in image_files:
        image = Image.open(img_path).convert("RGB")
        image = resize_image(image, target_height)
        records.append({
            "uuid": str(uuid.uuid4()),
            "filename": img_path.name,
            "base64_image": pil_to_base64(image),
        })
    return pd.DataFrame(records)


def load_images_from_pil(
    images: list[Image.Image],
    *,
    labels: list[str] | None = None,
    target_height: int = 512,
) -> pd.DataFrame:
    """Convert a list of PIL images into a seed DataFrame.

    Args:
        images: List of PIL Image objects.
        labels: Optional list of labels (same length as images).
        target_height: Resize images to this height.

    Returns:
        DataFrame with columns: uuid, label (if provided), base64_image.
    """
    records = []
    for i, image in enumerate(images):
        image = resize_image(image.convert("RGB"), target_height)
        record: dict = {
            "uuid": str(uuid.uuid4()),
            "base64_image": pil_to_base64(image),
        }
        if labels is not None:
            record["label"] = labels[i]
        records.append(record)
    return pd.DataFrame(records)


def load_images_from_hf_dataset(
    dataset_path: str,
    *,
    split: str = "train",
    image_column: str = "image",
    label_column: str | None = "label",
    max_images: int = 100,
    target_height: int = 512,
) -> pd.DataFrame:
    """Load images from a HuggingFace dataset into a seed DataFrame.

    Args:
        dataset_path: HuggingFace dataset identifier (e.g. "rokmr/pets").
        split: Dataset split to use.
        image_column: Name of the column containing PIL images.
        label_column: Name of the label column (None to skip).
        max_images: Maximum number of images to load.
        target_height: Resize images to this height.

    Returns:
        DataFrame with columns: uuid, label (if available), base64_image.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_path, split=split, streaming=True)

    records = []
    for i, row in enumerate(dataset):
        if i >= max_images:
            break
        image = resize_image(row[image_column].convert("RGB"), target_height)
        record: dict = {
            "uuid": str(uuid.uuid4()),
            "base64_image": pil_to_base64(image),
        }
        if label_column and label_column in row:
            record["label"] = str(row[label_column])
        records.append(record)
    return pd.DataFrame(records)
