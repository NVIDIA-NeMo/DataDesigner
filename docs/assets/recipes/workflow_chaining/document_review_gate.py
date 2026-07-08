# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "data-designer",
#     "numpy",
#     "pandas",
#     "pillow",
#     "pydantic",
#     "pyarrow",
# ]
# ///
"""Workflow Chaining Review Gate Recipe

Run a workflow to a named stage, export that intermediate dataset, simulate an
external review process, and resume downstream from the reviewed artifact.

Run:
    uv run document_review_gate.py --artifact-path ./workflow-artifacts --num-records 12 --overwrite
    uv run document_review_gate.py --artifact-path ./workflow-artifacts --num-records 12 --review-pages 4 --overwrite
    uv run document_review_gate.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pydantic import BaseModel, Field

import data_designer.config as dd
from data_designer.interface import DataDesigner

DEFAULT_ARTIFACT_DIR = Path("workflow-chaining-review-gate-artifacts")
PAGE_WIDTH = 1000
PAGE_HEIGHT = 1300
SUPPORTED_LABELS = (
    "invoice_number",
    "vendor",
    "date",
    "due_date",
    "total",
    "line_items",
    "signature",
)


@dataclass(frozen=True)
class DemoDirs:
    base: Path
    images: Path
    metadata: Path
    review: Path
    reviewed: Path
    outputs: Path


class ReviewSelectionParams(BaseModel):
    max_review_pages: int = Field(default=3, ge=1)
    jitter_px: int = Field(default=24, ge=0)


class CalibrationParams(BaseModel):
    confidence_bonus: float = Field(default=0.08, ge=0.0, le=0.25)


def ensure_demo_dirs(base_dir: Path) -> DemoDirs:
    dirs = DemoDirs(
        base=base_dir,
        images=base_dir / "images",
        metadata=base_dir / "metadata",
        review=base_dir / "review",
        reviewed=base_dir / "reviewed",
        outputs=base_dir / "outputs",
    )
    for path in (
        dirs.images,
        dirs.metadata,
        dirs.review,
        dirs.reviewed,
        dirs.outputs,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def metadata_path(base_dir: Path) -> Path:
    return ensure_demo_dirs(base_dir).metadata / "document_pages.parquet"


def review_candidates_path(base_dir: Path) -> Path:
    return ensure_demo_dirs(base_dir).review / "review_candidates.parquet"


def reviewed_candidates_path(base_dir: Path) -> Path:
    return ensure_demo_dirs(base_dir).reviewed / "reviewed_candidates.parquet"


def final_dataset_path(base_dir: Path) -> Path:
    return ensure_demo_dirs(base_dir).outputs / "final_dataset.parquet"


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _box(label: str, x: int, y: int, width: int, height: int, text: str = "") -> dict[str, Any]:
    if label not in SUPPORTED_LABELS:
        raise ValueError(f"Unsupported box label: {label}")
    return {
        "label": label,
        "x": int(max(0, min(PAGE_WIDTH - 1, x))),
        "y": int(max(0, min(PAGE_HEIGHT - 1, y))),
        "width": int(max(1, min(PAGE_WIDTH - max(0, x), width))),
        "height": int(max(1, min(PAGE_HEIGHT - max(0, y), height))),
        "text": text,
    }


def _text_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    label: str,
    *,
    fill: tuple[int, int, int] = (30, 30, 30),
    padding: int = 8,
) -> dict[str, Any]:
    draw.text(xy, text, font=font, fill=fill)
    left, top, right, bottom = draw.textbbox(xy, text, font=font)
    return _box(label, left - padding, top - padding, right - left + padding * 2, bottom - top + padding * 2, text)


def _serialize_boxes(boxes: list[dict[str, Any]]) -> str:
    return json.dumps(boxes, sort_keys=True)


def parse_boxes(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        if not value:
            return []
        parsed = json.loads(value)
    else:
        parsed = value
    if isinstance(parsed, dict):
        parsed = [parsed]
    return [dict(box) for box in parsed]


def validate_metadata_rows(rows: pd.DataFrame) -> None:
    required = {"page_id", "image_path", "document_type", "synthetic_metadata", "ground_truth_boxes"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    for record in rows.to_dict(orient="records"):
        image_path = Path(record["image_path"])
        if not image_path.exists():
            raise ValueError(f"Missing generated image: {image_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        boxes = parse_boxes(record["ground_truth_boxes"])
        if not boxes:
            raise ValueError(f"Missing ground-truth boxes for page: {record['page_id']}")
        for box in boxes:
            validate_box(box, width, height)


def validate_box(box: dict[str, Any], image_width: int, image_height: int) -> None:
    label = box.get("label")
    if label not in SUPPORTED_LABELS:
        raise ValueError(f"Unsupported box label: {label}")
    x = int(box["x"])
    y = int(box["y"])
    width = int(box["width"])
    height = int(box["height"])
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid box geometry: {box}")
    if x + width > image_width or y + height > image_height:
        raise ValueError(f"Box out of image bounds: {box}")


def generate_sample_pages(base_dir: Path, *, count: int = 12, seed: int = 11) -> pd.DataFrame:
    dirs = ensure_demo_dirs(base_dir)
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        image_path = dirs.images / f"page-{index:03d}.png"
        rows.append(_generate_page(index, image_path, rng))
    df = pd.DataFrame(rows)
    validate_metadata_rows(df)
    df.to_parquet(metadata_path(base_dir), index=False)
    return df


def load_or_generate_pages(base_dir: Path, *, count: int, seed: int, force: bool = False) -> pd.DataFrame:
    path = metadata_path(base_dir)
    if path.exists() and not force:
        df = pd.read_parquet(path)
        validate_metadata_rows(df)
        return df
    return generate_sample_pages(base_dir, count=count, seed=seed)


def _generate_page(index: int, image_path: Path, rng: random.Random) -> dict[str, Any]:
    image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (249, 248, 242))
    draw = ImageDraw.Draw(image)
    boxes: list[dict[str, Any]] = []
    document_type = rng.choice(("invoice", "intake_form", "service_form"))
    layout = rng.choice(("left_header", "right_header", "grid_form"))
    vendor = rng.choice(
        (
            "Northstar Office Supply",
            "Cedar Medical Billing",
            "Harbor Freight Services",
            "Aster Analytics Group",
            "Mesa Field Operations",
        )
    )
    invoice_number = f"{rng.choice(('INV', 'FORM', 'DOC'))}-{rng.randint(20000, 99999)}"
    date = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 24):02d}"
    due_date = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    total = f"${rng.randint(800, 8900):,}.{rng.randint(0, 99):02d}"
    field_values = {
        "invoice_number": invoice_number,
        "vendor": vendor,
        "date": date,
        "due_date": due_date,
        "total": total,
    }

    _draw_scan_background(draw, rng)
    if document_type == "invoice":
        boxes.extend(_draw_invoice(draw, rng, layout, field_values))
    else:
        boxes.extend(_draw_form(draw, rng, layout, field_values))

    image = _apply_scan_effects(image, rng)
    image.save(image_path)
    return {
        "page_id": f"synthetic-page-{index:03d}",
        "image_path": str(image_path),
        "document_type": document_type,
        "synthetic_metadata": json.dumps(
            {
                "layout": layout,
                "field_values": field_values,
                "generator": "document_review_gate",
            },
            sort_keys=True,
        ),
        "ground_truth_boxes": _serialize_boxes(boxes),
    }


def _draw_scan_background(draw: ImageDraw.ImageDraw, rng: random.Random) -> None:
    for _ in range(16):
        y = rng.randint(40, PAGE_HEIGHT - 40)
        shade = rng.randint(218, 238)
        draw.line((30, y, PAGE_WIDTH - 30, y + rng.randint(-1, 1)), fill=(shade, shade, shade), width=1)
    for _ in range(8):
        x = rng.randint(40, PAGE_WIDTH - 40)
        shade = rng.randint(225, 242)
        draw.line((x, 40, x + rng.randint(-1, 1), PAGE_HEIGHT - 40), fill=(shade, shade, shade), width=1)
    draw.rectangle((38, 38, PAGE_WIDTH - 38, PAGE_HEIGHT - 38), outline=(210, 205, 196), width=2)


def _draw_invoice(
    draw: ImageDraw.ImageDraw,
    rng: random.Random,
    layout: str,
    field_values: dict[str, str],
) -> list[dict[str, Any]]:
    title_font = _font(42, bold=True)
    label_font = _font(21, bold=True)
    body_font = _font(24)
    small_font = _font(18)
    boxes = []
    left_x = 82 if layout != "right_header" else 570
    right_x = 610 if layout != "right_header" else 88

    draw.text((left_x, 74), "INVOICE", font=title_font, fill=(25, 40, 60))
    boxes.append(_text_box(draw, (left_x, 146), field_values["vendor"], body_font, "vendor"))
    boxes.append(_text_box(draw, (right_x, 104), f"No. {field_values['invoice_number']}", body_font, "invoice_number"))
    boxes.append(_text_box(draw, (right_x, 170), f"Date {field_values['date']}", body_font, "date"))
    boxes.append(_text_box(draw, (right_x, 228), f"Due {field_values['due_date']}", body_font, "due_date"))

    bill_y = rng.randint(300, 380)
    draw.text((82, bill_y), "Bill To", font=label_font, fill=(45, 45, 45))
    draw.text(
        (82, bill_y + 36),
        rng.choice(("Delta Clinic", "Ridgeway Labs", "Oasis Regional", "Juniper Foods")),
        font=body_font,
        fill=(60, 60, 60),
    )

    table_x = rng.randint(70, 120)
    table_y = rng.randint(500, 575)
    table_w = rng.randint(780, 850)
    row_h = 54
    item_count = rng.randint(3, 5)
    table_h = row_h * (item_count + 1)
    draw.rectangle((table_x, table_y, table_x + table_w, table_y + table_h), outline=(80, 86, 95), width=2)
    draw.rectangle(
        (table_x, table_y, table_x + table_w, table_y + row_h), fill=(234, 238, 240), outline=(80, 86, 95), width=2
    )
    draw.text((table_x + 20, table_y + 15), "Description", font=small_font, fill=(20, 20, 20))
    draw.text((table_x + table_w - 170, table_y + 15), "Amount", font=small_font, fill=(20, 20, 20))
    items = []
    for row_index in range(item_count):
        y = table_y + row_h * (row_index + 1)
        draw.line((table_x, y, table_x + table_w, y), fill=(160, 165, 170), width=1)
        item = rng.choice(("Compliance review", "Field inspection", "Parts kit", "Data capture", "Service plan"))
        amount = f"${rng.randint(110, 1800):,}.00"
        items.append({"description": item, "amount": amount})
        draw.text((table_x + 20, y + 15), item, font=small_font, fill=(45, 45, 45))
        draw.text((table_x + table_w - 170, y + 15), amount, font=small_font, fill=(45, 45, 45))
    boxes.append(_box("line_items", table_x, table_y, table_w, table_h, json.dumps(items, sort_keys=True)))

    total_x = table_x + table_w - 260
    total_y = table_y + table_h + rng.randint(34, 76)
    boxes.append(_text_box(draw, (total_x, total_y), f"TOTAL {field_values['total']}", _font(28, bold=True), "total"))

    signature_y = min(total_y + rng.randint(145, 210), PAGE_HEIGHT - 170)
    draw.line((95, signature_y, 410, signature_y + rng.randint(-5, 5)), fill=(58, 58, 58), width=3)
    draw.text(
        (120, signature_y - 45), rng.choice(("A. Rivera", "M. Santos", "J. Kim")), font=_font(30), fill=(35, 35, 35)
    )
    draw.text((96, signature_y + 14), "Authorized signature", font=small_font, fill=(85, 85, 85))
    boxes.append(_box("signature", 90, signature_y - 58, 330, 96, "Authorized signature"))
    _draw_stamp(draw, rng)
    return boxes


def _draw_form(
    draw: ImageDraw.ImageDraw,
    rng: random.Random,
    layout: str,
    field_values: dict[str, str],
) -> list[dict[str, Any]]:
    title_font = _font(38, bold=True)
    label_font = _font(20, bold=True)
    body_font = _font(23)
    small_font = _font(18)
    boxes = []
    draw.text(
        (88, 72),
        "SERVICE INTAKE FORM" if layout == "grid_form" else "CLAIM SUMMARY",
        font=title_font,
        fill=(30, 48, 62),
    )
    boxes.append(_text_box(draw, (88, 148), field_values["vendor"], body_font, "vendor"))
    boxes.append(
        _text_box(draw, (620, 104), f"Reference {field_values['invoice_number']}", body_font, "invoice_number")
    )
    boxes.append(_text_box(draw, (620, 164), f"Received {field_values['date']}", body_font, "date"))
    boxes.append(_text_box(draw, (620, 224), f"Review by {field_values['due_date']}", body_font, "due_date"))

    grid_x = 86
    grid_y = rng.randint(330, 405)
    grid_w = 830
    section_h = 86
    section_count = rng.randint(4, 6)
    draw.rectangle((grid_x, grid_y, grid_x + grid_w, grid_y + section_h * section_count), outline=(80, 86, 95), width=2)
    line_items = []
    for row_index in range(section_count):
        y = grid_y + row_index * section_h
        draw.line((grid_x, y, grid_x + grid_w, y), fill=(150, 155, 160), width=1)
        field = rng.choice(("Account status", "Requested service", "Observed issue", "Assigned team", "Parts listed"))
        value = rng.choice(("Pending review", "Completed", "Requires follow up", "Matched", "Escalated"))
        line_items.append({"field": field, "value": value})
        draw.text((grid_x + 22, y + 16), field, font=label_font, fill=(35, 35, 35))
        draw.text((grid_x + 310, y + 18), value, font=body_font, fill=(55, 55, 55))
    boxes.append(
        _box("line_items", grid_x, grid_y, grid_w, section_h * section_count, json.dumps(line_items, sort_keys=True))
    )

    total_y = grid_y + section_h * section_count + rng.randint(50, 86)
    boxes.append(_text_box(draw, (620, total_y), f"Est. total {field_values['total']}", _font(26, bold=True), "total"))

    signature_y = min(total_y + rng.randint(120, 190), PAGE_HEIGHT - 165)
    draw.text((90, signature_y - 52), "Applicant approval", font=small_font, fill=(75, 75, 75))
    draw.line((90, signature_y, 445, signature_y + rng.randint(-3, 3)), fill=(50, 50, 50), width=3)
    draw.text(
        (120, signature_y - 42), rng.choice(("S. Patel", "C. Moreno", "T. Nguyen")), font=_font(31), fill=(35, 35, 35)
    )
    boxes.append(_box("signature", 84, signature_y - 62, 370, 100, "Applicant approval"))
    _draw_stamp(draw, rng)
    return boxes


def _draw_stamp(draw: ImageDraw.ImageDraw, rng: random.Random) -> None:
    if rng.random() < 0.75:
        center_x = rng.randint(660, 815)
        center_y = rng.randint(860, 1040)
        radius = rng.randint(58, 78)
        color = rng.choice(((140, 30, 40), (35, 90, 130), (30, 110, 80)))
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            outline=color,
            width=4,
        )
        draw.text(
            (center_x - 44, center_y - 12), rng.choice(("PAID", "SEEN", "FILED")), font=_font(21, bold=True), fill=color
        )


def _apply_scan_effects(image: Image.Image, rng: random.Random) -> Image.Image:
    angle = rng.uniform(-1.4, 1.4)
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(250, 249, 243))
    if rng.random() < 0.65:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 0.65)))
    np_rng = np.random.default_rng(rng.randrange(1_000_000_000))
    pixels = np.asarray(image).astype(np.int16)
    noise = np_rng.normal(0, rng.uniform(3.0, 7.0), pixels.shape)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _stable_rng(page_id: str) -> random.Random:
    digest = hashlib.sha256(page_id.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _jitter_box(box: dict[str, Any], rng: random.Random, jitter_px: int) -> dict[str, Any]:
    x = int(box["x"]) + rng.randint(-jitter_px, jitter_px)
    y = int(box["y"]) + rng.randint(-jitter_px, jitter_px)
    width = int(box["width"]) + rng.randint(-jitter_px, jitter_px)
    height = int(box["height"]) + rng.randint(-jitter_px, jitter_px)
    x = max(0, min(PAGE_WIDTH - 2, x))
    y = max(0, min(PAGE_HEIGHT - 2, y))
    width = max(8, min(PAGE_WIDTH - x, width))
    height = max(8, min(PAGE_HEIGHT - y, height))
    confidence = round(max(0.05, min(0.99, rng.uniform(0.45, 0.93))), 3)
    result = dict(box)
    result.update({"x": x, "y": y, "width": width, "height": height, "confidence": confidence})
    if box["label"] in {"due_date", "signature"}:
        result["confidence"] = round(max(0.05, confidence - rng.uniform(0.08, 0.2)), 3)
    return result


@dd.custom_column_generator(
    required_columns=["page_id"],
)
def mark_page_ready(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["page_ready"] = True
    return df


@dd.custom_column_generator(
    required_columns=["page_id", "image_path", "ground_truth_boxes"],
    side_effect_columns=["box_confidences", "uncertainty", "selected_for_review", "human_boxes"],
)
def select_review_candidates(df: pd.DataFrame, generator_params: ReviewSelectionParams) -> pd.DataFrame:
    df = df.copy()
    proposed = []
    confidences = []
    uncertainties = []
    for row in df.to_dict(orient="records"):
        rng = _stable_rng(str(row["page_id"]))
        boxes = []
        scores = []
        for truth_box in parse_boxes(row["ground_truth_boxes"]):
            if truth_box["label"] == "signature" and rng.random() < 0.22:
                continue
            predicted = _jitter_box(truth_box, rng, generator_params.jitter_px)
            boxes.append(predicted)
            scores.append(float(predicted["confidence"]))
        if not boxes:
            boxes = [_jitter_box(parse_boxes(row["ground_truth_boxes"])[0], rng, generator_params.jitter_px)]
            scores = [float(boxes[0]["confidence"])]
        uncertainty = round(1.0 - min(scores), 3)
        proposed.append(_serialize_boxes(boxes))
        confidences.append(json.dumps(scores))
        uncertainties.append(uncertainty)

    selected = set(pd.Series(uncertainties).sort_values(ascending=False).head(generator_params.max_review_pages).index)
    df["proposed_boxes"] = proposed
    df["box_confidences"] = confidences
    df["uncertainty"] = uncertainties
    df["selected_for_review"] = [index in selected for index in range(len(df))]
    df["human_boxes"] = [_serialize_boxes([]) for _ in range(len(df))]
    return df


@dd.custom_column_generator(
    required_columns=["page_id", "human_boxes", "proposed_boxes", "selected_for_review"],
)
def calibrate_from_reviewed_boxes(df: pd.DataFrame, generator_params: CalibrationParams) -> pd.DataFrame:
    df = df.copy()
    label_counts = {label: 0 for label in SUPPORTED_LABELS}
    reviewed_pages = []
    for row in df.to_dict(orient="records"):
        human_boxes = parse_boxes(row.get("human_boxes"))
        if not human_boxes:
            continue
        reviewed_pages.append(row["page_id"])
        for box in human_boxes:
            label_counts[str(box["label"])] += 1
    profile = {
        "reviewed_pages": reviewed_pages,
        "label_counts": {label: count for label, count in label_counts.items() if count},
        "confidence_bonus": generator_params.confidence_bonus if reviewed_pages else 0.0,
    }
    df["calibration_profile"] = json.dumps(profile, sort_keys=True)
    return df


@dd.custom_column_generator(
    required_columns=["calibration_profile", "human_boxes", "proposed_boxes", "uncertainty"],
    side_effect_columns=["extraction_confidence", "extraction_source", "final_boxes"],
)
def extract_with_calibrated_boxes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    extracted_fields = []
    confidences = []
    sources = []
    final_boxes = []
    for row in df.to_dict(orient="records"):
        human_boxes = parse_boxes(row.get("human_boxes"))
        profile = json.loads(row["calibration_profile"])
        if human_boxes:
            boxes = human_boxes
            confidence = 0.99
            source = "human_review"
        else:
            boxes = parse_boxes(row["proposed_boxes"])
            bonus = float(profile.get("confidence_bonus", 0.0))
            confidence = round(max(0.05, min(0.98, 1.0 - float(row["uncertainty"]) + bonus)), 3)
            source = "calibrated_weak_detector" if profile.get("reviewed_pages") else "weak_detector"
        fields = _fields_from_boxes(boxes)
        extracted_fields.append(json.dumps(fields, sort_keys=True))
        confidences.append(confidence)
        sources.append(source)
        final_boxes.append(_serialize_boxes(boxes))
    df["extracted_fields"] = extracted_fields
    df["extraction_confidence"] = confidences
    df["extraction_source"] = sources
    df["final_boxes"] = final_boxes
    return df


@dd.custom_column_generator(
    required_columns=[
        "page_id",
        "image_path",
        "document_type",
        "final_boxes",
        "extracted_fields",
        "extraction_confidence",
        "extraction_source",
        "human_boxes",
        "selected_for_review",
    ],
    side_effect_columns=["boxes", "fields", "source", "confidence", "provenance"],
)
def make_final_records(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    final_records = []
    boxes = []
    fields = []
    sources = []
    confidences = []
    provenance = []
    for row in df.to_dict(orient="records"):
        reviewed = bool(parse_boxes(row["human_boxes"]))
        record_provenance = {
            "page_id": row["page_id"],
            "document_type": row["document_type"],
            "selected_for_review": bool(row["selected_for_review"]),
            "reviewed": reviewed,
            "stage": "final_dataset",
        }
        record = {
            "image_path": row["image_path"],
            "boxes": parse_boxes(row["final_boxes"]),
            "fields": json.loads(row["extracted_fields"]),
            "source": row["extraction_source"],
            "confidence": float(row["extraction_confidence"]),
            "provenance": record_provenance,
        }
        final_records.append(json.dumps(record, sort_keys=True))
        boxes.append(row["final_boxes"])
        fields.append(row["extracted_fields"])
        sources.append(row["extraction_source"])
        confidences.append(float(row["extraction_confidence"]))
        provenance.append(json.dumps(record_provenance, sort_keys=True))
    df["final_record"] = final_records
    df["boxes"] = boxes
    df["fields"] = fields
    df["source"] = sources
    df["confidence"] = confidences
    df["provenance"] = provenance
    return df


def _fields_from_boxes(boxes: list[dict[str, Any]]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for box in boxes:
        label = str(box["label"])
        text = str(box.get("text") or "")
        if label == "line_items":
            try:
                fields[label] = json.loads(text) if text else []
            except json.JSONDecodeError:
                fields[label] = text
        else:
            fields[label] = text
    return fields


def build_workflow(
    base_dir: Path,
    *,
    count: int = 12,
    seed: int = 11,
    review_pages: int = 3,
    force_generate: bool = False,
) -> Any:
    dirs = ensure_demo_dirs(base_dir)
    pages = load_or_generate_pages(base_dir, count=count, seed=seed, force=force_generate)
    data_designer = DataDesigner(
        artifact_path=dirs.outputs / "artifacts",
        model_providers=[
            dd.ModelProvider(
                name="unused-local-provider",
                endpoint="http://localhost:9/v1",
                api_key="unused",
            )
        ],
    )
    workflow = data_designer.compose_workflow(name="document-hitl-layout-annotation")
    workflow.add_stage("document_pages", _document_pages_builder(metadata_path(base_dir)), num_records=len(pages))
    workflow.add_stage("review_candidates", _review_candidates_builder(review_pages))
    workflow.add_stage("calibrate_extractor", _calibration_builder())
    workflow.add_stage("extract_remaining", _extraction_builder())
    workflow.add_stage("final_dataset", _final_dataset_builder())
    return workflow


def _document_pages_builder(path: Path) -> dd.DataDesignerConfigBuilder:
    builder = dd.DataDesignerConfigBuilder(model_configs=[])
    builder.with_seed_dataset(dd.LocalFileSeedSource(path=str(path)))
    builder.add_column(
        dd.CustomColumnConfig(
            name="page_ready",
            generator_function=mark_page_ready,
            generation_strategy=dd.GenerationStrategy.FULL_COLUMN,
        )
    )
    return builder


def _review_candidates_builder(review_pages: int) -> dd.DataDesignerConfigBuilder:
    builder = dd.DataDesignerConfigBuilder(model_configs=[])
    builder.add_column(
        dd.CustomColumnConfig(
            name="proposed_boxes",
            generator_function=select_review_candidates,
            generation_strategy=dd.GenerationStrategy.FULL_COLUMN,
            generator_params=ReviewSelectionParams(max_review_pages=review_pages),
        )
    )
    return builder


def _calibration_builder() -> dd.DataDesignerConfigBuilder:
    builder = dd.DataDesignerConfigBuilder(model_configs=[])
    builder.add_column(
        dd.CustomColumnConfig(
            name="calibration_profile",
            generator_function=calibrate_from_reviewed_boxes,
            generation_strategy=dd.GenerationStrategy.FULL_COLUMN,
            generator_params=CalibrationParams(),
        )
    )
    return builder


def _extraction_builder() -> dd.DataDesignerConfigBuilder:
    builder = dd.DataDesignerConfigBuilder(model_configs=[])
    builder.add_column(
        dd.CustomColumnConfig(
            name="extracted_fields",
            generator_function=extract_with_calibrated_boxes,
            generation_strategy=dd.GenerationStrategy.FULL_COLUMN,
        )
    )
    return builder


def _final_dataset_builder() -> dd.DataDesignerConfigBuilder:
    builder = dd.DataDesignerConfigBuilder(model_configs=[])
    builder.add_column(
        dd.CustomColumnConfig(
            name="final_record",
            generator_function=make_final_records,
            generation_strategy=dd.GenerationStrategy.FULL_COLUMN,
        )
    )
    return builder


def run_to_review_stage(base_dir: Path, *, count: int = 12, seed: int = 11, review_pages: int = 3) -> Path:
    workflow = build_workflow(base_dir, count=count, seed=seed, review_pages=review_pages)
    results = workflow.run(targets="review_candidates")
    output_path = review_candidates_path(base_dir)
    results.export_stage("review_candidates", output_path)
    return output_path


def write_simulated_review_artifact(base_dir: Path) -> Path:
    review_path = review_candidates_path(base_dir)
    if not review_path.exists():
        raise FileNotFoundError(f"Review candidates parquet not found: {review_path}")

    df = pd.read_parquet(review_path)
    reviewed_rows = []
    for row in df.to_dict(orient="records"):
        reviewed_row = dict(row)
        if bool(reviewed_row.get("selected_for_review", False)):
            reviewed_row["human_boxes"] = reviewed_row["ground_truth_boxes"]
        reviewed_rows.append(reviewed_row)

    reviewed_df = pd.DataFrame(reviewed_rows)
    output_path = reviewed_candidates_path(base_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reviewed_df.to_parquet(output_path, index=False)
    return output_path


def resume_from_reviewed(base_dir: Path, *, count: int = 12, seed: int = 11, review_pages: int = 3) -> Path:
    reviewed_path = reviewed_candidates_path(base_dir)
    if not reviewed_path.exists():
        raise FileNotFoundError(f"Reviewed parquet not found: {reviewed_path}")
    workflow = build_workflow(base_dir, count=count, seed=seed, review_pages=review_pages)
    results = workflow.run(
        resume=dd.ResumeMode.ALWAYS,
        stage_output_overrides={"review_candidates": reviewed_path},
    )
    output_path = final_dataset_path(base_dir)
    results.export_stage("final_dataset", output_path)
    return output_path


def run_recipe(base_dir: Path, *, count: int, seed: int, review_pages: int, overwrite: bool = False) -> Path:
    if base_dir.exists():
        if overwrite:
            shutil.rmtree(base_dir)
        elif any(base_dir.iterdir()):
            raise FileExistsError(f"Artifact directory already exists: {base_dir}. Use --overwrite to replace it.")

    review_path = run_to_review_stage(base_dir, count=count, seed=seed, review_pages=review_pages)
    reviewed_path = write_simulated_review_artifact(base_dir)
    final_path = resume_from_reviewed(base_dir, count=count, seed=seed, review_pages=review_pages)
    final_df = pd.read_parquet(final_path)

    print(f"Review stage exported to: {review_path}")
    print(f"Reviewed artifact written to: {reviewed_path}")
    print(f"Final dataset exported to: {final_path}")
    print(f"Rows: {len(final_df)}")
    print(f"Rows selected for review: {int(final_df['selected_for_review'].sum())}")
    print(f"Rows using human_review source: {int((final_df['source'] == 'human_review').sum())}")
    print(f"Mean confidence: {final_df['confidence'].mean():.3f}")
    return final_path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Headless workflow chaining review gate recipe.")
    parser.add_argument("--model-alias", default="unused", help="Accepted for recipe runner compatibility.")
    parser.add_argument("--num-records", type=_positive_int, default=12)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--dataset-name", default="document_review_gate")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--review-pages", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    base_dir = args.artifact_path / args.dataset_name
    run_recipe(
        base_dir,
        count=args.num_records,
        seed=args.seed,
        review_pages=args.review_pages,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
