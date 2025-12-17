# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from huggingface_hub import DatasetCard

TEMPLATE_DATA_DESIGNER_DATASET_CARD_PATH = Path(__file__).parent / "dataset_card_template.md"


class DataDesignerDatasetCard(DatasetCard):
    """Dataset card for NeMo Data Designer datasets.

    This class extends Hugging Face's DatasetCard with a custom template
    specifically designed for Data Designer generated datasets.
    The template is located at `data_designer/integrations/huggingface/dataset_card_template.md`.
    """

    default_template_path = TEMPLATE_DATA_DESIGNER_DATASET_CARD_PATH
