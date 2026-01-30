# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from huggingface_hub import CardData, DatasetCard

TEMPLATE_DATA_DESIGNER_DATASET_CARD_PATH = Path(__file__).parent / "dataset_card_template.md"


class DataDesignerDatasetCard(DatasetCard):
    """Dataset card for NeMo Data Designer datasets.

    This class extends Hugging Face's DatasetCard with a custom template
    specifically designed for Data Designer generated datasets.
    The template is located at `data_designer/integrations/huggingface/dataset_card_template.md`.
    """

    default_template_path = TEMPLATE_DATA_DESIGNER_DATASET_CARD_PATH

    @classmethod
    def from_metadata(
        cls,
        metadata: dict,
        sdg_config: dict | None,
        repo_id: str,
        description: str,
    ) -> DataDesignerDatasetCard:
        """Create dataset card from metadata.json and sdg.json.

        Args:
            metadata: Contents of metadata.json
            sdg_config: Contents of sdg.json (optional)
            repo_id: HuggingFace repo ID
            description: Custom description text

        Returns:
            DataDesignerDatasetCard instance ready to upload
        """
        # Extract info from metadata
        target_num_records = metadata.get("target_num_records", 0)
        schema = metadata.get("schema", {})
        column_stats = metadata.get("column_statistics", [])

        # Get actual num_records from column_statistics if available
        if column_stats:
            actual_num_records = column_stats[0].get("num_records", target_num_records)
        else:
            actual_num_records = target_num_records

        # Compute size category
        size_categories = cls._compute_size_category(actual_num_records)

        # Extract column types from sdg.json if available
        config_types: dict[str, int] = {}
        num_columns_configured = 0
        if sdg_config:
            columns = sdg_config.get("data_designer", {}).get("columns", [])
            num_columns_configured = len(columns)
            for col in columns:
                col_type = col.get("column_type", "unknown")
                if isinstance(col_type, dict):
                    col_type = col_type.get("value", "unknown")
                config_types[col_type] = config_types.get(col_type, 0) + 1

        # Extract processor names from file_paths
        processor_names = []
        if "file_paths" in metadata and "processor-files" in metadata["file_paths"]:
            processor_names = list(metadata["file_paths"]["processor-files"].keys())

        # Prepare tags
        tags = ["synthetic", "datadesigner"]

        # Prepare CardData (metadata for YAML frontmatter)
        card_data = CardData(
            size_categories=size_categories,
            tags=tags,
        )

        # Prepare template variables
        template_vars = {
            "repo_id": repo_id,
            "num_records": actual_num_records,
            "target_num_records": target_num_records,
            "num_columns": len(schema),
            "size_categories": size_categories,
            "all_columns": schema,
            "column_statistics": column_stats,
            "num_columns_configured": num_columns_configured,
            "config_types": config_types,
            "percent_complete": 100 * actual_num_records / (target_num_records + 1e-10),
            "current_year": datetime.now().year,
            "has_processors": len(processor_names) > 0,
            "processor_names": processor_names,
            "tags": tags,
            "custom_description": description,
        }

        # Create card from template
        card = cls.from_template(card_data, template_path=str(cls.default_template_path), **template_vars)
        return card

    @staticmethod
    def _compute_size_category(num_records: int) -> str:
        """Compute HuggingFace size category from record count.

        Args:
            num_records: Number of records in the dataset

        Returns:
            Size category string for HuggingFace Hub tags
        """
        if num_records < 1000:
            return "n<1K"
        elif num_records < 10000:
            return "1K<n<10K"
        elif num_records < 100000:
            return "10K<n<100K"
        elif num_records < 1000000:
            return "100K<n<1M"
        elif num_records < 10000000:
            return "1M<n<10M"
        else:
            return "n>10M"
