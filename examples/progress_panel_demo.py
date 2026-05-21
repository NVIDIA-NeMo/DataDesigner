# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time

import data_designer.config as dd
from data_designer.engine.context import current_generation_column
from data_designer.engine.models.usage_events import TokenUsageEvent, emit_token_usage_event

_WORKERS = threading.Semaphore(6)


def _simulate_work(row: dict, *, column: str, base_delay: float) -> str:
    topic = str(row["topic"])
    jitter = (sum(ord(ch) for ch in topic + column) % 7) * 0.025
    input_tokens = 80 + (sum(ord(ch) for ch in topic) % 45)
    output_tokens = 12 + (sum(ord(ch) for ch in column) % 16)
    with _WORKERS:
        # This example is intentionally credential-free, so emit synthetic
        # token usage to exercise the progress panel's live token-rate columns.
        for _ in range(4):
            time.sleep((base_delay + jitter) / 4)
            emit_token_usage_event(
                TokenUsageEvent(
                    model_alias="progress-panel-demo",
                    model_name="synthetic-token-stream",
                    input_tokens=input_tokens // 4,
                    output_tokens=output_tokens // 4,
                    column=current_generation_column.get() or column,
                )
            )
    return f"{column}:{topic.lower().replace(' ', '-')}"


@dd.custom_column_generator(required_columns=["topic"])
def intent_label(row: dict) -> dict:
    row["intent_label"] = _simulate_work(row, column="intent", base_delay=0.12)
    return row


@dd.custom_column_generator(required_columns=["topic"])
def risk_signal(row: dict) -> dict:
    row["risk_signal"] = _simulate_work(row, column="risk", base_delay=0.16)
    return row


@dd.custom_column_generator(required_columns=["topic"])
def routing_bucket(row: dict) -> dict:
    row["routing_bucket"] = _simulate_work(row, column="route", base_delay=0.10)
    return row


def load_config_builder() -> dd.DataDesignerConfigBuilder:
    config_builder = dd.DataDesignerConfigBuilder()

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="topic",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=[
                    "Account recovery",
                    "GPU capacity",
                    "Invoice dispute",
                    "Model quality",
                    "Security review",
                    "Dataset cleanup",
                    "Latency regression",
                    "Documentation gap",
                ],
            ),
        )
    )
    config_builder.add_column(dd.CustomColumnConfig(name="intent_label", generator_function=intent_label))
    config_builder.add_column(dd.CustomColumnConfig(name="risk_signal", generator_function=risk_signal))
    config_builder.add_column(dd.CustomColumnConfig(name="routing_bucket", generator_function=routing_bucket))

    return config_builder
