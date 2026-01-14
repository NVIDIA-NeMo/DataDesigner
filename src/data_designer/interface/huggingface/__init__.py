# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Backward compatibility: re-export from new location
from data_designer.integrations.huggingface import (
    HubDatasetResults,
    HuggingFaceHubClient,
    resolve_hf_token,
)

# For backward compatibility, provide pull_from_hub as a function
pull_from_hub = HuggingFaceHubClient.pull_from_hub

# Legacy alias for mixin (deprecated, use HuggingFaceHubClient instead)
HuggingFaceHubMixin = HuggingFaceHubClient

__all__ = ["HuggingFaceHubMixin", "HuggingFaceHubClient", "pull_from_hub", "HubDatasetResults", "resolve_hf_token"]
