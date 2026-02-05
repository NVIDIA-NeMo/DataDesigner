# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-download the semantic dedup embedding model."""

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def main():
    """Download the embedding model to cache."""
    from sentence_transformers import SentenceTransformer

    print(f"Downloading model: {DEFAULT_MODEL}")
    SentenceTransformer(DEFAULT_MODEL)
    print("Model downloaded successfully!")


if __name__ == "__main__":
    main()
