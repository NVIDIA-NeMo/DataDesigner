/** Full source for prepare_corpus.py - used in ExpandableCode on deep-research-trajectories page */

export const prepare_corpusCode = `# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "huggingface_hub", "pyarrow"]
# ///

"""Prepare a retrieval corpus and question set for the OpenResearcher demo.

Builds corpus.jsonl and questions.jsonl from two sources:

    1. MuSiQue — multi-hop QA dataset (2/3/4-hop) with golden passages
    2. FineWeb — web documents as distractors (matches the OpenResearcher paper)

Golden passages (documents containing evidence for the answer) are mixed with
FineWeb distractors at roughly 1:100 ratio, so the model must search through
noise to find the signal.

Usage:
    uv run prepare_corpus.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_QUESTIONS = 192          # 64 per hop level (2, 3, 4)
NUM_FINEWEB_DISTRACTORS = 50_000
FINEWEB_SHARD = 0
OUTPUT_DIR = "data"


# ---------------------------------------------------------------------------
# MuSiQue extraction
# ---------------------------------------------------------------------------

def prepare_musique(num_questions: int) -> tuple[list[dict], list[dict]]:
    """Load MuSiQue and extract multi-hop questions with golden passages.

    Samples uniformly across hop counts (2, 3, 4) so the dataset has balanced
    difficulty. Golden passages (is_supporting=True) go into the corpus;
    non-golden passages from the same examples serve as additional distractors.

    Returns:
        (questions, corpus_docs) where corpus_docs have is_golden=True/False.
    """
    from datasets import load_dataset

    print("Loading MuSiQue (train split)...")
    dataset = load_dataset("bdsaglam/musique", split="train")

    # Bucket answerable examples by hop count
    hop_buckets: dict[int, list[dict]] = {}
    for example in dataset:
        if not example.get("answerable", False):
            continue
        num_hops = len(example.get("question_decomposition", []))
        if num_hops < 2:
            continue
        hop_buckets.setdefault(num_hops, []).append(example)

    # Sample uniformly: equal questions per hop level
    available_hops = sorted(hop_buckets.keys())
    per_hop = num_questions // len(available_hops)
    selected_examples = []
    for h in available_hops:
        bucket = hop_buckets[h]
        n = min(per_hop, len(bucket))
        selected_examples.extend(random.sample(bucket, n))

    print(f"  Selected {len(selected_examples)} questions across hops {available_hops}")

    # Build questions and corpus docs
    questions: list[dict] = []
    golden_titles: dict[str, str] = {}
    nongolden_titles: dict[str, str] = {}

    for example in selected_examples:
        num_hops = len(example["question_decomposition"])
        questions.append({
            "id": f"mq_{len(questions):06d}",
            "question": example["question"],
            "answer": example["answer"],
            "source": "musique",
            "num_hops": num_hops,
            "seed_id": 0,
        })

        for para in example.get("paragraphs", []):
            title = para.get("title", "").strip()
            content = para.get("paragraph_text", "").strip()
            if not title or not content:
                continue
            if para.get("is_supporting", False):
                if len(content) > len(golden_titles.get(title, "")):
                    golden_titles[title] = content
            else:
                if len(content) > len(nongolden_titles.get(title, "")):
                    nongolden_titles[title] = content

    # Golden passages
    corpus_docs = [
        {"title": t, "content": c, "source": "musique", "is_golden": True}
        for t, c in sorted(golden_titles.items())
    ]
    # Non-golden passages (skip titles already in golden set)
    corpus_docs.extend(
        {"title": t, "content": c, "source": "musique", "is_golden": False}
        for t, c in sorted(nongolden_titles.items())
        if t not in golden_titles
    )

    print(f"  Golden passages: {len(golden_titles)}")
    print(f"  Non-golden passages: {len(corpus_docs) - len(golden_titles)}")
    return questions, corpus_docs


# ---------------------------------------------------------------------------
# FineWeb distractor caching
# ---------------------------------------------------------------------------

def cache_fineweb(shard_index: int, max_docs: int) -> list[dict]:
    """Download a FineWeb parquet shard and extract English documents.

    Uses huggingface_hub for direct shard download (faster than load_dataset)
    and pyarrow for memory-efficient row-group-at-a-time reading.

    Returns:
        List of distractor documents with title (domain) and content (text).
    """
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    filename = f"sample/10BT/{shard_index:03d}_00000.parquet"
    print(f"Downloading FineWeb shard: {filename}")
    parquet_path = hf_hub_download(
        repo_id="HuggingFaceFW/fineweb",
        repo_type="dataset",
        filename=filename,
    )

    pf = pq.ParquetFile(parquet_path)
    print(f"  {pf.metadata.num_rows:,} rows in shard")

    docs: list[dict] = []
    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=["text", "url", "language", "token_count"])
        batch = table.to_pydict()

        for text, url, lang, tok_count in zip(
            batch["text"], batch["url"], batch["language"], batch["token_count"]
        ):
            if lang != "en" or tok_count < 50:
                continue
            text = text.strip()
            if not text:
                continue

            # Use domain as title
            try:
                domain = urlparse(url).netloc.removeprefix("www.")
            except Exception:
                domain = "unknown"

            docs.append({
                "title": domain,
                "content": text,
                "source": "fineweb",
                "is_golden": False,
            })
            if len(docs) >= max_docs:
                break

        if len(docs) >= max_docs:
            break

    print(f"  Extracted {len(docs):,} English documents (min 50 tokens)")
    return docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract MuSiQue questions and golden passages
    questions, corpus_docs = prepare_musique(NUM_QUESTIONS)

    # Download FineWeb distractors
    fineweb_docs = cache_fineweb(FINEWEB_SHARD, NUM_FINEWEB_DISTRACTORS)
    corpus_docs.extend(fineweb_docs)

    # Deduplicate by title (keep longest content)
    title_to_best: dict[str, dict] = {}
    for doc in corpus_docs:
        title = doc["title"]
        if title not in title_to_best or len(doc["content"]) > len(title_to_best[title]["content"]):
            title_to_best[title] = doc

    corpus = list(title_to_best.values())
    random.shuffle(corpus)

    # Assign stable IDs
    prefix_map = {"musique": "md", "fineweb": "fw"}
    source_counters: dict[str, int] = {}
    for doc in corpus:
        prefix = prefix_map.get(doc["source"], "xx")
        idx = source_counters.get(doc["source"], 0)
        doc["id"] = f"{prefix}_{idx:06d}"
        source_counters[doc["source"]] = idx + 1

    # Write corpus.jsonl
    corpus_path = output_dir / "corpus.jsonl"
    with open(corpus_path, "w") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Write questions.jsonl
    random.shuffle(questions)
    questions_path = output_dir / "questions.jsonl"
    with open(questions_path, "w") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Summary
    golden = sum(1 for d in corpus if d["is_golden"])
    nongolden = len(corpus) - golden
    print(f"\nCorpus: {len(corpus):,} docs ({golden} golden, {nongolden} distractors)")
    print(f"Questions: {len(questions)}")
    print(f"Output: {corpus_path.resolve()}")
    print(f"         {questions_path.resolve()}")


if __name__ == "__main__":
    main()`;
