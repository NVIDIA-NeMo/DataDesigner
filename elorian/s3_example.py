"""Example: read data from S3, run multimodal eval, write results back to S3.

Prerequisites:
    - AWS credentials configured (~/.aws/credentials or env vars)
    - uv pip install s3fs
    - export ANTHROPIC_API_KEY=...
    - export OPENAI_API_KEY=...

Run:
    uv run python -m elorian.s3_example
"""

from __future__ import annotations

import pandas as pd


# ------------------------------------------------------------------ #
# 1. Read from S3                                                     #
# ------------------------------------------------------------------ #

def read_from_s3() -> pd.DataFrame:
    """Read a dataset from a public S3 bucket (anonymous access)."""
    df = pd.read_csv(
        "s3://ray-example-data/sms_spam_collection_subset.txt",
        sep="\t",
        header=None,
        names=["label", "message"],
        storage_options={"anon": True},
    )
    print(f"Read {len(df)} rows from S3")
    print(df.head())
    return df


# ------------------------------------------------------------------ #
# 2. Write to S3                                                      #
# ------------------------------------------------------------------ #

def write_to_s3(df: pd.DataFrame, s3_path: str) -> None:
    """Write a DataFrame to your own S3 bucket.

    Args:
        df: DataFrame to write.
        s3_path: Full S3 URI, e.g. "s3://my-bucket/results/output.parquet".

    Uses default AWS credentials from ~/.aws/credentials or env vars
    (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION).
    """
    if s3_path.endswith(".csv"):
        df.to_csv(s3_path, index=False)
    else:
        df.to_parquet(s3_path, index=False)
    print(f"Wrote {len(df)} rows to {s3_path}")


# ------------------------------------------------------------------ #
# 3. Full pipeline: S3 → process → eval → S3                         #
# ------------------------------------------------------------------ #

def main() -> None:
    # Read
    df = read_from_s3()

    # Process — just a simple transform for illustration
    df["message_length"] = df["message"].str.len()
    df["is_spam"] = df["label"] == "spam"
    print(f"\nProcessed DataFrame:\n{df.head()}")

    # Write back (uncomment and replace with your bucket)
    # write_to_s3(df, "s3://your-bucket/results/sms_processed.parquet")
    # write_to_s3(df, "s3://your-bucket/results/sms_processed.csv")

    # --- Combining with the eval pipeline (images from S3) ---
    #
    # If your S3 bucket has images, read them into a seed DataFrame:
    #
    #   import boto3, base64, uuid, io
    #   from PIL import Image
    #   from elorian.image_utils import pil_to_base64, resize_image
    #
    #   s3 = boto3.client("s3")
    #   bucket, prefix = "my-bucket", "images/"
    #   objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    #
    #   records = []
    #   for obj in objects.get("Contents", []):
    #       body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
    #       image = resize_image(Image.open(io.BytesIO(body)).convert("RGB"), 512)
    #       records.append({
    #           "uuid": str(uuid.uuid4()),
    #           "filename": obj["Key"],
    #           "base64_image": pil_to_base64(image),
    #       })
    #   seed_df = pd.DataFrame(records)
    #
    #   # Run eval
    #   from elorian.pipeline import MultimodalEvalPipeline
    #   pipeline = MultimodalEvalPipeline(seed_df=seed_df)
    #   preview = pipeline.preview(num_records=3)
    #
    #   # Write results back to S3
    #   write_to_s3(preview.dataset, "s3://my-bucket/results/eval_results.parquet")


if __name__ == "__main__":
    main()
