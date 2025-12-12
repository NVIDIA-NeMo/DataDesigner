from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import pandas as pd

SeedTypeT = TypeVar("SeedTypeT", bound=str)

class SeedDatasetConfig(BaseModel, Generic[SeedTypeT]):
    """Base class for seed dataset configurations.

    All subclasses must define a `seed_type` field with a Literal value.
    This serves as a discriminated union discriminator.
    """
    seed_type: SeedTypeT = Field(..., description="Discriminator for seed dataset config type")


class LocalFileSeedConfig(SeedDatasetConfig):
    seed_type: Literal["local"] = "local"

    path: str | Path
    # Validate path exists?


class HuggingFaceSeedConfig(SeedDatasetConfig):
    seed_type: Literal["hf"] = "hf"

    dataset: str
    token: str | None = None
    # Validate dataset is of format "hf://..."


class DataFrameSeedConfig(SeedDatasetConfig):
    seed_type: Literal["df"] = "df"

    df: pd.DataFrame


SeedDatasetConfigT = Annotated[
    LocalFileSeedConfig | HuggingFaceSeedConfig | DataFrameSeedConfig,
    Field(discriminator="seed_type"),
]
