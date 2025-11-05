import pandas as pd

from data_designer.essentials import (
    CustomColumnConfig,
    DataDesigner,
    DataDesignerConfigBuilder,
    InferenceParameters,
    LoggingConfig,
    ModelConfig,
    configure_logging,
)

configure_logging(LoggingConfig.debug())

# Initialize NDD and add columns
MODEL_ALIAS = "nano"
SYSTEM_PROMPT = "/no_think"

model_configs = [
    ModelConfig(
        alias="nano",
        model="nvidia/nvidia-nemotron-nano-9b-v2",
        inference_parameters=InferenceParameters(
            temperature=0.5,
            top_p=1.0,
            max_tokens=1024,
            max_parallel_requests=4,
        ),
        provider="nvidia",
    )
]

builder = DataDesignerConfigBuilder(model_configs=model_configs)

builder.add_column(
    name="topic",
    column_type="sampler",
    sampler_type="category",
    params={
        "values": [
            "healthcare",
            "finance",
            "technology",
        ]
    },
)

builder.add_column(
    name="text",
    column_type="llm-text",
    model_alias=MODEL_ALIAS,
    prompt="Write me a paragraph about {{ topic }}.",
    system_prompt=SYSTEM_PROMPT,
)


def generator_function(df: pd.DataFrame) -> pd.DataFrame:
    df["length_frac"] = df["text"].apply(lambda x: len(x) / 1000)
    return df


builder.add_column(
    CustomColumnConfig(
        name="length_frac",
        generator_function=generator_function,
    )
)

# Generate dataset
dd = DataDesigner(artifact_path="./artifacts")
dd_preview = dd.preview(builder, num_records=10)
dd_preview.display_sample_record()

dd.create(builder, num_records=20)
