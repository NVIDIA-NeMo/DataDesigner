# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 👥 Data Designer Tutorial: Reproducing & Customizing Nemotron-Personas
#
# This notebook reproduces the [Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA) generation pipeline end to end with [🎨 NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner), and then shows how to customize that pipeline to generate personas for a specific use case. A similar approach was used to build every dataset in the [Nemotron-Personas HF collection](https://huggingface.co/collections/nvidia/nemotron-personas).
#
# We seed the pipeline with the **extended Nemotron-Personas-USA dataset on NGC**, which is a superset of the publicly released HuggingFace version. It includes additional demographic and persona fields used to ground synthetic generation. From those grounded seeds, two stages of LLM structured-output columns produce the persona attributes (cultural background, skills, career goals, hobbies) and the persona descriptions across professional, financial, healthcare, sports, arts, travel, and culinary dimensions.
#
# > **Prerequisites**: This tutorial seeds from the NGC-hosted [Nemotron-Personas](https://huggingface.co/collections/nvidia/nemotron-personas) dataset. See the NGC setup section below to install the NGC CLI and download the dataset before running the rest of the notebook.
#
# If this is your first time using Data Designer, we recommend starting with the [first notebook](https://nvidia-nemo.github.io/DataDesigner/latest/notebooks/1-the-basics/) in this tutorial series.
#
# <div align="center">
#   <img src="https://raw.githubusercontent.com/NVIDIA-NeMo/DataDesigner/main/docs/devnotes/posts/assets/nemotron-personas/nemotron_persona_via_ndd.png" alt="Nemotron Personas pipeline overview" width="600" />
# </div>

# %% [markdown]
# # 1. 📦 Imports

# %%
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

import data_designer.config as dd
from data_designer.interface import DataDesigner

# %% [markdown]
# ## 📥 NGC setup (one-time, before running this notebook)
#
# This notebook is seeded by the NGC-hosted **extended** Nemotron-Personas dataset, a superset of the public HuggingFace release. Make sure the dataset is on disk before running the cells below. Run the commands below in a terminal (on Colab, prefix each `!` for shell commands; `ngc config set` is interactive and can be run via `!ngc config set` in a Colab cell).
#
# 1. **Generate an NGC API key** at [ngc.nvidia.com/setup/api-key](https://ngc.nvidia.com/setup/api-key).
#
# 2. **Install the NGC CLI.** See [ngc.nvidia.com/setup/installers/cli](https://ngc.nvidia.com/setup/installers/cli) for platform-specific instructions. On Linux / Colab:
#
#    ```bash
#    wget -q --content-disposition "https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.164.0/files/ngccli_linux.zip" -O ngccli_linux.zip
#    unzip -q -o ngccli_linux.zip
#    chmod u+x ngc-cli/ngc
#    export PATH="$PWD/ngc-cli:$PATH"
#    ```
#
# 3. **Configure the NGC CLI** (interactive — answer the prompts; paste your API key, set `org=nvidia`, leave the rest at defaults):
#
#    ```bash
#    ngc config set
#    ```
#
# 4. **Download the persona dataset** (answer `Y` when prompted):
#
#    ```bash
#    data-designer download personas --locale en_US
#    ```
#
#    Lands at `~/.data-designer/managed-assets/datasets/en_US.parquet` (≈1.4 GB).
#
# Change the locale to any other [supported value](https://nvidia-nemo.github.io/DataDesigner/latest/concepts/person_sampling/) (`en_IN`, `en_SG`, `fr_FR`, `hi_Deva_IN`, `hi_Latn_IN`, `ja_JP`, `ko_KR`, `pt_BR`) to seed a regional pipeline instead.
#
# The cell below verifies the dataset is on disk; if not, it points back to the setup steps.

# %%
personas_locale = "en_US"

assets_dir = Path.home() / ".data-designer" / "managed-assets" / "datasets"
existing = list(assets_dir.glob(f"{personas_locale}*.parquet")) if assets_dir.exists() else []

if not existing:
    raise SystemExit(
        f"Nemotron-Personas-{personas_locale} not found at {assets_dir}.\n"
        "See the setup instructions in the cell above; this notebook expects the dataset to "
        "already be downloaded."
    )

print(f"Nemotron-Personas-{personas_locale} found ({len(existing)} parquet file(s)):")
for p in existing:
    print(f"  - {p.name}")

# %% [markdown]
# # 2. 🛠️ Define helpers
#
# These OCEAN Big-Five helpers come from the original Nemotron-Personas pipeline. They are **not invoked** in the default flow below, where OCEAN traits come directly from the NGC-hosted Nemotron-Personas-USA dataset via `with_synthetic_personas=True`. They are kept here for the `SAMPLE_FROM_SDG_PGM = True` reproduction path (see Section 4.2), since [NeMo SDG-PGMs](https://github.com/NVIDIA-NeMo/SDG-PGMs) handles demographic distributions but not Big Five personality scoring.


# %%
def get_trait_label(score: int) -> str:
    """Convert a Big Five T-score into a coarse label."""
    if score < 35:
        return "very low"
    if score < 45:
        return "low"
    if score < 55:
        return "average"
    if score < 65:
        return "high"
    return "very high"


def get_trait_description(trait: str, label: str) -> str:
    """Return a prose description for a (trait, label) pair."""
    descriptions: dict[str, dict[str, str]] = {
        "openness": {
            "very low": "Strongly prefers routine and the familiar. Traditional in thinking and values practicality over abstract ideas.",
            "low": "Generally prefers structure and predictability. Tends to be practical and focused on immediate realities.",
            "average": "Balances curiosity with practicality. Appreciates both new ideas and established methods.",
            "high": "Curious and appreciative of art, new ideas, and varied experiences. Open to unconventional thinking.",
            "very high": "Highly imaginative and intellectually curious. Strongly drawn to novelty, art, and abstract concepts.",
        },
        "conscientiousness": {
            "very low": "Spontaneous and flexible, often resisting structure. May struggle with organization and deadlines.",
            "low": "Often relaxed about obligations and somewhat disorganized. Values flexibility over strict planning.",
            "average": "Maintains balance between organization and flexibility. Reasonably reliable and attentive to responsibilities.",
            "high": "Organized, reliable, and methodical. Plans ahead and follows through on commitments.",
            "very high": "Exceptionally organized and disciplined. Strongly focused on achievement and meeting high standards.",
        },
        "extraversion": {
            "very low": "Strongly prefers solitude and quiet environments. May find social interaction draining.",
            "low": "Generally reserved and comfortable with solitude. Prefers small groups to large gatherings.",
            "average": "Balances social interaction with need for alone time. Moderately talkative in social situations.",
            "high": "Sociable, outgoing, and energetic. Enjoys group activities and being around others.",
            "very high": "Highly sociable and draws energy from others. Very talkative and comfortable being center of attention.",
        },
        "agreeableness": {
            "very low": "Critical, skeptical, and competitive. Prioritizes personal interests over group harmony.",
            "low": "Sometimes skeptical of others' intentions. More competitive than cooperative in approach.",
            "average": "Generally cooperative but can be assertive. Balances compassion with self-interest.",
            "high": "Kind, cooperative, and considerate. Prioritizes harmony and others' needs.",
            "very high": "Exceptionally compassionate and cooperative. Strongly motivated to help others and maintain harmony.",
        },
        "neuroticism": {
            "very low": "Exceptionally calm and resilient. Rarely experiences negative emotions like anxiety or sadness.",
            "low": "Emotionally stable and handles stress well. Not easily upset by challenging situations.",
            "average": "Experiences normal range of emotions. Moderately resilient but affected by significant challenges.",
            "high": "Experiences more negative emotions than average. Prone to worry and sensitive to stress.",
            "very high": "Highly emotionally reactive and prone to distress. Often experiences intense anxiety or sadness.",
        },
    }
    return descriptions[trait][label]


def generate_ocean_traits(num_records: int, base_seed: int | None = None) -> pd.DataFrame:
    """Generate synthetic OCEAN traits as a DataFrame with one JSON-encoded object per trait per row."""
    if num_records <= 0:
        return pd.DataFrame()

    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    rng = np.random.RandomState(base_seed) if base_seed is not None else np.random.RandomState()
    data: dict[str, list[str]] = {}

    for trait in traits:
        scores = rng.normal(50.0, 10.0, num_records) + rng.normal(0.0, 2.0, num_records)
        scores = np.clip(scores, 20.0, 80.0)
        t_scores = np.round(scores).astype(int)
        labels = [get_trait_label(int(score)) for score in t_scores]
        descriptions = [get_trait_description(trait, label) for label in labels]
        data[trait] = [
            json.dumps({"t_score": int(t), "label": l, "description": d})
            for t, l, d in zip(t_scores, labels, descriptions, strict=True)
        ]

    return pd.DataFrame(data)


# %% [markdown]
# # 3. 🎨 Set Up NeMo Data Designer (NDD)

# %% [markdown]
# ## 🪪 Specify Model ID and Alias
#
# - Use a [build.nvidia.com](https://build.nvidia.com/) model endpoint and model ID
# - Make sure your `NVIDIA_API_KEY` environment variable is set

# %%
MODEL_PROVIDER = "nvidia"
MODEL_ID = "openai/gpt-oss-20b"
MODEL_ALIAS = "gpt-oss-20b"

# %% [markdown]
# ## 🎛️ Adjust the model config
#
# > ⚠️ **Note**: You may need to adjust temperature and top_p settings depending on the model you use. Consult the model card on [build.nvidia.com](https://build.nvidia.com) for recommended settings.

# %%
model_configs = [
    dd.ModelConfig(
        alias=MODEL_ALIAS,
        model=MODEL_ID,
        provider=MODEL_PROVIDER,
        inference_parameters=dd.ChatCompletionInferenceParams(
            max_tokens=16384,
            temperature=dd.UniformDistribution(params=dd.UniformDistributionParams(low=0.9, high=1.1)),
            top_p=1.0,
            extra_body={"reasoning_effort": "high"},
            timeout=1200,
            max_parallel_requests=32,
        ),
    )
]

# %% [markdown]
# ## 🚀 Initialize Data Designer

# %%
data_designer = DataDesigner()

# %% [markdown]
# # 4. ✍️ Design the dataset
#
# The Nemotron-Personas pipeline has three main steps:
#
# - 1️⃣ Generate OCEAN Personality Traits
# - 2️⃣ Generate Persona Attributes by grounding in (PGM + OCEAN) details
# - 3️⃣ Generate Personas by grounding in (2)
#
# In this notebook Steps 1-2 are seeded from the released NGC artifact via `PersonSampler`; Step 3 runs LLM-driven generation. Section 4.2 also produces persona attributes via a separate LLM call.

# %% [markdown]
# ## 4.1 🌊 Generate OCEAN (Big Five) personality traits
# OCEAN is the most common scientific model for measuring and describing human personality traits.<br>
# See [Big Five personality traits Wikipedia article](https://en.wikipedia.org/wiki/Big_Five_personality_traits) for more context.
#
# In this notebook the OCEAN traits come straight from the **NGC-hosted Nemotron-Personas-USA dataset** in the next section (`with_synthetic_personas=True` exposes `person.openness`, `person.conscientiousness`, etc. as `struct<description, label, t_score>`). The helper functions in Section 2 are kept ready for the `SAMPLE_FROM_SDG_PGM = True` reproduction path.

# %% [markdown]
# ## 4.2 👩‍🎨👨‍🎨 Generate Persona Attributes
#
# We are focusing just on the part in the diagram below and seeding persona attributes with PGM + OCEAN details:
#
# <div align="center">
#   <img src="https://raw.githubusercontent.com/NVIDIA-NeMo/DataDesigner/main/docs/devnotes/posts/assets/nemotron-personas/nemotron_persona_via_ndd_step_2.png" alt="Stage 3: Persona attributes via structured outputs" width="600" />
# </div>

# %% [markdown]
# > ⚠️ **Note**:
# > Below, we show two different ways of seeding persona generation:
# >
# > When `SAMPLE_FROM_SDG_PGM = False` (default), we sample personal details and OCEAN traits from Data Designer's `PersonSampler` against the NGC-hosted Nemotron-Personas dataset (`PersonSamplerParams(locale=personas_locale, with_synthetic_personas=True)`).
# >
# > When `SAMPLE_FROM_SDG_PGM = True`, persons are generated from a custom Probabilistic Graphical Model via [NeMo SDG-PGMs](https://github.com/NVIDIA-NeMo/SDG-PGMs), and the OCEAN helpers from Section 2 layer the personality traits on top. **This branch is a hook** — bring your own `PGMGenerator` subclass (see the [`us_person` example](https://github.com/NVIDIA-NeMo/SDG-PGMs/tree/main/examples/us_person) and the cell below for the integration shape).
# >
# > To switch locales, change `personas_locale` in the Section 1 verify cell (and run `data-designer download personas --locale <code>` for the new locale first). All downstream prompts work unchanged across locales.

# %%
# Toggle the source of the base "person" record.
#   False (default) -- sample from the NGC-hosted Nemotron-Personas-USA artifact.
#   True            -- generate persons from a custom PGM via SDG-PGMs (hook; see below).
SAMPLE_FROM_SDG_PGM = False

# %%
config_builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

if SAMPLE_FROM_SDG_PGM:
    # Build the base person record via a custom Probabilistic Graphical Model using
    # NeMo SDG-PGMs (https://github.com/NVIDIA-NeMo/SDG-PGMs), then layer the OCEAN
    # Big-Five helpers from Section 2 on top. This matches the original four-stage
    # Nemotron-Personas pipeline (Stage 1 = OCEAN helpers, Stage 2 = PGM demographics).
    #
    # See the worked us_person example for a complete PGMGenerator subclass:
    # https://github.com/NVIDIA-NeMo/SDG-PGMs/tree/main/examples/us_person
    #
    # The integration shape:
    #
    #     from data_designer_plugins.pgm_generator_plugin import PGMGeneratorPluginConfig
    #     ocean_df = generate_ocean_traits(NUM_RECORDS)                # Stage 1 (OCEAN)
    #     config_builder.with_seed_dataset(
    #         dd.DataFrameSeedSource(df=ocean_df),
    #         sampling_strategy=dd.SamplingStrategy.ORDERED,
    #     )
    #     config_builder.add_column(                                   # Stage 2 (demographics)
    #         PGMGeneratorPluginConfig(
    #             name="person",
    #             generator_class="my_generators.UsPersonGenerator",
    #         )
    #     )
    raise NotImplementedError(
        "SDG-PGMs path is a hook in this notebook -- bring your own PGMGenerator subclass. "
        "See https://github.com/NVIDIA-NeMo/SDG-PGMs/tree/main/examples/us_person for a worked subclass "
        "and https://github.com/NVIDIA-NeMo/SDG-PGMs/blob/main/src/data_designer_plugins/pgm_generator_plugin.py "
        "for the column-generator interface."
    )

# Default path: sample synthetic personal details + OCEAN traits from the NGC-hosted asset.
# `with_synthetic_personas=True` exposes Big Five t-scores + labels + descriptions, plus
# `person.cultural_background`, hobbies, career goals, and context-specific personas (those
# extra fields stay nested in `person` and don't conflict with the columns we regenerate
# downstream). `drop=True` keeps `person` from leaking into the final dataset.
config_builder.add_column(
    dd.SamplerColumnConfig(
        name="person",
        sampler_type=dd.SamplerType.PERSON,
        params=dd.PersonSamplerParams(
            locale=personas_locale,
            age_range=[18, 114],
            with_synthetic_personas=True,
            # sex="Male"  # Optional: filter by sex
            # city=["New York", "Los Angeles"]  # Optional: filter by cities
        ),
        drop=True,
    )
)

# %%
# Add a unique identifier for each record
config_builder.add_column(name="uuid", column_type="sampler", sampler_type="uuid")

# Lift OCEAN traits to top-level so the original prompts can reference {{ openness.description }} etc.
for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
    config_builder.add_column(dd.ExpressionColumnConfig(name=trait, expr=f"{{{{ person.{trait} }}}}"))

# Add specific personal detail columns -- NOT included in the public release, but used for seeding Personas
config_builder.add_column(
    dd.ExpressionColumnConfig(
        name="ethnic_background",
        expr="{{ person.ethnic_background if person.ethnic_background else ' ' }}",
    )
)
config_builder.add_column(dd.ExpressionColumnConfig(name="first_name", expr="{{ person.first_name }}"))
config_builder.add_column(
    dd.ExpressionColumnConfig(
        name="middle_name",
        expr="{{ person.middle_name if person.middle_name else ' ' }}",
    )
)
config_builder.add_column(dd.ExpressionColumnConfig(name="last_name", expr="{{ person.last_name }}"))
# Note: the underlying field is `district`; the original Nemotron-Personas-USA dataset surfaces it as `county`.
config_builder.add_column(dd.ExpressionColumnConfig(name="county", expr="{{ person.district }}"))

# Add specific personal detail columns -- included in the public release
config_builder.add_column(dd.ExpressionColumnConfig(name="sex", expr="{{ person.sex }}"))
config_builder.add_column(dd.ExpressionColumnConfig(name="age", expr="{{ person.age }}", dtype="int"))
config_builder.add_column(dd.ExpressionColumnConfig(name="marital_status", expr="{{ person.marital_status }}"))
# These can legitimately be null in the source dataset; coerce to a single space so downstream
# Jinja templates stay safe (DD's validator rejects expression columns that render to "").
config_builder.add_column(
    dd.ExpressionColumnConfig(
        name="education_level",
        expr="{{ person.education_level if person.education_level else ' ' }}",
    )
)
config_builder.add_column(
    dd.ExpressionColumnConfig(
        name="bachelors_field",
        expr="{{ person.bachelors_field if person.bachelors_field else ' ' }}",
    )
)
config_builder.add_column(
    dd.ExpressionColumnConfig(
        name="occupation",
        expr="{{ person.occupation if person.occupation else ' ' }}",
    )
)
config_builder.add_column(dd.ExpressionColumnConfig(name="city", expr="{{ person.city }}"))
config_builder.add_column(dd.ExpressionColumnConfig(name="state", expr="{{ person.state }}"))
# Note: the underlying field is `postcode`; the original dataset surfaces it as `zipcode`.
config_builder.add_column(dd.ExpressionColumnConfig(name="zipcode", expr="{{ person.postcode }}"))
config_builder.add_column(dd.ExpressionColumnConfig(name="country", expr="{{ person.country }}"))

# %% [markdown]
# ### 👀 Generate a preview to see what we have so far (OCEAN + PGM columns only for now)

# %%
preview = data_designer.preview(config_builder, num_records=10)
preview.display_sample_record()

# %% [markdown]
# ### ➡️ Next, generate persona attributes grounded in OCEAN + PGM

# %%
PERSONA_ATTRIBUTES_SYSTEM_PROMPT = """\
You are a detailed persona generator specializing in creating realistic, nuanced, and diverse personal attributes. You should:
1. Generate attributes that are internally consistent and logically connected to the base persona details
2. Ensure cultural sensitivity and avoid stereotypes while acknowledging cultural influences
3. Create specific, detailed responses rather than generic ones
4. Base your responses on realistic correlations between personal attributes like ethnic background, age, sex, marital status, education, occupation, etc.
5. Always return your response in a valid JSON format
6. DO NOT include any explanations or reasoning for your choices

Your responses should be creative yet plausible, diverse yet consistent with the provided demographic information.
"""


# We define a PersonaAttributes schema so that all attributes are generated in one go,
# with the types and constraints as specified below. Pydantic is used to automatically validate the output.
class PersonaAttributes(BaseModel):
    cultural_background: str = Field(description="Description of the person's cultural background")
    skills_and_expertise: str = Field(description="Description of the person's skills and expertise")
    skills_and_expertise_list: list[str] = Field(description="List of the person's skills and expertise")
    career_goals_and_ambitions: str = Field(description="Description of the person's career goals and ambitions")
    hobbies_and_interests: str = Field(description="Description of the person's hobbies and interests")
    hobbies_and_interests_list: list[str] = Field(description="List of the person's hobbies and interests")


# %%
# Here we use a structured output column trick to generate all persona attributes
# in one shot, minimizing the number of API calls.
#
# Note how easy it is to access other fields in the dataset via Jinja templating.
# Doing so automatically infuses every record with row-specific details.
config_builder.add_column(
    dd.LLMStructuredColumnConfig(
        name="persona_attributes",
        system_prompt=PERSONA_ATTRIBUTES_SYSTEM_PROMPT,
        prompt="""\
Based on a person with the following profile:

Name: {{ first_name }} {{ middle_name if middle_name else '' }} {{ last_name }}
Sex: {{ sex }}
Age: {{ age }}
{{ 'Ethnic background: ' + ethnic_background if ethnic_background else ''}}
Marital status: {{ marital_status }}
Education: {{ education_level }}{{ ' in ' + bachelors_field if bachelors_field != 'no degree' else '' }}
Occupation: {{ occupation }}
Location: {{ city }}, {{ state }}, {{ county }}

Personality profile:
- {{ openness.description }}
- {{ conscientiousness.description }}
- {{ extraversion.description }}
- {{ agreeableness.description }}
- {{ neuroticism.description }}

Generate the following detailed persona attributes:
- cultural_background
- skills_and_expertise
- skills_and_expertise_list
- career_goals_and_ambitions
- hobbies_and_interests
- hobbies_and_interests_list

When generating attributes, make sure to incorporate the influences suggested by the personality profile description.
""",
        output_format=PersonaAttributes,
        model_alias=MODEL_ALIAS,
        drop=True,
    )
)

# Now we break up into multiple columns
config_builder.add_column(
    dd.ExpressionColumnConfig(name="cultural_background", expr="{{ persona_attributes.cultural_background }}")
)
config_builder.add_column(
    dd.ExpressionColumnConfig(name="skills_and_expertise", expr="{{ persona_attributes.skills_and_expertise }}")
)
config_builder.add_column(
    dd.ExpressionColumnConfig(
        name="skills_and_expertise_list", expr="{{ persona_attributes.skills_and_expertise_list }}"
    )
)
config_builder.add_column(
    dd.ExpressionColumnConfig(
        name="career_goals_and_ambitions", expr="{{ persona_attributes.career_goals_and_ambitions }}"
    )
)
config_builder.add_column(
    dd.ExpressionColumnConfig(name="hobbies_and_interests", expr="{{ persona_attributes.hobbies_and_interests }}")
)
config_builder.add_column(
    dd.ExpressionColumnConfig(
        name="hobbies_and_interests_list", expr="{{ persona_attributes.hobbies_and_interests_list }}"
    )
)

# %% [markdown]
# ### 🔍 Generate a preview and examine a sample record

# %%
preview = data_designer.preview(config_builder, num_records=10)

# %%
preview.dataset[0:3]

# %%
preview.display_sample_record()

# %% [markdown]
# ### 4.3 🦸‍♀️ 👩‍🎤 👩‍🍳 👩‍🔬 Generate Personas
#
# Now, let's focus on the second part shown in the diagram below:
#
# <div align="center">
#   <img src="https://raw.githubusercontent.com/NVIDIA-NeMo/DataDesigner/main/docs/devnotes/posts/assets/nemotron-personas/nemotron_persona_via_ndd_step_3.png" alt="Stage 4: Persona prose synthesis" width="600" />
# </div>

# %%
PERSONA_SYSTEM_PROMPT = """\
You are a specialized persona generator that creates fine-grained, creative and meaningful persona descriptions based on an individual's cultural background, skills, career goals, and interests. You should:
1. Synthesize a coherent persona that naturally emerges from these characteristics
2. Focus on how these attributes combine to create a unique perspective and approach to life
3. Ensure the persona description reflects the intersection of professional expertise, cultural values, and personal interests
4. Create a narrative that explains how these characteristics influence their worldview and decision-making
5. Always return your response in a valid JSON format
6. INCLUDE NAME IN EVERY PERSONA DESCRIPTION.
7. ALWAYS TAKE AGE INTO ACCOUNT TO INFORM INTERESTS, HABITS AND AFFINITY TO VARIOUS ASPECTS OF LIFE.
8. NEVER DIRECTLY MENTION THE CULTURAL HERITAGE. INSTEAD, INFUSE IT INTO PERSONA DESCRIPTIONS BY REFERRING TO CULTURAL PRACTICES, TRADITIONS, AND VALUES.

Each persona should be very specific, not a generic/bland description. Do not shy away from mentioning bad habits or quirks.

Here are examples of how each persona description may begin:
"An aspiring musician..."
"A renowned machine learning researcher..."
"A neonatal nurse with decades of experience..."
"An urban planner with a passion..."
"""


# We define a Personas schema so that all attributes are generated in one go,
# with the types and constraints specified. Again, Pydantic is used to automatically validate the output.
class Personas(BaseModel):
    professional_persona: str = Field(
        description="A one-sentence persona description including primary field of work, key professional skills, and how their unique personality traits manifest in their career"
    )
    finance_persona: str = Field(
        description="A one-sentence persona characterization of spending habits, relationship with money, saving and investment habits, and approach to financial decision-making, mentioning specific financial instruments and investment strategies they use."
    )
    healthcare_persona: str = Field(
        description="A one-sentence persona description of very specific health conditions they have as well as their approach to medical care, and their typical behavior as a patient. Include condition names and describe how the person proactively addresses/ completely neglects/ periodically manages/ actively monitors/ struggles with/ effectively controls/ inconsistently treats these conditions"
    )
    sports_persona: str = Field(
        description="A one-sentence persona description of athletic interests, seasonal sports, and their approach to fitness and exercise. Provide specific names of professional sports teams and club affiliations, based on the persona location"
    )
    arts_persona: str = Field(
        description="A one-sentence persona characterization of engagement with creative expression, artistic appreciation, cultural activities, and how the arts shape their identity and leisure time, if at all. Always provide specific artist/musician/actor/performer names"
    )
    travel_persona: str = Field(
        description="A one-sentence persona capturing travel interests and style, including planning preferences, adventure versus relaxation focus, and financial or family constraints. Always provide specific local and/or international destinations they have visited or wish to visit"
    )
    culinary_persona: str = Field(
        description="A one-sentence persona description of food/cuisine preferences, cooking skill level, and approach to dining experiences. Always provide specific names of dishes and names of ingredients they enjoy."
    )
    concise_persona: str = Field(
        description="A one-sentence description capturing the essence of this person's unique perspective and approach to life, highlighting unique quirks, facts, and/or bad habits"
    )
    detailed_persona: str = Field(
        description="A paragraph describing persona's cultural background, skills, goals, and interests shape their worldview and decision-making. Don't shy away from talking about bad habits or quirks"
    )


# %%
# Here we use a structured output column trick to generate all personas
# in one call, minimizing the number of API calls.
#
# As before, we can easily access other fields in the dataset via Jinja templating.
# Doing so automatically infuses every record with row-specific details.
config_builder.add_column(
    dd.LLMStructuredColumnConfig(
        name="personas",
        system_prompt=PERSONA_SYSTEM_PROMPT,
        prompt="""\
Based on a person with the following persona attributes and profile:

Age: {{ age }}
Cultural background: {{ cultural_background }}
{{ 'Hobbies and interests: ' + hobbies_and_interests if age >= 6 else '' }}
{{ 'Skills and expertise: ' + skills_and_expertise if age >= 16 else '' }}
{{ 'Career goals and ambitions: ' + career_goals_and_ambitions if age >= 16 else '' }}

Personality profile:
- {{ openness.description }}
- {{ conscientiousness.description }}
- {{ extraversion.description }}
- {{ agreeableness.description }}
- {{ neuroticism.description }}

Generate the following self-contained persona descriptions that capture how persona attributes and profile combine to create a unique individual's perspective and approach to various facets of life.

- professional_persona
- finance_persona
- healthcare_persona
- sports_persona
- arts_persona
- travel_persona
- culinary_persona
- concise_persona
- detailed_persona

Each requested persona description should be self-contained, meaning it can't begin with they/their as the reference wouldn't be clear.
When generating personas, make sure to incorporate the influences suggested by the personality profile description.

DO NOT USE THE RACE OF THE PERSONA IN YOUR RESPONSE.
NEVER DIRECTLY MENTION THE CULTURAL HERITAGE. INSTEAD, INFUSE IT INTO PERSONA DESCRIPTIONS BY REFERRING TO CULTURAL PRACTICES, TRADITIONS, AND VALUES.
INCLUDE NAME IN EVERY PERSONA DESCRIPTION.
ALWAYS TAKE AGE INTO ACCOUNT TO INFORM INTERESTS, HABITS AND AFFINITY TO VARIOUS ASPECTS OF LIFE.

Each persona description should be creative yet plausible and consistent with the provided demographic information and persona attributes.
Each persona should be very specific, not a generic/bland description. Do not shy away from mentioning bad habits or quirks.

Here are examples of how each description may begin:
"An aspiring musician..."
"A renowned machine learning researcher..."
"A neonatal nurse with decades of experience..."
"An urban planner with a passion..."
""",
        output_format=Personas,
        model_alias=MODEL_ALIAS,
        drop=True,
    )
)

# Now we break up into multiple columns
config_builder.add_column(
    dd.ExpressionColumnConfig(name="professional_persona", expr="{{ personas.professional_persona }}")
)
config_builder.add_column(dd.ExpressionColumnConfig(name="finance_persona", expr="{{ personas.finance_persona }}"))
config_builder.add_column(
    dd.ExpressionColumnConfig(name="healthcare_persona", expr="{{ personas.healthcare_persona }}")
)
config_builder.add_column(dd.ExpressionColumnConfig(name="sports_persona", expr="{{ personas.sports_persona }}"))
config_builder.add_column(dd.ExpressionColumnConfig(name="arts_persona", expr="{{ personas.arts_persona }}"))
config_builder.add_column(dd.ExpressionColumnConfig(name="travel_persona", expr="{{ personas.travel_persona }}"))
config_builder.add_column(dd.ExpressionColumnConfig(name="culinary_persona", expr="{{ personas.culinary_persona }}"))
config_builder.add_column(dd.ExpressionColumnConfig(name="concise_persona", expr="{{ personas.concise_persona }}"))
config_builder.add_column(dd.ExpressionColumnConfig(name="detailed_persona", expr="{{ personas.detailed_persona }}"))

# %% [markdown]
# ### 🔍 Generate a preview and examine a sample record

# %%
preview = data_designer.preview(config_builder, num_records=10)

# %%
preview.dataset[0:3]

# %%
preview.display_sample_record()

# %% [markdown]
# ### ↗️  Scale Up Persona Generation
# Scale up to the specified `NUM_RECORDS`. Bump this number for larger runs (the original Nemotron-Personas release scales to millions).

# %%
NUM_RECORDS = 50

scaled_persona_results = data_designer.create(config_builder, num_records=NUM_RECORDS, dataset_name="personas")

# Load the dataset into a pandas DataFrame
all_personas = scaled_persona_results.load_dataset()
all_personas.head(3)

# %% [markdown]
# ### 📄 View the evaluation report

# %%
analysis = scaled_persona_results.load_analysis()
analysis.to_report()

# %% [markdown]
# # 5. 🎯 Customize for your use case
#
# Everything above reproduces the **general-purpose** Nemotron-Personas-USA pipeline. In practice, enterprises will want personas grounded in their own domain (a healthcare provider needs persona dimensions a media company doesn't, and vice versa). With NeMo Data Designer, layering a custom attribute or persona on top of the released artifact is a few lines of config.
#
# To make the customization story concrete, the cell below adds a **`tech_persona`** dimension (with a specific list of `tech_tools` they use) that wasn't in the original Nemotron-Personas schema. The same pattern (one Pydantic schema + one `LLMStructuredColumnConfig` + one expression column per output field) generalizes to any domain-specific dimension you need.


# %%
class TechPersona(BaseModel):
    tech_persona: str = Field(
        description=(
            "A 2-3 sentence description of this person's relationship with technology: "
            "comfort with AI/digital tools, level of tech adoption (early-adopter / mainstream / late / "
            "skeptic), preferred devices, and one specific way technology shapes their daily routine. "
            "Be specific and consistent with the rest of the persona profile."
        )
    )
    tech_tools: list[str] = Field(
        description=(
            "List of 4-6 specific tech tools, apps, services, or devices this person uses regularly. "
            "Each entry should be a concrete named product, not a generic category."
        )
    )


config_builder.add_column(
    dd.LLMStructuredColumnConfig(
        name="custom_persona",
        system_prompt=(
            "You write nuanced, specific tech-relationship personas grounded in demographic "
            "and psychometric attributes. Avoid generic platitudes; ground every claim in the "
            "person's age, occupation, personality, and lifestyle."
        ),
        prompt="""\
Based on a person with the following persona profile:

Name: {{ first_name }} {{ last_name }}, Age: {{ age }}, Occupation: {{ occupation }}
Cultural background: {{ cultural_background }}
Career goals: {{ career_goals_and_ambitions }}
Hobbies: {{ hobbies_and_interests }}
Concise persona: {{ concise_persona }}

Personality profile:
- {{ openness.description }}
- {{ conscientiousness.description }}
- {{ extraversion.description }}
- {{ agreeableness.description }}
- {{ neuroticism.description }}

Generate the `tech_persona` and `tech_tools` fields as described in the schema. Be specific and consistent with the profile above.
""",
        output_format=TechPersona,
        model_alias=MODEL_ALIAS,
        drop=True,
    )
)

config_builder.add_column(dd.ExpressionColumnConfig(name="tech_persona", expr="{{ custom_persona.tech_persona }}"))
config_builder.add_column(dd.ExpressionColumnConfig(name="tech_tools", expr="{{ custom_persona.tech_tools }}"))

# %%
preview = data_designer.preview(config_builder, num_records=5)
preview.display_sample_record()

# %% [markdown]
# # ⏭️ Next Steps
#
# 1. Everything above is just an example of personas that can be generated. These personas are not set in stone and can be easily adjusted. For example, if your downstream model needs a different type of persona, tweak or extend the pipeline (Section 5 demonstrates the pattern).
#
# 2. You should be able to use this notebook as is to generate Nemotron-Personas for any of the [supported locales](https://nvidia-nemo.github.io/DataDesigner/latest/concepts/person_sampling/) by changing `personas_locale` (and running `data-designer download personas --locale <code>` once for the new locale). For a brand-new region without an NGC dataset, flip `SAMPLE_FROM_SDG_PGM = True` and provide a custom [SDG-PGMs](https://github.com/NVIDIA-NeMo/SDG-PGMs) generator (the OCEAN helpers in Section 2 are the Stage 1 scaffolding for that path).
#     - You may need to adjust and/or translate prompts to your region's language(s)
#     - You may need to work with a different LLM that is better suited for your region
