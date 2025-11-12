from data_designer.essentials import (
    CategorySamplerParams,
    DataDesigner,
    DataDesignerConfigBuilder,
    InferenceParameters,
    LLMTextColumnConfig,
    ModelConfig,
    PersonSamplerParams,
    SamplerColumnConfig,
    Score,
    SubcategorySamplerParams,
    ToJsonlProcessorConfig,
)

# define model aliases
model_alias_generator = "content_generator"
model_configs = [
    ModelConfig(
        alias=model_alias_generator,
        provider="nvidia",
        model="deepseek-ai/deepseek-r1-distill-qwen-14b",
        inference_parameters=InferenceParameters(
            max_tokens=8000,
            temperature=0.7,
            top_p=0.95,
        ),
    )
]

config_builder = DataDesignerConfigBuilder(model_configs=model_configs)

# ESI levels
ESI_LEVELS = [
    "ESI 1: Resuscitation",
    "ESI 2: Emergency",
    "ESI 3: Urgent",
    "ESI 4: Less Urgent",
    "ESI 5: Non-urgent",
]

# Unique record ID
config_builder.add_column(
    name="record_id", column_type="sampler", sampler_type="uuid", params={"short_form": True, "uppercase": True}
)

# ESI level (balanced sampling)
config_builder.add_column(
    SamplerColumnConfig(
        name="esi_level_description",
        sampler_type="category",
        params=CategorySamplerParams(
            values=ESI_LEVELS,
        ),
    )
)

# Clinical scenario (conditioned on ESI level)
config_builder.add_column(
    SamplerColumnConfig(
        name="clinical_scenario",
        sampler_type="subcategory",
        params=SubcategorySamplerParams(
            category="esi_level_description",
            values={
                ESI_LEVELS[0]: [
                    "Cardiac arrest",
                    "Unresponsive with no pulse",
                    "Severe respiratory distress",
                    "Major trauma with signs of shock",
                    "Suspected narcotic overdose with shallow respirations",
                ],
                ESI_LEVELS[1]: [
                    "Crushing substernal chest pain radiating to the left arm",
                    "Sudden onset of facial droop and arm weakness",
                    "New onset confusion in an elderly patient",
                    "Active suicidal ideation with a plan",
                    "High-speed motor vehicle accident",
                    "Severe abdominal pain in a patient with a history of aortic aneurysm",
                ],
                ESI_LEVELS[2]: [
                    "Abdominal pain with fever and nausea",
                    "High fever with a productive cough and history of COPD",
                    "Displaced fracture with visible deformity",
                    "Asthma attack, responsive to initial treatment",
                    "Vaginal bleeding in a pregnant patient",
                    "Head injury with brief loss of consciousness",
                ],
                ESI_LEVELS[3]: [
                    "Simple laceration requiring sutures",
                    "Twisted ankle, unable to bear weight",
                    "Sore throat with fever",
                    "Symptoms of a urinary tract infection",
                    "Painful ear with fever in a child",
                ],
                ESI_LEVELS[4]: [
                    "Request for a prescription refill",
                    "Suture removal",
                    "Minor rash present for several days",
                    "Common cold symptoms",
                    "Follow-up for a minor wound check",
                ],
            },
        ),
    )
)

# Synthetic patient info
config_builder.add_column(
    SamplerColumnConfig(
        name="patient",
        sampler_type="person",
        params=PersonSamplerParams(age_range=[18, 70]),
    )
)

# Triage note writing style (captures range from poor to best quality notes)
config_builder.add_column(
    SamplerColumnConfig(
        name="writing_style",
        sampler_type="category",
        params=CategorySamplerParams(values=["Draft", "Adequate", "Polished"]),
    )
)

# LLM-generated triage note
config_builder.add_column(
    LLMTextColumnConfig(
        name="content",
        prompt=(
            "You are an experienced triage nurse in a busy Emergency Department writing a draft note. "
            "Write a realistic, concise triage note in a telegraphic style using common medical abbreviations. "
            "The note is for a {{ patient.age }} y/o {{ 'M' if patient.sex == 'Male' else 'F' }}. "
            "Triage classification: '{{ esi_level_description }}'. "
            "Reason for visit: '{{ clinical_scenario }}'. "
            "Desired writing style: '{{ writing_style }}'. "
            "Structure the note with 'CC:' and 'HPI:'. "
            "Adjust the style and level of clinical detail based on the 'writing_style': "
            "- Draft: Use minimal structure, brief statements, and omit some details; clinical indicators may be less clear. "
            "- Adequate: Use complete sentences, include all relevant clinical indicators, but avoid excessive detail. "
            "- Polished: Be thorough, precise, and clear; include nuanced or subtle signs and show strong clinical reasoning. "
            "Also, adjust level of detail based on urgency (ESI 1 is always brief). "
            "Respond with ONLY the note text, starting with 'CC:'."
        ),
        model_alias=model_alias_generator,
    )
)

# Rubric: clinical coherence
clinical_coherence_rubric = Score(
    name="Clinical Coherence",
    description="Evaluates how well the clinical details in the triage note align with the assigned ESI level and scenario.",
    options={
        "5": "Note is perfectly aligned with the ESI level and scenario; details are clinically plausible and specific.",
        "4": "Note is well-aligned, with only minor details that might be slightly inconsistent.",
        "3": "Note is generally consistent, but some key clinical indicators are missing or don't fully match the ESI level.",
        "2": "Note shows significant inconsistency between the clinical details and the assigned ESI level.",
        "1": "Note is clinically incoherent and does not reflect the assigned ESI level or scenario at all.",
    },
)

# Rubric: ESI level complexity (reduced to 3 levels: Simple, Moderate, Complex)
esi_level_complexity_rubric = Score(
    name="ESI Level Complexity",
    description="Evaluates how difficult it is to infer the correct ESI level from the note. Higher scores indicate greater complexity, which is desirable for creating a challenging dataset.",
    options={
        "Complex": "Note contains subtle or conflicting information, requiring clinical reasoning to distinguish between ESI levels.",
        "Moderate": "Note requires some clinical inference; indicators are present but not always immediately obvious.",
        "Simple": "Note uses clear, direct, or textbook indicators that make the ESI level obvious.",
    },
)

jsonl_entry_template = {
    "messages": [
        {
            "role": "system",
            "content": (
                "You are an expert ER triage nurse. Your task is to classify the following triage note into one of the five Emergency Severity Index (ESI) levels."
                f" The possible levels are: {', '.join([repr(level) for level in ESI_LEVELS])}."
                " Carefully analyze the clinical details in the triage note, focusing on patient acuity, resource needs, and risk of rapid deterioration."
                " Respond with only the selected ESI level description, exactly matching one of the listed possibilities. Do not provide extra text or explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                "Triage Note: {{ content }}\n"
                "Classify the ESI level for this note based on the provided definitions."
                ' Respond in JSON format only: { "esi_level_description": "..." }'
            ),
        },
        {"role": "assistant", "content": ('{ "esi_level_description": "{{ esi_level_description }}" }')},
    ],
}

config_builder.add_processor(
    ToJsonlProcessorConfig(
        template=jsonl_entry_template,
        folder_name="jsonl_files",
        fraction_per_file={
            "train.jsonl": 0.8,
            "validation.jsonl": 0.2,
        },
    )
)

dd = DataDesigner(
    artifact_path="./artifacts", blob_storage_path="/Users/amanoel/Data/nemotron-personas-datasets_v0.0.6"
)
preview = dd.preview(config_builder, num_records=10)

dd.create(config_builder, num_records=20)
