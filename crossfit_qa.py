# /// script
# dependencies = [
#   "data-designer",
# ]
# ///

import data_designer.config as dd
from data_designer.interface import DataDesigner


def build_config() -> dd.DataDesignerConfigBuilder:
    config_builder = dd.DataDesignerConfigBuilder()

    config_builder.add_column(dd.SamplerColumnConfig(
        name="category",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(values=[
            "Movements & Technique",
            "Programming & Workouts",
            "Nutrition & Recovery",
            "Competition & History",
            "Equipment & Gym Setup",
            "Scaling & Injuries",
        ]),
    ))

    config_builder.add_column(dd.SamplerColumnConfig(
        name="topic",
        sampler_type=dd.SamplerType.SUBCATEGORY,
        params=dd.SubcategorySamplerParams(
            category="category",
            values={
                "Movements & Technique": [
                    "Olympic lifts (snatch, clean & jerk)",
                    "Gymnastics movements (muscle-ups, handstand walks)",
                    "Kettlebell techniques",
                    "Double-unders and jump rope",
                    "Rowing and assault bike mechanics",
                ],
                "Programming & Workouts": [
                    "Benchmark WODs (Fran, Murph, Grace)",
                    "AMRAP vs EMOM vs For Time",
                    "Strength cycle programming",
                    "CrossFit Open workouts",
                    "Hero WODs",
                ],
                "Nutrition & Recovery": [
                    "Zone diet vs macros",
                    "Pre and post workout nutrition",
                    "Mobility and stretching routines",
                    "Sleep and recovery protocols",
                    "Supplements for CrossFit athletes",
                ],
                "Competition & History": [
                    "CrossFit Games history",
                    "Notable athletes (Mat Fraser, Tia-Clair Toomey)",
                    "Qualifying process and semifinals",
                    "Team vs individual competition",
                    "CrossFit affiliates and community",
                ],
                "Equipment & Gym Setup": [
                    "Barbell and bumper plate selection",
                    "Rig and pull-up bar setup",
                    "Wrist wraps, knee sleeves, and belts",
                    "Home gym essentials",
                    "Shoes (lifters vs metcons)",
                ],
                "Scaling & Injuries": [
                    "Scaling options for beginners",
                    "Common CrossFit injuries and prevention",
                    "Rhabdomyolysis awareness",
                    "Shoulder and knee prehab",
                    "Returning to training after injury",
                ],
            },
        ),
    ))

    config_builder.add_column(dd.SamplerColumnConfig(
        name="audience_level",
        sampler_type=dd.SamplerType.CATEGORY,
        params=dd.CategorySamplerParams(
            values=["beginner", "intermediate", "advanced"],
            weights=[0.4, 0.4, 0.2],
        ),
    ))

    config_builder.add_column(dd.LLMTextColumnConfig(
        name="question",
        model_alias="nvidia-text",
        system_prompt=(
            "You generate realistic questions that a CrossFit athlete would ask. "
            "The question should sound natural, like something posted in a CrossFit forum or asked to a coach. "
            "Output only the question, nothing else."
        ),
        prompt=(
            "Generate a {{ audience_level }}-level question about {{ topic }} "
            "in the context of {{ category }}."
        ),
        with_trace=dd.TraceType.ALL_MESSAGES,
    ))

    config_builder.add_column(dd.LLMTextColumnConfig(
        name="answer",
        model_alias="nvidia-text",
        system_prompt=(
            "You are an experienced CrossFit Level 3 coach. "
            "Give accurate, practical answers in 2-4 sentences. "
            "Be direct and succinct -- no bullet lists, no tables, no markdown formatting. "
            "Adjust your language to match the audience level."
        ),
        prompt=(
            "Answer this CrossFit question for a {{ audience_level }} athlete in 2-4 sentences:\n\n"
            "{{ question }}"
        ),
        with_trace=dd.TraceType.ALL_MESSAGES,
    ))

    return config_builder


if __name__ == "__main__":
    config_builder = build_config()
    designer = DataDesigner()
    preview = designer.preview(config_builder=config_builder, num_records=10)

    preview.dataset.to_parquet("crossfit_qa_preview.parquet")
    print(f"Saved {len(preview.dataset)} records to crossfit_qa_preview.parquet")

    from data_designer.web.server import run_server
    run_server(data_file="crossfit_qa_preview.parquet", open_browser=True)
