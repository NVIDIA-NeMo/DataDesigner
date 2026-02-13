/** Full source for openresearcher_demo.py - used in ExpandableCode on deep-research-trajectories page */

export const openresearcherDemoCode = `import data_designer.config as dd
from data_designer.interface import DataDesigner

# Models
config = dd.DataDesignerConfigBuilder()
config.add_model_config(
    dd.ModelConfig(
        alias="search_rollout_model",
        model="nvidia/nemotron-3-nano-30b-a3b",
        provider="nvidia",
        inference_parameters=dd.ChatCompletionInferenceParams(
            temperature=1.0,
            top_p=0.95,
            max_tokens=16384,
        ),
    )
)
config.add_model_config(
    dd.ModelConfig(
        alias="judge",
        model="nvidia/nemotron-3-nano-30b-a3b",
        provider="nvidia",
    )
)

# MCP retriever
tool_config = dd.ToolConfig(
    tool_alias="knowledge-base",
    providers=["corpus-retriever"],
    max_tool_call_turns=150,
)
config.add_tool_config(tool_config)

# Seed questions with reference answers
config.with_seed_dataset(
    dd.LocalFileSeedSource(path="questions.jsonl"),
)

config.add_column(
    dd.ExpressionColumnConfig(
        name="research_question",
        expr="{{ question }}",
    )
)

# Research trajectory generation
config.add_column(
    dd.LLMTextColumnConfig(
        name="research_answer",
        prompt="Research and answer thoroughly:\n\n{{ research_question }}",
        model_alias="search_rollout_model",
        system_prompt=SYSTEM_PROMPT,
        tool_alias="knowledge-base",
        with_trace=dd.TraceType.ALL_MESSAGES,
        extract_reasoning_content=True,
    )
)

# Rejection sampling judge
config.add_column(
    dd.LLMJudgeColumnConfig(
        name="correctness",
        model_alias="judge",
        prompt=(
            "Question: {{ research_question }}\n"
            "Reference answer: {{ answer }}\n"
            "Generated answer: {{ research_answer }}\n"
            "Does the generated answer correctly address the question?"
        ),
        scores=[
            dd.Score(
                name="correct",
                description="Is the answer factually correct?",
                options={
                    1: "Correct",
                    0: "Incorrect",
                },
            ),
        ],
    )
)

# Run
mcp_provider = dd.LocalStdioMCPProvider(
    name="corpus-retriever",
    command="uv",
    args=["run", "retriever_mcp.py", "serve"],
    env={"CORPUS_PATH": "corpus.jsonl"},
)
data_designer = DataDesigner(mcp_providers=[mcp_provider])
results = data_designer.create(
    config_builder=config,
    num_records=1000,
    dataset_name="research-trajectories",
)
`;
