# Use Case Recipes

Recipes are a collection of code examples that demonstrate how to leverage Data Designer in specific use cases.
Each recipe is a self-contained example that can be run independently.

!!! question "New to Data Designer?"
    Recipes provide working code for specific use cases without detailed explanations. If you're learning Data Designer for the first time, we recommend starting with our [tutorial notebooks](../../notebooks/), which offer step-by-step guidance and explain core concepts. Once you're familiar with the basics, return here for practical, ready-to-use implementations.

!!! tip Prerequisite
    These recipes use the Open AI model provider by default. Ensure your OpenAI model provider has been set up using the Data Designer CLI before running a recipe.

<div class="grid cards" markdown>

-   :material-snake:{ .lg .middle } **Text to Python**

    Generate a dataset of natural language instructions paired with Python code implementations, with varying complexity levels and industry focuses.

    ---

    **Demonstrates:**

    - Python code generation
    - Python code validation
    - LLM-as-judge

    ---

    [:material-book-open-page-variant: View Recipe](code_generation/text_to_python.md){ .md-button }
    [Download Code :octicons-download-24:](../assets/recipes/code_generation/text_to_python.py){ .md-button download="text_to_python.py" }

-   :material-database:{ .lg .middle } **Text to SQL**

    Generate a dataset of natural language instructions paired with SQL code implementations, with varying complexity levels and industry focuses.

    ---

    **Demonstrates:**

    - SQL code generation
    - SQL code validation
    - LLM-as-judge

    ---

    [:material-book-open-page-variant: View Recipe](code_generation/text_to_sql.md){ .md-button }
    [Download Code :octicons-download-24:](../assets/recipes/code_generation/text_to_sql.py){ .md-button download="text_to_sql.py" }


-   :material-chat:{ .lg .middle } **Product Info QA**

    Generate a dataset that contains information about products and associated question/answer pairs.

    ---

    **Demonstrates:**

    - Structured outputs
    - Expression columns
    - LLM-as-judge

    ---

    [:material-book-open-page-variant: View Recipe](qa_and_chat/product_info_qa.md){ .md-button }
    [Download Code :octicons-download-24:](../assets/recipes/qa_and_chat/product_info_qa.py){ .md-button download="product_info_qa.py" }


-   :material-chat:{ .lg .middle } **Multi-Turn Chat**

    Generate a dataset of multi-turn chat conversations between a user and an AI assistant.

    ---

    **Demonstrates:**

    - Structured outputs
    - Expression columns
    - LLM-as-judge

    ---

    [:material-book-open-page-variant: View Recipe](qa_and_chat/multi_turn_chat.md){ .md-button }
    [Download Code :octicons-download-24:](../assets/recipes/qa_and_chat/multi_turn_chat.py){ .md-button download="multi_turn_chat.py" }

-   :material-file-document:{ .lg .middle } **W-2 Tax Forms**

    Generate synthetic W-2 tax form datasets with realistic employee and employer information using person samplers and statistical distributions.

    ---

    **Demonstrates:**

    - Person samplers
    - Expression columns
    - Statistical distributions
    - Realistic PII generation

    ---

    [:material-book-open-page-variant: View Recipe](forms/w2_forms.md){ .md-button }
    [Download Code :octicons-download-24:](../assets/recipes/forms/w2_forms.py){ .md-button download="w2_forms.py" }

-   :material-hospital-box:{ .lg .middle } **Clinical Trials**

    Create synthetic clinical trial datasets with trial metadata, participant demographics, investigator details, and medical observations.

    ---

    **Demonstrates:**

    - Person samplers for multiple roles
    - Healthcare data generation
    - Conditional sampling
    - Medical documentation

    ---

    [:material-book-open-page-variant: View Recipe](healthcare/clinical_trials.md){ .md-button }
    [Download Code :octicons-download-24:](../assets/recipes/healthcare/clinical_trials.py){ .md-button download="clinical_trials.py" }

-   :material-brain:{ .lg .middle } **Reasoning Traces**

    Build synthetic reasoning traces demonstrating step-by-step problem-solving.

    ---

    **Demonstrates:**

    - Reasoning trace generation
    - Structured thinking patterns
    - Empathic AI responses
    - Multi-step reasoning

    ---

    [:material-book-open-page-variant: View Recipe](reasoning/reasoning_traces.md){ .md-button }
    [Download Code :octicons-download-24:](../assets/recipes/reasoning/reasoning_traces.py){ .md-button download="reasoning_traces.py" }

-   :material-file-search:{ .lg .middle } **RAG Evaluation**

    Generate comprehensive evaluation datasets for Retrieval-Augmented Generation systems with question-answer pairs and quality metrics.

    ---

    **Demonstrates:**

    - Seed dataset usage
    - Structured Q&A generation
    - LLM-as-judge evaluation
    - Multiple difficulty levels

    ---

    [:material-book-open-page-variant: View Recipe](rag/rag_evaluation.md){ .md-button }
    [Download Code :octicons-download-24:](../assets/recipes/rag/rag_evaluation.py){ .md-button download="rag_evaluation.py" }

-   :material-image-text:{ .lg .middle } **Visual Question Answering**

    Create visual question answering datasets using Vision Language Models with image context and multiple choice questions.

    ---

    **Demonstrates:**

    - Multimodal generation
    - Image context handling
    - Structured outputs
    - Multiple choice Q&A

    ---

    [:material-book-open-page-variant: View Recipe](multimodal/visual_qa.md){ .md-button }
    [Download Code :octicons-download-24:](../assets/recipes/multimodal/visual_qa.py){ .md-button download="visual_qa.py" }

-   :material-hospital:{ .lg .middle } **Insurance Claims**

    Generate synthetic insurance claims datasets with realistic policyholder information, claim details, and adjuster notes.

    ---

    **Demonstrates:**

    - Person samplers for multiple roles
    - Insurance domain data
    - Conditional sampling
    - PII generation

    ---

    [:material-book-open-page-variant: View Recipe](healthcare/insurance_claims.md){ .md-button }
    [Download Code :octicons-download-24:](../assets/recipes/healthcare/insurance_claims.py){ .md-button download="insurance_claims.py" }

-   :material-stethoscope:{ .lg .middle } **Physician Notes**

    Create realistic physician notes with patient demographics, vital signs, chief complaints, and detailed notes.

    ---

    **Demonstrates:**

    - Medical documentation
    - Person samplers
    - Vital signs generation
    - SOAP note format

    ---

    [:material-book-open-page-variant: View Recipe](healthcare/physician_notes.md){ .md-button }
    [Download Code :octicons-download-24:](../assets/recipes/healthcare/physician_notes.py){ .md-button download="physician_notes.py" }

</div>
