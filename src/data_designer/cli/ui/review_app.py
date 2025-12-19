# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from data_designer.cli.services.review_service import ReviewService

# Page config
st.set_page_config(
    page_title="DataDesigner Review",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Comprehensive CSS styling
st.markdown(
    """
    <style>
    /* Main app styling */
    .stApp {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    header {visibility: hidden;}

    /* All text elements */
    * {
        color: #e0e0e0;
    }

    /* Headers - adjusted hierarchy */
    h1 {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.75rem !important;
        line-height: 1.2 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    h2 {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.75rem !important;
        line-height: 1.2 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        line-height: 1.2 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    h4 {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        line-height: 1.2 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    h5, h6 {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        line-height: 1.2 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Paragraphs and labels */
    p {
        color: #e0e0e0 !important;
        font-size: 0.9rem !important;
        line-height: 1.4 !important;
        margin: 0 !important;
    }

    label, span, div {
        color: #e0e0e0 !important;
    }

    /* Strong/bold text */
    strong, b {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #2d2d2d;
        color: #ffffff !important;
        border: 1px solid #505050;
        border-radius: 4px;
        padding: 0.4rem 0.8rem;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.2s;
        white-space: nowrap;
        height: 36px;
        min-height: 36px;
        max-height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .stButton > button:hover {
        background-color: #3d3d3d;
        border-color: #707070;
    }

    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background-color: #0e7fc1;
        border-color: #1a8cd8;
        color: #ffffff !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #1a8cd8;
    }

    /* Disabled buttons - maintain same size */
    .stButton > button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
        height: 36px;
        min-height: 36px;
        max-height: 36px;
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border: 1px solid #505050 !important;
        border-radius: 4px;
        font-size: 0.9rem !important;
    }

    /* Number input - match button height */
    .stNumberInput > div > div > input {
        height: 36px !important;
        min-height: 36px !important;
        max-height: 36px !important;
        padding: 0 0.8rem !important;
    }

    /* Input labels */
    .stTextInput > label,
    .stTextArea > label,
    .stNumberInput > label {
        color: #e0e0e0 !important;
        font-size: 0.85rem !important;
        margin-bottom: 4px !important;
    }

    /* Code blocks */
    .stCodeBlock,
    [data-testid="stCodeBlock"] {
        background-color: #252525 !important;
    }

    pre {
        background-color: #252525 !important;
        border: 1px solid #505050 !important;
        border-radius: 4px;
        padding: 0.75rem !important;
        color: #e0e0e0 !important;
        font-size: 13px !important;
        line-height: 1.5 !important;
    }

    code {
        background-color: transparent !important;
        font-family: 'Monaco', 'Menlo', 'Consolas', monospace !important;
    }

    /* Streamlit code syntax highlighting - Python */
    code .hljs-keyword,
    code .token.keyword { color: #569cd6 !important; }

    code .hljs-string,
    code .token.string { color: #ce9178 !important; }

    code .hljs-number,
    code .token.number { color: #b5cea8 !important; }

    code .hljs-comment,
    code .token.comment { color: #6a9955 !important; }

    code .hljs-function,
    code .token.function { color: #dcdcaa !important; }

    code .hljs-class-name,
    code .token.class-name { color: #4ec9b0 !important; }

    code .hljs-built_in,
    code .token.builtin { color: #4ec9b0 !important; }

    code .hljs-operator,
    code .token.operator { color: #d4d4d4 !important; }

    /* SQL specific */
    code .hljs-type,
    code .token.type { color: #4ec9b0 !important; }

    /* Line numbers */
    .stCodeBlock [data-line-number]::before {
        color: #858585 !important;
    }

    /* Tables - FIXED */
    table {
        background-color: #2d2d2d !important;
        color: #e0e0e0 !important;
    }

    thead tr th {
        background-color: #3d3d3d !important;
        color: #ffffff !important;
        border-bottom: 2px solid #505050 !important;
        padding: 0.5rem !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }

    tbody tr td {
        background-color: #2d2d2d !important;
        color: #e0e0e0 !important;
        border-bottom: 1px solid #404040 !important;
        padding: 0.5rem !important;
        font-size: 0.85rem !important;
    }

    tbody tr:hover td {
        background-color: #353535 !important;
    }

    /* DataFrame specific */
    .dataframe {
        color: #e0e0e0 !important;
        background-color: #2d2d2d !important;
    }

    .dataframe th {
        color: #ffffff !important;
        background-color: #3d3d3d !important;
    }

    .dataframe td {
        color: #e0e0e0 !important;
        background-color: #2d2d2d !important;
    }

    /* Streamlit dataframe widget */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] > div,
    .stDataFrame {
        background-color: #2d2d2d !important;
    }

    [data-testid="stDataFrame"] table {
        background-color: #2d2d2d !important;
        color: #e0e0e0 !important;
    }

    [data-testid="stDataFrame"] thead th {
        background-color: #3d3d3d !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }

    [data-testid="stDataFrame"] tbody td {
        background-color: #2d2d2d !important;
        color: #e0e0e0 !important;
        font-size: 0.85rem !important;
    }

    [data-testid="stDataFrame"] tbody tr:hover td {
        background-color: #353535 !important;
    }

    /* Dividers */
    hr {
        border-color: #505050 !important;
        margin: 0.5rem 0 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border: 1px solid #505050;
        font-size: 0.9rem !important;
    }

    .streamlit-expanderContent {
        background-color: #252525 !important;
        color: #e0e0e0 !important;
        border: 1px solid #404040;
        border-top: none;
    }

    /* Messages */
    .stSuccess {
        background-color: #1e3a1e !important;
        color: #66bb6a !important;
        border: 1px solid #2e5a2e;
        font-size: 0.9rem !important;
    }

    .stError {
        background-color: #3a1e1e !important;
        color: #ef5350 !important;
        border: 1px solid #5a2e2e;
        font-size: 0.9rem !important;
    }

    .stInfo {
        background-color: #1e2a3a !important;
        color: #42a5f5 !important;
        border: 1px solid #2e3a5a;
        font-size: 0.9rem !important;
    }

    /* Markdown */
    .stMarkdown {
        color: #e0e0e0 !important;
    }

    /* Metrics */
    .stMetric {
        background-color: #2d2d2d;
        padding: 0.5rem;
        border-radius: 4px;
        border: 1px solid #404040;
    }

    .stMetric label {
        color: #b0b0b0 !important;
        font-size: 0.75rem !important;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1rem !important;
    }

    /* JSON viewer - aggressive override */
    .stJson,
    .stJson > div,
    .stJson > div > div,
    .stJson pre,
    [data-testid="stJson"],
    [data-testid="stJson"] > div,
    [data-testid="stJson"] pre {
        background-color: #252525 !important;
        color: #e0e0e0 !important;
    }

    /* JSON syntax elements */
    .stJson .json-key {
        color: #9cdcfe !important;
    }

    .stJson .json-value {
        color: #ce9178 !important;
    }

    .stJson .json-string {
        color: #ce9178 !important;
    }

    /* JSON inside expander */
    .streamlit-expanderContent .stJson,
    .streamlit-expanderContent [data-testid="stJson"] {
        background-color: #252525 !important;
    }

    /* All JSON-like structures */
    pre[class*="json"],
    div[class*="json"] {
        background-color: #252525 !important;
        color: #e0e0e0 !important;
    }

    /* All text displays */
    .stText {
        background-color: #2d2d2d !important;
        color: #e0e0e0 !important;
        padding: 0.4rem;
        border-radius: 4px;
    }

    /* Pre-formatted text */
    .stMarkdown pre {
        background-color: #252525 !important;
        color: #e0e0e0 !important;
        border: 1px solid #505050 !important;
    }

    /* Caption */
    .stCaption {
        color: #999 !important;
        font-size: 0.75rem !important;
    }

    /* Block container - reduced padding */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        max-width: 100% !important;
    }

    /* Override any white/light backgrounds */
    div[style*="background-color: rgb(255, 255, 255)"],
    div[style*="background-color: white"],
    div[style*="background: white"] {
        background-color: #252525 !important;
    }

    /* Ensure all container backgrounds are dark */
    .element-container,
    [data-testid="element-container"] {
        background-color: transparent !important;
    }

    /* Streamlit specific overrides */
    .stMarkdownContainer {
        background-color: transparent !important;
    }

    /* Any remaining white backgrounds */
    * {
        scrollbar-color: #404040 #1e1e1e;
    }

    ::-webkit-scrollbar {
        background-color: #1e1e1e;
    }

    ::-webkit-scrollbar-thumb {
        background-color: #404040;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session_state(dataset_path: Path, reviewer: str) -> None:
    """Initialize Streamlit session state with proper defaults."""
    if "service" not in st.session_state:
        st.session_state.service = ReviewService(dataset_path)
        st.session_state.reviewer = reviewer
        st.session_state.current_index = 0


def get_rating_key(index: int) -> str:
    """Get session state key for rating at index."""
    return f"rating_{index}"


def get_comment_key(index: int) -> str:
    """Get session state key for comment at index."""
    return f"comment_{index}"


def render_header(service: ReviewService) -> None:
    """Render compact header with progress."""
    progress = service.get_review_progress()

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.markdown("## DataDesigner Review")

    with col2:
        st.markdown(
            f'<p style="line-height: 36px; margin: 0; font-size: 0.85rem; color: #b0b0b0;">Progress: {progress["reviewed_records"]}/{progress["total_records"]} ({progress["progress_percent"]:.0f}%)</p>',
            unsafe_allow_html=True,
        )

    with col3:
        if st.button("End Session", use_container_width=True):
            st.success("✓ Reviews saved")
            st.stop()


def render_navigation(service: ReviewService) -> None:
    """Render navigation controls."""
    info = service.get_dataset_info()

    st.markdown('<div style="margin-bottom: 8px;"><strong>Navigation</strong></div>', unsafe_allow_html=True)

    col1, col2, col3, col_spacer, col4 = st.columns([1, 1, 1, 0.5, 2])

    with col1:
        if st.button("← Prev", disabled=st.session_state.current_index == 0):
            st.session_state.current_index -= 1
            st.rerun()

    with col2:
        if st.button("Next →", disabled=st.session_state.current_index >= info["num_records"] - 1):
            st.session_state.current_index += 1
            st.rerun()

    with col3:
        if st.button("Random"):
            import random

            st.session_state.current_index = random.randint(0, info["num_records"] - 1)
            st.rerun()

    with col4:
        new_index = st.number_input(
            "Jump to Record Index",
            min_value=0,
            max_value=info["num_records"] - 1,
            value=st.session_state.current_index,
            step=1,
            help=f"Enter a record index (0-{info['num_records'] - 1}) to jump to that record",
        )
        if new_index != st.session_state.current_index:
            st.session_state.current_index = new_index
            st.rerun()


def classify_column(name: str, value: Any) -> str:
    """Classify column by content type."""
    if isinstance(value, dict) and "is_valid" in value:
        return "validation"

    if isinstance(value, dict):
        all_judge = all(
            isinstance(v, dict) and "score" in v and "reasoning" in v for v in value.values() if isinstance(v, dict)
        )
        if all_judge and len(value) > 0:
            return "llm_judge"

    if isinstance(value, str) and len(value) > 50:
        code_keywords = ["def ", "class ", "function ", "SELECT ", "FROM ", "import "]
        if any(kw in value for kw in code_keywords):
            return "code"

    return "generated"


def render_code_column(name: str, value: str) -> None:
    """Render code with syntax highlighting."""
    value_upper = value.upper()

    if any(kw in value_upper for kw in ["SELECT ", "INSERT ", "UPDATE ", "DELETE ", "CREATE ", "FROM ", "WHERE "]):
        lang = "sql"
    elif any(kw in value for kw in ["def ", "import ", "class ", "return ", "elif ", "from ", "async ", "await "]):
        lang = "python"
    elif any(kw in value for kw in ["function ", "const ", "let ", "var ", "=>", "console.log"]):
        lang = "javascript"
    elif any(kw in value for kw in ["public class ", "private ", "protected ", "void ", "static "]):
        lang = "java"
    elif any(kw in value for kw in ["#include", "int main(", "printf(", "cout <<", "std::"]):
        lang = "cpp"
    elif any(kw in value for kw in ["#!/bin/bash", "#!/bin/sh", "echo ", "export ", "${"]):
        lang = "bash"
    elif value.strip().startswith("{") and value.strip().endswith("}"):
        lang = "json"
    else:
        lang = "python"

    st.markdown(f"**{name}**")
    st.code(value, language=lang, line_numbers=True)


def render_validation_column(name: str, value: dict) -> None:
    """Render validation results as a table."""
    st.markdown(f"**{name}**")

    if "is_valid" in value:
        table_data = []
        for key, val in value.items():
            if key == "is_valid":
                display_val = "✅ Valid" if val else "❌ Invalid"
            else:
                display_val = str(val)
            table_data.append({"Field": key, "Value": display_val})

        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    else:
        results = []
        for col_name, val_output in value.items():
            if isinstance(val_output, dict):
                is_valid = val_output.get("is_valid")
                status = "✅ Valid" if is_valid else "❌ Invalid"
                results.append({"Column": col_name, "Status": status})

        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)


def render_llm_judge_column(name: str, value: dict) -> None:
    """Render LLM judge scores as a table."""
    st.markdown(f"**{name}**")

    table_data = []
    for measure, results in value.items():
        if isinstance(results, dict):
            score = results.get("score", "N/A")
            reasoning = results.get("reasoning", "No reasoning provided")
            table_data.append({"Measure": measure, "Score": str(score), "Reasoning": reasoning})

    if table_data:
        st.dataframe(
            pd.DataFrame(table_data),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Measure": st.column_config.TextColumn("Measure", width="small"),
                "Score": st.column_config.TextColumn("Score", width="small"),
                "Reasoning": st.column_config.TextColumn("Reasoning", width="large"),
            },
        )


def render_record_display(service: ReviewService, index: int) -> None:
    """Render record with smart formatting."""
    try:
        record = service.get_record_by_index(index)

        code_cols = []
        validation_cols = []
        judge_cols = []
        generated_cols = []

        for col_name in record.index:
            value = record[col_name]
            col_type = classify_column(col_name, value)

            if col_type == "code":
                code_cols.append((col_name, value))
            elif col_type == "validation":
                validation_cols.append((col_name, value))
            elif col_type == "llm_judge":
                judge_cols.append((col_name, value))
            else:
                generated_cols.append((col_name, value))

        if generated_cols:
            st.markdown("#### Generated Columns")
            data = []
            for name, value in generated_cols:
                if isinstance(value, (dict, list)):
                    val_str = json.dumps(value, indent=2)
                else:
                    val_str = str(value)
                if len(val_str) > 200:
                    val_str = val_str[:200] + "..."
                data.append({"Column": name, "Value": val_str})

            df = pd.DataFrame(data)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Column": st.column_config.TextColumn("Column", width="small"),
                    "Value": st.column_config.TextColumn("Value", width="large"),
                },
            )
            st.markdown('<div style="margin-bottom: 16px;"></div>', unsafe_allow_html=True)

        if code_cols:
            st.markdown("#### Code")
            for name, value in code_cols:
                render_code_column(name, value)
                st.markdown('<div style="margin-bottom: 12px;"></div>', unsafe_allow_html=True)

        if validation_cols:
            st.markdown("#### Validation")
            for name, value in validation_cols:
                render_validation_column(name, value)
                st.markdown('<div style="margin-bottom: 12px;"></div>', unsafe_allow_html=True)

        if judge_cols:
            st.markdown("#### LLM Judge")
            for name, value in judge_cols:
                render_llm_judge_column(name, value)
                st.markdown('<div style="margin-bottom: 12px;"></div>', unsafe_allow_html=True)

        with st.expander("View Raw JSON"):
            st.json(record.to_dict())

    except Exception as e:
        st.error(f"Error loading record: {e}")


def render_review_panel(service: ReviewService, index: int) -> None:
    """Render review panel with rating and comments."""
    st.markdown("### Rating")

    rating_key = get_rating_key(index)
    if rating_key not in st.session_state:
        st.session_state[rating_key] = None

    st.markdown('<div style="margin-bottom: 8px;"><strong>Select rating:</strong></div>', unsafe_allow_html=True)
    cols = st.columns(6)

    ratings = [("Reject", "❌"), ("1", "1"), ("2", "2"), ("3", "3"), ("4", "4"), ("5", "5")]
    for i, (rating_value, rating_label) in enumerate(ratings):
        with cols[i]:
            is_selected = st.session_state[rating_key] == rating_value
            button_label = f"**{rating_label}**" if is_selected else rating_label

            if st.button(
                button_label,
                key=f"rating_btn_{rating_value}_{index}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                st.session_state[rating_key] = rating_value
                st.rerun()

    if st.session_state[rating_key]:
        if st.session_state[rating_key] == "Reject":
            st.caption("**Selected:** ❌ Reject")
        else:
            st.caption(f"**Selected:** {st.session_state[rating_key]}/5")

    st.markdown('<div style="margin: 16px 0;"><hr style="margin: 0;"></div>', unsafe_allow_html=True)

    st.markdown("### Comments")
    comment_key = get_comment_key(index)
    comment = st.text_area(
        "Add notes (optional)",
        key=comment_key,
        height=100,
        placeholder="Add feedback...",
        label_visibility="collapsed",
    )

    if st.button("Submit Review", use_container_width=True, type="primary"):
        selected_rating = st.session_state[rating_key]

        if not selected_rating:
            st.error("Please select a rating")
            return

        rating = "thumbs_down" if selected_rating == "Reject" else "thumbs_up"

        rating_text = f"[Rating: {selected_rating}]"
        full_comment = f"{rating_text} {comment}" if comment else rating_text

        try:
            service.submit_review(
                record_index=index,
                rating=rating,
                comment=full_comment,
                reviewer=st.session_state.reviewer,
            )

            st.session_state[rating_key] = None
            if comment_key in st.session_state:
                del st.session_state[comment_key]

            info = service.get_dataset_info()
            if st.session_state.current_index < info["num_records"] - 1:
                st.session_state.current_index += 1
                st.rerun()
            else:
                st.success("✓ All records reviewed!")

        except Exception as e:
            st.error(f"Submit failed: {e}")


def main() -> None:
    """Main app entry point."""
    if len(sys.argv) < 3:
        st.error("Launch via: data-designer review --dataset <path>")
        return

    dataset_path = Path(sys.argv[1])
    reviewer = sys.argv[2]

    initialize_session_state(dataset_path, reviewer)
    service = st.session_state.service

    render_header(service)
    st.markdown('<div style="margin-bottom: 16px;"></div>', unsafe_allow_html=True)

    render_navigation(service)
    st.markdown('<div style="margin-bottom: 32px;"></div>', unsafe_allow_html=True)

    col_content, col_review = st.columns([2.5, 1])

    with col_content:
        st.markdown(f"### Record {st.session_state.current_index}")
        st.markdown('<div style="margin-bottom: 16px;"></div>', unsafe_allow_html=True)
        render_record_display(service, st.session_state.current_index)

    with col_review:
        render_review_panel(service, st.session_state.current_index)

    st.markdown('<div style="margin-top: 32px;"></div>', unsafe_allow_html=True)
    st.caption(f"💾 Auto-saved to: `{service.repository.review_file_path}`")


if __name__ == "__main__":
    main()
