# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
import os
from pathlib import Path

from rich.theme import Theme

DEFAULT_NUM_RECORDS = 10

EPSILON = 1e-8
REPORTING_PRECISION = 2

DEFAULT_REPR_HTML_STYLE = "nord"

REPR_HTML_FIXED_WIDTH = 1000
REPR_HTML_TEMPLATE = """
<meta charset="UTF-8">
<style>
{{css}}

.code {{{{
  padding: 4px;
  border: 1px solid grey;
  border-radius: 4px;
  max-width: {fixed_width}px;
  width: 100%;
  display: inline-block;
  box-sizing: border-box;
  text-align: left;
  vertical-align: top;
  line-height: normal;
  overflow-x: auto;
}}}}

.code pre {{{{
  white-space: pre-wrap;       /* CSS 3 */
  white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
  white-space: -pre-wrap;      /* Opera 4-6 */
  white-space: -o-pre-wrap;    /* Opera 7 */
  word-wrap: break-word;
  overflow-wrap: break-word;
  margin: 0;
}}}}
</style>
{{highlighted_html}}
""".format(fixed_width=REPR_HTML_FIXED_WIDTH)


class NordColor(Enum):
    NORD0 = "#2E3440"  # Darkest gray (background)
    NORD1 = "#3B4252"  # Dark gray
    NORD2 = "#434C5E"  # Medium dark gray
    NORD3 = "#4C566A"  # Lighter dark gray
    NORD4 = "#D8DEE9"  # Light gray (default text)
    NORD5 = "#E5E9F0"  # Very light gray
    NORD6 = "#ECEFF4"  # Almost white
    NORD7 = "#8FBCBB"  # Teal
    NORD8 = "#88C0D0"  # Light cyan
    NORD9 = "#81A1C1"  # Soft blue
    NORD10 = "#5E81AC"  # Darker blue
    NORD11 = "#BF616A"  # Red
    NORD12 = "#D08770"  # Orange
    NORD13 = "#EBCB8B"  # Yellow
    NORD14 = "#A3BE8C"  # Green
    NORD15 = "#B48EAD"  # Purple


RICH_CONSOLE_THEME = Theme(
    {
        "repr.number": NordColor.NORD15.value,  # Purple for numbers
        "repr.string": NordColor.NORD14.value,  # Green for strings
        "repr.bool_true": NordColor.NORD9.value,  # Blue for True
        "repr.bool_false": NordColor.NORD9.value,  # Blue for False
        "repr.none": NordColor.NORD11.value,  # Red for None
        "repr.brace": NordColor.NORD7.value,  # Teal for brackets/braces
        "repr.comma": NordColor.NORD7.value,  # Teal for commas
        "repr.ellipsis": NordColor.NORD7.value,  # Teal for ellipsis
        "repr.attrib_name": NordColor.NORD3.value,  # Light gray for dict keys
        "repr.attrib_equal": NordColor.NORD7.value,  # Teal for equals signs
        "repr.call": NordColor.NORD10.value,  # Darker blue for function calls
        "repr.function_name": NordColor.NORD10.value,  # Darker blue for function names
        "repr.class_name": NordColor.NORD12.value,  # Orange for class names
        "repr.module_name": NordColor.NORD8.value,  # Light cyan for module names
        "repr.error": NordColor.NORD11.value,  # Red for errors
        "repr.warning": NordColor.NORD13.value,  # Yellow for warnings
    }
)

DEFAULT_HIST_NAME_COLOR = "medium_purple1"

DEFAULT_HIST_VALUE_COLOR = "pale_green3"


DEFAULT_AGE_RANGE = [18, 114]
MIN_AGE = 0
MAX_AGE = 114

LOCALES_WITH_MANAGED_DATASETS = ["en_US", "ja_JP", "en_IN", "hi_IN"]

US_STATES_AND_MAJOR_TERRITORIES = {
    # States
    "AK",
    "AL",
    "AR",
    "AZ",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "IA",
    "ID",
    "IL",
    "IN",
    "KS",
    "KY",
    "LA",
    "MA",
    "MD",
    "ME",
    "MI",
    "MN",
    "MO",
    "MS",
    "MT",
    "NC",
    "ND",
    "NE",
    "NH",
    "NJ",
    "NM",
    "NV",
    "NY",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VA",
    "VT",
    "WA",
    "WI",
    "WV",
    "WY",
    # D.C.
    "DC",
    # Territories
    "AS",
    "GU",
    "MP",
    "PR",
    "VI",
}

MAX_TEMPERATURE = 2.0
MIN_TEMPERATURE = 0.0
MAX_TOP_P = 1.0
MIN_TOP_P = 0.0
MIN_MAX_TOKENS = 1
REASONING_TRACE_COLUMN_POSTFIX = "__reasoning_trace"

AVAILABLE_LOCALES = [
    "ar_AA",
    "ar_AE",
    "ar_BH",
    "ar_EG",
    "ar_JO",
    "ar_PS",
    "ar_SA",
    "az_AZ",
    "bg_BG",
    "bn_BD",
    "bs_BA",
    "cs_CZ",
    "da_DK",
    "de",
    "de_AT",
    "de_CH",
    "de_DE",
    "dk_DK",
    "el_CY",
    "el_GR",
    "en",
    "en_AU",
    "en_BD",
    "en_CA",
    "en_GB",
    "en_IE",
    "en_IN",
    "en_NZ",
    "en_PH",
    "en_TH",
    "en_US",
    "es",
    "es_AR",
    "es_CA",
    "es_CL",
    "es_CO",
    "es_ES",
    "es_MX",
    "et_EE",
    "fa_IR",
    "fi_FI",
    "fil_PH",
    "fr_BE",
    "fr_CA",
    "fr_CH",
    "fr_FR",
    #    "fr_QC", deprecated, use fr_CA instead
    "ga_IE",
    "he_IL",
    "hi_IN",
    "hr_HR",
    "hu_HU",
    "hy_AM",
    "id_ID",
    "it_CH",
    "it_IT",
    "ja_JP",
    "ka_GE",
    "ko_KR",
    "la",
    "lb_LU",
    "lt_LT",
    "lv_LV",
    "mt_MT",
    "ne_NP",
    "nl_BE",
    "nl_NL",
    "no_NO",
    "or_IN",
    "pl_PL",
    "pt_BR",
    "pt_PT",
    "ro_RO",
    "ru_RU",
    "sk_SK",
    "sl_SI",
    "sq_AL",
    "sv_SE",
    "ta_IN",
    "th",
    "th_TH",
    "tl_PH",
    "tr_TR",
    "tw_GH",
    "uk_UA",
    "vi_VN",
    "zh_CN",
    "zh_TW",
    "zu_ZA",
]

DATA_DESIGNER_HOME_ENV_VAR = "DATA_DESIGNER_HOME"

DATA_DESIGNER_HOME = Path(os.getenv(DATA_DESIGNER_HOME_ENV_VAR, Path.home() / ".data-designer"))

MANAGED_ASSETS_PATH_ENV_VAR = "DATA_DESIGNER_MANAGED_ASSETS_PATH"

MANAGED_ASSETS_PATH = Path(os.getenv(MANAGED_ASSETS_PATH_ENV_VAR, DATA_DESIGNER_HOME / "managed-assets"))

MODEL_CONFIGS_FILE_NAME = "model_configs.yaml"

MODEL_CONFIGS_FILE_PATH = DATA_DESIGNER_HOME / MODEL_CONFIGS_FILE_NAME

MODEL_PROVIDERS_FILE_NAME = "model_providers.yaml"

MODEL_PROVIDERS_FILE_PATH = DATA_DESIGNER_HOME / MODEL_PROVIDERS_FILE_NAME

NVIDIA_PROVIDER_NAME = "nvidia"

NVIDIA_API_KEY_ENV_VAR_NAME = "NVIDIA_API_KEY"

OPENAI_PROVIDER_NAME = "openai"

OPENAI_API_KEY_ENV_VAR_NAME = "OPENAI_API_KEY"

PREDEFINED_PROVIDERS = [
    {
        "name": NVIDIA_PROVIDER_NAME,
        "endpoint": "https://integrate.api.nvidia.com/v1",
        "provider_type": "openai",
        "api_key": NVIDIA_API_KEY_ENV_VAR_NAME,
    },
    {
        "name": OPENAI_PROVIDER_NAME,
        "endpoint": "https://api.openai.com/v1",
        "provider_type": "openai",
        "api_key": OPENAI_API_KEY_ENV_VAR_NAME,
    },
]

PREDEFINED_PROVIDERS_MODEL_MAP = {
    NVIDIA_PROVIDER_NAME: {
        "text": "nvidia/nvidia-nemotron-nano-9b-v2",
        "reasoning": "openai/gpt-oss-20b",
        "vision": "nvidia/nemotron-nano-12b-v2-vl",
    },
    OPENAI_PROVIDER_NAME: {
        "text": "gpt-4.1",
        "reasoning": "gpt-5",
        "vision": "gpt-5",
    },
}
