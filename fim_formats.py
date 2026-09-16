import re

FORMATS = [
    {
        "name": "starcoder",
        "protocol": "openai_completions",
        "template": "<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>",
    },
    {
        "name": "codellama",
        "protocol": "openai_completions",
        "template": "<PRE> {prefix} <SUF>{suffix} <MID>",
    },
    {
        "name": "deepseek_tokens",
        "protocol": "openai_completions",
        "template": "<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>",
    },
    {
        "name": "qwen_openai_tags",
        "protocol": "openai_completions",
        "template": "<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>",
    },
    {
        "name": "native_suffix_param",
        "protocol": "deepseek_fim",
        "template": None,
    },
    {
        "name": "llamacpp_infill",
        "protocol": "llamacpp_infill",
        "template": None,
    },
]

TOKEN_PATTERN = re.compile(r"<[^<>]+>")

LEAK_TOKENS = sorted({
    token
    for fmt in FORMATS if fmt["template"]
    for token in TOKEN_PATTERN.findall(fmt["template"])
})


def format_names():
    return [fmt["name"] for fmt in FORMATS]


def applicable_formats(model, selected=None):
    formats = [
        fmt for fmt in FORMATS
        if not (fmt["name"] == "llamacpp_infill" and model.fim_endpoint is None)
    ]
    if selected:
        formats = [fmt for fmt in formats if fmt["name"] in selected]
    return formats
