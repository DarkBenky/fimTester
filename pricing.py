PRICES = {
    "deepseek-chat": (0.28, 0.42),
    "deepseek-reasoner": (0.55, 2.19),
    "deepseek-flash": (0.14, 0.28),
    "deepseek-v3": (0.28, 0.42),
    "deepseek-r1": (0.55, 2.19),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
    "qwen/qwen3-coder": (1.50, 6.00),
    "qwen/qwen3-coder-next": (1.50, 6.00),
    "qwen/qwen3-32b": (0.30, 1.20),
    "qwen/qwen3-235b-a22b": (0.20, 1.20),
    "qwen/qwen3-30b-a3b": (0.15, 0.60),
    "qwen/qwen2.5-coder-32b-instruct": (0.30, 1.20),
    "qwen/qwen2.5-coder-30b-a3b-instruct": (0.15, 0.60),
    "qwen/qwen2.5-coder-14b-instruct": (0.15, 0.60),
    "qwen/qwen2.5-coder-7b-instruct": (0.05, 0.20),
    "qwen/qwen2.5-72b-instruct": (0.90, 3.60),
    "qwen/qwen2.5-32b-instruct": (0.30, 1.20),
    "qwen/qwen2.5-14b-instruct": (0.15, 0.60),
    "qwen/qwen2.5-7b-instruct": (0.05, 0.20),
    "qwen/qwen2.5-3b-instruct": (0.02, 0.08),
    "qwen/qwen2.5-1.5b-instruct": (0.01, 0.04),
    "qwen/qwen2.5-0.8b-instruct": (0.01, 0.04),
    "qwen/qwen2.5-0.5b-instruct": (0.01, 0.04),
}


def price_for(model, pricing_override):
    if pricing_override:
        return pricing_override.get("input"), pricing_override.get("output")
    return PRICES.get(model)


def cost_usd(model, prompt_tokens, completion_tokens, pricing_override, live_cost=None):
    if live_cost is not None:
        return live_cost
    price = price_for(model, pricing_override)
    if price is None:
        return None
    input_price, output_price = price
    return (prompt_tokens or 0) * input_price / 1_000_000 + (completion_tokens or 0) * output_price / 1_000_000
