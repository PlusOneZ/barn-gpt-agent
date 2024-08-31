from .constants import USD_PRICE_PER_TOKEN, DALLE_MODEL_COSTS


def token_to_usd(token_usage, model: str) -> float:
    # token usage {
    #     "completion_tokens": 17,
    #     "prompt_tokens": 57,
    #     "total_tokens": 74
    #   }
    try:
        if model not in USD_PRICE_PER_TOKEN:
            model = "gpt-4"
        # print(USD_PRICE_PER_TOKEN[model]["input"], USD_PRICE_PER_TOKEN[model]["output"])
        return USD_PRICE_PER_TOKEN[model]["input"] * token_usage.prompt_tokens + \
            USD_PRICE_PER_TOKEN[model]["output"] * token_usage.completion_tokens
    except Exception as e:
        print(e)
        return 0


def get_price_from_resp(resp):
    return token_to_usd(resp.usage, resp.model)


def get_image_gen_price_from_model(model, resolution):
    if model not in DALLE_MODEL_COSTS:
        model = "dall-e-3"
    if resolution not in DALLE_MODEL_COSTS[model]:
        resolution = list(DALLE_MODEL_COSTS[model].keys())[-1]
    return DALLE_MODEL_COSTS[model].get(resolution, 0.080)
