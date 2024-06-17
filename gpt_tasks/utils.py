from .constants import USD_PRICE_PER_TOKEN, USD_TO_CNY


def usd_to_cny(amount: float) -> float:
    return amount * USD_TO_CNY


def token_to_usd(token_usage, model: str) -> float:
    # token usage {
    #     "completion_tokens": 17,
    #     "prompt_tokens": 57,
    #     "total_tokens": 74
    #   }
    try:
        for k in USD_PRICE_PER_TOKEN.keys():
            if model.startswith(k):
                model = k
                break
        print(USD_PRICE_PER_TOKEN[model]["input"], USD_PRICE_PER_TOKEN[model]["output"])
        return USD_PRICE_PER_TOKEN[model]["input"] * token_usage.prompt_tokens + \
            USD_PRICE_PER_TOKEN[model]["output"] * token_usage.completion_tokens
    except Exception as e:
        print(e)
        return 0


def get_price_from_resp(resp):
    return usd_to_cny(token_to_usd(resp.usage, resp.model))
