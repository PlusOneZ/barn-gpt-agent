CHAT_SYSTEM_PROMPT_ZH = '''你是一个为用户提供帮助的助手，请尽量提供内容丰富有帮助的输出。请使用Markdown语法来格式化你的输出。'''
VISION_MAX_LENGTH = 512

DALLE_MODEL_COSTS = {
    "dall-e-2": {
        "standard": {
            "1024x1024": 0.020,
            "512x512": 0.018,
            "256x256": 0.016,
        },
    },
    "dall-e-3": {
        "standard": {
            "1024x1024": 0.040,
            "1024x1792": 0.080,
            "1792x1024": 0.080,
        },
        "hd": {
            "1024x1024": 0.080,
            "1024x1792": 0.120,
            "1792x1024": 0.120,
        },
    }
}

DEFAULT_SIZE = "1024x1024"
DEFAULT_QUALITY = "standard"


def check_size_valid(model, size):
    if model not in DALLE_MODEL_COSTS:
        return DEFAULT_SIZE
    if size not in DALLE_MODEL_COSTS[model][DEFAULT_QUALITY]:
        return DEFAULT_SIZE
    return size


def check_quality_valid(quality):
    if quality not in ["standard", "hd"]:
        return DEFAULT_QUALITY
    return quality


ONE_MILLION = 1000000

DEFAULT_MODELS = {
    "chat": "gpt-4o-mini",
    "image-generation": "dall-e-2",
    "image-recognition": "gpt-4o-2024-08-06",
    "audio-generation": "tts-1",
    "audio-recognition": "whisper-1"
}

USD_PRICE_PER_TOKEN = {
    "gpt-4o-2024-08-06": {
        "input": 2.5 / ONE_MILLION,
        "output": 10. / ONE_MILLION,
    },
    "chatgpt-4o-latest": {
        "input": 5. / ONE_MILLION,
        "output": 15. / ONE_MILLION,
    },
    "gpt-4o-mini": {
        "input": .15 / ONE_MILLION,
        "output": .60 / ONE_MILLION,
    },
    "gpt-3.5-turbo": {
        "input": 0.5 / ONE_MILLION,
        "output": 1.5 / ONE_MILLION,
    },
    "gpt-4o": {
        "input": 5. / ONE_MILLION,
        "output": 15. / ONE_MILLION,
    },
    "gpt-4-turbo": {
        "input": 10. / ONE_MILLION,
        "output": 30. / ONE_MILLION,
    },
    "gpt-4-32k": {
        "input": 60. / ONE_MILLION,
        "output": 120. / ONE_MILLION,
    },
    "gpt-4": {
        "input": 30. / ONE_MILLION,
        "output": 60. / ONE_MILLION,
    },
}
