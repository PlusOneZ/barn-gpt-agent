CHAT_SYSTEM_PROMPT_ZH = '''你是一个为用户提供帮助的助手，请尽量提供内容丰富有帮助的输出。请使用Markdown语法来格式化你的输出。'''
VISION_MAX_LENGTH = 200

DALLE_MODEL_COSTS = {
    "dall-e-2": {
        "1024x1024":    0.020,
        "512x512":      0.018,
        "256x256":      0.016,
    },
    "dall-e-3": {
        "1024x1024":    0.040,
        "1024x1792":    0.080,
        "1792x1024":    0.080,
    }
}

ONE_MILLION = 1000000

USD_TO_CNY = 10.

USD_PRICE_PER_TOKEN = {
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
    "gpt-4": {
        "input": 30. / ONE_MILLION,
        "output": 60. / ONE_MILLION,
    },
}