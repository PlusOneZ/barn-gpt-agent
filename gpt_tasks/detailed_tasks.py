from openai import OpenAI
from dotenv import load_dotenv
from gradio_client import Client
import logging
import time
import os

from .constants import CHAT_SYSTEM_PROMPT_ZH, VISION_MAX_LENGTH, DEFAULT_MODELS, HF_FLUX_MODE_COST, check_size_valid, check_quality_valid
from .utils import get_image_gen_price_from_model

load_dotenv()

# 初始化OpenAI客户端
client_openai = OpenAI()
# 初始化huggingface的flux模型客户端
client_flux1_schnell = Client("BarnGPT/FLUX.1-schnell", auth=[os.getenv("HF_USERNAME"), os.getenv("HF_PASSWORD")])
client_flux1_dev = Client("BarnGPT/FLUX.1-dev", auth=[os.getenv("HF_USERNAME"), os.getenv("HF_PASSWORD")])


def chat(user_input: list, model, _):
    """
    调用OpenAI的聊天API

    参数:
    user_input: 用户输入的聊天内容
    model: 使用的模型名称
    _: 未使用
    返回:
    
    """
    logging.info("Chat completion called with user input: '{}'".format(user_input))
    # check user_input style
    send_input = [{"role": "system", "content": CHAT_SYSTEM_PROMPT_ZH}]
    send_input.extend([
        {"role": "user", "content": i} if isinstance(i, str)
        else {"role": i["role"], "content": i["content"]}
        for i in user_input
    ])
    completion = client_openai.chat.completions.create(
        model=model if model else DEFAULT_MODELS["chat"],
        messages=send_input
    )
    return completion


def image_generation(user_prompt: str, model, options=None):
    """
    调用huggingface的flux模型生成图像

    参数:
    user_prompt: 用户输入的提示文本
    model: 使用的模型名称
    options: 可选参数，包含生成图像的其他选项，quality, size
    返回:
    
    """
    if model == "se1-flux1-schnell":
        return huggingface_flux_image_generation(client_flux1_schnell, user_prompt, options)
    elif model == "se1-flux1-dev":
        return huggingface_flux_image_generation(client_flux1_dev, user_prompt, options)
    else:
        return image_generation_openai(user_prompt, model, options)


# 调用huggingface的flux模型生成图像
# 参数:
#   - client: 客户端对象
#   - user_prompt: 用户输入的提示文本
#   - options: 可选参数，包含生成图像的其他选项
# 返回:
#   根据不同模型调用相应的图像生成函数并返回结果
def huggingface_flux_image_generation(client: Client, user_prompt: str, options=None):
    _size = options.get("size", '1024x1024') if options else '1024x1024'
    _w, _h = _size.split('x')
    result = client.predict(
        prompt=user_prompt,
        randomize_seed=True,
        width=int(_w),
        height=int(_h),
        num_inference_steps=28,
        api_name="/infer"
    )
    logging.info(f"Result: {result}, prompt: {user_prompt}")
    return return_result(result, user_prompt, HF_FLUX_MODE_COST)


# 创建一个类似于OpenAI的Result类
class OpenaiLikeResult:
    class Data:
        """
        数据子类，用于存储生成图像的详细信息
        """
        def __init__(self, b64_json, revised_prompt, url, seed):
            self.b64_json = b64_json
            self.revised_prompt = revised_prompt
            self.url = url
            self.seed = seed

    def __init__(self, url, seed, prompt, usage):
        """
        初始化OpenaiLikeResult类

        参数:
        url: 图像的URL
        seed: 随机种子
        prompt: 用户输入的提示文本
        usage: 使用传入的参数
        """
        current_timestamp = int(time.time())
        # 创建一个包含图像生成详细信息的字典，用于 json 输出
        self._data = {
            "created": current_timestamp,
            "data": [
                {
                    "b64_json": None,  # 假设 b64_json 为 None
                    "revised_prompt": prompt,
                    "url": 'file://' + url,  # 使用解析的文件路径作为 URL 并添加 "file://" 前缀
                    "seed": seed
                }
            ],
            "usage": usage  # 使用传入的参数
        }
        self.created = current_timestamp
        self.data = []
        self.data.append(self.Data(None, prompt, 'file://' + url, seed))
        self.usage = usage

    def __getitem__(self, item):
        return self._data[item]

    def json(self):
        return self._data



def return_result(result: tuple, prompt: str, cost: float):
    """
    返回一个类似于OpenAI的Result类

    参数:
    result: 生成图像的结果
    prompt: 用户输入的提示文本
    返回:
    返回一个类似于OpenAI的Result类
    """
    return OpenaiLikeResult(result[0], result[1], prompt, cost)


def image_generation_openai(user_prompt: str, model, options=None):
    """
    调用OpenAI的图像生成API

    参数:
    user_prompt: 用户输入的提示文本
    model: 使用的模型名称
    options: 可选参数，包含生成图像的其他选项
    返回:
    
    """
    _size = options.get("size", '1024x1024') if options else '1024x1024'
    _quality = options.get("quality", 'standard') if options else 'standard'
    _model = model if model else DEFAULT_MODELS["image-generation"]
    _size = check_size_valid(_model, _size)
    _quality = check_quality_valid(_quality)
    logging.info(f"size and quality: {_size}, {_quality}")
    logging.info(f"Image Generation called with prompt: {user_prompt}")
    response = client_openai.images.generate(
        model=_model,
        prompt=user_prompt,
        size=_size,
        quality=_quality,
        n=1
    )
    response.usage = get_image_gen_price_from_model(model, _size, _quality)
    return response


def vision(user_input: list, model, _):
    """
    调用OpenAI的图像识别API

    参数:
    user_input: 用户输入的图像
    model: 使用的模型名称
    _: 未使用
    返回:
    
    """
    logging.info("Vision chat called with user input: '{}'".format(user_input))
    completion = client_openai.chat.completions.create(
        model=model if model else DEFAULT_MODELS["image-recognition"],
        messages=user_input,
        max_tokens=VISION_MAX_LENGTH,
    )
    return completion


def tts(_):
    logging.info("TTS called for rate limit checking")
    return


def transcribe_to_text(_):
    logging.info("STT called for rate limit checking")
    return


