from openai import OpenAI
from dotenv import load_dotenv
from gradio_client import Client
import logging
import time
import os

from .constants import CHAT_SYSTEM_PROMPT_ZH, VISION_MAX_LENGTH, DEFAULT_MODELS, check_size_valid, check_quality_valid
from .utils import get_image_gen_price_from_model

load_dotenv()

client_openai = OpenAI()
client_flux1_schnell = Client("BarnGPT/FLUX.1-schnell", auth=[os.getenv("HF_USERNAME"), os.getenv("HF_PASSWORD")])
client_flux1_dev = Client("BarnGPT/FLUX.1-dev", auth=[os.getenv("HF_USERNAME"), os.getenv("HF_PASSWORD")])


# Calling to OpenAI chat completion API with
# Params:
#   - user_input: list of strings or dict with role and content
def chat(user_input: [], model, _):
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
    if model == "se1-flux1-schnell":
        return huggingface_flux_image_generation(client_flux1_schnell, user_prompt, options)
    elif model == "se1-flux1-dev":
        return huggingface_flux_image_generation(client_flux1_dev, user_prompt, options)
    else:
        return image_generation_openai(user_prompt, model, options)


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
    return return_result(result, user_prompt)


class OpenaiLikeResult:
    class Data:
        def __init__(self, b64_json, revised_prompt, url, seed):
            self.b64_json = b64_json
            self.revised_prompt = revised_prompt
            self.url = url
            self.seed = seed

    def __init__(self, url, seed, prompt, usage):
        current_timestamp = int(time.time())
        self._data = {
            "created": current_timestamp,
            "data": [
                {
                    "b64_json": None,  # Assuming b64_json is None
                    "revised_prompt": prompt,
                    "url": 'file://' + url,  # Use the parsed file path as the URL with "file://" prefix
                    "seed": seed
                }
            ],
            "usage": 0.04  # FIXME here
        }
        self.created = current_timestamp
        self.data = []
        self.data.append(self.Data(None, prompt, 'file://' + url, seed))
        self.usage = usage

    def __getitem__(self, item):
        return self._data[item]

    def json(self):
        return self._data


def return_result(result: tuple, prompt: str):
    # Get the current timestamp as an integer
    return OpenaiLikeResult(result[0], result[1], prompt, 0.04)
    # Convert the dictionary to a JSON string
    # json_string = json.dumps(data)
    # Print the resulting JSON string
    # logging.debug(json_string)


def image_generation_openai(user_prompt: str, model, options=None):
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


def vision(user_input: [], model, _):
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


