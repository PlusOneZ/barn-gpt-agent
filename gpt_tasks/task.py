from backoff import on_exception, runtime
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import logging
from time import sleep
from threading import Thread

from ratelimit import limits, RateLimitException

from call_hook.send_results import call_hook_with_result

from .constants import CHAT_SYSTEM_PROMPT_ZH, VISION_MAX_LENGTH, DEFAULT_MODELS, check_size_valid, check_quality_valid
from .utils import get_price_from_resp, get_image_gen_price_from_model

from random import random

load_dotenv()

client = OpenAI()


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
    completion = client.chat.completions.create(
        model=model if model else DEFAULT_MODELS["chat"],
        messages=send_input
    )
    return completion


def image_generation(user_prompt: str, model, options=None):
    _size = options.get("size", '1024x1024') if options else '1024x1024'
    _quality = options.get("quality", 'standard') if options else 'standard'
    _model = model if model else DEFAULT_MODELS["image-generation"]
    _size = check_size_valid(_model, _size)
    _quality = check_quality_valid(_quality)
    logging.info(f"size and quality: {_size}, {_quality}")
    logging.info(f"Image Generation called with prompt: {user_prompt}")
    response = client.images.generate(
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
    completion = client.chat.completions.create(
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


def hook_callback_for_task(task, task_type, get_result, rate_control_only=False):
    match task_type:
        case "chat":
            call_per_period = 3500
        case "image-generation":
            call_per_period = 5
        case "image-recognition":
            call_per_period = 3500
        case "audio-generation":
            call_per_period = 100
        case "audio-recognition":
            call_per_period = 100
        case _:
            call_per_period = 5  # for testing

    def rate_limit_exceeded(params):
        logging.warning(f"Rate limit exceeded for {task_type}")
        call_hook_with_result(
            params['args'][1],
            [{
                "type": "error_rate_limit",
                "content": "Rate limit exceeded",
                "target_task": task_type,
            }],
            status="rejected")

    if not rate_control_only:
        @on_exception(
            runtime,
            RateLimitException,
            max_tries=0,
            max_time=5,
            value=lambda _: random() * 5 + 1,
            on_giveup=rate_limit_exceeded,
            raise_on_giveup=False)
        @limits(calls=call_per_period, period=60)
        def inner_func(data, hook, model=None, options=None):
            try:
                api_resp = task(data, model, options)
                logging.debug(api_resp.json() if api_resp else "dummy response")
                result, usage = get_result(api_resp)
                if hook:
                    call_hook_with_result(hook, [{
                        "type": task_type,
                        "content": result,
                        # "tokens_used": usage,
                        "usage": usage
                    }], api_response=api_resp)
            except OpenAIError as e:
                logging.error(f"{task_type}: task not finished!")
                logging.error(f"Data: {data}")
                logging.error(f"Hook: {hook}")
                logging.error(f"With error: {e}")
                call_hook_with_result(
                    hook, [{
                        "type": "3rd_party_error",
                        "content": str(e),
                        "err_body": e.body if e.body else None,
                        "target_task": task_type,
                    }],
                    status="failed")
            except Exception as e:
                logging.error(f"{task_type}: task not finished!")
                logging.error(f"Data: {data}")
                logging.error(f"With error: {str(e)}")
                call_hook_with_result(
                    hook, [{
                        "type": "system_error",
                        "content": "Error while dealing with inner logic, this call is not charged",
                        "err_body": str(e),
                        "target_task": task_type,
                    }],
                    status="failed"
                )
    else:
        @limits(calls=call_per_period, period=60)
        def inner_func(_d, _h):
            pass

    return inner_func


class DoTask:
    def __init__(self):
        self.actual_tasks = {
            "chat": hook_callback_for_task(
                chat,
                "chat",
                lambda x: (x.choices[0].message.content, get_price_from_resp(x))
            ),
            "image-generation":
                hook_callback_for_task(
                    image_generation,
                    "image-generation",
                    lambda x: (x.data[0].url, x.usage)
                ),
            "image-recognition":
                hook_callback_for_task(
                    vision,
                    "image-recognition",
                    lambda x: (x.choices[0].message.content, get_price_from_resp(x))
                ),
            # dummy task
            "dummy":
                hook_callback_for_task(
                    lambda x, m: sleep(5),
                    "dummy",
                    lambda x: ("this is a dummy task", 0)
                )
        }
        self.rated_tasks = {
            # only for rate limit checking.
            "audio-generation":
                hook_callback_for_task(
                    tts,
                    "audio-generation",
                    lambda _: "OK", rate_control_only=True
                ),
            "audio-recognition":
                hook_callback_for_task(
                    transcribe_to_text,
                    "audio-recognition",
                    lambda _: "OK",
                    rate_control_only=True
                ),
        }

    def create(self, task_type, data, hook, model=None, options=None):
        if task_type not in self.actual_tasks:
            logging.error(f"Task type '{task_type}' not found")
            return False
        thread = Thread(target=self.actual_tasks[task_type], args=(data, hook, model, options))
        thread.start()
        return True

    def check_limit(self, task_type):
        try:
            self.rated_tasks[task_type](None, None)
        except RateLimitException:
            return False
        return True
