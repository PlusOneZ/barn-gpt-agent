import logging
from random import random
from threading import Thread
from time import sleep

from backoff import on_exception, runtime
from openai import OpenAIError
from gradio_client.exceptions import AppError
from ratelimit import RateLimitException, limits

from call_hook.send_results import call_hook_with_result
from gpt_tasks.detailed_tasks import chat, image_generation, vision, tts, transcribe_to_text
from gpt_tasks.utils import get_price_from_resp


def hook_callback_for_task(task, task_type, get_result, rate_control_only=False):
    """
    根据任务类型设置调用频率，并设置调用频率限制

    参数:
    task (function): 要执行的任务函数
    task_type (str): 任务类型，用于确定调用频率限制
    get_result (function): 从API响应中提取结果的函数
    rate_control_only (bool): 是否仅进行速率控制，默认为False

    返回:
    function: 包装了速率限制和错误处理的内部函数

    说明:
    - 根据不同的任务类型设置不同的调用频率限制
    - 使用装饰器实现速率限制和错误处理
    - 如果超出速率限制，会调用错误回调函数
    - 如果不是仅进行速率控制，还会执行实际的任务并处理结果
    """
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
        """
        当速率限制超出时，调用错误回调函数
        """
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
            '''
            执行实际的任务并处理结果
            '''
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
            except AppError as e:
                logging.error(f"{task_type}: Gradio错误!")
                logging.error(f"数据: {data}")
                logging.error(f"错误信息: {str(e)}")
                call_hook_with_result(
                    hook, [{
                        "type": "3rd_party_error",
                        "content": "HF app error, you may encounter a rate limit control. No charge for this call.",
                        "err_body": f"HF app error, you may encounter a rate limit control: {e.__class__.__name__}.",
                        "target_task": task_type,
                    }],
                    status="failed"
                )
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
            '''
            仅进行速率控制
            '''
            pass

    return inner_func


class DoTask:
    """
    根据任务类型设置调用频率，并设置调用频率限制
    发放实际任务的类
    有6个任务，分别是：
    1. chat
    2. image-generation
    3. image-recognition
    4. audio-generation
    5. audio-recognition
    6. dummy
    详情任务在detailed_tasks.py中

    TODO: 为不同的模型设置不同的调用频率限制
    """
    def __init__(self):
        """
        初始化实际任务和限速任务
        """
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
        """
        创建一个线程来执行实际任务
        """
        if task_type not in self.actual_tasks:
            logging.error(f"Task type '{task_type}' not found")
            return False
        thread = Thread(target=self.actual_tasks[task_type], args=(data, hook, model, options))
        thread.start()
        return True

    def check_limit(self, task_type):
        """
        检查任务类型是否超出速率限制
        """
        try:
            self.rated_tasks[task_type](None, None)
        except RateLimitException:
            return False
        return True
