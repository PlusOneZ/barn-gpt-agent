from backoff import on_exception, constant
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import logging
from time import sleep
from threading import Thread

from ratelimit import limits, RateLimitException

from call_hook.send_results import call_hook_with_result

load_dotenv()

client = OpenAI()


# Calling to OpenAI chat completion API with
# Params:
#   - user_input: list of strings or dict with role and content
def chat(user_input: []):
    logging.info("Chat completion called with user input: '{}'".format(user_input))
    # check user_input style
    send_input = [{"role": "system", "content": "You're a helpful assistant."}]
    send_input.extend([
        {"role": "user", "content": i} if isinstance(i, str)
        else {"role": i["role"], "content": i["content"]}
        for i in user_input
    ])
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=send_input
    )
    return completion


def image_generation(user_prompt: str):
    logging.info(f"Image Generation called with prompt: {user_prompt}")
    response = client.images.generate(
        model='dall-e-2',
        prompt=user_prompt,
        size='1024x1024',
        quality='standard',
        n=1
    )
    return response


def vision(user_input: []):
    logging.info("Vision chat called with user input: '{}'".format(user_input))
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=user_input,
        max_tokens=200,
    )
    return completion


def tts(input_text: str):
    logging.info(f"Text to Speech called with text: {input_text}")
    response = client.audio.speech.create(
        model='tts-1',
        voice='alloy',
        input=input_text
    )
    return response


def transcribe_to_text(audio_file):
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    return transcript


def hook_callback_for_task(task, task_type, get_result):
    match task_type:
        case "chat":
            call_per_second = 50
        case "image-generation":
            call_per_second = 5
        case "image-recognition":
            call_per_second = 20
        case "audio-generation":
            call_per_second = 2
        case "audio-recognition":
            call_per_second = 2
        case _:
            call_per_second = 1  # for testing

    def rate_limit_exceeded(params):
        logging.warning(f"Rate limit exceeded for {task_type}")
        call_hook_with_result(params['args'][1], [{"type": "error", "content": "Rate limit exceeded"}], status="rejected")

    @on_exception(constant, RateLimitException, max_tries=3, on_giveup=rate_limit_exceeded, interval=10)
    @limits(calls=call_per_second, period=1)
    def inner_func(data, hook):
        try:
            api_resp = task(data)
            logging.debug(api_resp.json() if api_resp else "dummy response")
            result = get_result(api_resp)
            call_hook_with_result(hook, [{"type": task_type, "content": result}], api_response=api_resp)
        except OpenAIError as e:
            logging.error(f"{task_type}: task not finished!")
            logging.error(f"Data: {data}")
            logging.error(f"Hook: {hook}")
            logging.error(f"With error: {e}")
            call_hook_with_result(hook, [{"type": task_type, "content": str(e)}], status="failed")

    return inner_func


class DoTask:
    def __init__(self):
        self.tasks = {
            "chat": hook_callback_for_task(chat, "chat", lambda x: x.choices[0].message.content),
            "image-generation": hook_callback_for_task(image_generation, "image-generation", lambda x: x.data[0].url),
            "image-recognition": hook_callback_for_task(vision, "image-recognition", lambda x: x.choices[0].message.content),
            # "audio-generation": hook_callback_for_task(tts, "audio-generation", lambda x: x.choices[0].message.content),
            # "audio-recognition": hook_callback_for_task(transcribe_to_text, "audio-recognition", lambda x: x.choices[0].message.content),
            "dummy": hook_callback_for_task(lambda x: sleep(5), "dummy", lambda x: "this is a dummy task")
        }

    def create(self, task_type, data, hook):
        if task_type not in self.tasks:
            logging.error(f"Task type '{task_type}' not found")
            return False
        thread = Thread(target=self.tasks[task_type], args=(data, hook))
        thread.start()
        return True

