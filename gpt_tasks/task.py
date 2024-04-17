from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import logging

from call_hook.send_results import call_hook_with_result

load_dotenv()

client = OpenAI()


# Calling to OpenAI chat completion API with
# Params:
#   - user_input: str
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


def hook_callback_for_task(task, task_type, data, hook, get_result):
    def inner_func():
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

