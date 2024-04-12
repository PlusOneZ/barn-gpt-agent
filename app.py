import asyncio

from flask import Flask
from time import sleep
from flask import request
import logging
from openai import OpenAIError

from threading import Thread

import gpt_tasks.task as gt
from call_hook.send_results import call_hook_with_result


logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
loop = asyncio.get_event_loop()


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!\n'


# /api/dumb/task
# request body: {json: {data: data, hook: hook}
@app.route('/api/dummy/task', methods=['POST'])
def api():
    # get data and hook from request body
    data = request.json['data']
    hook = request.json['hook']
    print(data, hook)

    def some_async_task(api_hook):
        sleep(5)
        res = call_hook_with_result(api_hook, ["result from flask async task"])
        return res

    # start the async task
    thread = Thread(target=some_async_task, kwargs={"api_hook": hook})
    thread.start()
    return "OK"


# /api/chat
# request body : {json: {data: {content: "some content" * required}, hook: hook}
@app.route('/api/task/chat', methods=['POST'])
def chatting_api():
    data = request.json['data']
    hook = request.json['hook']

    # defined for asynchronous task
    def chat_and_call_hook(user_input, api_hook):
        try:
            ai_resp = gt.chat(user_input)  # results from calling GPT chat
            logging.debug(ai_resp)
            result = ai_resp.choices[0].message.content
            # todo save the result some where trackable
            call_hook_with_result(api_hook, [result])
        except OpenAIError as e:
            logging.error(f"chatting_api: task [{user_input}] with hook '{hook}' not finished!")
            logging.error(f"With error: {e}")

    thread = Thread(target=chat_and_call_hook, kwargs={'user_input': data["content"]["prompts"], "api_hook": hook})
    thread.start()
    return "OK"


# /api/task/image/generation
# request body : {json: {data: {image_prompt: * required}, hook: hook}
@app.route('/api/task/image/generation', methods=['POST'])
def image_generation_api():
    data = request.json['data']
    hook = request.json['hook']

    def generate_image_and_call_hook(prompt, api_hook):
        try:
            ai_resp = gt.image_generation(prompt)  # results from calling GPT chat
            logging.debug(ai_resp)
            result = ai_resp.data[0].url
            # todo save the result some where trackable
            call_hook_with_result(api_hook, [{"type": "image_generation", "url": result}])
        except OpenAIError as e:
            logging.error(f"image_generation_api: task [{prompt}] with hook '{hook}' not finished!")
            logging.error(f"With error: {e}")

    thread = Thread(target=generate_image_and_call_hook, kwargs={'prompt': data["image_prompt"], "api_hook": hook})
    thread.start()
    return "OK"


if __name__ == '__main__':
    app.run(port=5000, debug=True)
