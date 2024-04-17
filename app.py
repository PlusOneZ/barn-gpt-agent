from flask import Flask
from time import sleep
from flask import request
import logging

from threading import Thread

import gpt_tasks.task as gt
from call_hook.send_results import call_hook_with_result


logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!\n'


# /api/dumb/task
# request body: {json: {data: data, hook: hook}
@app.route('/api/task/dummy', methods=['POST'])
def api():
    # get data and hook from request body
    data = request.json['data']
    hook = request.json['hook']
    print(data, hook)

    # dummy tasks that takes 5 seconds to complete
    cb = gt.hook_callback_for_task(
        lambda x: sleep(5),
        "dummy",
        data,
        hook,
        lambda x: "this is a dummy task"
    )
    # start the async task
    thread = Thread(target=cb)
    thread.start()
    return "OK"


# /api/chat
# request body : {json: {data: {content: "some content" * required}, hook: hook}
@app.route('/api/task/chat', methods=['POST'])
def chatting_api():
    data = request.json['data']
    hook = request.json['hook']

    cb = gt.hook_callback_for_task(
        gt.chat,
        "chat",
        data, hook,
        lambda x: x.choices[0].message.content
    )
    thread = Thread(target=cb)
    thread.start()
    return "OK"


# /api/task/image/generation
# request body : {json: {data: {image_prompt: * required}, hook: hook}
@app.route('/api/task/image/generation', methods=['POST'])
def image_generation_api():
    data = request.json['data']
    hook = request.json['hook']

    cb = gt.hook_callback_for_task(
        gt.image_generation,
        "image-generation",
        data["image_prompt"],
        hook,
        lambda x: x.data[0].url
    )
    thread = Thread(target=cb)
    thread.start()
    return "OK"


@app.route('/api/task/vision', methods=['POST'])
def vision_api():
    data = request.json['data']
    hook = request.json['hook']

    cb = gt.hook_callback_for_task(
        gt.vision,
        "image-recognition",
        data,
        hook,
        lambda x: x.choices[0].message.content
    )
    thread = Thread(target=cb)
    thread.start()
    return "OK"


if __name__ == '__main__':
    app.run(port=5000, debug=True)
