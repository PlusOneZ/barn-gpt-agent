from flask import Flask
from flask import request

import logging

from gpt_tasks import DoTask


logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

task_manager = DoTask()


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!\n'


# /api/dumb/task
# request body: {json: {data: data, hook: hook}
@app.route('/api/task/dummy', methods=['POST'])
def dummy_api():
    # get data and hook from request body
    hook = request.json['hook']

    # dummy tasks that takes 5 seconds to complete
    ret = task_manager.create("dummy", "dummy", hook)
    return "OK" if ret else ("Failed", 400)


# /api/chat
# request body : {json: {data: {content: "some content" * required}, hook: hook}
@app.route('/api/task/chat', methods=['POST'])
def chatting_api():
    data = request.json['data']
    hook = request.json['hook']

    ret = task_manager.create("chat", data, hook)
    return "OK" if ret else ("Failed", 400)


# /api/task/image/generation
# request body : {json: {data: {image_prompt: * required}, hook: hook}
@app.route('/api/task/image/generation', methods=['POST'])
def image_generation_api():
    data = request.json['data']
    hook = request.json['hook']

    ret = task_manager.create("image-generation", data["image_prompt"], hook)
    return "OK" if ret else ("Failed", 400)


@app.route('/api/task/audio/generation', methods=['POST'])
def tts_api():
    if task_manager.check_limit("audio-generation"):
        return "OK"
    else:
        return "Rate limit exceeded", 429


@app.route('/api/task/audio/recognition', methods=['POST'])
def transcribe_to_text_api():
    if task_manager.check_limit("audio-recognition"):
        return "OK"
    else:
        return "Rate limit exceeded", 429


@app.route('/api/task/vision', methods=['POST'])
def vision_api():
    data = request.json['data']
    hook = request.json['hook']

    ret = task_manager.create("image-recognition", data, hook)
    return "OK" if ret else ("Failed", 400)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
