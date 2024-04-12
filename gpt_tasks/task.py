from openai import OpenAI
from dotenv import load_dotenv
import logging

load_dotenv()

client = OpenAI()


# Calling to OpenAI chat completion API with
# Params:
#   - user_input: str
def chat(user_input: []):
    logging.info("Chat completion called with user input: '{}'".format(user_input))
    # check user_input style
    user_input = [
        {"role": "user", "content": i} if isinstance(i, str)
        else {"role": i["role"], "content": i["content"]}
        for i in user_input
    ]
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=user_input
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
