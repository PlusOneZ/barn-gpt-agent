from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    client = OpenAI()
    # defaults to getting the key using os.environ.get("OPENAI_API_KEY")
    # if you saved the key under a different environment variable name, you can do something like:
    # client = OpenAI(
    #   api_key=os.environ.get("CUSTOM_ENV_NAME"),
    # )
    mode = "tts"
    if mode == "chat":
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content":
                     "This is a test, just say 'Hello'"
                 },
                {"role": "user",
                 "content":
                     "Hello!"
                 }
            ]
        )
        print(completion.choices[0].message)

    elif mode == "vision":
        from openai import OpenAI

        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What’s in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "",
                            },
                        },
                    ],
                }
            ],
            max_tokens=200,
        )

        print(response.choices[0])

    elif mode == 'tts':
        from pathlib import Path
        import openai

        speech_file_path = Path(__file__).parent / "speech.mp3"
        response = openai.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input="Read this aloud for me please!"
        )
        response.stream_to_file(speech_file_path)

