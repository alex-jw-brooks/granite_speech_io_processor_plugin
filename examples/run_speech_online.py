"""
Example for running the pass simplification plugin in online mode.

NOTE: You need to first start the server with the plugin active;
you can do this as shown below.

vllm serve ibm-granite/granite-speech-3.3-2b \
    --api-key token-abc123 \
    --max-model-len 2048 \
    --enable-lora  \
    --default-mm-loras '{"audio":"ibm-granite/granite-speech-3.3-2b"}' \
    --max-lora-rank 64 \
    --io-processor-plugin granite_speech_pass_simplification \
    --enforce-eager
"""
from io import BytesIO
import base64
import os

import librosa
from openai import OpenAI
import soundfile


# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "token-abc123"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

base_model_name = "ibm-granite/granite-speech-3.3-2b"


# Audio clip for "What is 2 + 2?"
SAMPLE_FILE = os.path.join(os.path.dirname(__file__), "math.mp3")
audio, sr = librosa.load(SAMPLE_FILE)


with BytesIO() as buffer:
    soundfile.write(buffer, audio, sr, format="WAV")
    data = buffer.getvalue()
    audio_base64 = base64.b64encode(data).decode("utf-8")


### 1. Example with Audio
# NOTE: We do not actually need to pass a text prompt here,
# because the plugin will inject one for transcription.

chat_completion_with_audio = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        # Any format supported by librosa is supported
                        "url": f"data:audio/ogg;base64,{audio_base64}"
                    },
                },
            ],
        }
    ],
    temperature=0.2,
    max_tokens=64,
    model=base_model_name,
)


print(f"Generated text: {chat_completion_with_audio.choices[0].message.content}")
