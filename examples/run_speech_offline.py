"""
Example for running the pass simplification plugin in offline mode.
"""
import os
import librosa
from vllm import LLM, SamplingParams

# Audio clip for "What is 2 + 2?"
SAMPLE_FILE = os.path.join(os.path.dirname(__file__), "math.mp3")
audio = librosa.load(SAMPLE_FILE)

model_id = "ibm-granite/granite-speech-3.3-2b"

model = LLM(
    model=model_id,
    trust_remote_code=True,
    enforce_eager=True,
    enable_lora=True,
    max_lora_rank=64,
    max_model_len=2048,  # This may be needed for lower resource devices.
    limit_mm_per_prompt={"audio": 1},
    default_mm_loras={
        "audio": model_id
    },  # Always use the colocated lora if we have audio
    io_processor_plugin="granite_speech_pass_simplification",  # Use the plugin!
)

inputs = {
    # NOTE - `prompt` is not needed with this plugin, because we inject a transcription
    # prompt in if audio is present. Any existing prompt will be ignored if we have audio
    # at the moment!
    "multi_modal_data": {
        "audio": audio,
    }
}

outputs = model.generate(
    inputs,
    sampling_params=SamplingParams(
        temperature=0.2,
        max_tokens=64,
    ),
)
print(f"Generated text: {outputs[0].outputs[0].text}")
