
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.config import VllmConfig
from vllm.utils import Counter

from .utils import submit_audio_prompts_and_run_engine

TRANSCRIPTION_PROMPT = "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\nToday's Date: October 27, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|><|audio|>can you transcribe the speech into a written format?<|end_of_text|>\n"

class GraniteSpeechProcessor(IOProcessor):
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        # We use an independent request counter for things
        # submitted by this plugin to avoid potentially collisions
        # with the outer loop.
        self.request_counter = Counter()

    def parse_request(self, request):
        return request

    def pre_process(self, prompt, *, params, lora_request, llm_engine, processor):
        """Checks if the request has audio, and if it does, runs it in the engine
        and sets the output as the input prompt.
        
        TODO: This currently is not handling batching correctly, but with this
        approach, we will also be able to handle mixed audio + text batches
        correctly.
        """
        if isinstance(prompt, dict) and "multi_modal_data" in prompt:
            # If the request contains audio, we call generate
            # once ahead of time with a prompt for transcription
            mm_data = prompt["multi_modal_data"]
            prompt["prompt"] = TRANSCRIPTION_PROMPT
            if "audio" in mm_data:
                print("Prompt contains audio; it will be processed and resubmitted")
                # Add the request to the LLM engine and run it
                outputs = submit_audio_prompts_and_run_engine(
                    prompt,
                    params,
                    lora_request,
                    llm_engine,
                    processor,
                    self.request_counter,
                )

                prompt = outputs[0].outputs[0].text
                print(f"Intermediate output: {prompt}")

        return prompt

    # TODO: need to integrate post processing for generate still, this may be
    # a tiny bit different than pooling due to the output interfaces
    def output_to_response(self, plugin_output):
        raise NotImplementedError("Output to response not integrated to generate")

    def post_process(self, model_output, request_id = None, **kwargs):
        return model_output

    async def pre_process_async(
        self,
        prompt,
        request_id = None,
        **kwargs,
    ):
        raise NotImplementedError("Async preprocess is not implemented for this plugin")