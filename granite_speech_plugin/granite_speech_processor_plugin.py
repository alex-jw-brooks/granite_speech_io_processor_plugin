
from vllm.plugins.io_processors.interface import IOProcessor

from vllm.config import VllmConfig

TRANSCRIPTION_PROMPT = "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\nToday's Date: October 27, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|><|audio|>can you transcribe the speech into a written format?<|end_of_text|>\n"

class GraniteSpeechProcessor(IOProcessor):
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config

    def parse_request(self, request):
        return request

    def pre_process(self, prompt, *, llm_instance, params, lora_request, priority):
        """Checks if the request has audio, and if it does, runs it in the engine
        and sets the output as the input prompt."""
        if isinstance(prompt, dict) and "multi_modal_data" in prompt:
            # If the request contains audio, we call generate
            # once ahead of time with a prompt for transcription
            mm_data = prompt["multi_modal_data"]
            prompt["prompt"] = TRANSCRIPTION_PROMPT
            if "audio" in mm_data:
                print("Prompt contains audio; it will be processed and resubmitted")
                # add the request to the engine
                llm_instance._validate_and_add_requests(
                    prompt,
                    params=params,
                    lora_request=lora_request,
                    priority=priority,
                    use_tqdm=False,
                )
            outputs = llm_instance._run_engine(use_tqdm=False)
            prompt = outputs[0].outputs[0].text
            print(f"Intermediate output: {prompt}")

        return prompt

    # TODO: need to integrate post processing for generate still, this may be
    # a tiny bit different than pooling due to the output interfaces
    def output_to_response(self, plugin_output):
        raise NotImplementedError("Output to response not integrated to generate")

    def post_process(self, model_output, request_id = None, **kwargs):
        raise NotImplementedError("Post process not integrated to generate")
