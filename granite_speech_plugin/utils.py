"""
Helpers for validating requests, adding to them to the engine,
and running the engine.

NOTE: The helpers below are largely derived from the LLM class in
vLLM, but for now copied here to avoid adding new interfaces to LLM.
The implementation here is used to back the synchronous implementatio
of the plugin; for the analogous async implementation, see async_utils.
"""
from typing import Sequence
from vllm.inputs.parse import get_prompt_components
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.logger import init_logger

# TODO - would be better to dynamically fetch this from the model
TRANSCRIPTION_PROMPT = "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\nToday's Date: October 27, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|><|audio|>can you transcribe the speech into a written format?<|end_of_text|>\n"

# FIXME - This is a hack, and also, the token count does not have the audio tokens expanded.
# It would be better
def get_transcription_tokens():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("ibm-granite/granite-speech-3.3-2b")
    return tok.encode(TRANSCRIPTION_PROMPT)
TRANSCRIPTION_TOKENS = get_transcription_tokens()
TRANSCRIPTION_INPUT_LENGTH = len(TRANSCRIPTION_TOKENS)

logger = init_logger(__name__)

def submit_audio_prompts_and_run_engine(prompt, params, lora_request, engine, processor, request_counter):
    """Given an engine instance, prompt(s), params, and lora request(s),
    validate and submit the cleaned prompts into the engine.
    
    NOTE: Things submitted to this are assumed to be multimodal with audio
    so that we can transcribe them directly.
    """
    prompts = validate_requests(prompt, params, lora_request)
    add_requests_to_engine(engine, processor, request_counter, prompts, params, lora_request)
    return run_llm_engine(engine)

def validate_requests(prompts, params, lora_request):
    if isinstance(prompts, (str, dict)):
        # Convert a single prompt to a list.
        prompts = [prompts]  # type: ignore[list-item]

    num_requests = len(prompts)
    if isinstance(params, Sequence) and len(params) != num_requests:
        raise ValueError("The lengths of prompts and params must be the same.")
    if isinstance(lora_request, Sequence) and len(lora_request) != num_requests:
        raise ValueError(
            "The lengths of prompts and lora_request must be the same."
        )

    return prompts


def add_requests_to_engine(engine, processor, request_counter, prompts, params, lora_request):
    # Used for all transcription requests, since the prompt is basically const
    tokenization_kwargs = {}
    priority = 0
    for sp in params if isinstance(params, Sequence) else (params,):
        if isinstance(sp, SamplingParams):
            # We only care about the final output
            sp.output_kind = RequestOutputKind.FINAL_ONLY

    # Add requests to the engine.
    for i, prompt in enumerate(prompts):
        request_id = f"transcription-{next(request_counter)}"
        prompt_params = params[i] if isinstance(params, Sequence) else params
        prompt_lora = lora_request=lora_request[i] if isinstance(lora_request, Sequence) else lora_request
        prompt_text, _, _ = get_prompt_components(prompt)

        engine_request = processor.process_inputs(
            request_id,
            prompt,
            params,
            lora_request=prompt_lora,
            tokenization_kwargs=tokenization_kwargs,
            priority=priority,
        )

        _log_engine_request(engine_request)

        engine.add_request(
            request_id,
            engine_request,
            prompt_params,
            lora_request=prompt_lora,
            tokenization_kwargs=tokenization_kwargs,
            priority=priority,
            prompt_text=prompt_text,
        )




def _log_engine_request(engine_request):
    logger.info(
        "****************************************************\n"
        "Submitting engine request: \n"
        f"\tRequest ID: {engine_request.request_id}\n"
        f"\tPrompt Token IDs: {engine_request.prompt_token_ids}\n"
        f"\tSampling Params: {engine_request.sampling_params}\n"
        # f"\tMultimodal Features: {engine_request.mm_features}\n" # useful, but hard to look at
        f"\tLoRA Request: {engine_request.lora_request}\n"
        "****************************************************\n"
    )
    

def run_llm_engine(engine):
    """Runs the LLM engine until all unfinished requests are finished."""
    outputs = []
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
    return sorted(
        outputs,
        key=lambda x: int(x.request_id.split("-")[-1]),
    )
