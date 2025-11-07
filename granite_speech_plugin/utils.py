"""
Helpers for validating requests, adding to them to the engine,
and running the engine.

NOTE: The helpers below are largely derived from the LLM class in
vLLM, but for now copied here to avoid adding new interfaces to LLM.
"""
from typing import Sequence
from vllm.inputs.parse import get_prompt_components
from vllm.sampling_params import RequestOutputKind, SamplingParams

def submit_audio_prompts_and_run_engine(prompt, params, lora_request, engine, processor, request_counter):
    """Given an engine instance, prompt(s), params, and lora request(s),
    validate and submit the cleaned prompts into the engine.
    
    NOTE: Things submitted to this are assumed to be multimodal with audio
    so that we can transcribe them directly.
    """
    prompts = validate_requests(prompt, params, lora_request)
    add_requests_to_engine(engine, processor, request_counter, prompts, params, lora_request)
    return run_llm_engine(engine, processor)

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
    for sp in params if isinstance(params, Sequence) else (params,):
        if isinstance(sp, SamplingParams):
            # We only care about the final output
            sp.output_kind = RequestOutputKind.FINAL_ONLY

    # Add requests to the engine.
    for i, prompt in enumerate(prompts):
        add_requests_to_engine(
            engine,
            processor,
            request_counter,
            prompt,
            params[i] if isinstance(params, Sequence) else params,
            lora_request=lora_request[i]
            if isinstance(lora_request, Sequence)
            else lora_request,
            priority=0,
        )

def add_request_to_engine(engine, processor, request_counter, prompt, params, lora_request):
    """Add a request to the engine.
    
    NOTE: Processor is invoked in the engine also, but we use
    the explicitly passed object, as calling processor through
    the engine is deprecated & will be removed in the future.
    """
    prompt_text, _, _ = get_prompt_components(prompt)
    request_id = f"transcription-{next(request_counter)}"

    # Submit all intermediate transcription requests as priority 0
    # We also skip truncation kwarg validation here, because for this
    # plugin, the prompt is hard-coded for transcription, so it will
    # be const length.
    tokenization_kwargs = {}
    engine_request = processor.process_inputs(
        request_id,
        prompt,
        params,
        lora_request=lora_request,
        tokenization_kwargs=tokenization_kwargs,
        priority=0,
    )

    engine.add_request(
        request_id,
        engine_request,
        params,
        lora_request=lora_request,
        tokenization_kwargs=tokenization_kwargs,
        priority=0,
        prompt_text=prompt_text,
    )

def run_llm_engine(engine):
    """Runs the LLM engine until all unfinished requests are finished."""
    outputs = []
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
    return sorted(outputs, key=lambda x: int(x.request_id))