"""
Helpers for validating requests, adding to them to the engine,
and running the engine.

NOTE: The helpers below are largely derived from the base OpenAIServing
interface in vLLM, as well as some of its subclasses (notably the chat
completions one).
"""
from vllm.inputs.parse import get_prompt_components, PromptComponents

from .utils import _log_engine_request, TRANSCRIPTION_PROMPT, TRANSCRIPTION_TOKENS


def _get_prompt_components(prompt):
    if isinstance(prompt, list):
        return PromptComponents(token_ids=prompt)
    return get_prompt_components(prompt)  # type: ignore[arg-type]


async def run_async_generate(
    request,
    preprocess_partial,
    request_counter,
    processor,
    engine_client,
    sampling_params,
    lora_request,
):
    # Schedule the request and get the result generator.
    request_id = f"transcription-{next(request_counter)}"
    priority = 0
    tokenization_kwargs = {}

    (
        _,
        _,
        engine_prompts,
    ) = await preprocess_partial(request)
    # HACK - we can *probably* be more efficient here and remove the call to
    # the partial since the input is fixed, but for now use it preprocess the
    # multimodal part...

    if len(engine_prompts) > 1:
        raise ValueError(
            "We expected to have one engine prompt but have {}!".format(
                len(engine_prompts)
            )
        )

    request_prompt = TRANSCRIPTION_PROMPT
    engine_prompt = engine_prompts[0]
    engine_prompt["prompt_token_ids"] = TRANSCRIPTION_TOKENS

    # TODO - this is guaranteed to be len 1, so we can remove the loop here and simplify
    # TODO - check if we really need this wrapper here
    prompt_text, _, _ = _get_prompt_components(request_prompt)

    engine_request = processor.process_inputs(
        request_id,
        engine_prompt,
        sampling_params,
        lora_request=lora_request,
        tokenization_kwargs=tokenization_kwargs,
        priority=priority,
    )

    _log_engine_request(engine_request)

    # NOTE: the engine client handles the io processor piece here
    result_generator = engine_client.generate(
        engine_request,
        sampling_params,
        request_id,
        lora_request=lora_request,
        priority=priority,
        prompt_text=prompt_text,
        tokenization_kwargs=tokenization_kwargs,
    )

    # yield the result from the async generator
    async for res in result_generator:
        return res
