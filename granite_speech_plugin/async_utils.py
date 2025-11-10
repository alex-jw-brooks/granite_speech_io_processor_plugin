"""
Helpers for validating requests, adding to them to the engine,
and running the engine.

NOTE: The helpers below are largely derived from the base OpenAIServing
interface in vLLM, as well as some of its subclasses (notably the chat
completions one).
"""
from collections.abc import AsyncGenerator, AsyncIterator
from vllm.inputs.parse import get_prompt_components, PromptComponents
from vllm.entrypoints.utils import get_max_tokens
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.outputs import RequestOutput

from .utils import _log_engine_request, TRANSCRIPTION_PROMPT, TRANSCRIPTION_TOKENS
def _get_prompt_components(prompt):
    if isinstance(prompt, list):
        return PromptComponents(token_ids=prompt)
    return get_prompt_components(prompt)  # type: ignore[arg-type]


async def run_async_generate(request, preprocess_partial, request_counter, processor, engine_client, sampling_params, lora_request):
    # Schedule the request and get the result generator.
    request_id = f"transcription-{next(request_counter)}"
    priority = 0
    tokenization_kwargs = {}
    generators: list[AsyncGenerator[RequestOutput, None]] = []

    (
        _,
        request_prompts,
        engine_prompts,
    ) = await preprocess_partial(request)
    # HACK - ^ Would be better to skip the preprocess patial
    # entirely, but we do need to build the engine prompt.
    request_prompts = [TRANSCRIPTION_PROMPT]
    engine_prompts[0]["prompt_token_ids"] = TRANSCRIPTION_TOKENS

    # NOTE - this is guaranteed to be len 1, so we can remove the loop here and simplify
    for i, engine_prompt in enumerate(engine_prompts):
        # TODO - do we even need this wrapper in this case?
        prompt_text, _, _ = _get_prompt_components(request_prompts[i])

        # TODO - this plugin should handle validation and creating the sampling params,
        # and it should also blow up if beam search is used, since we don't allow it yet
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
        generator = engine_client.generate(
            engine_request,
            sampling_params,
            request_id,
            lora_request=lora_request,
            priority=priority,
            prompt_text=prompt_text,
            tokenization_kwargs=tokenization_kwargs,
        )
        generators.append(generator)

    assert len(generators) == 1
    (result_generator,) = generators

    async for res in result_generator:
        return res
