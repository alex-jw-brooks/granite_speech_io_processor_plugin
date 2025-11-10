import asyncio
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.entrypoints.chat_utils import ConversationMessage
from vllm.entrypoints.utils import get_max_tokens
from vllm.config import VllmConfig
from vllm.utils import Counter
from vllm.inputs.data import TokensPrompt as EngineTokensPrompt
from vllm.plugins.io_processors.interface import IOProcessorPluginType
from vllm.sampling_params import SamplingParams
from vllm.entrypoints.openai.serving_engine import AnyRequest
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from .utils import submit_audio_prompts_and_run_engine, TRANSCRIPTION_INPUT_LENGTH, TRANSCRIPTION_TOKENS
from .async_utils import run_async_generate


class GraniteSpeechProcessor(IOProcessor):
    plugin_type = IOProcessorPluginType.INPUT_ONLY

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.max_model_len = self.model_config.max_model_len
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        # We use an independent request counter for things
        # submitted by this plugin to avoid potentially collisions
        # with the outer loop.
        self.request_counter = Counter()

    def parse_request(self, request: ChatCompletionRequest, has_preprocess_partial=False):
        # Offline case for .generate() calls
        if not has_preprocess_partial:
            if not isinstance(request, (str, dict)):
                raise TypeError(f"Parse failed - expected string or dict, got {type(request)}")
            return request

        if isinstance(request, ChatCompletionRequest):
            # Currently we leave it as a request since the preprocess
            # partial is async, so it's cleaner to just call it from the
            # async preprocess implementation
            return request
        else:
            # TODO - really this should probably be a warning and forward for other types,
            # which would allow us to avoid unwanted functionality for other endpoints
            raise TypeError(f"Unsupported request type! {type(request)}")

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
            prompt["prompt_token_ids"] = TRANSCRIPTION_TOKENS
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

    async def pre_process_async(
        self,
        prompt,
        request_id,
        preprocess_partial,  ### Actually we may not even need this partial, since we can just create the inputs directly
        lora_request,
        engine_client,
        processor,
        **kwargs,
    ):
        # Run generation asynchronously (non-streaming) and pull out the transcription
        request = prompt
        result = await run_async_generate(
            request,
            preprocess_partial,
            self.request_counter,
            processor,
            engine_client,
            sampling_params=self.validate_or_generate_params(request),
            lora_request=lora_request,
        )

        # Instead of rebuilding a new request and reapplying the partial,
        # we just build the corresponding conversion, request_prompts, and
        # engine_prompts directly
        request_prompt = result.outputs[0].text
        conversation = [ConversationMessage(role="user", content=request_prompt)]
        engine_prompt = EngineTokensPrompt(prompt_token_ids=result.outputs[0].token_ids)

        return conversation, [request_prompt], [engine_prompt]

    def validate_or_generate_params(
        self,
        request: AnyRequest | None = None,
        params: SamplingParams | None = None,
    ) -> SamplingParams:
        # NOTE - passing the request type allows us to potentially
        # customize params by task.
        if params is not None:
            return params
        if not isinstance(request, ChatCompletionRequest):
            raise ValueError("Unable to generate params from anything but chat completions currently")
        return self._build_chat_completions_params(request)

    def _build_chat_completions_params(self, request):
        max_tokens = get_max_tokens(
            max_model_len=self.max_model_len,
            request=request,
            input_length=TRANSCRIPTION_INPUT_LENGTH, #FIXME - we should use the expanded toks here
            default_sampling_params=self.default_sampling_params,
        )

        sampling_params = request.to_sampling_params(
            max_tokens,
            self.model_config.logits_processor_pattern,
            self.default_sampling_params,
        )
        return sampling_params

    def get_modified_lora_request(self, engine_prompts, lora_request):
        # Based on where this is called in the lifecycle,
        # for now always drop the lora, because this is
        # after transcription, so we don't want to use
        # the audio anymore!
        # TODO - this should be done more carefully
        return None
