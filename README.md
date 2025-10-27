# Granite Speech Plugin for vLLM

### Background
Granite speech `3.x` models take a two-stage pass to inference; to process an audio file and respond to its content, e.g., an audio clip stating `what is 2 + 2?`, we first transcribe the audio, and then process the transcription result with the underlying LLM.

In order to treat inference the way you may treat other audio models, which understand and respond to the audio directly in one generate call, this generally means that you need to call it in vLLM twice, which can be inconvenient. This plugin solves this problems by essentially doing the following:

- If the request has audio submits the multimodal request to the vLLM engine and processes the request, then returns the text back to vLLM, which will then process it as a text request (i.e., calling the engine twice in one request)*
- If the request has no audio, the plugin doesn't do anything, and request is processed normally

### Installation and Usage
To install this plugin, clone it and install with `pip install .`; you can find an example for how to use it in `examples/run_speech_plugin.py`, which ingests an audio clip speaking `what is 2 + 2?` and outputs the answer `4` in a single generate call.


* This is still on a feature branch and not supported in vLLM yet, since IO plugins are currently only used in pooling models. The corresponding branch in my fork can be found here: https://github.com/alex-jw-brooks/vllm/tree/generate_io_plugins
