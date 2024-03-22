from typing import List, Optional, Union
import queue, threading
import time

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter

class Request:
    request_counter = Counter()
    def __init__(self, prompt_token_ids: Optional[List[int]], sampling_params: SamplingParams) -> None:
        self.request_id = str(next(Request.request_counter))
        self.prompt_token_ids = prompt_token_ids
        self.sampling_params = sampling_params
        self.output = None
        self.finished = threading.Event()

class MyLLMEngine:
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    NOTE: This class is intended to be used for offline inference. For online
    serving, use the `AsyncLLMEngine` class instead.
    NOTE: For the comprehensive list of arguments, see `EngineArgs`.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq", "gptq" and "squeezellm". If None, we first check
            the `quantization_config` attribute in the model config file. If
            that is None, we assume the model weights are not quantized and use
            `dtype` to determine the data type of the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Otherwise, too small values may cause out-of-memory (OOM) errors.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode.
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: int = 8192,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            **kwargs,
        )
        self.llm_engine = LLMEngine.from_engine_args(engine_args)
        self.queue = queue.Queue()
        self.request_map = {}
        self.stopped = True
        self.thread = None

    def generate(
        self,
        prompt_token_ids: Optional[List[List[int]]] = None,
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[RequestOutput]:
        # Add requests to the engine.
        req = Request(prompt_token_ids, sampling_params)
        self.request_map[req.request_id] = req
        self.queue.put(req)
        req.finished.wait()
        return req.output

    def _run_engine(self):                    
        while not self.stopped:
            while not self.queue.empty():
                req: Request = self.queue.get()
                self.llm_engine.add_request(req.request_id, None, req.sampling_params, req.prompt_token_ids)
  
            if self.llm_engine.has_unfinished_requests():
                step_outputs: List[RequestOutput] = self.llm_engine.step()
                for output in step_outputs:
                    if output.finished:
                        req = self.request_map.pop(output.request_id)
                        req.output = output
                        self.llm_engine.abort_request(output.request_id)
                        req.finished.set()      
            else:
                try:
                    req: Request = self.queue.get(timeout=1)
                    self.llm_engine.add_request(req.request_id, None, req.sampling_params, req.prompt_token_ids)
                except queue.Empty:
                    pass

    def start(self):
        if not self.stopped:
            raise RuntimeError(f"model host is already started")

        self.stopped = False
        self.thread = threading.Thread(target=self._run_engine)
        self.thread.start()
        
    def stop(self):
        self.stopped = True
        self.thread.join()
