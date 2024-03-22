import torch
import queue
import threading
import time
from types import TracebackType
from typing import Any, List, Optional, Type

from .sequence import Sequence
from .llama_hf import LlamaConfig, LlamaForCausalLM
from ..logger import logger


class ModelHost:
    def __init__(
        self,
        model_obj,
        kv_caches,
        max_batch_size=32,
    ):
        self.model_obj: LlamaForCausalLM = model_obj
        self.kv_caches = kv_caches
        self.running_seqs = []
        self.max_batch_size = max_batch_size
        self.peak_batch_size = 0
        
        self.stopped = True
        self.cycle_time = 1.0
        
        # double queue size to avoid blocking
        self.batch_queue = queue.Queue(maxsize=2 * max_batch_size)
        self.thread = None

    def predict_async(self, prompt_tokens: List[int], max_tokens: int) -> Sequence:
        if self.stopped:
            raise RuntimeError(f"model host is stopped, can't predict anymore")
        seq = Sequence(prompt_tokens=prompt_tokens, max_tokens=max_tokens)
        self.batch_queue.put(seq)
        return seq
    
    def predict(self, prompt_tokens: List[int], max_tokens: int) -> List[int]:
        seq = self.predict_async(prompt_tokens, max_tokens)
        tokens = seq.get_result()
        return tokens

    def start(self):
        if not self.stopped:
            raise RuntimeError(f"model host is already started")

        self.stopped = False
        self.thread = threading.Thread(target=self.run_iterations)
        self.thread.start()

    def stop(self):
        self.stopped = True
        logger.debug(f"notify worker thread to stop")
        self.thread.join()
        logger.debug(f"worker thread is stopped")

    def get_new_requests(self) -> List[Sequence]:
        # logger.debug(f'enter get_new_requests')
        new_seqs = []
        # blocking until get at least one request
        if len(self.running_seqs) == 0:
            try:
                new_seqs.append(self.batch_queue.get(block=True, timeout=self.cycle_time))
                logger.info(f'received new seq {new_seqs[-1].id}')
            except queue.Empty:
                pass
        
        # get more already arrived requests
        while (
            len(new_seqs) + len(self.running_seqs) < self.max_batch_size and not self.batch_queue.empty()
        ):
            try:
                new_seqs.append(self.batch_queue.get(block=False))
                logger.info(f'received new seq {new_seqs[-1].id}')
            except queue.Empty:
                break

        return new_seqs

    @torch.inference_mode()
    def run_iterations(self):
        while not self.stopped:
            # logger.debug(f'enter run_iterations')
            # remove finished requests
            remove_seqs = []
            for idx, seq in enumerate(self.running_seqs):
                if len(seq) >= seq.max_tokens + len(seq.prompt_tokens):
                    remove_seqs.append(idx)
                    logger.info(f'finished the seq {seq.id}')
                    seq.set_result_ready()
            n_removed = 0
            for i in remove_seqs:
                i = i - n_removed
                self.running_seqs.pop(i)
                for kv_cache in self.kv_caches:
                    kv_cache.remove_seq(i)
                # logger.debug(f'remove seq {i} from cache')
                n_removed += 1
            
            # check new requests
            sequences: List[Sequence] = self.get_new_requests()
            # logger.debug(f'get {len(sequences)} new sequences')
            if len(sequences) != 0:
                new_tokens = self.model_obj(sequences, self.kv_caches, prefill=True)
                for seq, new_token in zip(sequences, new_tokens):
                    seq.append(new_token)
                for prefill_seq in sequences:
                    logger.debug(f'running seqs: {len(self.running_seqs)}, insert seq into index {prefill_seq.index}')
                    self.running_seqs.insert(prefill_seq.index, prefill_seq)
            
            # run single decoding iteration
            #logger.debug(f'run one decode iteration for {len(self.running_seqs)} seqs')
            current_batch_size = len(self.running_seqs)
            if current_batch_size > self.peak_batch_size:
                self.peak_batch_size = current_batch_size
            if len(self.running_seqs) != 0:               
                new_tokens = self.model_obj(self.running_seqs, self.kv_caches, prefill=False)
                #new_tokens = [42] * len(self.running_seqs)
                #time.sleep(1)
                for seq, new_token in zip(self.running_seqs, new_tokens):
                    seq.append(new_token)

    def __enter__(self):
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.stop()