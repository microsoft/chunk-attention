import torch
import time
import random
import uuid
import asyncio
import threading, queue
from typing import List, Optional, Union
import my_llm_engine
# https://github.com/vllm-project/vllm/blob/665c48963be11b2e5cb7209cd25f884129e5c284/tests/kernels/test_attention.py
from vllm._C import ops
from vllm import LLM, SamplingParams
from vllm.engine.llm_engine import LLMEngine
from model_perf.server import ServerModelRunner
import multiprocessing


n_heads, d_head = 32, 128

def gen_dataset_tokens_only(n_seqs, seq_len, n_shared):
    shared_tokens = [42] * n_shared
    res = []
    for _ in range(n_seqs):
        tokens = [i for i in shared_tokens] + [random.randint(10, 100) for _ in range(seq_len - n_shared)]
        res.append(tokens)
    return res

def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


def ref_single_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    num_queries_per_kv: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    alibi_slopes,
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    n_seqs = query.shape[0]

    block_tables = block_tables.cpu().tolist()
    context_lens = context_lens.cpu().tolist()
    for i in range(n_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_kv_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(context_len, device="cuda").int()
            alibi_bias = (position_ids - context_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(
                1, 1, -1)

        out = ref_masked_attention(q, keys, values, scale, alibi_bias)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)

def create_block_tables(block_tables):
    max_block_len = max([len(block_table) for block_table in block_tables])
    paddings = [[-1] * (max_block_len - len(block_table)) for block_table in block_tables]
    new_block_tables = [block_table + padding for block_table, padding in zip(block_tables, paddings)]
    return torch.tensor(new_block_tables, dtype=torch.int)

def gen_dataset(n_prompt, n_completion, n_shared, batch_size, block_size, force_share=False):
    seq_len = n_prompt + n_completion
    n_blocks = seq_len * batch_size // block_size
    assert seq_len % block_size == 0
    assert n_shared % block_size == 0
    
    # make the last dimension always 16 bytes
    x = 16 // torch.tensor([]).element_size()
    assert d_head % x == 0
    key_cache = torch.randn(size=(n_blocks, n_heads, d_head // x, block_size, x))
    value_cache = torch.randn(size=(n_blocks, n_heads, d_head, block_size))
    print(f'key_cache.shape:{key_cache.shape} value_cache.shape:{value_cache.shape}')
    
    # place shared prompt tokens
    n_shared_blocks = 0
    if force_share:
        n_shared_blocks = n_shared // block_size
    block_tables = [list(range(n_shared_blocks)) for _ in range(batch_size)]
    n_used_blocks = n_shared_blocks

    # place non-shared prompt tokens
    context_lens = [] 
    for i in range(batch_size):
        n = n_prompt // block_size - n_shared_blocks
        context_lens.append((n_shared_blocks + n) * block_size)
        for _ in range(n):
            block_tables[i].append(n_used_blocks)
            n_used_blocks += 1
    
    query = torch.randn(batch_size, n_heads, d_head)
    scale = float(1.0 / (d_head ** 0.5))
    output = torch.empty(batch_size, n_heads, d_head)
    num_kv_heads = n_heads
    
    # check correctness
    ops.paged_attention_v1(
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            create_block_tables(block_tables),
            torch.tensor(context_lens, dtype=torch.int),
            block_size,
            max(context_lens),
            None)
    # Run the reference implementation.
    ref_output = torch.empty_like(query)
    ref_single_query_cached_kv_attention(
        ref_output,
        query,
        1,
        key_cache,
        value_cache,
        create_block_tables(block_tables),
        torch.tensor(context_lens, dtype=torch.int),
        scale,
        None,
    )
    assert torch.allclose(output, ref_output, atol=1e-3, rtol=1e-3)
    
    return query, key_cache, value_cache, output, num_kv_heads, block_tables, context_lens, scale, n_used_blocks
    
@torch.inference_mode()
def run_paged_attention_tps(n_prompt, n_completion, n_shared, batch_size, force_share):
    print(f'\n[PagedAttn]')
    print(f'{torch.randn(1).device} dtype: {torch.randn(1).dtype}')
    print(f'prompt:{n_prompt} completion:{n_completion} shared:{n_shared} batch_size:{batch_size}')
    block_size = 32 # tokens in each block
    query, key_cache, value_cache, output, num_kv_heads, block_tables, context_lens, scale, n_used_blocks = \
        gen_dataset(n_prompt, n_completion, n_shared, batch_size, block_size, force_share=force_share)

    # start decoding
    ret = []
    latency = 0.0
     
    for t in range(n_prompt, n_prompt + n_completion):
        if t % block_size == 0:
            for i in range(batch_size):
                block_tables[i].append(n_used_blocks)
                n_used_blocks += 1
        for i in range(batch_size): context_lens[i] += 1
        block_tables_tensor = create_block_tables(block_tables)
        context_lens_tensor = torch.tensor(context_lens, dtype=torch.int)
        max_context_len = max(context_lens)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        ops.paged_attention_v1(
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables_tensor,
            context_lens_tensor,
            block_size,
            max_context_len,
            None)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        latency += (end_time - start_time)
        ret.append(latency)
        #print(f'iter {t}, latency {end_time - start_time}')
    return ret

@torch.inference_mode()
def run_paged_attention_latency(seq_len, n_shared, batch_size, force_share):
    print(f'\n[PagedAttn]')
    print(f'{torch.randn(1).device} dtype: {torch.randn(1).dtype}')
    print(f'seq_len:{seq_len} n_shared:{n_shared}')
    
    block_size = 32 # tokens in each block
    query, key_cache, value_cache, output, num_kv_heads, block_tables, context_lens, scale, _ = \
        gen_dataset(seq_len, seq_len, n_shared, batch_size, block_size, force_share=force_share)
    
    block_tables_tensor = create_block_tables(block_tables)
    context_lens_tensor = torch.tensor(context_lens, dtype=torch.int)
    max_context_len = max(context_lens)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    n_repeat = 100
    for _ in range(n_repeat):          
        ops.paged_attention_v1(
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables_tensor,
            context_lens_tensor,
            block_size,
            max_context_len,
            None)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    t = (end_time - start_time)/n_repeat * 1e3
    print(f"PagedAttn: {t:.2f} ms")
    return t

@torch.inference_mode()
def run_llama_offline(model_path, n_prompt, n_completion, n_shared, batch):
    print(f'\n[vllm]')
    print(f'prompt:{n_prompt} completion:{n_completion} shared:{n_shared} batch_size:{batch}')
    seq_ids = gen_dataset_tokens_only(batch, n_prompt, n_shared)
    
    llm = LLM(model=model_path,
              enforce_eager=True,
              max_num_seqs=batch,
              block_size=32,
              max_num_batched_tokens=batch * n_prompt,
              swap_space=0, #cpu memory in GB for offloading
              disable_log_stats=True,
              )
    sampling_params = SamplingParams(stop=["ca2c8e7c9"],
                                     #use_beam_search=True,
                                     n=1,
                                     #best_of=2,
                                     max_tokens=n_completion,
                                     early_stopping=False,
                                     ignore_eos=True,
                                     temperature=0.0,
                                     top_p=1.0)
    torch.cuda.synchronize()
    start = time.perf_counter()
    results = llm.generate(prompt_token_ids=seq_ids, sampling_params=sampling_params)
    torch.cuda.synchronize()
    end = time.perf_counter()
    total_tokens = sum([len(res.outputs[0].token_ids) for res in results])
    assert total_tokens == batch * n_completion
    tps = total_tokens / (end - start)
    print(f'time: {end-start} tps: {tps}')
    return tps

def llm_engine_step_hook(foo):   
    def collect_metrics(self:LLMEngine):
        ret = foo(self)
        total_num_gpu_blocks = self.cache_config.num_gpu_blocks
        num_free_gpu_blocks = (
            self.scheduler.block_manager.get_num_free_gpu_blocks())
        num_used_gpu_blocks = total_num_gpu_blocks - num_free_gpu_blocks
        # block_size: Size of a cache block in number of tokens.
        mem_used_gb = num_used_gpu_blocks * self.cache_config.block_size * 32 * 128 * 2 * 2 * 32 / 1024 / 1024 / 1024
        if mem_used_gb > collect_metrics.peak_kv_cache_mem:
            collect_metrics.peak_kv_cache_mem = mem_used_gb
        batch_size = len(self.scheduler.running)
        if batch_size > collect_metrics.peak_batch_size:
            collect_metrics.peak_batch_size = batch_size
        return ret
    collect_metrics.peak_kv_cache_mem = 0
    collect_metrics.peak_batch_size = 0
    return collect_metrics

LLMEngine.step = llm_engine_step_hook(LLMEngine.step)

class SystemUnderTest:
    def __init__(self, model_path, n_completion: int) -> None:
        self.engine = my_llm_engine.MyLLMEngine(
            model=model_path,
            #max_num_seqs=max_batch, # max batch size
            swap_space=0, #cpu memory in GB for offloading
            enforce_eager=True,
            block_size=32,
            disable_log_stats=True)     
        self.sampling_params = SamplingParams(stop=["ca2c8e7c9"],
                                     #use_beam_search=True,
                                     n=1,
                                     #best_of=2,
                                     max_tokens=n_completion,
                                     early_stopping=False,
                                     ignore_eos=True,
                                     temperature=0.0, # GREEDY search
                                     top_p=1.0)
    
    def run(self, prompt_tokens: List[int]):
        start = time.perf_counter()
        result = self.engine.generate(
            sampling_params=self.sampling_params,
            prompt_token_ids=prompt_tokens)
        end = time.perf_counter()
        output_tokens = len(result.outputs[0].token_ids)
        assert output_tokens == self.sampling_params.max_tokens
        #return {'latency_ms': (end - start) * 1e3}
        
    def start(self):
        self.engine.start()

    def stop(self):
        print(f"peak KV memory(GB): {LLMEngine.step.peak_kv_cache_mem}, peak batch size: {LLMEngine.step.peak_batch_size}")
        self.engine.stop()

def run_llama_server(model_path, n_prompt, n_completion, n_shared, max_batch, rps):
    global peak_kv_cache_mem
    print(f'\n[vllm]')
    print(f'prompt:{n_prompt} completion:{n_completion} shared:{n_shared} max_batch_size:{max_batch}')
    seq_ids = gen_dataset_tokens_only(10000, n_prompt, n_shared)
    
    with ServerModelRunner(SystemUnderTest,
                           async_worker=False,
                           num_workers=1,
                           num_threads=max_batch,
                           tensorboard=True)(model_path, n_completion) as model_runner:
        report = model_runner.benchmark(
            queries=[(x, ) for x in seq_ids],
            target_qps=rps,
            min_query_count=300, min_duration_ms=60000)
        while report['#queries/issued'] > report['#queries/succeeded'] + report['#queries/failed']:
            time.sleep(1)
            report = model_runner.get_report()
        
        print(report)
        assert report['#queries/issued'] == report['#queries/succeeded']
        return report


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    model_path = "/tmp/models/open_llama_7b_v2"
    #model_path = 'openlm-research/open_llama_7b'
    #run_llama_offline(model_path, n_prompt=1024, n_completion=512, n_shared=0, batch=32)
    run_llama_server(model_path, n_prompt=1024, n_completion=512, n_shared=1024, max_batch=32, rps=1)
    #sut = SystemUnderTest(model_path, n_completion=512, max_batch=32)
    #sut.run([42] * 1024)
