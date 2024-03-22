from typing import List
import torch
import time
import random
from chunk_attn import Attention
from chunk_attn.models.llama_hf import LlamaConfig, LlamaTokenizer, LlamaForCausalLM
from chunk_attn.models.model_host import ModelHost, Sequence
import benchmark_attn_pytorch
import multiprocessing
try:
    from model_perf.server import ServerModelRunner
except:
    pass

n_heads, d_head = 32, 128
is_cuda = torch.cuda.is_available()
device = torch.randn(1, device='cuda').device if is_cuda else torch.device('cpu')
torch.set_default_device(device)
dtype = torch.float16 if is_cuda else torch.float32
torch.set_default_dtype(dtype)

def gen_dataset(n_heads, d_head, batch_size, n_prompt, n_shared):
    keys, values, qs, seqs = benchmark_attn_pytorch.gen_dataset(n_heads, d_head, batch_size, n_prompt, n_shared)
    q = torch.cat(qs, dim=1) # (n_heads, n_seqs, d_head)
    q = q.transpose(0, 1).contiguous()
    keys = [key.transpose(0, 1).contiguous() for key in keys]
    values = [value.transpose(0, 1).contiguous() for value in values]
    return keys, values, q, seqs

def gen_dataset_tokens_only(n_seqs, seq_len, n_shared):
    shared_tokens = [42] * n_shared
    res = []
    for _ in range(n_seqs):
        tokens = [i for i in shared_tokens] + [random.randint(10, 100) for _ in range(seq_len - n_shared)]
        res.append(tokens)
    return res

def run_chunk_attn_tps(n_prompt, n_completion, n_shared, batch_size):
    chunk_size = 64 
    print(f'\n[ChunkAttn]\nnum_of_threads:{torch.get_num_threads()} chunk_size:{chunk_size}')
    print(f'{device} {dtype}')
    print(f'prompt:{n_prompt} completion:{n_completion} shared:{n_shared} batch_size:{batch_size}')
    
    keys, values, q, seqs = gen_dataset(n_heads, d_head, batch_size, n_prompt, n_shared)  
    attn = Attention(n_heads=n_heads, d_head=d_head, chunk_size=chunk_size, 
                     memory_mb=8192*4,
                     dtype=dtype, device=device)
    new_tokens = list(range(batch_size))
    new_k = torch.randn((batch_size, n_heads, d_head))
    new_v = torch.randn((batch_size, n_heads, d_head))
    
    ret = []
    latency = 0.0
    for i in range(batch_size):
        attn.add_seq(tokens=seqs[i], k=keys[i], v=values[i])
    
    # warm up
    attn.forward(q=q)
    
    for i in range(n_prompt, n_prompt + n_completion):
        attn.append_token(tokens=new_tokens, k=new_k, v=new_v)
        attn.refresh_kernel_context(force=False)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        output = attn.forward(q=q)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        latency += (end_time - start_time)
        ret.append(latency)

    return ret

def run_chunk_attn_latency(n_prompt, n_shared, batch_size, chunk_size, partition=0):
    print(f'\n[ChunkAttn]\ninterop_threads:{torch.get_num_interop_threads()} intraop_threads:{torch.get_num_threads()}')
    print(f'{device} {dtype}')   
    print(f'n_prompt:{n_prompt} n_shared:{n_shared} chunk_size:{chunk_size} partition:{partition}')

    keys, values, q, seqs = gen_dataset(n_heads, d_head, batch_size, n_prompt, n_shared)
    attn = Attention(n_heads=n_heads, d_head=d_head, chunk_size=chunk_size, 
                     memory_mb=8192,
                     dtype=dtype, device=device)
    for i in range(batch_size):
        attn.add_seq(tokens=seqs[i], k=keys[i], v=values[i])   
    # warm up
    attn.forward(q=q, partition=partition)
    
    if is_cuda: torch.cuda.synchronize()
    n_repeat = 100
    t_total = 0.0
    start_time = time.perf_counter()
    for step in range(n_repeat):
        output = attn.forward(q=q, partition=partition)
    if is_cuda: torch.cuda.synchronize()
    end_time = time.perf_counter() 
    t = (end_time - start_time)/n_repeat * 1e3  # in ms
    print(f"ChunkAttn: {t:.2f} ms")
    return t

def run_chunk_attn_cmp_chunk_seq(seq_len, n_shared, batch_size, chunk_size):    
    print(f'\n[ChunkAttn]\ninterop_threads:{torch.get_num_interop_threads()} intraop_threads:{torch.get_num_threads()}')
    print(f'{device} {dtype}')
    print(f'seq_len:{seq_len}, n_shared:{n_shared} chunk_size:{chunk_size}')
    
    keys, values, q, seqs = gen_dataset(n_heads, d_head, batch_size, seq_len, n_shared)
    n_seqs = len(seqs)
    attn = Attention(n_heads=n_heads, d_head=d_head, chunk_size=chunk_size, memory_mb=8192,
                     dtype=dtype, device=device)
    for i in range(n_seqs):
        attn.add_seq(tokens=seqs[i], k=keys[i], v=values[i])  
    # warm up
    attn.forward(q=q)
    n_repeat = 100
    
    # chunk first   
    t_total = 0.0
    for _ in range(n_repeat):
        output = attn.forward(q=q)
    t_chunk = t_total/n_repeat/1e3
    
    # sequence first
    t_total = 0.0
    for _ in range(n_repeat):
        output = attn.forward(q=q, partition=2)
    t_seq = t_total/n_repeat/1e3
       
    print(f'Chunk: {t_chunk:.2f} ms, Seq: {t_seq:.2f} ms')
    return (t_chunk, t_seq)


@torch.inference_mode()
def run_llama_offline(model_path, n_prompt, n_completion, n_shared, batch):   
    print(f'\n[ChunkAttn]\nnum_of_threads:{torch.get_num_threads()}')
    print(f'prompt:{n_prompt} completion:{n_completion} shared:{n_shared} batch_size:{batch}')

    config = LlamaConfig.from_pretrained(model_path)
    config.max_position_embeddings = 10240
    config.num_hidden_layers = 1
    if torch.cuda.is_available():
        config.torch_dtype = torch.float16
        config.torch_device = torch.device('cuda:0')       
    else:
        config.torch_dtype = torch.float32
        config.torch_device = torch.device('cpu')
    
    model = LlamaForCausalLM.from_pretrained(model_path, 
                                             config=config,
                                             torch_dtype=config.torch_dtype,
                                             device_map=config.torch_device)
    print(f'model {model_path} loaded')
    model([Sequence(prompt_tokens=[42])], None, prefill=True)
    kv_caches = model.create_kv_caches()

    seq_ids = gen_dataset_tokens_only(batch, n_prompt, n_shared)
    
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    seqs = [Sequence(prompt_tokens=seq) for seq in seq_ids]
    new_tokens = model(seqs, kv_caches, prefill=True)
    
    for idx, seq in enumerate(seqs):
        seq.append(new_tokens[idx])
  
    tps = 0.0         
    for i in range(n_completion):    
        new_tokens = model(seqs, kv_caches, prefill=False)
        #print(f'iteration {i} done')
        for idx, seq in enumerate(seqs):
            seq.append(new_tokens[idx])
    
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    
    tps = batch * n_completion / (t3 - t1)
    print(f'prefill+decode: {t3 - t1:.2f}, tps {tps:.2f}')
    return tps


class SystemUnderTest:
    def __init__(self, model_path, n_completion: int) -> None:
        self.model_path = model_path        
        self.n_completion = n_completion
        config = LlamaConfig.from_pretrained(self.model_path)
        config.torch_dtype = dtype
        config.torch_device = device
        #config.num_hidden_layers = 1
        
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype=config.torch_dtype,
            device_map=config.torch_device)
        print(f'model {self.model_path} loaded')
        self.kv_caches = self.model.create_kv_caches()
        self.host = ModelHost(self.model, self.kv_caches, max_batch_size=64) 
        # warm-up
        self.model([Sequence(prompt_tokens=[42])], None, prefill=True)

    def start(self):
        self.host.start()
    
    def stop(self):
        print(f'peak KV memory(GB): {self.kv_caches[0].peak_memory_allocated()/1024/1024/1024}, peak batch size: {self.host.peak_batch_size}')
        self.host.stop()
    
    def run(self, prompt_tokens: List[int]) -> List[int]:
        f = self.host.predict_async(prompt_tokens, self.n_completion)
        all_tokens = f.get_result()
        assert len(all_tokens) == len(prompt_tokens) + self.n_completion

def run_llama_server(model_path, n_prompt, n_completion, n_shared, max_batch, rps):
    print(f'\n[ChunkAttn]\nprompt:{n_prompt} completion:{n_completion} shared:{n_shared} max_batch_size:{max_batch}')
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


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    #torch.set_num_threads(32)
    #run_chunk_attn_cmp_chunk_seq(256, 256, 32, 64)
    #run_chunk_attn_cmp_chunk_seq(320, 256, 32, 64)
    #run_chunk_attn_latency(256, 256, 32, 64, 0)
    #run_chunk_attn_latency(256, 256, 32, 64, 2)
    #run_chunk_attn_tps(n_prompt=1024, n_completion=256, n_shared=0, batch_size=16)
    model_path = 'openlm-research/open_llama_7b'
    #model_path = "/tmp/models/open_llama_7b_v2"
    run_llama_offline(model_path, n_prompt=1024, n_completion=64, n_shared=0, batch=64)
    #run_llama_server(model_path, n_prompt=1024, n_completion=512, n_shared=0, max_batch=32, rps=2.1)
