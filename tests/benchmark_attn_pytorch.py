import torch
from torch.nn.functional import scaled_dot_product_attention
import random
import time

n_heads, d_head = 32, 128
is_cuda = torch.cuda.is_available()
device = torch.randn(1, device='cuda').device if is_cuda else torch.device('cpu')
torch.set_default_device(device)
dtype = torch.float16 if is_cuda else torch.float32
torch.set_default_dtype(dtype)


def gen_dataset(n_heads, d_head, n_seqs, seq_len, n_shared):
    keys = [torch.randn((n_heads, seq_len, d_head)) for _ in range(n_seqs)]
    shared_keys = torch.randn((n_heads, n_shared, d_head))
    for key in keys:
        key[:, :n_shared, :] = shared_keys
    values = [torch.randn((n_heads, seq_len, d_head)) for _ in range(n_seqs)]
    shared_values = torch.randn((n_heads, n_shared, d_head))
    for value in values:
        value[:, :n_shared, :] = shared_values     
    qs = [torch.randn((n_heads, 1, d_head)) for _ in range(n_seqs)]
    seqs = []
    for _ in range(n_seqs):
        seqs.append(list(range(n_shared)) + [random.randint(n_shared, seq_len) for _ in range(seq_len - n_shared)])
        
    return keys, values, qs, seqs

@torch.inference_mode()
def run_pytorch_tps(n_prompt, n_completion, n_shared, batch_size):
    print(f'\n[PyTorch]\ninterop_threads:{torch.get_num_interop_threads()} intraop_threads:{torch.get_num_threads()}')
    print(f'{torch.randn(1).device} {torch.randn(1).dtype}')
    print(f'prompt:{n_prompt} completion:{n_completion} n_shared:{n_shared} batch_size:{batch_size}')
    keys, values, qs, seqs = gen_dataset(n_heads, d_head, batch_size, n_prompt+n_completion, n_shared)
    keys = torch.stack(keys, dim=0)
    values = torch.stack(values, dim=0)
    qs = torch.stack(qs, dim=0)

    # warm up
    attn_weight = torch.matmul(qs, keys.transpose(-1, -2))
    attn_weight = torch.softmax(attn_weight, dim=-1)
    output = torch.matmul(attn_weight, values)
    
    ret = []
    latency = 0.0
    for i in range(n_prompt, n_prompt + n_completion):
        k, v = keys[:,:,:i+1,:].transpose(-1, -2), values[:,:,:i+1,:]
        k, v = k.contiguous(), v.contiguous()
        if is_cuda: torch.cuda.synchronize()
        start_time = time.perf_counter()
        attn_weight = torch.matmul(qs, k)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        output = torch.matmul(attn_weight, v)
        #scaled_dot_product_attention(qs, keys[:,:,:i+1,:], values[:,:,:i+1,:], attn_mask=None, dropout_p=0.0)
        if is_cuda: torch.cuda.synchronize()
        end_time = time.perf_counter()
        latency += (end_time - start_time)  # in seconds
        ret.append(latency)
        # print(f'iter {i+1}, latency {end_time - start_time}') 
    return ret

@torch.inference_mode()
def run_pytorch_latency(seq_len, n_shared, batch_size):
    print(f'\n[PyTorch]')
    print(f'{torch.randn(1).device} {torch.randn(1).dtype} num_of_threads:{torch.get_num_threads()}')
    print(f'seq_len:{seq_len} n_shared:{n_shared}')
    
    keys, values, qs, seqs = gen_dataset(n_heads, d_head, batch_size, seq_len, n_shared)
    keys = torch.stack(keys, dim=0)
    values = torch.stack(values, dim=0)
    qs = torch.stack(qs, dim=0)
    
    # warm up
    attn_weight = torch.matmul(qs, keys.transpose(-1, -2))
    attn_weight = torch.softmax(attn_weight, dim=-1)
    output = torch.matmul(attn_weight, values)

    n_repeat = 100
    if is_cuda: torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(n_repeat):
        attn_weight = torch.matmul(qs, keys.transpose(-1, -2))
        attn_weight = torch.softmax(attn_weight, dim=-1)
        output = torch.matmul(attn_weight, values)
    if is_cuda: torch.cuda.synchronize()
    end_time = time.perf_counter()
    t = (end_time - start_time)/n_repeat * 1e3
    print(f"PyTorch: {t:.2f} ms")
    return t


if __name__ == '__main__':
    latency = run_pytorch_latency(512, 256, 32)
    print(latency)
