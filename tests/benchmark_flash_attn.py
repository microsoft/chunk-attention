import torch
import time
# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
from torch.nn.functional import scaled_dot_product_attention


n_heads, d_head = 32, 128
is_cuda = torch.cuda.is_available()
assert is_cuda
device = torch.randn(1, device='cuda').device if is_cuda else torch.device('cpu')
torch.set_default_device(device)
dtype = torch.float16 if is_cuda else torch.float32
torch.set_default_dtype(dtype)


def gen_dataset(n_heads, d_head, n_seqs, seq_len, n_shared):
    keys = torch.randn((n_seqs, n_heads, seq_len, d_head))
    values = torch.randn((n_seqs, n_heads, seq_len, d_head))
    qs = torch.randn((n_seqs, n_heads, 1, d_head))
    return keys, values, qs

@torch.inference_mode()
def run_flash_tps(n_prompt, n_completion, n_shared, batch_size):
    print(f'\n[FlashAttn]')
    print(f'{torch.randn(1).device} {torch.randn(1).dtype}')
    print(f'prompt:{n_prompt} completion:{n_completion} n_shared:{n_shared} batch_size:{batch_size}')
    keys, values, qs = gen_dataset(n_heads, d_head, batch_size, n_prompt+n_completion, n_shared)
 
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, 
        enable_math=False, 
        enable_mem_efficient=False
    ):
        print(f'torch.backends.cuda.flash_sdp_enabled={torch.backends.cuda.flash_sdp_enabled()}')
        # warm up
        scaled_dot_product_attention(qs, keys, values, attn_mask=None, dropout_p=0.0)
        ret = []
        latency = 0.0
        for i in range(n_prompt, n_prompt + n_completion):
            k, v = keys[:,:,:i+1,:], values[:,:,:i+1,:]
            k, v = k.contiguous(), v.contiguous()
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            scaled_dot_product_attention(qs, k, v, attn_mask=None, dropout_p=0.0)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            latency += (end_time - start_time)  # in seconds
            ret.append(latency)
            # print(f'iter {i+1}, latency {end_time - start_time}') 
    return ret

@torch.inference_mode()
def run_flash_latency(seq_len, n_shared, batch_size):
    print(f'\n[FlashAttn]')
    print(f'seq_len:{seq_len} n_shared:{n_shared}')
    
    keys, values, qs = gen_dataset(n_heads, d_head, batch_size, seq_len, n_shared)
    
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, 
        enable_math=False, 
        enable_mem_efficient=False
    ):
        print(f'torch.backends.cuda.flash_sdp_enabled={torch.backends.cuda.flash_sdp_enabled()}')
        # warm up
        scaled_dot_product_attention(qs, keys, values, attn_mask=None, dropout_p=0.0)

        n_repeat = 100
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(n_repeat):
            scaled_dot_product_attention(qs, keys, values, attn_mask=None, dropout_p=0.0)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        t = (end_time - start_time)/n_repeat * 1e3
        print(f"FlashAttn: {t:.2f} ms")
    return t


if __name__ == '__main__':
    latency = run_flash_latency(512, 256, 32)
    print(latency)
