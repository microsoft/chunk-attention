import torch
import sys


# 1. use sdp in pytorch
# https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/fc133e4ffc6275f9d1c3a74ddd10e0a2/scaled_dot_product_attention_tutorial.ipynb#scrollTo=8pt80K3RScx2                   
# https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
if sys.platform == 'win32':
    def casual_attn(q, k, v, output_shape):  
        attn_output: torch.Tensor = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), 
            k.transpose(1, 2), 
            v.transpose(1, 2),
            dropout_p=0.0,
            is_causal=True,)
        # print(f'o1.shape: {o1.shape}, s.tokens: {len(s)}, hidden_size: {self.hidden_size}')
        attn_output = attn_output.transpose(1, 2).contiguous().view(*output_shape)
        return attn_output
# 2. use xformers
# https://facebookresearch.github.io/xformers/components/ops.html
else:
    import xformers.ops as xops
    def casual_attn(q, k, v, output_shape):
        attn_output = xops.memory_efficient_attention(
                    q, k, v,
                    attn_bias=xops.LowerTriangularMask()
            )
        attn_output = attn_output.view(*output_shape)
        return attn_output
