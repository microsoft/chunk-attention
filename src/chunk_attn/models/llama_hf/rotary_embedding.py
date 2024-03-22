import torch

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, head_size, rotary_dim, max_position_embeddings=10240, base=10000, device='cpu', dtype=torch.float32):
        super().__init__()
        self.dim = rotary_dim
        self.base = base

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device, dtype=dtype)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[:, None, :].to(dtype)
        self.sin_cached = emb.sin()[:, None, :].to(dtype)

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, positions, q, k):
        # q,k: [n_seqs, n_tokens, num_attention_heads * head_dim]
        embed_dim = q.size(-1)
        all_but_last_dim = q.size()[:-1]
        q = q.view(*all_but_last_dim, embed_dim // self.dim, self.dim)
        k = k.view(*all_but_last_dim, embed_dim // self.dim, self.dim)
        max_pos = torch.max(positions).item()
        positions = positions.tolist()        
        if max_pos > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=max_pos, device=q.device, dtype=q.dtype)
        q_results, k_results = [], []
        for seq_idx, position in enumerate(positions):
            q_tmp, k_tmp = [], []
            for pos_idx, pos in enumerate(position):
                cos = self.cos_cached[pos]
                sin = self.sin_cached[pos]
                q_tmp.append((q[seq_idx][pos_idx] * cos) + (self.rotate_half(q[seq_idx][pos_idx]) * sin))
                k_tmp.append((k[seq_idx][pos_idx] * cos) + (self.rotate_half(k[seq_idx][pos_idx]) * sin))
            q_results.append(torch.stack(q_tmp, dim=0))
            k_results.append(torch.stack(k_tmp, dim=0))
        q = torch.stack(q_results, dim=0).view(*all_but_last_dim, embed_dim)
        k = torch.stack(k_results, dim=0).view(*all_but_last_dim, embed_dim)
        return q, k

LlamaRotaryEmbedding = RotaryEmbedding
if torch.cuda.is_available():
    from .rotary_embedding_vllm import RotaryEmbedding as RotaryEmbeddingVLLM
    LlamaRotaryEmbedding = RotaryEmbeddingVLLM