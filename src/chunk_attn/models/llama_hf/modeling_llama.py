import torch
from torch import nn

from transformers.activations import ACT2FN
from typing import List, Tuple
from transformers import PreTrainedModel
from .configuration_llama import LlamaConfig
from .rotary_embedding import LlamaRotaryEmbedding
from .layernorm import LlamaRMSNorm
from .casual_attn import casual_attn
import chunk_attn
import chunk_attn.nvtx as nvtx
from chunk_attn.models.sequence import Sequence


class LlamaChunkAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        # `num_key_value_heads=num_attention_heads` by deafult, the model will use Multi Head Attention (MHA)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        if self.config.rope_scaling is not None:
            raise NotImplementedError('rope_scaling is not supported')
        self.rotary_emb = LlamaRotaryEmbedding(
            head_size=self.head_dim, rotary_dim=self.head_dim, max_position_embeddings=10240,
            base=10000)
                
    def forward(self, 
                hidden_states: torch.Tensor,   # [n_seqs, n_tokens, n_heads*d_head]
                seqs: List[Sequence],          # [n_seqs, n_tokens]
                position_ids: torch.Tensor,    # [n_seqs, n_tokens]
                kv_cache,
                prefill: bool):
        batch, n_tokens, d_hidden = hidden_states.shape
        
        nvtx.range_push("qkv proj")
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        nvtx.range_pop()

        # add positional embedding
        nvtx.range_push("rope")
        query_states, key_states = self.rotary_emb(position_ids, query_states, key_states)
        nvtx.range_pop()

        query_states:torch.Tensor = query_states.view(batch, n_tokens, self.num_heads, self.head_dim)
        key_states:torch.Tensor = key_states.view(batch, n_tokens, self.num_key_value_heads, self.head_dim)
        value_states:torch.Tensor = value_states.view(batch, n_tokens, self.num_key_value_heads, self.head_dim)
    
        attn_output = None
        
        if prefill:
            # for prefill tokens
            nvtx.range_push("sdp of prefill")
            attn_output = casual_attn(query_states, key_states, value_states, output_shape=(batch, n_tokens, d_hidden))
            nvtx.range_pop()
            if kv_cache:
                for idx, seq in enumerate(seqs):
                    nvtx.range_push("add_seq")
                    k, v = key_states[idx], value_states[idx]
                    # print(f'seq len:{len(seq.tokens)}, k.size:{k.size()}, v.size:{v.size()}')
                    idx_in_seqs = kv_cache.add_seq(tokens=seq.tokens, k=k, v=v)
                    nvtx.range_pop()           
                    seq.index = idx_in_seqs
        else:
            # for decode tokens, process before adding prompts to chunk attn        
            assert hidden_states.shape[1] == 1
            q = query_states.squeeze(dim=1)
            k = key_states.squeeze(dim=1)
            v = value_states.squeeze(dim=1)         
            nvtx.range_push("append_token")         
            kv_cache.append_token(tokens=[seq[-1] for seq in seqs], k=k, v=v, fused=True)  # 261 us
            nvtx.range_pop()    
            # [n_decode_tokens, n_heads, d_head]
            nvtx.range_push("attn")
            attn_output = kv_cache.forward(q=q)    # 482 us
            nvtx.range_pop()                   
            attn_output = attn_output.view(batch, 1, d_hidden) # 83 us

        nvtx.range_push("o_proj")
        output = self.o_proj(attn_output)
        nvtx.range_pop()
        if kv_cache:
            nvtx.range_push("reserve")
            kv_cache.reserve()
            nvtx.range_pop()
            kv_cache.refresh_kernel_context(force=False)
        return output


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        if self.config.pretraining_tp > 1:
            raise NotImplementedError('pretraining_tp > 1 is not supported yet')

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaChunkAttention(config=config) if config.use_chunk_attn else LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self,
                hidden_states: torch.Tensor,    # [n_seqs, n_tokens, n_heads*d_head]
                seqs: List[Sequence],    # [n_seqs, n_tokens]
                position_ids: torch.Tensor,    # [n_seqs, n_tokens]
                kv_cache,
                prefill: bool):
        residual = hidden_states

        nvtx.range_push("1.input ln")
        hidden_states = self.input_layernorm(hidden_states)
        nvtx.range_pop()
        
        # Self Attention
        nvtx.range_push("2.self attn")
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            seqs=seqs,
            position_ids=position_ids,
            kv_cache=kv_cache,
            prefill=prefill)
        nvtx.range_pop()
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        nvtx.range_push("3.post ln")
        hidden_states = self.post_attention_layernorm(hidden_states)
        nvtx.range_pop()

        nvtx.range_push("4.mlp")
        hidden_states = self.mlp(hidden_states)
        nvtx.range_pop()

        nvtx.range_push("5.residual")
        hidden_states = residual + hidden_states
        nvtx.range_pop()

        return hidden_states


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_unexpected = [".*rotary_emb.inv_freq.*"]
    _supports_flash_attn_2 = False


class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

    def forward(self,
                seqs: List[Sequence],
                kv_caches:List[chunk_attn.Attention],
                prefill: bool):
        input_ids = []
        position_ids = []
        if prefill:
            input_ids = [seq.tokens for seq in seqs]
            position_ids = [list(range(len(seq))) for seq in seqs]
        else:
            input_ids = [[seq[-1]] for seq in seqs]
            position_ids = [[len(seq)-1] for seq in seqs]
        max_seq_len = max([len(x) for x in input_ids])
        input_ids = [x + [self.padding_idx] * (max_seq_len - len(x)) for x in input_ids]
        position_ids = [x + [0] * (max_seq_len - len(x)) for x in position_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.norm.weight.device)
        position_ids = torch.tensor(position_ids, dtype=torch.int64, device=self.norm.weight.device)

        hidden_states = self.embed_tokens(input_ids)

        # decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                seqs=seqs,
                position_ids=position_ids,              
                kv_cache=kv_caches[idx] if kv_caches else None,
                prefill=prefill
            )
            hidden_states = layer_outputs
      
        hidden_states = self.norm(hidden_states)       
        return hidden_states

    def detect_free_memory(self):
        if self.config.chunk_attn_kvcache_gb > 0:
            free_mem = self.config.chunk_attn_kvcache_gb * 1024
        elif torch.device(self.config.torch_device).type == 'cuda':
            device = self.config.torch_device
            t = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024
            a = torch.cuda.memory_allocated(device) / 1024 / 1024
            free_mem = t - a
            print(f'total_mem:{t/1024:.2f} GB, allocated:{a/1024:.2f} GB, free_mem:{free_mem/1024:.2f} GB')
        else:
            #raise NotImplementedError('detect_free_memory is not implemented for cpu')
            free_mem = 8192
        return 0.8 * free_mem
    
    def create_kv_caches(self):
        free_mem = self.detect_free_memory() / len(self.layers)
        kv_caches = []
        kv_cache_1 = chunk_attn.Attention(
            n_heads=self.config.num_attention_heads,
            d_head=self.config.hidden_size // self.config.num_attention_heads,
            chunk_size=64,
            memory_mb = int(free_mem),         
            share_prefix=True,
            tpp_threshold=1,
            dtype=self.config.torch_dtype,
            device=self.config.torch_device)
        kv_caches.append(kv_cache_1)
        for layer in self.layers[1:]:
            new_kv_cache = chunk_attn.Attention(kv_cache_1)
            kv_caches.append(new_kv_cache)
        return kv_caches
    
    def clear_kv_cache(self):
        for cache in self.kv_caches:
            cache.clear()
            

class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(self, 
                seqs: List[Sequence],
                kv_caches:List[chunk_attn.Attention],
                prefill: bool):
        hidden_states = self.model(seqs, kv_caches, prefill)
        logits = self.lm_head(hidden_states)
        tokens = torch.argmax(logits, dim=-1).tolist()
        next_tokens = [x[-1] for x in tokens]
        return next_tokens

    def create_kv_caches(self) -> List[chunk_attn.Attention]:
        return self.model.create_kv_caches()