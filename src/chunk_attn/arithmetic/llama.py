from typing import List, Tuple
from .tensor import Tensor
from .linear import Linear
from .module import Module
from transformers import LlamaConfig


class Sequence:
    def __init__(self, tokens=[]) -> None:
        self.tokens = tokens
        self.is_prefill = True
        self.is_decode = False
        self.idx = 0


class LlamaAttention(Module):
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
        self.q_proj = Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor, 
                prefill_seqs: List[Sequence],
                decode_seqs: List[Sequence]):
        bsz, n_tokens, _ = tuple(hidden_states.shape)
        n_prefill_tokens = sum([len(s.tokens) for s in prefill_seqs])
        n_decode_tokens = len(decode_seqs)

        if self.config.pretraining_tp > 1:
            raise NotImplementedError('pretraining_tp > 1 is not supported yet')
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        
        attn_output = Tensor(hidden_states.size()) # [bsz, seq_len, hidden_size]
        attn_output = self.o_proj(attn_output)

        return attn_output

class LlamaMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            raise NotImplementedError('pretraining_tp > 1 is not supported yet')
        else:
            down_proj = self.down_proj(self.gate_proj(x) * self.up_proj(x))

        return down_proj

class LlamaRMSNorm(Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = Tensor(hidden_size, metrics=True)
        self.variance_epsilon = eps

    '''
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    '''

class LlamaDecoderLayer(Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    '''
    def forward(self, 
                hidden_states: torch.Tensor,
                prefill_seqs: List[Sequence], 
                decode_seqs: List[Sequence]):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            prefill_seqs=prefill_seqs,
            decode_seqs=decode_seqs)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = hidden_states
        return outputs
    '''

class LlamaModel(Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        #self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    '''
    def forward(self,
                prefill_seqs: List[Sequence],
                decode_seqs: List[Sequence]
                ):
        input_ids = []
        for s in prefill_seqs:
            input_ids.extend(s.tokens)
        for s in decode_seqs:
            input_ids.append(s.tokens[-1])
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        print(input_ids)
        hidden_states = self.embed_tokens(input_ids)

        # decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                prefill_seqs=prefill_seqs,
                decode_seqs=decode_seqs,
            )
            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)
        return hidden_states 
    '''

class LlamaForCausalLM(Module):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
    
    '''
    def forward(self,
                prefill_seqs: List[Sequence],
                decode_seqs: List[Sequence]):
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        hidden_states = self.model(prefill_seqs=prefill_seqs, decode_seqs=decode_seqs)

        logits = self.lm_head(hidden_states)
        logits = logits.float()
        return logits
    '''