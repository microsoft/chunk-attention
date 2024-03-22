from chunk_attn.arithmetic import llama
from transformers import LlamaConfig

if __name__ == '__main__':
    config = LlamaConfig.from_pretrained('openlm-research/open_llama_3b')
    model = llama.LlamaForCausalLM(config)
    print(f'total:{model.metrics_total}')
    print(f'self:{model.metrics_self}')