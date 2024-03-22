
# https://kipp.ly/blog/transformer-inference-arithmetic/
# d_head: The "d_head" represents the dimensionality of each attention head. In other words, it denotes the size of the projected embeddings for each head. For example, if a GPT model uses 8 attention heads and has an embedding dimension of 512 (d_embd=512), then "d_head" would be 512 divided by 8, resulting in a value of 64.
# d_model(d_embd): The "d_model" represents the dimensionality of the model. In other words, it denotes the size of the input and output embeddings. For example, if a GPT model has an embedding dimension of 512 (d_embd=512), then "d_model" would be 512.
# d_inner: The "d_inner" represents the dimensionality of the inner feed-forward layer. In other words, it denotes the size of the hidden layer in the feed-forward network. For example, if a GPT model has an embedding dimension of 512 (d_embd=512), then "d_inner" would be 4 times 512, resulting in a value of 2048.
class GPTArithmetic:  
    def __init__(self,
                 vocab_size=50257,
                 max_position_embeddings=1024,
                 d_embd=768,
                 n_layer=12, 
                 n_heads=12,
                 fp=16):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.n_layer = n_layer
        self.n_heads = n_heads
        self.d_embd = d_embd
        self.d_model = self.d_embd
        self.d_head = d_embd // n_heads
        self.d_inner = 4 * self.d_embd
        self.fp = fp
    
    @staticmethod
    def from_gpt2():
        return GPTArithmetic(
            max_position_embeddings=1024,
            d_embd=768,
            n_layer=12, 
            n_heads=12,
        )
    
    @staticmethod
    def from_gpt3():
        return GPTArithmetic(
            max_position_embeddings=2048,
            d_embd=12288,
            n_layer=96, 
            n_heads=96,
        )
    
    @staticmethod
    def from_gpt35_turbo():
        return GPTArithmetic(
            max_position_embeddings=4096,
            d_embd=12288,
            n_layer=96, 
            n_heads=96,
        )
    
    def kvcache_per_token(self):
        # Numbers every LLM Developer should know:
        # https://github.com/ray-project/llm-numbers#1-mb-gpu-memory-required-for-1-token-of-output-with-a-13b-parameter-model
        return 2 * self.n_layer * self.d_embd * (self.fp // 8)
    
    def num_params(self):
        # https://aizi.substack.com/p/how-does-gpt-3-spend-its-175b-parameters
        # https://docs.google.com/spreadsheets/d/10Y4GLc28UgeKr2qSYEZuRqELn1D-w5EiQpAGg-_y4Xg/edit#gid=899002403
        # vocab emebedding
        vocab = self.vocab_size * self.d_model
        # postion embedding
        pos = self.max_position_embeddings * self.d_model
        # attention
        attn = 4 * self.d_model * self.d_head * self.n_heads
        # ffn
        ffn = 8 * self.d_model * self.d_model + 5 * self.d_model
        return vocab + pos + (attn + ffn) * self.n_layer
    
    def mem_params(self):
        return self.num_params() * (self.fp // 8)
        
if __name__ == "__main__":
    gpt = GPTArithmetic.from_gpt3()
    print(f'num of params: {gpt.num_params() / 1000 / 1000 / 1000:.0f}B')
    token_kvcache = gpt.kvcache_per_token()
    print(f'each token takes {token_kvcache / 1024 / 1024:.2f} MB memory')
    total_mem = 640 * 1024 * 1024 * 1024    # A100*8: 640 GB
    mem_params = gpt.mem_params()
    print(f'params takes up to {mem_params / 1024 / 1024 / 1024:.2f} GB memory')
    num_tokens = (total_mem - mem_params)/token_kvcache
    print(f'max tokens to hold: {num_tokens:.0f}, mapping to {num_tokens / gpt.max_position_embeddings:.0f} served queries')