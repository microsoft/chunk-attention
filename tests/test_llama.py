import time
import sys
import torch
import unittest
from chunk_attn.models.llama_hf import LlamaConfig, LlamaTokenizer, LlamaForCausalLM
from chunk_attn.models.model_host import ModelHost, Sequence
from benchmark_chunk_attn import gen_dataset_tokens_only


class TestLlama(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        #cls.model_path = 'openlm-research/open_llama_3b'
        cls.model_path = 'openlm-research/open_llama_7b'
        #cls.model_path = "/tmp/models/open_llama_7b_v2"
        # cls.model_path = 'openlm-research/open_llama_13b'
        
        cls.tokenizer = LlamaTokenizer.from_pretrained(cls.model_path)
        config = LlamaConfig.from_pretrained(cls.model_path)
        #config.num_hidden_layers = 1
        if torch.cuda.is_available():
            config.torch_dtype = torch.float16
            config.torch_device = torch.device('cuda:0')       
        else:
            config.torch_dtype = torch.float32
            config.torch_device = torch.device('cpu')
        
        cls.model = LlamaForCausalLM.from_pretrained(
            cls.model_path,
            config=config,
            torch_dtype=config.torch_dtype,
            device_map=config.torch_device)
        print(f'model {cls.model_path} loaded')
        cls.kv_caches = cls.model.create_kv_caches()

        # warm-up
        cls.model(seqs=[Sequence(prompt_tokens=[42])], kv_caches=None, prefill=True)
        print('warm-up done')     

    @classmethod
    def tearDownClass(self) -> None:
        pass
    
    @torch.inference_mode()
    def test_iterative_decoding(self):
        system_prompt = '''You must act as a friendly agent in charge of collecting a clear idea of what went wrong with the order, you need to ask them. Ask only one question at a time and be friendly. Don't create any information, it must be given by the client. You must be smart!\n'''
        # prompts = [system_prompt + 'Q: What is the black hole?\nA:',
        #            system_prompt + 'Q: What is the largest animal?\nA:',
        #            system_prompt + 'Q: What is the smallest animal?\nA:',
        #            system_prompt + 'Q: What is the largest country?\nA:',
        #            system_prompt + 'Q: What is the smallest country?\nA:',
        #            system_prompt + 'Q: What is the largest city?\nA:']
        prompts = ['Q: What is the black hole?\nA:'] * 8
  
        seqs = []
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.tolist()[0]
            seqs.append(Sequence(prompt_tokens=input_ids))
        new_tokens = self.model(seqs, self.kv_caches, prefill=True)
        
        for idx, seq in enumerate(seqs):
            seq.append(new_tokens[idx])

        start = time.time()
        for i in range(32):
            new_tokens = self.model(seqs, self.kv_caches, prefill=False)
            for idx, seq in enumerate(seqs):
                seq.append(new_tokens[idx])
                print(self.tokenizer.decode(seq.tokens) + '\n')
        
        end = time.time()
        print(f'elapsed time: {end - start}')
        
    def test_model_host(self):
        with ModelHost(self.model, self.kv_caches, max_batch_size=16) as host:
            futures = []
            # 'Q: Tell me what is the largest animal in the world.\nA:'
            for i in range(17):
                prompt_tokens = self.tokenizer(f'Q{i+1}: What is black hole?\nA: ', return_tensors="pt").input_ids.tolist()[0]
                f = host.predict_async(prompt_tokens, 32)
                time.sleep(0.2)
                futures.append(f)
            
            for f in futures:
                print(self.tokenizer.decode(f.get_result()) + '\n')
    
    @torch.inference_mode()
    def test_tps(self):
        n_prompt, n_completion, n_shared, batch = 256, 256, 0, 16   
        seqs = gen_dataset_tokens_only(batch, n_prompt, n_shared)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        seqs = [Sequence(prompt_tokens=seq) for seq in seqs]
        new_tokens = self.model(seqs, self.kv_caches, prefill=True)
        
        for idx, seq in enumerate(seqs):
            seq.append(new_tokens[idx])
  
        tps = 0.0      
        for i in range(n_completion - 1):    
            new_tokens = self.model(seqs, self.kv_caches, prefill=False)
            # print(f'iteration {i} done')
            for idx, seq in enumerate(seqs):
                seq.append(new_tokens[idx])
        
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        
        for idx, seq in enumerate(seqs):
            assert(len(seq) == n_prompt + n_completion)
        
        tps = batch * n_completion / (t3 - t1)
        print(f'prefill+decode: {t3 - t1:.2f}, tps {tps:.2f}')
        return tps


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestLlama("test_model_host"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
