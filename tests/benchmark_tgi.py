import random
import time
import io
import asyncio
import signal
import subprocess
from typing import List, Optional, Union
from text_generation import AsyncClient
from model_perf.server import ServerModelRunner
import transformers


def generate_request(num_request, prompt_length, shared_length):
    prompt_length -= 1
    words = ['sun ', 'app ', 'play ', 'good ', 'at ', 'old ']
 
    random.seed(10)
    requests = []
    for i in range(num_request):
        request = ''
        for j in range(shared_length):
            request += words[1]
        for k in range(prompt_length - shared_length):
            index = k % len(words)
            request += words[index]
        requests.append(request)
    random.shuffle(requests)
    return requests


class SystemUnderTest:
    def __init__(self, n_completion: int) -> None:
        self.client = AsyncClient("http://127.0.0.1:3000")
        self.n_completion = n_completion
        # command to start TGI server:
        # text-generation-launcher --model-id /tmp/models/open_llama_7b_v2 --port 3000 --max-input-length=4096 --max-total-tokens=8192 --max-batch-total-tokens=61440 --waiting-served-ratio=1.2
    
    async def run(self, request: str):
        response = await self.client.generate(request, 
                                              max_new_tokens=self.n_completion)
        assert response.details.generated_tokens == self.n_completion
        
    async def start(self):    
        pass

    async def stop(self):
        pass


def run_llama_server(n_prompt, n_completion, n_shared, max_batch, rps):
    print(f'\n[tgi]')
    print(f'prompt:{n_prompt} completion:{n_completion} shared:{n_shared} max_batch_size:{max_batch}')
    requests = generate_request(32, n_prompt, n_shared)
    
    with ServerModelRunner(SystemUnderTest,
                           async_worker=True,
                           num_workers=1,
                           num_tasks=max_batch,
                           tensorboard=True)(n_completion) as model_runner:

        report = model_runner.benchmark(
            queries=[(x, ) for x in requests],
            target_qps=rps,
            min_query_count=100, min_duration_ms=60000)
        while report['#queries/issued'] > report['#queries/succeeded'] + report['#queries/failed']:
            time.sleep(1)
            report = model_runner.get_report()
        
        print(report)
        assert report['#queries/issued'] == report['#queries/succeeded']
        return report
    

if __name__ == '__main__':
    model_path = 'openlm-research/open_llama_7b'
    run_llama_server(n_prompt=2048, n_completion=512, n_shared=0, max_batch=32, rps=0.6)
