import unittest
import torch
from chunk_attn import Attention
import random
import math


class TestChunkAttnGPU(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.randn(1, device="cuda").device
        self.dtype = torch.float16
        torch.set_default_dtype(self.dtype)
        torch.set_default_device(self.device)
        print(f"device:{self.device} dtype:{self.dtype}")

    def tearDown(self) -> None:
        pass

    def test_check_result_with_pytorch(self):
        seq_len = 2000
        n_shared = round(seq_len * 0.5)
        n_heads, d_head = 2, 128
        n_requests = 32
        print(f"\nseq_len: {seq_len}, n_shared: {n_shared}, n_requests: {n_requests}")
        print(f"{torch.randn(1).device} {torch.randn(1).dtype}")

        keys = [torch.randn((seq_len, n_heads, d_head)) for _ in range(n_requests)]
        shared_keys = torch.randn((n_shared, n_heads, d_head))
        for key in keys:
            key[:n_shared, :, :] = shared_keys
        values = [torch.randn((seq_len, n_heads, d_head)) for _ in range(n_requests)]
        shared_values = torch.randn((n_shared, n_heads, d_head))
        for value in values:
            value[:n_shared, :, :] = shared_values
        qs = [torch.randn((1, n_heads, d_head)) for _ in range(n_requests)]

        # Implementation in PyTorch
        outputs = []
        for i in range(n_requests):
            score = torch.matmul(qs[i].transpose(0, 1), keys[i].transpose(0, 1).transpose(1, 2))
            score = torch.softmax(score.to(torch.float32) / math.sqrt(d_head), dim=-1)
            score = score.to(torch.float16)
            outputs.append(torch.matmul(score, values[i].transpose(0, 1)).transpose(0, 1))
        output_ref = torch.cat(outputs, dim=0)

        q = torch.cat(qs, dim=0)
        chunks = [64]
        for chunk_size in chunks:
            print(f"chunk_size: {chunk_size}")
            attn = Attention(
                n_heads=n_heads,
                d_head=d_head,
                chunk_size=chunk_size,
                memory_mb=8192,
                dtype=self.dtype,
                device=self.device,
            )
            for i in range(n_requests):
                attn.add_seq(
                    tokens=list(range(n_shared)) + [random.randint(n_shared, seq_len) for _ in range(seq_len - n_shared)],
                    k=keys[i], v=values[i],)
            # attn.print()
            output2 = attn.forward(q=q)  # chunk+seq by default
            # print(output2[0][0])
            # print(output_ref[0][0])
            self.assertTrue(torch.allclose(output_ref, output2, atol=1e-3))

            # seq-first only, for perf testing purpose
            output3 = attn.forward(q=q, partition=2)
            # print(output3[2][1])
            # print(output_ref[2][1])
            self.assertTrue(torch.allclose(output_ref, output3, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
