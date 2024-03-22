import unittest
import torch
from chunk_attn import Attention
import random
import math

# torch.set_num_threads(1)
print(
    f"interop_threads:{torch.get_num_interop_threads()} intraop_threads:{torch.get_num_threads()}"
)

@unittest.skipIf(torch.cuda.is_available(), "test runs on CPU only.")
class TestChunkAttnCPU(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        torch.set_default_dtype(self.dtype)
        torch.set_default_device(self.device)
        print(f"device:{self.device} dtype:{self.dtype}")

    def tearDown(self) -> None:
        pass

    def test_add_seq(self):
        n_heads, d_head = 2, 4
        attn = Attention(
            n_heads=n_heads,
            d_head=d_head,
            chunk_size=2,
            memory_mb=16,
            dtype=self.dtype,
            device=self.device,
        )

        # root -> n1 -> n2 -> n3
        attn.add_seq(
            tokens=[1, 2, 3, 4, 5],
            k=torch.ones((5, n_heads, d_head)),
            v=torch.ones((5, n_heads, d_head)),
        )
        trie = attn.get_trie()
        self.assertEqual((trie.n_seqs, trie.n_tokens, trie.tokens), (1, 0, []))
        n1 = trie.children[0]
        self.assertEqual((n1.n_seqs, n1.n_tokens, n1.tokens), (1, 2, [1, 2]))
        n2 = n1.children[0]
        self.assertEqual((n2.n_seqs, n2.n_tokens, n2.tokens), (1, 2, [3, 4]))
        n3 = n2.children[0]
        self.assertEqual((n3.n_seqs, n3.n_tokens, n3.tokens), (1, 1, [5]))
        self.assertEqual(attn.tails, [n3])

        # root -> n1 -> n2 -> n3
        #                  -> n4
        attn.add_seq(
            tokens=[1, 2, 3, 4, 6],
            k=torch.ones((5, n_heads, d_head)),
            v=torch.ones((5, n_heads, d_head)),
        )
        self.assertEqual((trie.n_seqs, trie.n_tokens, trie.tokens), (2, 0, []))
        n1 = trie.children[0]
        self.assertEqual((n1.n_seqs, n1.n_tokens, n1.tokens), (2, 2, [1, 2]))
        n2 = n1.children[0]
        self.assertEqual((n2.n_seqs, n2.n_tokens, n2.tokens), (2, 2, [3, 4]))
        n3 = n2.children[0]
        self.assertEqual((n3.n_seqs, n3.n_tokens, n3.tokens), (1, 1, [5]))
        n4 = n2.children[1]
        self.assertEqual((n4.n_seqs, n4.n_tokens, n4.tokens), (1, 1, [6]))
        self.assertEqual(attn.tails, [n3, n4])

        # root -> n1 -> n2 -> n3
        #                  -> n4
        #                  -> n5
        attn.add_seq(
            tokens=[1, 2, 3, 4],
            k=torch.ones((4, n_heads, d_head)),
            v=torch.ones((4, n_heads, d_head)),
        )
        self.assertEqual((trie.n_seqs, trie.n_tokens, trie.tokens), (3, 0, []))
        n1 = trie.children[0]
        self.assertEqual((n1.n_seqs, n1.n_tokens, n1.tokens), (3, 2, [1, 2]))
        n2 = n1.children[0]
        self.assertEqual((n2.n_seqs, n2.n_tokens, n2.tokens), (3, 2, [3, 4]))
        n3 = n2.children[0]
        self.assertEqual((n3.n_seqs, n3.n_tokens, n3.tokens), (1, 1, [5]))
        n4 = n2.children[1]
        self.assertEqual((n4.n_seqs, n4.n_tokens, n4.tokens), (1, 1, [6]))
        n5 = n2.children[2]
        self.assertEqual((n5.n_seqs, n5.n_tokens, n5.tokens), (1, 0, []))
        self.assertEqual(attn.tails, [n3, n4, n5])

        # root -> n1 -> n2 -> n3
        #                  -> n4
        #                  -> n5
        #      -> n6 -> n7
        attn.add_seq(
            tokens=[1, 7],
            k=torch.ones((2, n_heads, d_head)),
            v=torch.ones((2, n_heads, d_head)),
        )
        self.assertEqual((trie.n_seqs, trie.n_tokens, trie.tokens), (4, 0, []))
        n1 = trie.children[0]
        self.assertEqual((n1.n_seqs, n1.n_tokens, n1.tokens), (3, 2, [1, 2]))
        n2 = n1.children[0]
        self.assertEqual((n2.n_seqs, n2.n_tokens, n2.tokens), (3, 2, [3, 4]))
        n3 = n2.children[0]
        self.assertEqual((n3.n_seqs, n3.n_tokens, n3.tokens), (1, 1, [5]))
        n4 = n2.children[1]
        self.assertEqual((n4.n_seqs, n4.n_tokens, n4.tokens), (1, 1, [6]))
        n5 = n2.children[2]
        self.assertEqual((n5.n_seqs, n5.n_tokens, n5.tokens), (1, 0, []))
        n6 = trie.children[1]
        self.assertEqual((n6.n_seqs, n6.n_tokens, n6.tokens), (1, 2, [1, 7]))
        n7 = n6.children[0]
        self.assertEqual((n7.n_seqs, n7.n_tokens, n7.tokens), (1, 0, []))
        self.assertEqual(attn.tails, [n3, n4, n5, n7])

        # root -> n1 -> n2 -> n3
        #                  -> n4
        #                  -> n5
        #            -> n8
        #      -> n6 -> n7
        attn.add_seq(
            tokens=[1, 2],
            k=torch.ones((2, n_heads, d_head)),
            v=torch.ones((2, n_heads, d_head)),
        )
        self.assertEqual((trie.n_seqs, trie.n_tokens, trie.tokens), (5, 0, []))
        n1 = trie.children[0]
        self.assertEqual((n1.n_seqs, n1.n_tokens, n1.tokens), (4, 2, [1, 2]))
        n2 = n1.children[0]
        self.assertEqual((n2.n_seqs, n2.n_tokens, n2.tokens), (3, 2, [3, 4]))
        n3 = n2.children[0]
        self.assertEqual((n3.n_seqs, n3.n_tokens, n3.tokens), (1, 1, [5]))
        n4 = n2.children[1]
        self.assertEqual((n4.n_seqs, n4.n_tokens, n4.tokens), (1, 1, [6]))
        n5 = n2.children[2]
        self.assertEqual((n5.n_seqs, n5.n_tokens, n5.tokens), (1, 0, []))
        n8 = n1.children[1]
        self.assertEqual((n8.n_seqs, n8.n_tokens, n8.tokens), (1, 0, []))
        n6 = trie.children[1]
        self.assertEqual((n6.n_seqs, n6.n_tokens, n6.tokens), (1, 2, [1, 7]))
        n7 = n6.children[0]
        self.assertEqual((n7.n_seqs, n7.n_tokens, n7.tokens), (1, 0, []))
        self.assertEqual(attn.tails, [n3, n4, n5, n8, n7])

        # root -> n1 -> n2 -> n3
        #                  -> n4
        #                  -> n5
        #                  -> n9
        #            -> n8
        #      -> n6 -> n7
        attn.add_seq(
            tokens=[1, 2, 3, 4, 5],
            k=torch.ones((5, n_heads, d_head)),
            v=torch.ones((5, n_heads, d_head)),
        )
        self.assertEqual((trie.n_seqs, trie.n_tokens, trie.tokens), (6, 0, []))
        n1 = trie.children[0]
        self.assertEqual((n1.n_seqs, n1.n_tokens, n1.tokens), (5, 2, [1, 2]))
        n2 = n1.children[0]
        self.assertEqual((n2.n_seqs, n2.n_tokens, n2.tokens), (4, 2, [3, 4]))
        n3 = n2.children[0]
        self.assertEqual((n3.n_seqs, n3.n_tokens, n3.tokens), (1, 1, [5]))
        n4 = n2.children[1]
        self.assertEqual((n4.n_seqs, n4.n_tokens, n4.tokens), (1, 1, [6]))
        n5 = n2.children[2]
        self.assertEqual((n5.n_seqs, n5.n_tokens, n5.tokens), (1, 0, []))
        n9 = n2.children[3]
        self.assertEqual((n9.n_seqs, n9.n_tokens, n9.tokens), (1, 1, [5]))
        n8 = n1.children[1]
        self.assertEqual((n8.n_seqs, n8.n_tokens, n8.tokens), (1, 0, []))
        n6 = trie.children[1]
        self.assertEqual((n6.n_seqs, n6.n_tokens, n6.tokens), (1, 2, [1, 7]))
        n7 = n6.children[0]
        self.assertEqual((n7.n_seqs, n7.n_tokens, n7.tokens), (1, 0, []))
        self.assertEqual(attn.tails, [n3, n4, n5, n9, n8, n7])

    def test_append_token(self):
        n_heads, d_head = 2, 4
        attn = Attention(
            n_heads=n_heads,
            d_head=d_head,
            chunk_size=2,
            memory_mb=16,
            dtype=self.dtype,
            device=self.device,
        )
        attn.add_seq(
            tokens=[1, 2, 3, 4, 5],
            k=torch.randn((5, n_heads, d_head)),
            v=torch.randn((5, n_heads, d_head)),
        )
        attn.add_seq(
            tokens=[1, 2, 3, 4, 6, 7],
            k=torch.randn((6, n_heads, d_head)),
            v=torch.randn((6, n_heads, d_head)),
        )
        attn.append_token(
            tokens=[8, 9],
            k=torch.randn((2, n_heads, d_head)),
            v=torch.randn((2, n_heads, d_head)),
        )
        # attn.print()
        trie = attn.get_trie()
        self.assertEqual((trie.n_seqs, trie.n_tokens, trie.tokens), (2, 0, []))
        n1 = trie.children[0]
        self.assertEqual((n1.n_seqs, n1.n_tokens, n1.tokens), (2, 2, [1, 2]))
        n2 = n1.children[0]
        self.assertEqual((n2.n_seqs, n2.n_tokens, n2.tokens), (2, 2, [3, 4]))
        n3 = n2.children[0]
        self.assertEqual((n3.n_seqs, n3.n_tokens, n3.tokens), (1, 2, [5, 8]))
        n4 = n2.children[1]
        self.assertEqual((n4.n_seqs, n4.n_tokens, n4.tokens), (1, 2, [6, 7]))
        n5 = n4.children[0]
        self.assertEqual((n5.n_seqs, n5.n_tokens, n5.tokens), (1, 1, [9]))
        self.assertEqual(attn.tails, [n3, n5])

    def test_forward_simple(self):
        n_heads, d_head = 2, 4
        attn = Attention(
            n_heads=n_heads,
            d_head=d_head,
            chunk_size=2,
            memory_mb=16,
            dtype=self.dtype,
            device=self.device,
        )
        attn.add_seq(
            tokens=[1, 2, 3, 4, 5],
            k=torch.ones((5, n_heads, d_head)),
            v=torch.ones((5, n_heads, d_head)),
        )
        attn.forward(q=torch.ones((1, n_heads, d_head)))

    def test_forward(self):
        n_heads, d_head = 2, 4
        k1, v1 = torch.randn((n_heads, 5, d_head)), torch.randn((n_heads, 5, d_head))
        k2, v2 = torch.randn((n_heads, 6, d_head)), torch.randn((n_heads, 6, d_head))
        k2[:, 0:4, :] = k1[:, 0:4, :]
        v2[:, 0:4, :] = v1[:, 0:4, :]
        q = torch.randn((n_heads, 2, d_head))

        score1 = torch.matmul(q[:, 0:1, :], k1.transpose(-1, -2))
        score1 = torch.softmax(score1 / math.sqrt(d_head), dim=-1)
        output1 = torch.matmul(score1, v1)

        score2 = torch.matmul(q[:, 1:2, :], k2.transpose(-1, -2))
        score2 = torch.softmax(score2 / math.sqrt(d_head), dim=-1)
        output2 = torch.matmul(score2, v2)

        output_ref = torch.cat([output1, output2], dim=1).transpose(0, 1)

        attn = Attention(
            n_heads=n_heads,
            d_head=d_head,
            chunk_size=2,
            memory_mb=16,
            dtype=self.dtype,
            device=self.device,
        )
        attn.add_seq(tokens=[1, 2, 3, 4, 5], k=k1.transpose(0, 1).contiguous(), v=v1.transpose(0, 1).contiguous())
        attn.add_seq(tokens=[1, 2, 3, 4, 6, 7], k=k2.transpose(0, 1).contiguous(), v=v2.transpose(0, 1).contiguous())
        attn.print()
        output = attn.forward(q=q.transpose(0, 1).contiguous())
        
        self.assertTrue(torch.allclose(output_ref, output, atol=1e-3))

    def test_remove(self):
        n_heads, d_head = 2, 4
        attn = Attention(
            n_heads=n_heads,
            d_head=d_head,
            chunk_size=2,
            memory_mb=16,
            dtype=self.dtype,
            device=self.device,
        )

        # root -> n1 -> n2 -> n3
        attn.add_seq(
            tokens=[1, 2, 3, 4, 5],
            k=torch.ones((5, n_heads, d_head)),
            v=torch.ones((5, n_heads, d_head)),
        )

        # root -> n1 -> n2 -> n3
        #                  -> n4
        attn.add_seq(
            tokens=[1, 2, 3, 4, 6],
            k=torch.ones((5, n_heads, d_head)),
            v=torch.ones((5, n_heads, d_head)),
        )
        
        attn.remove_seq(0)
        
        # attn.print()
        trie = attn.get_trie()
        self.assertEqual((trie.n_seqs, trie.n_tokens, trie.tokens), (1, 0, []))
        n1 = trie.children[0]
        self.assertEqual((n1.n_seqs, n1.n_tokens, n1.tokens), (1, 2, [1, 2]))
        n2 = n1.children[0]
        self.assertEqual((n2.n_seqs, n2.n_tokens, n2.tokens), (1, 2, [3, 4]))
        n3 = n2.children[0]
        self.assertEqual((n3.n_seqs, n3.n_tokens, n3.tokens), (1, 1, [6]))
        
    @unittest.skip("skip")
    def test_duplicate(self):
        n_heads, d_head = 2, 4
        attn = Attention(
            n_heads=n_heads,
            d_head=d_head,
            chunk_size=2,
            k=torch.ones((n_heads, 5, d_head)),
            v=torch.ones((n_heads, 5, d_head)),
        )
        attn.add_seq(
            tokens=[1, 2, 3, 4, 5],
            k=torch.randn((n_heads, 5, d_head)),
            v=torch.randn((n_heads, 5, d_head)),
        )
        attn.add_seq(
            tokens=[1, 2, 3, 4, 6, 7],
            k=torch.randn((n_heads, 6, d_head)),
            v=torch.randn((n_heads, 6, d_head)),
        )
        attn.print()
        attn.duplicate(0, 2)
        attn.print()
        attn.duplicate(3, 1)
        attn.print()
        attn.remove_seq(2)
        attn.print()

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
            outputs.append(torch.matmul(score, values[i].transpose(0, 1)).transpose(0, 1))
        output_ref = torch.cat(outputs, dim=0)

        q = torch.cat(qs, dim=0)
        chunks = [32, 64, 128, 256, 512, 1024]
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
