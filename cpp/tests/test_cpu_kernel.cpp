#include <algorithm>
#include <random>

#include "gtest/gtest.h"
#include "attention.h"

TEST(TestPerf, Basic) {
    // torch::set_num_threads(1);

    int n_tokens = 1024;
    int n_seqs = 32;
    int n_heads = 32;
    int d_head = 128;
    int chunk_size = 16;
    int n_decode_steps = 512;

    std::vector<int> tokens(n_tokens);
    std::iota(std::begin(tokens), std::end(tokens), 0);
    auto rng = std::default_random_engine{};

    for (int i = 0; i < 8; i++) {
        GPT::Attention attn(
          n_heads, d_head, chunk_size, 2048, true, 1, torch::kFloat32, torch::Device(torch::kCPU));
        for (int i = 0; i < n_seqs; ++i) {
            std::shuffle(std::begin(tokens), std::end(tokens), rng);
            torch::Tensor k = torch::rand({ n_heads, n_tokens, d_head });
            torch::Tensor v = torch::rand({ n_heads, n_tokens, d_head });
            attn.add_seq(tokens, k, v);
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto chunk_infos = attn.get_chunks();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "chunk_infos.size(): " << chunk_infos.size() << " "
                  << "get_chunks(us): " << duration.count() << std::endl;
    }

    /*
    std::vector<int> new_tokens(n_seqs);
    std::iota(std::begin(new_tokens), std::end(new_tokens), 0);
    torch::Tensor new_k = torch::rand({ n_heads, n_seqs, d_head });
    torch::Tensor new_v = torch::rand({ n_heads, n_seqs, d_head });

    torch::Tensor q = torch::rand({ n_heads, n_seqs, d_head });
    for (int i = 0; i < n_decode_steps; ++i) {
        attn.forward(q);
        //attn.append_token(new_tokens, new_k, new_v);
    }
    */
}