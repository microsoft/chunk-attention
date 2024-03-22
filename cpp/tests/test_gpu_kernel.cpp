#include "kernel_cuda.h"
#include "attention.h"
#include <string>
#include "gtest/gtest.h"
#include <random>


TEST(TestGPUKernel, attn_chunks_first) {
    constexpr uint32_t d_head = 128;
    constexpr uint32_t chunk_size = 64;
    constexpr uint32_t n_heads = 16;
    constexpr uint32_t n_seqs = 15;
    std::vector<GPT::ChunkInfo> chunk_infos;
    auto fp16_options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16);
    constexpr uint32_t num_chunks = 256;
    for (int i = 0; i < num_chunks; i++) {
        auto key = torch::randn({ n_heads, chunk_size, d_head }, fp16_options);
        auto value = torch::randn({ n_heads, chunk_size, d_head }, fp16_options);
        GPT::Chunk* chunk = new GPT::Chunk(key, value);
        GPT::ChunkInfo chunk_info(chunk, 0, n_seqs);
        chunk_infos.push_back(chunk_info);
    }
    GPT::KernelContext kernel(n_heads, d_head, chunk_size, fp16_options);

    torch::randn({ 4, 4 }).to(torch::kCUDA, true);
    auto start_t1 = std::chrono::high_resolution_clock::now();
    auto& kernel_context = refresh_kernel_context(kernel, chunk_infos, n_seqs);
    auto end_t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_t1 - start_t1);
    std::cout << "gen_kernel_context: " << duration.count() << " us" << std::endl;

    torch::Tensor query = torch::randn({ n_seqs, n_heads, d_head }, fp16_options);
    attn_chunks_first(kernel_context, query);
    int sum = 0;
    torch::Tensor begins_ends = kernel_context.begins_ends_offsets.to(torch::kCPU);
    for (int i = 0; i < num_chunks; i++) {
        torch::Tensor score =
          torch::matmul(query.transpose(0, 1).to(torch::kFloat32),
                        chunk_infos[i].chunk->key().to(torch::kFloat32).transpose(1, 2)) /
          std::sqrt(d_head);
        torch::Tensor weight_max = std::get<0>(score.max(2));
        torch::Tensor weight_exp = torch::exp((score - weight_max.unsqueeze(2)));
        torch::Tensor weight_sum = weight_exp.sum(2);
        torch::Tensor output =
          torch::matmul(weight_exp.to(torch::kFloat16), chunk_infos[i].chunk->value());
        int n = begins_ends[1][i].item<int>() - begins_ends[0][i].item<int>();
        ASSERT_TRUE(torch::allclose(
          weight_max,
          kernel_context.maxs_sums[0].slice(0, sum, sum + n).reshape({ n_heads, n }),
          1e-3,
          1e-3));
        ASSERT_TRUE(torch::allclose(
          weight_sum,
          kernel_context.maxs_sums[1].slice(0, sum, sum + n).reshape({ n_heads, n }),
          1e-3,
          1e-3));
        ASSERT_TRUE(torch::allclose(
          output,
          kernel_context.attns.slice(0, sum, sum + n).reshape({ n_heads, n, d_head }),
          1e-2,
          1e-2));
        sum += n;
    }
}

//TEST(TestGPUKernel, attn_seq_first) {
//    constexpr uint32_t d_head = 128;
//    constexpr uint32_t chunk_size = 64;
//    constexpr uint32_t n_heads = 16;
//    constexpr uint32_t n_seqs = 64;
//
//    constexpr uint32_t n_shared_chunks = 5;
//    constexpr uint32_t n_unshared_chunks = 5;
//
//    auto fp16_options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16);
//    std::vector<GPT::ChunkInfo> chunk_infos;
//    for (int i = 0; i < n_shared_chunks; i++) {
//        auto key = torch::randn({ n_heads, chunk_size, d_head }, fp16_options);
//        auto value = torch::randn({ n_heads, chunk_size, d_head }, fp16_options);
//        GPT::Chunk* chunk =
//          new GPT::Chunk(chunk_size, n_heads, d_head, key, value, 0, chunk_size, fp16_options);
//        GPT::ChunkInfo chunk_info(chunk, 0, n_seqs);
//        chunk_infos.push_back(chunk_info);
//    }
//    for (int i = 0; i < n_unshared_chunks; i++) {
//        for (int j = 0; j < n_seqs; j++) {
//            auto key = torch::randn({ n_heads, chunk_size, d_head }, fp16_options);
//            auto value = torch::randn({ n_heads, chunk_size, d_head }, fp16_options);
//            GPT::Chunk* chunk =
//              new GPT::Chunk(chunk_size, n_heads, d_head, key, value, 0, chunk_size, fp16_options);
//            GPT::ChunkInfo chunk_info(chunk, j, j + 1);
//            chunk_infos.push_back(chunk_info);
//        }
//    }
//    GPT::GPUKernel kernel(n_heads, d_head, chunk_size, fp16_options);
//
//    torch::randn({ 4, 4 }).to(torch::kCUDA, true);
//    cudaDeviceSynchronize();
//    auto start_t1 = std::chrono::high_resolution_clock::now();
//    auto& kernel_context = kernel.gen_kernel_context(chunk_infos, n_seqs);
//    cudaDeviceSynchronize();
//
//    //std::cout << "attns: " << kernel_context.attns.sizes() << std::endl;
//    //std::cout << "begins_ends: " << kernel_context.begins_ends.sizes() << std::endl;
//    //std::cout << "offsets: \n" << kernel_context.offsets << std::endl;
//    //std::cout << "maxs_sums: " << kernel_context.maxs_sums.sizes() << std::endl;
//    //std::cout << "maxs_sums.strides: " << kernel_context.maxs_sums.strides() << std::endl;
//    //std::cout << "n_shared_chunks: " << kernel_context.n_shared_chunks << std::endl;
//    //std::cout << "seq_chunk_map: " << kernel_context.seq_chunk_map.sizes() << std::endl;
//    //std::cout << "seq_n_tokens: " << kernel_context.seq_n_tokens.sizes() << std::endl;
//
//    auto end_t1 = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_t1 - start_t1);
//    std::cout << "gen_kernel_context: " << duration.count() << " us" << std::endl;
//
//    torch::Tensor query = torch::randn({ n_seqs, n_heads, d_head }, fp16_options);
//    torch::Tensor output = torch::empty({ n_seqs, n_heads, d_head }, fp16_options);
//    kernel.attn_chunks_first(query, kernel_context);
//    kernel.attn_seqs_first(query, output, kernel_context);
//
//    for (int i = 0; i < n_seqs; i++) {
//        std::vector<torch::Tensor> chunked_keys;
//        std::vector<torch::Tensor> chunked_values;
//        for (auto& chunk_info: chunk_infos) {
//            if (i >= chunk_info.seq_idx_begin && i < chunk_info.seq_idx_end) {
//                chunked_keys.push_back(chunk_info.chunk->key());
//                chunked_values.push_back(chunk_info.chunk->value());
//            }
//        }
//        torch::Tensor single_q = query.slice(0, i, i + 1);
//        torch::Tensor key = torch::cat(chunked_keys, 1);
//        torch::Tensor value = torch::cat(chunked_values, 1);
//        torch::Tensor qk =
//          torch::matmul(single_q.transpose(0, 1), key.transpose(1, 2)) / std::sqrt(d_head);
//        torch::Tensor score = torch::softmax(qk.to(torch::kFloat32), 2);
//        torch::Tensor expect_output = torch::matmul(score.to(torch::kFloat16), value).transpose(0, 1);
//        ASSERT_TRUE(torch::allclose(expect_output, output.slice(0, i, i + 1), 1e-3, 1e-3));
//    }
//}


TEST(TestGPUKernel, forward_test) {
    int n_tokens = 1024;
    int n_requests = 48;
    int n_heads = 32;
    int d_head = 128;
    int chunk_size = 64;
    int n_decode_steps = 512;

    auto fp16_options = at::device(at::Device(c10::DeviceType::CUDA)).dtype(torch::kFloat16);

    std::vector<int> tokens(n_tokens);
    std::iota(std::begin(tokens), std::end(tokens), 0);
    auto rng = std::default_random_engine{};

    GPT::Attention attn(n_heads, d_head, chunk_size, 2048, true, 1, torch::kFloat16, torch::kCUDA);

    for (int i = 0; i < n_requests; ++i) {
        std::shuffle(std::begin(tokens), std::end(tokens), rng);
        torch::Tensor k = torch::randn({ n_tokens, n_heads, d_head }, fp16_options);
        torch::Tensor v = torch::randn({ n_tokens, n_heads, d_head }, fp16_options);
        attn.add_seq(tokens, k, v);
    }

    torch::Tensor q = torch::randn({ n_requests, n_heads, d_head }, fp16_options);
    for (int i = 0; i < n_decode_steps; ++i) {
        attn.forward(q);
    }
}
