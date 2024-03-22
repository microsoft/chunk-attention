#pragma once

#include <cstdint>
#include "chunk_info.h"
#include "logging.h"

namespace GPT {

struct KernelContext {
    KernelContext(int n_heads, int d_head, int chunk_size, torch::TensorOptions& options) {
        this->n_heads = n_heads;
        this->d_head = d_head;
        this->chunk_size = chunk_size;
        this->options = options;

        valid = false;
        delta_tokens = 0;

        if (!options.device().is_cuda()) {
            throw std::runtime_error("GPUKernel is enabled. But the device is not CUDA.");
        }

        LOG_INFO("GPUKernel: Runs on cuda:{}", options.device().index());

        const int init_seq_num = 128;
        const int init_chunk_num = 128;

        attns = torch::empty({ init_seq_num * init_chunk_num, n_heads, d_head }, options);
        maxs_sums = torch::empty({ 2, init_seq_num * init_chunk_num, n_heads },
                                 options.dtype(torch::kFloat32));
    };

    KernelContext(KernelContext& other) = default;

    int n_heads;
    int d_head;
    int chunk_size;
    torch::TensorOptions options;

    int tpp_threshold;

    bool valid;
    int n_chunks;
    int n_shared_chunks;
    int delta_tokens;

    torch::Tensor keys_values;
    torch::Tensor begins_ends_offsets;
    torch::Tensor seq_chunk_map;
    torch::Tensor seq_n_tokens;

    // temp result storage
    torch::Tensor attns; // [chunk_num, n_seqs*num_chunks, d_head]
    torch::Tensor maxs_sums;
};

void attention(KernelContext& context,
               torch::Tensor q,
               torch::Tensor& output,
               int partition = 0 // 0: auto, 1: chunk first, 2: sequence first
);

void attn_chunks_first(KernelContext& context, torch::Tensor& query);

void attn_seqs_first(KernelContext& context, const torch::Tensor& query, torch::Tensor& output);

void append_kv(KernelContext& context, const torch::Tensor& keys, const torch::Tensor& values);

KernelContext& refresh_kernel_context(KernelContext& context,
                                      std::vector<ChunkInfo>& chunk_infos,
                                      int n_seqs);

} // namespace GPT