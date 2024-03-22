#pragma once

#include "chunk_info.h"
#include "logging.h"

namespace GPT {

struct KernelContext {
    KernelContext(int n_heads, int d_head, int chunk_size, torch::TensorOptions& options) {
        this->n_heads = n_heads;
        this->d_head = d_head;
        this->chunk_size = chunk_size;
        this->options = options;

        this->tpp_threshold = 1;

        valid = false;
        delta_tokens = 0;

        if (!options.device().is_cpu()) {
            throw std::runtime_error("CPUKernel is enabled. But the device is not CPU.");
        }
        int cpu_num_threads = torch::get_num_threads();
        LOG_DEBUG("CPUKernel: Set omp num of threads to {}", cpu_num_threads);
        omp_set_dynamic(0); // Explicitly disable dynamic teams
        omp_set_num_threads(cpu_num_threads);
    };
    KernelContext(KernelContext&) = default;

    int tpp_threshold;

    int n_heads;
    int d_head;
    int chunk_size;
    torch::TensorOptions options;
    int delta_tokens;

    bool valid;
    std::vector<ChunkInfo> chunk_infos;
};

void attention(KernelContext& context, torch::Tensor& q, torch::Tensor& output, int partition);

KernelContext& refresh_kernel_context(KernelContext& context,
                                      std::vector<ChunkInfo>& chunk_infos,
                                      int n_seqs);
} // namespace GPT