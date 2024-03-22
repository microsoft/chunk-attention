#pragma once

#include "chunk.h"
#include "chunk_allocator.h"
#include "chunk_info.h"
#include "task_executor.h"
# ifdef USE_CUDA
#include "kernel_cuda.h"
# endif
#ifdef  USE_MKL
#include "kernel_cpu_mkl.h"
#endif

namespace GPT {

class Attention {
  public:
    Attention(int n_heads,
              int d_head,
              int chunk_size,
              int memory_mb = 2048,
              bool share_prefix = true,
              int tpp_threshold = 1,
              torch::Dtype dtype = torch::kFloat16,
              torch::Device device = torch::Device(torch::kCUDA));
    Attention(Attention&);
    virtual ~Attention() = default; 
    void operator=(Attention const& x) = delete;

    int add_seq(const std::vector<int>& tokens, torch::Tensor k, torch::Tensor v);
    std::shared_ptr<std::future<int>> add_seq_async(const std::vector<int>& tokens,
                                                    torch::Tensor k,
                                                    torch::Tensor v);
    void remove_seq(int seq_idx);
    // append tokens for all sequences, one token per sequence.
    std::vector<Chunk*> append_token(const std::vector<int>& tokens, torch::Tensor k, torch::Tensor v, bool fused=true);
    void refresh_kernel_context(bool force=false);
    std::vector<Chunk*> reserve();
    void clear();
    int64_t peak_memory_allocated() const { return allocator_->peak_memory_allocated(); }

    void duplicate(int seq_idx, int copies);
    // duplicate a seq and remove a seq in beam search
    int append_completion(int seq_idx, std::vector<int>& tokens, torch::Tensor k, torch::Tensor v);

    // partation 0: auto(chunk_seq), 1: chunk first, 2: sequence first
    torch::Tensor forward(torch::Tensor q, int partation = 0);

    Chunk* get_trie() { return root_.get(); }
    std::vector<Chunk*> tails() { return tails_; }
    void print(Chunk* root = nullptr, int level = 0);
    std::vector<ChunkInfo> get_chunks();

  private:
    Chunk* at(int seq_idx);

  private:
    int n_heads_;
    int d_head_;
    int chunk_size_;
    bool share_prefix_;
    std::shared_ptr<Chunk> root_;
    std::vector<Chunk*> tails_;
    torch::TensorOptions t_options_;
    int memory_mb_;
    std::shared_ptr<ChunkAllocator> allocator_;

#ifdef USE_MKL
    KernelContext kernel_ctx_;
#endif

#ifdef USE_CUDA
    KernelContext kernel_ctx_;
#endif
};

}