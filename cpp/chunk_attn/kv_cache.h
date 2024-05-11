#pragma once

#include "chunk.h"
#include "chunk_allocator.h"
#include "chunk_info.h"

namespace GPT {

class KVCache {
  public:
    KVCache(int n_heads,
            int d_head,
            int chunk_size,
            int n_layers,
            int memory_mb = 2048,
            bool share_prefix = true,
            torch::Dtype dtype = torch::kFloat16,
            torch::Device device = torch::Device(torch::kCUDA));
    KVCache(KVCache&) = delete;
    virtual ~KVCache() = default;
    void operator=(KVCache const& x) = delete;

    int add_seq(const std::vector<int>& tokens);
    void remove_seq(int seq_idx);
    // append tokens for all sequences, one token per sequence.
    std::vector<Chunk*> append_token(const std::vector<int>& tokens);

    // k/v: [n_tokens, n_heads, d_head]
    int assign_kv(int seq_idx, int layer, torch::Tensor k, torch::Tensor v);
    // append tokens for all sequences, one token per sequence.
    std::vector<Chunk*> assign_kv_last(int seq_idx,
                                       int layer,
                                       torch::Tensor k,
                                       torch::Tensor v,
                                       bool fused = true);

    std::vector<Chunk*> reserve();
    void clear();
    int64_t peak_memory_allocated() const { return allocator_->peak_memory_allocated(); }

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
    int n_layers_;
    bool share_prefix_;
    std::shared_ptr<Chunk> root_;
    std::vector<Chunk*> tails_;
    torch::TensorOptions t_options_;
    int memory_mb_;
    std::shared_ptr<ChunkAllocator> allocator_;

    bool changed_ = false;
};

}