#pragma once

#include <memory>
#include <set>
#include "chunk.h"

namespace GPT {

class ChunkAllocator {
  public:
    ChunkAllocator(int memory_mb,
                   int chunk_size,
                   int n_heads,
                   int d_head,
                   torch::TensorOptions& options);
    ChunkAllocator(ChunkAllocator const& other) = delete;
    virtual ~ChunkAllocator() = default;

    Chunk* allocate();
    Chunk* allocate(Chunk& other);
    Chunk* allocate(const std::vector<int>& ids, torch::Tensor& k, torch::Tensor& v, int start, int end);
    Chunk* allocate(torch::Tensor& k, torch::Tensor& v, int start, int end);
    void free(Chunk* chunk);

    bool full() const;
    int64_t peak_memory_allocated() const { return peak_memory_allocated_; }

  private:
    int memory_mb_;
    int chunk_size_;
    int n_heads_;
    int d_head_;
    torch::TensorOptions t_options_;

    torch::Tensor key_storage_;
    torch::Tensor value_storage_;
    std::list<std::shared_ptr<Chunk>> chunks_; // all reserved chunks
    std::set<Chunk*> free_set_;

    int max_chunks_;
    int64_t chunk_kv_bytes_;
    int64_t chunk_total_bytes_;
    int64_t peak_memory_allocated_;
};

}