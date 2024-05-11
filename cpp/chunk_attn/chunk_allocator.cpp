#include "chunk_allocator.h"
#include <utility>
#include "str_utils.h"
#include "logging.h"

namespace GPT {

ChunkAllocator::ChunkAllocator(int memory_mb,
                               int chunk_size,
                               int n_heads,
                               int d_head,
                               int n_layers,
                               torch::TensorOptions& options)
  : memory_mb_(memory_mb)
  , chunk_size_(chunk_size)
  , n_heads_(n_heads)
  , d_head_(d_head)
  , n_layers_(n_layers)
  , t_options_(options)
  , peak_memory_allocated_(0) {
    float size_of_data = 4;
    if (options.dtype() == torch::kFloat64) {
        size_of_data = 8;
    } else if (options.dtype() == torch::kFloat32 || options.dtype() == torch::kFloat) {
        size_of_data = 4;
    } else if (options.dtype() == torch::kFloat16) {
        size_of_data = 2;
    } else if (options.dtype() == torch::kInt8) {
        size_of_data = 1;
    } else {
        throw std::runtime_error(fmt_str("unsupported dtype {}", options.dtype()));
    }

    int64_t chunk_kv_elements = chunk_size_ * n_heads_ * d_head_;
    chunk_kv_bytes_ = chunk_kv_elements * size_of_data;
    chunk_total_bytes_ = chunk_kv_bytes_ * 2; // 2 for key and value
    max_chunks_ = std::floor(float(memory_mb_) * 1024 * 1024 / chunk_total_bytes_);
    LOG_INFO("ChunkAllocator: memory capacity {} MB, each chunk requires {} KB, reserved chunks {}",
             memory_mb,
             chunk_total_bytes_ / 1024,
             max_chunks_);

    // pre allocate chunks
    for (int i = 0; i < max_chunks_; ++i) {
        auto chunk = std::make_shared<Chunk>(chunk_size_, n_heads_, d_head_, n_layers_, t_options_);
        chunks_.push_back(chunk);
        free_set_.insert(chunk.get());
    }
}

Chunk* ChunkAllocator::allocate() {
    if (free_set_.size() > 0) {
        auto ite = free_set_.begin();
        free_set_.erase(ite);
        int64_t used_memory = int64_t(chunks_.size() - free_set_.size()) * chunk_total_bytes_;
        if (used_memory > peak_memory_allocated_) {
            peak_memory_allocated_ = used_memory;
        }
        return *ite;
    }

    if (full()) {
        LOG_ERROR("ChunkAllocator: free chunks {}, allocated chunks {}, reserved chunks {}",
                  free_set_.size(),
                  chunks_.size() - free_set_.size(),
                  max_chunks_);
        throw std::runtime_error("ChunkAllocator: capacity reached");
    } else {
        auto chunk = std::make_shared<Chunk>(chunk_size_, n_heads_, d_head_, t_options_);
        chunks_.push_back(chunk);
        int64_t used_memory = int64_t(chunks_.size() - free_set_.size()) * chunk_total_bytes_;
        if (used_memory > peak_memory_allocated_) {
            peak_memory_allocated_ = used_memory;
        }
        return chunk.get();
    }
}

Chunk* ChunkAllocator::allocate(Chunk& other) {
    Chunk* chunk = allocate();
    chunk->deep_copy(other);
    return chunk;
}

Chunk* ChunkAllocator::allocate(const std::vector<int>& ids,
                                int start,
                                int end) {
    Chunk* chunk = allocate();
    chunk->append_tokens(ids, start, end);
    return chunk;
}

Chunk* ChunkAllocator::allocate(torch::Tensor& k, torch::Tensor& v, int start, int end) {
    Chunk* chunk = allocate();
    chunk->append_tokens(k, v, start, end);
    return chunk;
}

void ChunkAllocator::free(Chunk* chunk) {
    auto ite = free_set_.find(chunk);
    if (ite != free_set_.end()) {
        throw std::runtime_error("free is called, but chunk is in free list");
    }
    chunk->clear();
    free_set_.insert(chunk);
}

bool ChunkAllocator::full() const {
    return chunks_.size() >= max_chunks_;
}

} // namespace GPT
