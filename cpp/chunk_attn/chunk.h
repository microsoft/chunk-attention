#pragma once

#include <memory>
#include <torch/torch.h>
#include "str_utils.h"

namespace GPT {

class Chunk {
  public:
    Chunk(int capacity, int n_heads, int d_head, int n_layers, torch::TensorOptions& options);
    Chunk(torch::Tensor& key_storage, torch::Tensor& value_storage);
    Chunk(Chunk const& other);
    virtual ~Chunk() = default;

    int capacity() { return key_.size(1); }
    inline int n_tokens() { return n_tokens_; }

    bool equal(Chunk& other);
    bool equal(const std::vector<int>& ids, int start, int end);

    bool full() { return tokens.size() == key_.size(1); }

    torch::Tensor& key() { return key_; }
    torch::Tensor& value() { return value_; }

    inline void* key_ptr() { return key_ptr_; }
    inline void* value_ptr() { return value_ptr_; }

    void append_tokens(const std::vector<int>& ids,
                       torch::Tensor& k,
                       torch::Tensor& v,
                       int start,
                       int end);
    void append_tokens(torch::Tensor& k, torch::Tensor& v, int start, int end);
    void append_tokens(const std::vector<int>& ids, int start, int end);
    void deep_copy(Chunk const& other);
    void clear() {
        n_tokens_ = 0;
        tokens.clear();
        n_seqs = 1;
        children.clear();
        parent = nullptr;
    }

    Chunk* add_child(Chunk* child);
    Chunk* insert_child(int idx, Chunk* child);

    std::string to_string(bool brief = false);

  public:
    std::vector<int> tokens;
    // Non-owning pointers. The ownership is managed by ChunkAllocator.
    std::vector<Chunk*> children;
    int n_seqs;
    Chunk* parent;

  private:
    void init(torch::Tensor& key_storage, torch::Tensor& value_storage);
    torch::Tensor key_;
    void* key_ptr_;
    torch::Tensor value_;
    void* value_ptr_;
    int n_tokens_;
};

}
