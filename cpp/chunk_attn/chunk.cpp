#include <utility>
#include "chunk.h"

namespace GPT {

Chunk::Chunk(int capacity, int n_heads, int d_head, int n_layers, torch::TensorOptions& options) {
    key_ = torch::zeros({ n_heads, capacity, d_head }, options);
    value_ = torch::zeros({ n_heads, capacity, d_head }, options);
    init(key_, value_);
}

Chunk::Chunk(torch::Tensor& key_storage, torch::Tensor& value_storage) {
    init(key_storage, value_storage);
}

void Chunk::init(torch::Tensor& key_storage, torch::Tensor& value_storage) {
    assert(key_storage.size(0) == value_storage.size(0) &&
           key_storage.size(1) == value_storage.size(1) &&
           key_storage.size(2) == value_storage.size(2));
    int capacity = key_storage.size(1);
    n_seqs = 1;
    tokens.reserve(capacity);
    n_tokens_ = 0;
    parent = nullptr;
    key_ = key_storage;
    key_ptr_ = key_.data_ptr();
    value_ = value_storage;
    value_ptr_ = value_.data_ptr();
}

Chunk::Chunk(Chunk const& other) {
    deep_copy(other);
}

bool Chunk::equal(Chunk& other) {
    if (this->n_tokens() != other.n_tokens()) {
        return false;
    }

    for (int i = 0; i < this->n_tokens(); i++) {
        if (this->tokens[i] != other.tokens[i]) {
            return false;
        }
    }

    return true;
}

// start: inclusive, end: exclusive
bool Chunk::equal(const std::vector<int>& ids, int start, int end) {
    int n = end - start;
    if (this->n_tokens() != n || end > ids.size()) {
        return false;
    }
    for (int i = 0; i < n; i++) {
        if (this->tokens[i] != ids[start + i]) {
            return false;
        }
    }
    return true;
}

void Chunk::append_tokens(torch::Tensor& k, torch::Tensor& v, int start, int end) {
    int n = end - start;
    if (n > capacity() - n_tokens()) {
        throw std::runtime_error("no enough space in Chunk");
    }

    key_.slice(1, n_tokens(), n_tokens() + n) = k.slice(1, start, end);
    value_.slice(1, n_tokens(), n_tokens() + n) = v.slice(1, start, end);
    n_tokens_ += n;
}

void Chunk::append_tokens(const std::vector<int>& ids,
                          torch::Tensor& k,
                          torch::Tensor& v,
                          int start,
                          int end) {
    append_tokens(k, v, start, end);
    tokens.insert(tokens.end(), ids.begin() + start, ids.begin() + end);
}

void Chunk::append_tokens(const std::vector<int>& ids, int start, int end) {
    int n = end - start;
    if (n > capacity() - n_tokens()) {
        throw std::runtime_error("no enough space in Chunk");
    }

    tokens.insert(tokens.end(), ids.begin() + start, ids.begin() + end);
    n_tokens_ += n;
}

void Chunk::deep_copy(Chunk const& other) {
    if (this == &other) {
        return;
    }

    if (other.children.size() > 0) {
        throw std::runtime_error("Copy a non tailing chunk is forbidden");
    }

    n_seqs = 1;
    tokens = other.tokens;
    n_tokens_ = other.n_tokens_;
    key_.copy_(other.key_);
    value_.copy_(other.value_.clone());
}

Chunk* Chunk::add_child(Chunk* child) {
    this->children.push_back(child);
    child->parent = this;
    return child;
}

Chunk* Chunk::insert_child(int idx, Chunk* child) {
    std::vector<Chunk*>::iterator it = this->children.begin() + idx;
    this->children.insert(it, child);
    child->parent = this;
    return child;
}

std::string Chunk::to_string(bool brief) {
    std::vector<std::string> string_tokens;
    if (brief && tokens.size() > 3) {
        string_tokens.push_back(std::to_string(tokens[0]));
        string_tokens.push_back("...");
        string_tokens.push_back(std::to_string(tokens[tokens.size() - 1]));
    } else {
        std::transform(tokens.begin(),
                       tokens.end(),
                       std::back_inserter(string_tokens),
                       [](int num) { return std::to_string(num); });
    }
    std::string s = fmt_str("%d: [%s]", n_seqs, join_str(string_tokens, ",").c_str());
    return s;
}

} // namespace GPT