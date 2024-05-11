#include "kv_cache.h"
#include "logging.h"
#include <limits>
#include <stack>

namespace GPT {

KVCache::KVCache(int n_heads,
                 int d_head,
                 int chunk_size,
                 int n_layers,
                 int memory_mb,
                 bool share_prefix,
                 torch::Dtype dtype,
                 torch::Device device)
  : n_heads_(n_heads)
  , d_head_(d_head)
  , chunk_size_(chunk_size)
  , n_layers_(n_layers)
  , share_prefix_(share_prefix)
  , t_options_(torch::TensorOptions()
                 .dtype(dtype)
                 .layout(torch::kStrided)
                 .device(device)
                 .requires_grad(false))
  , memory_mb_(memory_mb) {
    root_ = std::make_shared<Chunk>(0, n_heads, d_head, t_options_);
    root_->n_seqs = 0;

    if (!t_options_.device().is_cpu() && !t_options_.device().is_cuda()) {
        throw std::runtime_error("unsupported device");
    }

    allocator_ =
      std::make_shared<ChunkAllocator>(memory_mb, chunk_size, n_heads, d_head, t_options_);
}

// k/v: [n_tokens, n_heads, d_head]
int KVCache::add_seq(const std::vector<int>& tokens) {
    int seq_idx = 0;
    int total_tokens = tokens.size();
    Chunk* prev = root_.get();
    int start = 0;

    // skip the common prefix
    if (share_prefix_) {
        while (start < total_tokens) {
            prev->n_seqs += 1;
            start += prev->n_tokens();
            bool found = false;
            for (auto ite = prev->children.begin(); ite != prev->children.end(); ite++) {
                Chunk* child = *ite;
                if (child->children.size() != 0 && // don't share with leaf
                    child->equal(tokens, start, start + child->n_tokens())) {
                    prev = child;
                    found = true;
                    break;
                }
                seq_idx += child->n_seqs;
            }

            if (!found) {
                break;
            }
        }
    } else {
        prev->n_seqs += 1;
    }

    for (; start < total_tokens; start += chunk_size_) {
        int end = std::min(start + chunk_size_, total_tokens);
        auto child = allocator_->allocate(tokens, start, end);
        prev = prev->add_child(child);
    }

    // make sure full chunks are not the tails.
    // reason: all non-tail chunks are possible to be shared by other sequences.
    // each sequence needs a unique chunk to distinguish itself from others.
    if (prev->full()) {
        auto child = allocator_->allocate();
        prev = prev->add_child(child);
    }

    tails_.insert(tails_.begin() + seq_idx, prev);
    kernel_ctx_.valid = false;
    return seq_idx;
}

std::vector<Chunk*> KVCache::append_token(const std::vector<int>& tokens) {
    if (tails_.size() != tokens.size()) {
        throw std::runtime_error("num of seqs is not equal to num of tokens");
    }

    std::vector<Chunk*> new_chunks = reserve();

    for (auto ite = 0; ite < tails_.size(); ite++) {
        Chunk* tail = tails_[ite];
        tail->append_tokens(tokens, ite, ite + 1);
    }

    kernel_ctx_.delta_tokens += 1;
    return new_chunks;
}

std::vector<Chunk*> KVCache::reserve() {
    std::vector<Chunk*> new_chunks;
    for (auto ite = 0; ite < tails_.size(); ite++) {
        Chunk* tail = tails_[ite];
        if (tail->full()) {
            auto child = allocator_->allocate();
            new_chunks.push_back(child);
            tail = tail->add_child(child);
            tails_[ite] = tail;
        }
    }
    if (new_chunks.size() > 0) {
        kernel_ctx_.valid = false;
    }
    return new_chunks;
}

Chunk* KVCache::at(int seq_idx) {
    Chunk* chunk = root_.get();
    int start = 0;
    while (chunk->children.size() > 0) {
        for (auto& child : chunk->children) {
            int end = start + child->n_seqs;
            if (start <= seq_idx && seq_idx < end) {
                chunk = child;
                break;
            }
            start = end;
        }
    }
    return chunk;
}

void KVCache::remove_seq(int seq_idx) {
    Chunk* seq_tail = tails_[seq_idx];
    while (seq_tail != nullptr) {
        seq_tail->n_seqs -= 1;
        Chunk* parent = seq_tail->parent;
        if (seq_tail->n_seqs <= 0 && parent != nullptr) {
            parent->children.erase(
              std::remove(parent->children.begin(), parent->children.end(), seq_tail),
              parent->children.end());
            allocator_->free(seq_tail);
        }
        seq_tail = parent;
    }
    tails_.erase(tails_.begin() + seq_idx);
    kernel_ctx_.valid = false;
}

void KVCache::clear() {
    int n_seqs = tails_.size();
    for (auto i = 0; i < n_seqs; i++) {
        remove_seq(i);
    }
}

/*
def print_tree(node, level=0):
    indent = "  " * level
    print(indent + node.data)

    for child in node.children:
        print_tree(child, level + 1)

*/
void KVCache::print(Chunk* root, int level) {
    if (root == nullptr) {
        root = root_.get();
    }

    std::string indent;
    for (int i = 0; i < level; ++i) {
        indent += "    ";
    }
    std::cout << indent;

    std::cout << root->to_string(true) << std::endl;

    for (auto child : root->children) {
        print(child, level + 1);
    }
}

std::vector<ChunkInfo> KVCache::get_chunks() {
    std::vector<ChunkInfo> chunk_infos;
    chunk_infos.reserve(8192);
    std::vector<std::tuple<int, Chunk*>> stack;
    stack.reserve(8192);

    stack.emplace_back(0, root_.get());
    while (!stack.empty()) {
        auto& item = stack.back();
        int seq_begin = std::get<0>(item);
        auto chunk = std::get<1>(item);
        int seq_end = seq_begin + chunk->n_seqs;
        stack.pop_back();

        int start = seq_begin;
        for (auto& child : chunk->children) {
            stack.emplace_back(start, child);
            start += child->n_seqs;
        }

        // note: don't skip empty chunks (chunk->n_tokens() == 0),
        // they are reserved to avoid next append_kv call triggerring tree change.
        if (chunk == root_.get()) {
            continue;
        }

        chunk_infos.emplace_back(chunk, seq_begin, seq_end);
    }

    return chunk_infos;
}

} // namespace GPT