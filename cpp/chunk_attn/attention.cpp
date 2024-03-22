#include <limits>
#include <stack>
#include "attention.h"
#include "logging.h"

namespace GPT {

Attention::Attention(int n_heads,
                     int d_head,
                     int chunk_size,
                     int memory_mb,
                     bool share_prefix,
                     int tpp_threshold,
                     torch::Dtype dtype,
                     torch::Device device)
  : n_heads_(n_heads)
  , d_head_(d_head)
  , chunk_size_(chunk_size)
  , share_prefix_(share_prefix)
  , t_options_(torch::TensorOptions()
                 .dtype(dtype)
                 .layout(torch::kStrided)
                 .device(device)
                 .requires_grad(false))
  , memory_mb_(memory_mb)
  , kernel_ctx_(n_heads, d_head, chunk_size, t_options_) {
    kernel_ctx_.tpp_threshold = tpp_threshold;
    root_ = std::make_shared<Chunk>(0, n_heads, d_head, t_options_);
    root_->n_seqs = 0;

    if (!t_options_.device().is_cpu() && !t_options_.device().is_cuda()) {
        throw std::runtime_error("unsupported device");
    }

    allocator_ =
      std::make_shared<ChunkAllocator>(memory_mb, chunk_size, n_heads, d_head, t_options_);
}

Attention::Attention(Attention& other): kernel_ctx_(other.kernel_ctx_) {
    n_heads_ = other.n_heads_;
    d_head_ = other.d_head_;
    chunk_size_ = other.chunk_size_;
    share_prefix_ = other.share_prefix_;
    t_options_ = other.t_options_;
    allocator_ = std::make_shared<ChunkAllocator>(
      other.memory_mb_, other.chunk_size_, other.n_heads_, other.d_head_, other.t_options_);

    root_ = std::make_shared<Chunk>(0, kernel_ctx_.n_heads, kernel_ctx_.d_head, t_options_);
    root_->n_seqs = 0;

    if (!t_options_.device().is_cpu() && !t_options_.device().is_cuda()) {
        throw std::runtime_error("unsupported device");
    }
}

// q: [n_seqs, n_heads, d_head]
torch::Tensor Attention::forward(torch::Tensor q, int partition) {
    // we manipulate tensor data by raw data pointer: data_ptr<float>().
    // so we need to make sure the tensor is contiguous.
    // but q might be a transpose view and is non-contiguous.
    // A transpose of a tensor creates a view of the original tensor
    // which follows non-contiguous order.
    // https://www.tutorialspoint.com/pytorch-how-to-check-if-a-tensor-is-contiguous-or-not
    if (!q.is_contiguous()) {
        throw std::runtime_error("q is not contiguous");
    }

#ifdef USE_MKL
    torch::Tensor output = torch::zeros({ q.size(0), n_heads_, d_head_ }, t_options_);
#else
    torch::Tensor output = torch::empty({ q.size(0), n_heads_, d_head_ }, t_options_);
#endif

    this->refresh_kernel_context();
    GPT::attention(kernel_ctx_, q, output, partition);

    return output;
}

std::vector<Chunk*> Attention::append_token(const std::vector<int>& tokens,
                                            torch::Tensor k,
                                            torch::Tensor v,
                                            bool fused) {
    if (tails_.size() != tokens.size()) {
        throw std::runtime_error("num of seqs is not equal to num of tokens");
    }

    std::vector<Chunk*> new_chunks = reserve();

#ifdef USE_MKL
    fused = false;
#endif // USE_MKL

    if (fused) {
#ifdef USE_CUDA
        this->refresh_kernel_context();
        GPT::append_kv(kernel_ctx_, k, v);
        for (auto ite = 0; ite < tails_.size(); ite++) {
            Chunk* tail = tails_[ite];
            tail->append_tokens(tokens, ite, ite + 1);
        }
#endif
    } else {
        k = k.transpose(0, 1);
        v = v.transpose(0, 1);
        for (auto ite = 0; ite < tails_.size(); ite++) {
            Chunk* tail = tails_[ite];
            tail->append_tokens(tokens, k, v, ite, ite + 1);
        }
    }

    kernel_ctx_.delta_tokens += 1;
    return new_chunks;
}

std::vector<Chunk*> Attention::reserve() {
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

void Attention::refresh_kernel_context(bool force) {
    if (force || !kernel_ctx_.valid) {
        std::vector<ChunkInfo> chunk_infos = get_chunks();
        GPT::refresh_kernel_context(kernel_ctx_, chunk_infos, root_->n_seqs);
    }
}

// k/v: [n_tokens, n_heads, d_head]
int Attention::add_seq(const std::vector<int>& tokens, torch::Tensor k, torch::Tensor v) {
    assert(k.size(0) == v.size(0));
    k = k.transpose(0, 1);
    v = v.transpose(0, 1);

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
        auto child = allocator_->allocate(tokens, k, v, start, end);
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

std::shared_ptr<std::future<int>> Attention::add_seq_async(const std::vector<int>& tokens,
                                                           torch::Tensor k,
                                                           torch::Tensor v) {
    throw std::runtime_error("the implementation has bugs, still have problem in capturing torch tensors in lambda");
    std::shared_ptr<std::promise<int>> promise = std::make_shared<std::promise<int>>();
    std::shared_ptr<std::future<int>> future =
      std::make_shared<std::future<int>>(promise->get_future());
    /*
    task_executor_.enqueue([this, promise, tokens, k, v]() {
        int seq_idx = this->add_seq(tokens, k, v);
        promise->set_value(seq_idx);
    });
    */
    return future;
}

Chunk* Attention::at(int seq_idx) {
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

int Attention::append_completion(int seq_idx,
                                 std::vector<int>& tokens,
                                 torch::Tensor k,
                                 torch::Tensor v) {
    // locate the seq.
    auto chunk = at(seq_idx);
    // the tail chunk can not be full for sure
    assert(!chunk->full());
    assert(chunk->children.size() == 0);
    assert(chunk->n_seqs == 1);

    if (tokens.size() > 1) {
        auto it = std::find(chunk->parent->children.begin(), chunk->parent->children.end(), chunk);
        int distance = it - chunk->parent->children.begin() + 1;
        for (int i = 1; i < tokens.size(); i++) {
            auto chunk_copy = allocator_->allocate(*chunk);
            chunk_copy->append_tokens(tokens, k, v, i, i + 1);
            chunk->parent->insert_child(distance, chunk_copy);
            distance += 1;

            // make sure full chunks are not the tails
            if (chunk_copy->full()) {
                auto next = allocator_->allocate();
                chunk_copy->add_child(next);
            }
        }
    }
    chunk->append_tokens(tokens, k, v, 0, 1);
    // make sure full chunks are not the tails
    if (chunk->full()) {
        auto next = allocator_->allocate();
        chunk->add_child(next);
    }
    return seq_idx;
}

void Attention::remove_seq(int seq_idx) {
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

void Attention::clear() {
    int n_seqs = tails_.size();
    for (auto i = 0; i < n_seqs; i++) {
        remove_seq(i);
    }
}

void Attention::duplicate(int seq_idx, int copies) {
    if (copies < 1) {
        return;
    }

    Chunk* chunk = root_.get();
    int start = 0;
    while (chunk->children.size() > 0) {
        chunk->n_seqs += copies;
        for (auto& child : chunk->children) {
            int end = start + child->n_seqs;
            if (start <= seq_idx && seq_idx < end) {
                chunk = child;
                break;
            }
            start = end;
        }
    }

    auto it = std::find(chunk->parent->children.begin(), chunk->parent->children.end(), chunk);
    int distance = it - chunk->parent->children.begin() + 1;
    for (int i = 0; i < copies; i++) {
        auto chunk_copy = allocator_->allocate(*chunk);
        chunk->parent->insert_child(distance, chunk_copy);
        distance += 1;
    }
}

/*
def print_tree(node, level=0):
    indent = "  " * level
    print(indent + node.data)

    for child in node.children:
        print_tree(child, level + 1)

*/
void Attention::print(Chunk* root, int level) {
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

std::vector<ChunkInfo> Attention::get_chunks() {
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