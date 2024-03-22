#pragma once

#include "small_vector.h"
#include "spin_lock.h"
#include "logging.h"

namespace GPT {

struct ThreadLocalStorage {
    ThreadLocalStorage(int n_seqs, int n_tokens, int d_head, SpinLock* l = nullptr)
      : weight_(n_seqs * n_tokens)
      , weight_max_(n_seqs)
      , output_(n_seqs * d_head)
      , scale1_(n_seqs)
      , scale2_(n_seqs)
      , lock(l) {
        if (n_seqs * n_tokens > MAX_SEQS * MAX_CHUNK_SIZE || n_seqs > MAX_SEQS ||
            n_seqs * d_head > MAX_SEQS * 128) {
            LOG_WARN("ThreadLocalStorage: memory is allocated on heap, which hurts performance. weight size {}, weight_max size {}, output size {}, "
                     "scale1 size {}, scale2 size {}",
                     weight_.size(),
                     weight_max_.size(),
                     output_.size(),
                     scale1_.size(),
                     scale2_.size());
        }
    }

    float* weight() { return weight_.data(); }
    float* weight_max() { return weight_max_.data(); }
    float* output() { return output_.data(); }
    float* scale1() { return scale1_.data(); }
    float* scale2() { return scale2_.data(); }

    SpinLock* lock;

  private:
      static const int MAX_SEQS = 512;
      static const int MAX_CHUNK_SIZE = 128;
      sv::small_vector<float, MAX_SEQS * MAX_CHUNK_SIZE> weight_;
      sv::small_vector<float, MAX_SEQS> weight_max_;
      sv::small_vector<float, MAX_SEQS * 128> output_;
      sv::small_vector<float, MAX_SEQS> scale1_;
      sv::small_vector<float, MAX_SEQS> scale2_;
    
};

} // namespace GPT