#include <limits>
#include "mkl_cblas.h"
#include "mkl_vml_functions.h"
#include <omp.h>
#include "logging.h"
#include "kernel_cpu_mkl.h"
#include "kernel_cpu_tls.h"
#include "chunk_info.h"

// https://repository.prace-ri.eu/git/CodeVault/hpc-kernels/dense_linear_algebra/-/blob/master/gemm/mklblas/main.cpp
// https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/cblas-gemm-001.html
// https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-macos/2023-1/cmake-config-for-onemkl.html

namespace GPT {

void attn_one_chunk(float* q,
                    float* k,
                    float* v,
                    float* o,
                    int n_seqs,
                    int chunk_tokens, // chunk_tokens is the actual tokens count.
                                      // some chunks are padded, chunk_tokens <= chunk_size
                    int n_heads,
                    int d_head,
                    float d_sqrt, // avoid recomputing sqrt(d_head), time-consuming(verified)
                    float* weight_max,
                    float* weight_expsum,
                    ThreadLocalStorage* tls) {

    if (chunk_tokens <= 0) {
        return;
    }

    float* weight_local = tls->weight();
    float* weight_max_local = tls->weight_max();
    float* output_local = tls->output();
    float* scale1 = tls->scale1();
    float* scale2 = tls->scale2();

    // score_tmp = q * k^T
    cblas_sgemm(CblasRowMajor,    // CBLAS_LAYOUT layout,
                CblasNoTrans,     // CBLAS_TRANSPOSE TransA,
                CblasTrans,       // CBLAS_TRANSPOSE TransB,
                n_seqs,           // const int M,
                chunk_tokens,     // const int N,
                d_head,           // const int K,
                1.0,              // const float alpha,
                q,                // const float *A,
                d_head * n_heads, // const int lda,
                k,                // const float *B,
                d_head,           // const int ldb,
                0.0,              // const float beta,
                weight_local,     // float *C,
                chunk_tokens      // const int ldc
    );

    // weight_tmp = exp(weight_tmp - weight_max_tmp)
    for (int i = 0; i < n_seqs; i++) {
        float m = std::numeric_limits<float>::lowest();
        for (int j = 0; j < chunk_tokens; j++) {
            m = std::max(m, weight_local[i * chunk_tokens + j]);
        }

        // weight tmp is reused to store exp(weight_tmp - m)
        // std::exp((weight_local[i * chunk_tokens + j] - m) / d_sqrt);
        for (int j = 0; j < chunk_tokens; j++) {
            weight_local[i * chunk_tokens + j] = (weight_local[i * chunk_tokens + j] - m) / d_sqrt;
        }
        vsExp(chunk_tokens, weight_local + i * chunk_tokens, weight_local + i * chunk_tokens);

        weight_max_local[i] = m / d_sqrt;
    }

    cblas_sgemm(CblasRowMajor, // CBLAS_LAYOUT layout,
                CblasNoTrans,  // CBLAS_TRANSPOSE TransA,
                CblasNoTrans,  // CBLAS_TRANSPOSE TransB,
                n_seqs,        // const int M,
                d_head,        // const int N,
                chunk_tokens,  // const int K,
                1.0,           // const float alpha,
                weight_local,  // const float *A,
                chunk_tokens,  // const int lda,
                v,             // const float *B,
                d_head,        // const int ldb,
                0.0,           // const float beta,
                output_local,  // float *C,
                d_head         // const int ldc
    );

    if (tls->lock != nullptr) {
        tls->lock->lock();
    }
    for (int i = 0; i < n_seqs; i++) {
        float m = std::max(weight_max[i], weight_max_local[i]);
        scale1[i] = weight_max_local[i] - m;
        scale2[i] = weight_max[i] - m;
        weight_max[i] = m;
    }
    vsExp(n_seqs, scale1, scale1);
    vsExp(n_seqs, scale2, scale2);

    // output = scale1 * output_local + scale2 * output_final
    for (int i = 0; i < n_seqs; i++) {
        // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/cblas-axpby.html
        cblas_saxpby(
          d_head, scale1[i], output_local + i * d_head, 1, scale2[i], o + i * d_head * n_heads, 1);
    }

    // weight_expsum = scale1 * sum(weight_local) + scale2 * weight_expsum
    for (int i = 0; i < n_seqs; i++) {
        float s = 0;
        for (int j = 0; j < chunk_tokens; j++) {
            s += weight_local[i * chunk_tokens + j];
        }
        weight_expsum[i] = scale1[i] * s + scale2[i] * weight_expsum[i];
    }
    if (tls->lock != nullptr) {
        tls->lock->unlock();
    }
}

void attn_one_seq(float* q,
                  int seq_index,
                  int head_idx,
                  std::vector<ChunkInfo>& chunk_infos,
                  float* o,
                  int chunk_size,
                  int n_heads,
                  int d_head,
                  float d_sqrt // avoid recomputing sqrt(d_head), time-consuming(verified)
) {
    sv::small_vector<float, 2048> weight;
    float* weight_local = weight.data();
    float weight_max = std::numeric_limits<float>::lowest();
    float weight_expsum = 0;

    for (auto& chunk_info : chunk_infos) {
        int start = chunk_info.seq_idx_begin;
        int end = chunk_info.seq_idx_end;
        if (seq_index < start || seq_index >= end) {
            continue;
        }

        Chunk* chunk = chunk_info.chunk;
        int chunk_tokens = chunk->n_tokens(); // actual tokens in this chunk;
        if (chunk_tokens <= 0) {
            continue;
        }
        float* k = (float*)chunk->key_ptr() + head_idx * chunk_size * d_head;
        float* v = (float*)chunk->value_ptr() + head_idx * chunk_size * d_head;

        // score_tmp = q * k^T
        cblas_sgemm(CblasRowMajor,    // CBLAS_LAYOUT layout,
                    CblasNoTrans,     // CBLAS_TRANSPOSE TransA,
                    CblasTrans,       // CBLAS_TRANSPOSE TransB,
                    1,                // const int M,
                    chunk_tokens,     // const int N,
                    d_head,           // const int K,
                    1.0,              // const float alpha,
                    q,                // const float *A,
                    d_head * n_heads, // const int lda,
                    k,                // const float *B,
                    d_head,           // const int ldb,
                    0.0,              // const float beta,
                    weight_local,     // float *C,
                    chunk_tokens      // const int ldc
        );

        // weight_tmp = exp(weight_tmp - weight_max_tmp)
        float m = weight_max;
        for (int j = 0; j < chunk_tokens; j++) {
            m = std::max(m, weight_local[j]);
        }

        // weight tmp is reused to store exp(weight_tmp - m)
        // std::exp((weight_local[j] - m) / d_sqrt);
        for (int j = 0; j < chunk_tokens; j++) {
            weight_local[j] = (weight_local[j] - m) / d_sqrt;
        }
        vsExp(chunk_tokens, weight_local, weight_local);

        float scale2 = std::exp((weight_max - m) / d_sqrt);

        // C=alpha*AB + beta*C
        cblas_sgemm(CblasRowMajor,   // CBLAS_LAYOUT layout,
                    CblasNoTrans,    // CBLAS_TRANSPOSE TransA,
                    CblasNoTrans,    // CBLAS_TRANSPOSE TransB,
                    1,               // const int M,
                    d_head,          // const int N,
                    chunk_tokens,    // const int K,
                    1.0,             // const float alpha,
                    weight_local,    // const float *A,
                    chunk_tokens,    // const int lda,
                    v,               // const float *B,
                    d_head,          // const int ldb,
                    scale2,          // const float beta,
                    o,               // float *C,
                    d_head * n_heads // const int ldc
        );

        // weight_expsum = scale1 * sum(weight_local) + scale2 * weight_expsum
        float s = 0;
        for (int j = 0; j < chunk_tokens; j++) {
            s += weight_local[j];
        }
        weight_expsum = s + scale2 * weight_expsum;
        weight_max = m;
    }

    for (int i = 0; i < d_head; i++) {
        o[i] /= weight_expsum;
    }
}

void seq_first(KernelContext& context, torch::Tensor& q, torch::Tensor& output) {
    int d_head = context.d_head;
    float d_sqrt = std::sqrt(d_head);
    int n_seqs = q.size(0);
    int n_heads = context.n_heads;
    int chunk_size = context.chunk_size;

    float* output_ptr = output.data_ptr<float>();
    float* q_ptr = q.data_ptr<float>();
    int total = n_seqs * n_heads;

#pragma omp parallel for
    for (int i = 0; i < total; i++) {
        int head_idx = i / n_seqs;
        int seq_idx = i % n_seqs;

        LOG_DEBUG("omp: {} total threads, current thread {}, head {} seq {}",
                  omp_get_num_threads(),
                  omp_get_thread_num(),
                  head_idx,
                  seq_idx);

        float* q = q_ptr + head_idx * d_head + seq_idx * d_head * n_heads;
        float* o = output_ptr + head_idx * d_head + seq_idx * d_head * n_heads;
        attn_one_seq(
          q, seq_idx, head_idx, context.chunk_infos, o, chunk_size, n_heads, d_head, d_sqrt);
    }
}

// q: [n_seqs, n_heads, d_head]
void chunk_first(KernelContext& context, torch::Tensor& q, torch::Tensor& output) {
    int n_heads = context.n_heads;
    int chunk_size = context.chunk_size;
    torch::Tensor weight_max =
      torch::full({ n_heads, q.size(0) }, std::numeric_limits<float>::lowest());
    torch::Tensor weight_expsum = torch::zeros({ n_heads, q.size(0) });

    int d_head = context.d_head;
    float d_sqrt = std::sqrt(d_head);
    int n_seqs = q.size(0);
    int n_tasks = context.chunk_infos.size();

    float* weight_max_ptr = weight_max.data_ptr<float>();
    float* weight_expsum_ptr = weight_expsum.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* q_ptr = q.data_ptr<float>();
    int n_threads = omp_get_max_threads();

    if (n_heads < n_threads) {
        LOG_DEBUG(
          "ChunkAttn: partition by head and chunk. n_heads {}, n_threads {}", n_heads, n_threads);
        sv::small_vector<SpinLock, 96> locks(n_heads);
        int total = n_heads * n_tasks;

#pragma omp parallel for
        for (int i = 0; i < total; i++) {
            int chunk_idx = i / n_heads;
            int head_idx = i % n_heads;

            // LOG_INFO("omp: {} total threads, current thread {}, head {}, task {}",
            //          omp_get_num_threads(),
            //          omp_get_thread_num(),
            //          head_idx, task_idx);

            float* weight_max_head = weight_max_ptr + head_idx * n_seqs;
            float* weight_expsum_head = weight_expsum_ptr + head_idx * n_seqs;
            float* output_head = output_ptr + head_idx * d_head;
            float* q_head = q_ptr + head_idx * d_head;

            ChunkInfo& task = context.chunk_infos[chunk_idx];
            Chunk* chunk = task.chunk;
            int chunk_tokens = chunk->n_tokens(); // actual tokens in this chunk
            int seq_begin = task.seq_idx_begin;
            int seq_end = task.seq_idx_end;
            SpinLock* lock = &locks[head_idx];
            ThreadLocalStorage tls(seq_end - seq_begin, chunk_tokens, d_head, lock);

            float* key = (float*)chunk->key_ptr() + head_idx * chunk_size * d_head;
            float* value = (float*)chunk->value_ptr() + head_idx * chunk_size * d_head;

            attn_one_chunk(q_head + seq_begin * d_head * n_heads,
                           key,
                           value,
                           output_head + seq_begin * d_head * n_heads,
                           seq_end - seq_begin,
                           chunk_tokens,
                           n_heads,
                           d_head,
                           d_sqrt,
                           weight_max_head + seq_begin,
                           weight_expsum_head + seq_begin,
                           &tls);
        }
    } else {
        LOG_DEBUG("ChunkAttn: partition by head only, lock free. n_heads {}, n_threads {}",
                  n_heads,
                  n_threads);

#pragma omp parallel for
        for (int head_idx = 0; head_idx < n_heads; head_idx++) {

            LOG_DEBUG("omp: {} total threads, current thread {}, head {}",
                      omp_get_num_threads(),
                      omp_get_thread_num(),
                      head_idx);

            float* weight_max_head = weight_max_ptr + head_idx * n_seqs;
            float* weight_expsum_head = weight_expsum_ptr + head_idx * n_seqs;
            float* output_head = output_ptr + head_idx * d_head;
            float* q_head = q_ptr + head_idx * d_head;

            for (int task_idx = 0; task_idx < n_tasks; task_idx++) {
                ChunkInfo& task = context.chunk_infos[task_idx];
                Chunk* chunk = task.chunk;
                int chunk_tokens = chunk->n_tokens(); // actual tokens in this chunk
                int seq_begin = task.seq_idx_begin;
                int seq_end = task.seq_idx_end;
                ThreadLocalStorage tls(seq_end - seq_begin, chunk_tokens, d_head);

                float* key = (float*)chunk->key_ptr() + head_idx * chunk_size * d_head;
                float* value = (float*)chunk->value_ptr() + head_idx * chunk_size * d_head;

                attn_one_chunk(q_head + seq_begin * d_head * n_heads,
                               key,
                               value,
                               output_head + seq_begin * d_head * n_heads,
                               seq_end - seq_begin,
                               chunk_tokens,
                               n_heads,
                               d_head,
                               d_sqrt,
                               weight_max_head + seq_begin,
                               weight_expsum_head + seq_begin,
                               &tls);
            }
        }
    }
    output.div_(weight_expsum.transpose(0, 1).unsqueeze(2));
}

KernelContext& refresh_kernel_context(KernelContext& context,
                                      std::vector<ChunkInfo>& chunk_infos,
                                      int n_seqs) {
    context.valid = true;
    context.chunk_infos = chunk_infos;
    context.delta_tokens = 0;
    return context;
}

void attention(KernelContext& context, torch::Tensor& q, torch::Tensor& output, int partition) {
    if (partition == 0 || partition == 1) {
        chunk_first(context, q, output);
    } else if (partition == 2) {
        seq_first(context, q, output);
    } else {
        throw std::runtime_error("Unsupported partition type");
    }
}

} // namespace GPT