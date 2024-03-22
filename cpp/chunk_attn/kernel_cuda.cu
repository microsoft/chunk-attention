#include "kernel_cuda.h"
#include <ATen/cuda/CUDAContext.h>
#include <mma.h>
#include "logging.h"
#include <nvtx3/nvtx3.hpp>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
#define CUDA_ARCH_LESS_THAN_800
#endif

namespace GPT {
using namespace nvcuda::wmma;

#define CHECK_SHAPE(x, ...)                                                                        \
    TORCH_CHECK(x.sizes() == torch::IntArrayRef({ __VA_ARGS__ }),                                  \
                #x " must have shape (" #__VA_ARGS__ ")")

__inline__ __device__ void commit_async_cp_group() {
    #ifndef CUDA_ARCH_LESS_THAN_800
    asm volatile("cp.async.commit_group;\n" ::);
    #endif
}

template<int n>
__inline__ __device__ void wait_async_cp_group() {
    #ifndef CUDA_ARCH_LESS_THAN_800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
    #endif
}

template<uint32_t n>
__inline__ __device__ void cp_async_cg_shared_global(void* __restrict__ smem_ptr,
                                                     const void* __restrict__ gmem_ptr) {
    #ifdef CUDA_ARCH_LESS_THAN_800
    uint32_t k = n >> 2; // 4 bytes per int. n must be a multiple of 4
    for (uint32_t i = 0; i < k; i++) {
        ((int*)smem_ptr)[i] = ((int*)gmem_ptr)[i];
    }
    #else
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(
                   static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr))),
                 "l"(gmem_ptr),
                 "n"(n));
    #endif
}

template<typename scalar_t>
class TypeTraits {};

template<typename scalar, int N, int M>
__inline__ __device__ void print_matrix(const char* name,
                                        int block_x,
                                        int block_y,
                                        const scalar* matrix,
                                        int stride) {
    if (blockIdx.x != block_x || blockIdx.y != block_y) {
        return;
    }

    if (threadIdx.x == 0) {
        printf("%s on block [%d, %d]\n", name, block_x, block_y);
        for (int i = 0; i < N; i++) {
            printf("[");
            for (int j = 0; j < M; j++) {
                printf("%f, ", TypeTraits<scalar>::to_float(matrix[i * stride + j]));
            }
            printf("],\n");
        }
    }
    __syncthreads();
}

template<typename scalar_t>
struct ShortVec {
    using scalar2_t = typename TypeTraits<scalar_t>::scalar2_t;
    scalar_t elements[16 / sizeof(scalar_t)];

  public:
    __inline__ __device__ float dot(const ShortVec<scalar_t>& other) const {
        const auto* elements1 = reinterpret_cast<const scalar2_t*>(elements);
        const auto* elements2 = reinterpret_cast<const scalar2_t*>(other.elements);
        scalar2_t result = TypeTraits<scalar_t>::mul2(elements1[0], elements2[0]);

#pragma unroll
        for (int i = 1; i < 16 / sizeof(scalar2_t); i++) {
            result = TypeTraits<scalar_t>::fma2(elements1[i], elements2[i], result);
        }
        return TypeTraits<scalar_t>::add_two_to_float(result);
    }
};

template<>
class TypeTraits<half> {
  public:
    using scalar_t = half;
    using scalar2_t = half2;
    using vector_t = ShortVec<scalar_t>;
    __inline__ __device__ static scalar2_t mul2(const scalar2_t& a, const scalar2_t& b) {
        return __hmul2(a, b);
    }

    __inline__ __device__ static scalar2_t fma2(const scalar2_t& a,
                                                const scalar2_t& b,
                                                const scalar2_t& c) {
        return __hfma2(a, b, c);
    }

    __inline__ __device__ static float add_two_to_float(const scalar2_t& a) {
        return __half2float(a.x + a.y);
    }

    __inline__ __device__ static scalar_t float_to_scalar(const float& a) {
        return __float2half(a);
    }

    __inline__ __device__ static scalar2_t float_to_scalar2(const float& a) {
        return __float2half2_rn(a);
    }

    __inline__ __device__ static float to_float(const half& a) { return __half2float(a); }
};

template<>
class TypeTraits<float> {
  public:
    __inline__ __device__ static float to_float(const float& a) { return a; }
    __inline__ __device__ static float float_to_scalar(const float& a) { return a; }
};

template<typename scalar_t, int length, int a_stride, int b_stride>
__inline__ __device__ float vector_dot(const half* __restrict__ A, const half* __restrict__ B) {
    using scalar2_t = typename TypeTraits<scalar_t>::scalar2_t;
    scalar_t r_a[length];
    scalar_t r_b[length];
#pragma unroll
    for (int i = 0; i < length; i++) {
        r_a[i] = A[i * a_stride];
        r_b[i] = B[i * b_stride];
    }

    scalar2_t* r_a2 = reinterpret_cast<scalar2_t*>(r_a);
    scalar2_t* r_b2 = reinterpret_cast<scalar2_t*>(r_b);
    scalar2_t c = TypeTraits<half>::mul2(r_a2[0], r_b2[0]);
#pragma unroll
    for (int i = 1; i < length / 2; i++) {
        c = TypeTraits<half>::fma2(r_a2[i], r_b2[i], c);
    }
    return TypeTraits<half>::add_two_to_float(c);
}

// only support col partition
template<typename scalar_t,
         uint32_t row,
         uint32_t d_stride,
         uint32_t tile_col_size,
         uint32_t thread_num>
__inline__ __device__ void load_tile_col_partition(const scalar_t* __restrict__ src,
                                                   uint32_t src_stride,
                                                   scalar_t* __restrict__ dst,
                                                   int partition_idx) {
    constexpr uint32_t threads_per_row = thread_num > row ? thread_num / row : 1;
    constexpr uint32_t rows_per_thread = thread_num > row ? 1 : row / thread_num;
    constexpr uint32_t thread_load_size = tile_col_size / threads_per_row;
    constexpr uint32_t thread_load_bytes_each_time =
      thread_load_size * sizeof(scalar_t) > 16 ? 16 : thread_load_size * sizeof(scalar_t);
    constexpr uint32_t thread_load_times =
      thread_load_size * sizeof(scalar_t) / thread_load_bytes_each_time;

    uint32_t thread_idx = threadIdx.x % thread_num;
    uint32_t thread_row_idx = thread_idx % row; // thread partition by row
    uint32_t thread_col_idx = thread_idx / row;
#pragma unroll
    for (int i = 0; i < rows_per_thread; i++) {
#pragma unroll
        for (int j = 0; j < thread_load_times; j++) {
            uint32_t src_offset = (i * thread_num + thread_row_idx) * src_stride +
                                  partition_idx * tile_col_size +
                                  thread_col_idx * thread_load_size +
                                  j * thread_load_bytes_each_time / sizeof(scalar_t);
            uint32_t dst_offset = (i * thread_num + thread_row_idx) * d_stride +
                                  partition_idx * tile_col_size +
                                  thread_col_idx * thread_load_size +
                                  j * thread_load_bytes_each_time / sizeof(scalar_t);

            cp_async_cg_shared_global<thread_load_bytes_each_time>(dst + dst_offset,
                                                                   src + src_offset);
        }
    }
}

template<typename scalar_t,
         uint32_t col,
         uint32_t s_stride,
         uint32_t d_stride,
         uint32_t tile_row_size,
         uint32_t thread_num>
__inline__ __device__ void load_tile_row_partition(const scalar_t* __restrict__ src,
                                                   scalar_t* __restrict__ dst,
                                                   int partition_idx) {
    constexpr uint32_t thread_load_bytes = tile_row_size * col / thread_num * sizeof(scalar_t);
    constexpr uint32_t thread_load_bytes_each_time =
      thread_load_bytes > 16 ? 16 : thread_load_bytes;
    constexpr uint32_t thread_load_elem_num = thread_load_bytes_each_time / sizeof(scalar_t);
    constexpr uint32_t threads_one_col = col / thread_load_elem_num;
    constexpr uint32_t row_num_one_time = thread_num / threads_one_col;
    constexpr uint32_t thread_load_times = thread_load_bytes / thread_load_bytes_each_time;

    uint32_t thread_idx = threadIdx.x % thread_num;
    uint32_t thread_row_idx = thread_idx / threads_one_col; // thread partition bycol
    uint32_t thread_col_idx = thread_idx % threads_one_col;
#pragma unroll
    for (int i = 0; i < thread_load_times; i++) {
        uint32_t src_offset = partition_idx * tile_row_size * s_stride + (i * row_num_one_time + thread_row_idx) * s_stride +
                              thread_col_idx * thread_load_elem_num;
        uint32_t dst_offset = partition_idx * tile_row_size * d_stride + (i * row_num_one_time + thread_row_idx) * d_stride +
                              thread_col_idx * thread_load_elem_num;
        cp_async_cg_shared_global<thread_load_bytes_each_time>(dst + dst_offset, src + src_offset);
    }
}

template<int M, int N>
class PartitionCal {
  public:
    static constexpr __device__ int get_n_warp_size() {
        if constexpr (M < 64 || N < 64) {
            if constexpr (M < N) {
                return 4;
            } else if (M == N) {
                return 2;
            } else {
                return 1;
            }
        } else {
            return 2;
        }
    }
    static constexpr __device__ int get_m_warp_size() { return 4 / get_n_warp_size(); }
};

template<typename scalar_t, int M, int N, int K>
__inline__ __device__ void matrix_multiply_gAB_sC(const scalar_t* __restrict__ gA,
                                                  uint32_t gA_stride,
                                                  scalar_t* __restrict__ sA,
                                                  const scalar_t* __restrict__ gB,
                                                  scalar_t* __restrict__ sB,
                                                  float* __restrict__ sC,
                                                  uint32_t sC_stride) {
    constexpr int warp_size = 32;
    constexpr int m_warp_size = PartitionCal<M, N>::get_m_warp_size();
    constexpr int n_warp_size = PartitionCal<M, N>::get_n_warp_size();
    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / warp_size;
    const int warp_on_m =
      m_warp_size == n_warp_size ? warp_idx / m_warp_size : warp_idx % m_warp_size;
    const int warp_on_n = warp_idx % n_warp_size;

    constexpr int mma_m = 16;
    constexpr int mma_n = 16;
    constexpr int mma_k = 16;
    constexpr int padding_k = K + 16 / sizeof(scalar_t);
    constexpr int M_Wrap = M / m_warp_size;
    constexpr int N_Wrap = N / n_warp_size;
    constexpr int m_mma_count = M_Wrap / mma_m;
    constexpr int n_mma_count = N_Wrap / mma_n;
    constexpr int k_mma_count = K / mma_k;
    constexpr int pre_load = 1;

    using wmma_row_a = fragment<matrix_a, mma_m, mma_n, mma_k, scalar_t, row_major>;
    using wmma_row_b = fragment<matrix_b, mma_m, mma_n, mma_k, scalar_t, col_major>;
    using wmma_row_c = fragment<accumulator, mma_m, mma_n, mma_k, float>;

    wmma_row_a a_frag[m_mma_count];
    wmma_row_b b_frag[n_mma_count];
    wmma_row_c c_frag[m_mma_count * n_mma_count];

#pragma unroll
    for (int i = 0; i < m_mma_count * n_mma_count; i++) {
        fill_fragment(c_frag[i], 0);
    }

#pragma unroll
    for (int k = 0; k < k_mma_count; k++) {
        // pre-load tile to smem
        if (k == 0) {
#pragma unroll
            for (int i = 0; i < pre_load; i++) {
                load_tile_col_partition<scalar_t, M, padding_k, mma_k, 128>(gA, gA_stride, sA, i);
                load_tile_col_partition<scalar_t, N, padding_k, mma_k, 128>(gB, K, sB, i);
                commit_async_cp_group();
            }
        }
        if (k + pre_load < k_mma_count) {
            load_tile_col_partition<scalar_t, M, padding_k, mma_k, 128>(
              gA, gA_stride, sA, k + pre_load);
            load_tile_col_partition<scalar_t, N, padding_k, mma_k, 128>(gB, K, sB, k + pre_load);
            commit_async_cp_group();
        }

        // loading finishes
        if (k + pre_load < k_mma_count) {
            wait_async_cp_group<pre_load>();
        } else {
            wait_async_cp_group<0>();
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < m_mma_count; i++) {
            int a_offset = warp_on_m * M_Wrap * padding_k + i * mma_m * padding_k + k * mma_k;
            load_matrix_sync(a_frag[i], sA + a_offset, padding_k);
        }

#pragma unroll
        for (int i = 0; i < n_mma_count; i++) {
            int b_offset = warp_on_n * N_Wrap * padding_k + i * mma_n * padding_k + k * mma_k;
            load_matrix_sync(b_frag[i], sB + b_offset, padding_k);
        }

#pragma unroll
        for (int i = 0; i < m_mma_count; i++) {
#pragma unroll
            for (int j = 0; j < n_mma_count; j++) {
                mma_sync(
                  c_frag[i * n_mma_count + j], a_frag[i], b_frag[j], c_frag[i * n_mma_count + j]);
            }
        }
    }

    int c_base_offset = warp_on_m * M_Wrap * sC_stride + warp_on_n * N_Wrap;
#pragma unroll
    for (int i = 0; i < m_mma_count; i++) {
#pragma unroll
        for (int j = 0; j < n_mma_count; j++) {
            int c_offset = c_base_offset + i * mma_m * sC_stride + j * mma_n;
            store_matrix_sync(sC + c_offset, c_frag[i * n_mma_count + j], sC_stride, mem_row_major);
        }
    }
    __syncthreads();
}

template<typename scalar_t, int M, int N, int K>
__inline__ __device__ void matrix_multiply_sA_gBC(const scalar_t* __restrict__ sA,
                                                  const scalar_t* __restrict__ gB,
                                                  scalar_t* __restrict__ sB,
                                                  half* __restrict__ gC,
                                                  uint32_t sA_stride,
                                                  uint32_t sC_stride) {
    constexpr int warp_size = 32;
    constexpr int m_warp_size = PartitionCal<M, N>::get_m_warp_size();
    constexpr int n_warp_size = PartitionCal<M, N>::get_n_warp_size();
    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / warp_size;
    const int warp_on_m =
      n_warp_size == m_warp_size ? warp_idx / m_warp_size : warp_idx % m_warp_size;
    const int warp_on_n = warp_idx % n_warp_size;

    constexpr int mma_m = 16;
    constexpr int mma_n = 16;
    constexpr int mma_k = 16;
    constexpr int padding_n = N + 16 / sizeof(scalar_t);
    constexpr int M_Wrap = M / m_warp_size;
    constexpr int N_Wrap = N / n_warp_size;
    constexpr int m_mma_count = M_Wrap / mma_m;
    constexpr int n_mma_count = N_Wrap / mma_n;
    constexpr int k_mma_count = K / mma_k;
    constexpr int pre_load = 1;

    using wmma_row_a = fragment<matrix_a, mma_m, mma_n, mma_k, scalar_t, row_major>;
    ;
    using wmma_row_b = fragment<matrix_b, mma_m, mma_n, mma_k, scalar_t, row_major>;
    using wmma_row_c = fragment<accumulator, mma_m, mma_n, mma_k, scalar_t>;
    wmma_row_a a_frag[m_mma_count];
    wmma_row_b b_frag[n_mma_count];
    wmma_row_c c_frag[m_mma_count * n_mma_count];

    for (int i = 0; i < m_mma_count * n_mma_count; i++) {
        fill_fragment(c_frag[i], 0);
    }

#pragma unroll
    for (int k = 0; k < k_mma_count; k++) {
        // pre-load tile to smem
        if (k == 0) {
#pragma unroll
            for (int i = 0; i < pre_load; i++) {
                load_tile_row_partition<scalar_t, N, N, padding_n, mma_k, 128>(gB, sB, i);
                commit_async_cp_group();
            }
        }
        if (k + pre_load < k_mma_count) {
            load_tile_row_partition<scalar_t, N, N, padding_n, mma_k, 128>(gB, sB, k + pre_load);
            commit_async_cp_group();
        }

        // loading finishes
        if (k + pre_load < k_mma_count) {
            wait_async_cp_group<pre_load>();
        } else {
            wait_async_cp_group<0>();
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < m_mma_count; i++) {
            int a_offset = warp_on_m * M_Wrap * sA_stride + i * mma_m * sA_stride + k * mma_k;
            load_matrix_sync(a_frag[i], sA + a_offset, sA_stride);
        }

#pragma unroll
        for (int i = 0; i < n_mma_count; i++) {
            int b_offset = warp_on_n * N_Wrap + k * mma_k * padding_n + i * mma_n;
            load_matrix_sync(b_frag[i], sB + b_offset, padding_n);
        }

        // load to registry and compute
#pragma unroll
        for (int i = 0; i < m_mma_count; i++) {
#pragma unroll
            for (int j = 0; j < n_mma_count; j++) {
                mma_sync(
                  c_frag[i * n_mma_count + j], a_frag[i], b_frag[j], c_frag[i * n_mma_count + j]);
            }
        }
    }

    int c_base_offset = warp_on_m * M_Wrap * sC_stride + warp_on_n * N_Wrap;
#pragma unroll
    for (int i = 0; i < m_mma_count; i++) {
#pragma unroll
        for (int j = 0; j < n_mma_count; j++) {
            int c_offset = c_base_offset + i * mma_m * sC_stride + j * mma_n;
            store_matrix_sync(gC + c_offset, c_frag[i * n_mma_count + j], sC_stride, mem_row_major);
        }
    }
    __syncthreads();
}

template<int length>
__inline__ __device__ float warp_vector_scale(float* __restrict__ val, float scale) {
    constexpr int warp_size = 32;
    int lane_id = threadIdx.x % warp_size;
#pragma unroll
    for (int i = lane_id; i < length; i += warp_size) {
        val[i] *= scale;
    }
    __syncwarp();
}

template<int N>
__inline__ __device__ float warp_vector_sum(const float* __restrict__ val) {
    constexpr int warp_size = 32;
    int thread_idx = threadIdx.x % warp_size;
    float sum = 0;
#pragma unroll
    for (int i = thread_idx; i < N; i += warp_size) {
        sum += val[i];
    }
    for (int i = warp_size / 2; i >= 1; i /= 2) {
        sum += __shfl_xor_sync(uint32_t(-1), sum, i);
    }
    return sum;
}

template<int vector_length>
__inline__ __device__ float warp_vector_max(const float* __restrict__ val) {
    constexpr int warp_size = 32;
    int thread_idx = threadIdx.x % warp_size;
    float max_val = -FLT_MAX;
#pragma unroll
    for (int i = thread_idx; i < vector_length; i += warp_size) {
        max_val = fmaxf(max_val, val[i]);
    }
    for (int i = warp_size / 2; i >= 1; i /= 2) {
        max_val = fmaxf(max_val, __shfl_xor_sync(uint32_t(-1), max_val, i));
    }
    return max_val;
}

template<typename scalar_t, int N>
__inline__ __device__ float warp_cal_exp(float* __restrict__ val,
                                         scalar_t* __restrict__ cpy,
                                         const float max) {
    constexpr int warp_size = 32;
    int thread_idx = threadIdx.x % warp_size;
    for (int i = thread_idx; i < N; i += warp_size) {
        float exp = __expf(val[i] - max);
        val[i] = exp;
        cpy[i] = TypeTraits<scalar_t>::float_to_scalar(exp);
    }
    __syncwarp();
}

template<typename scalar_t, int N>
__inline__ __device__ void warp_vector_merge(scalar_t* a,
                                             scalar_t* b,
                                             float scale_a,
                                             float scale_b) {
    constexpr int warp_size = 32;
    constexpr int each_thread_merge = (N / 2) / warp_size;
    int lane_idx = threadIdx.x % warp_size;
    using scalar2_t = typename TypeTraits<scalar_t>::scalar2_t;
    scalar2_t scale_a2 = TypeTraits<scalar_t>::float_to_scalar2(scale_a);
    scalar2_t scale_b2 = TypeTraits<scalar_t>::float_to_scalar2(scale_b);
    scalar2_t* a_2 = reinterpret_cast<scalar2_t*>(a);
    scalar2_t* b_2 = reinterpret_cast<scalar2_t*>(b);

#pragma unroll
    for (int i = 0; i < each_thread_merge; i++) {
        int offset = i * warp_size + lane_idx;
        a_2[offset] = TypeTraits<scalar_t>::fma2(
          b_2[offset], scale_b2, TypeTraits<scalar_t>::mul2(a_2[offset], scale_a2));
    }
    __syncwarp();
}

template<typename scalar_t, int N, int K>
__inline__ __device__ void warp_vect_mul_raw_major_matrix(const scalar_t* vec,
                                                          const scalar_t* mat,
                                                          scalar_t* shared_mat,
                                                          float* result,
                                                          float scale) {
    using vector_t = typename TypeTraits<scalar_t>::vector_t;

    constexpr int warp_size = 32;
    constexpr int load_size = 64;
    constexpr int load_times = K / load_size;
    constexpr int padding_k = K + 16 / sizeof(scalar_t);
    constexpr int pre_load = 1;
    constexpr int vector_size = sizeof(vector_t) / sizeof(scalar_t);
    constexpr int vector_count = load_size / (16 / sizeof(scalar_t));
    constexpr int n_pre_thread = N / warp_size;
    const int lane_idx = threadIdx.x % warp_size;

#pragma unroll
    for (int k = 0; k < load_times; k++) {
        // pre-load tile to smem
        if (k == 0) {

#pragma unroll
            for (int i = 0; i < pre_load; i++) {
                load_tile_col_partition<scalar_t, N, padding_k, load_size, 32>(
                  mat, K, shared_mat, i);
                commit_async_cp_group();
            }
        }
        if (k + pre_load < load_times) {
            load_tile_col_partition<scalar_t, N, padding_k, load_size, 32>(
              mat, K, shared_mat, k + pre_load);
            commit_async_cp_group();
        }

        // loading finishes
        if (k + pre_load < load_times) {
            wait_async_cp_group<pre_load>();
        } else {
            wait_async_cp_group<0>();
        }

        __syncwarp();

#pragma unroll
        for (int i = 0; i < vector_count; i++) {
            vector_t vec_reg;
            vector_t mat_reg;
            vec_reg = reinterpret_cast<const vector_t*>(vec)[k * vector_count + i];
#pragma unroll
            for (int j = 0; j < n_pre_thread; j++) {
                mat_reg = reinterpret_cast<const vector_t*>(
                  shared_mat + (j * warp_size + lane_idx) * padding_k)[k * vector_count + i];
                result[j] += vec_reg.dot(mat_reg);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < n_pre_thread; i++) {
        result[i] = result[i] * scale;
    }
    __syncwarp();
}

template<typename scalar_t, int N, int K>
__inline__ __device__ void warp_vect_mul_raw_major_matrix_v2(const scalar_t* vec,
                                                          const scalar_t* mat,
                                                          scalar_t* shared_mat,
                                                          float* result,
                                                          float scale) {
    using vector_t = typename TypeTraits<scalar_t>::vector_t;

    constexpr int warp_size = 32;
    constexpr int n_load_elems = warp_size;
    constexpr int n_load_times = N  / warp_size;
    constexpr int k_load_size = 128;
    constexpr int k_load_elems = k_load_size / sizeof(scalar_t);
    constexpr int k_load_times = K / k_load_elems;
    constexpr int load_times = n_load_times * k_load_times;
    constexpr int vector_size = sizeof(vector_t) / sizeof(scalar_t);
    constexpr int vector_count = k_load_elems / vector_size;


    constexpr int padding_k = K + 16 / sizeof(scalar_t);
    constexpr int pre_load = 1;
    const int lane_idx = threadIdx.x % warp_size;

#pragma unroll
    for (int l = 0; l < load_times; l++) {
        // pre-load tile to smem
        int k = l / n_load_times;
        int n = l % n_load_times;

        if (l == 0) {

#pragma unroll
            for (int i = 0; i < pre_load; i++) {
                load_tile_row_partition<scalar_t, k_load_elems, K, padding_k, n_load_elems, 32>(mat + k * k_load_elems, shared_mat + k * k_load_elems, n);
                commit_async_cp_group();
            }
        }
        if (l + pre_load < load_times) {
            int k_next = (l + pre_load) / n_load_times;
            int n_next = (l + pre_load) % n_load_times;
            load_tile_row_partition<scalar_t, k_load_elems, K, padding_k, n_load_elems, 32>(mat + k_next * k_load_elems, shared_mat + k_next * k_load_elems, n_next);
            commit_async_cp_group();
        }

        // loading finishes
        if (l + pre_load < load_times) {
            wait_async_cp_group<pre_load>();
        } else {
            wait_async_cp_group<0>();
        }

        __syncwarp();

#pragma unroll
        for (int i = 0; i < vector_count; i++) {
            vector_t vec_reg;
            vector_t mat_reg;
            vec_reg = reinterpret_cast<const vector_t*>(vec + k * k_load_elems)[i];
            mat_reg = reinterpret_cast<const vector_t*>(
              shared_mat + (n * warp_size + lane_idx) * padding_k + k * k_load_elems)[i];
            result[n] += vec_reg.dot(mat_reg);
        }
    }

#pragma unroll
    for (int i = 0; i < n_load_times; i++) {
        result[i] = result[i] * scale;
    }
    __syncwarp();
}

template<typename scalar_t, int N, int K>
__inline__ __device__ void warp_vect_mul_col_major_matrix(const scalar_t *vec, const scalar_t *mat, scalar_t *shared_mat, scalar_t *result, scalar_t scale)
{
    using vector_t = typename TypeTraits<scalar_t>::vector_t;

    constexpr int warp_size = 32;
    constexpr int load_size = 8;
    constexpr int load_times = K / load_size;
    constexpr int padding_n = N + 16 / sizeof(scalar_t);
    constexpr int pre_load = 1;
    constexpr int vector_length = sizeof(vector_t) / sizeof(scalar_t);
    constexpr int vector_count = load_size / (16 / sizeof(scalar_t));
    constexpr int n_pre_thread = N / warp_size;
    const int lane_idx = threadIdx.x % warp_size;

    float reg_result[n_pre_thread] = {0};

#pragma unroll
    for (int k = 0; k < load_times; k++) {
        // pre-load tile to smem
        if (k == 0) {
#pragma unroll
            for (int i = 0; i < pre_load; i++) {
                load_tile_row_partition<scalar_t, N, N, padding_n, load_size, 32>(
                  mat, shared_mat, i);
                commit_async_cp_group();
            }
        }
        if (k + pre_load < load_times) {
            load_tile_row_partition<scalar_t, N, N, padding_n, load_size, 32>(
              mat, shared_mat, k + pre_load);
            commit_async_cp_group();
        }

        // loading finishes
        if (k + pre_load < load_times) {
            wait_async_cp_group<pre_load>();
        } else {
            wait_async_cp_group<0>();
        }

        __syncwarp();

#pragma unroll
        for (int i = 0; i < vector_count; i++) {
            vector_t vec_reg;
            vector_t mat_reg;
            vec_reg = reinterpret_cast<const vector_t*>(vec)[k * vector_count + i];
#pragma unroll
            for (int j = 0; j < n_pre_thread; j++) {
#pragma unroll
                for(int l = 0; l < vector_length; l++) {
                    mat_reg.elements[l] = shared_mat[(k * load_size + i * vector_length + l) * padding_n + j * warp_size + lane_idx];
                }
                reg_result[j] += vec_reg.dot(mat_reg);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < n_pre_thread; i++)
    {
        result[i * warp_size + lane_idx] = result[i * warp_size + lane_idx] * scale + TypeTraits<scalar_t>::float_to_scalar(reg_result[i]);
    }
    __syncwarp();
}

template<typename scalar_t, int n_seqs, int chunk_size, int d_head>
__global__ void attn_chunk_first_kernel(
  const scalar_t* __restrict__ query, // [n_heads, n_seqs, head_size]
  void** __restrict__ keys,           // chunk_num<[n_heads, chunk_size, d_head]>
  void** __restrict__ values,         // chunk_num<[n_heads, chunk_size, d_head]>
  scalar_t* __restrict__ attns,       // chunk_num<[n_heads, n_seqs , d_head]>
  float* __restrict__ maxs,           // chunk_num<[n_heads, n_seqs]>
  float* __restrict__ sums,           // chunk_num<[n_heads, n_seqs]>
  int* __restrict__ offsets,          // where each chunk's results(attns, maxs, sums) starts
  const int* __restrict__ begins,
  const int* __restrict__ ends,
  float dim_scale,
  int n_heads) {
    constexpr uint32_t padded_chunk_size = chunk_size + 16 / sizeof(scalar_t);
    constexpr uint32_t padded_head_dim = d_head + 16 / sizeof(scalar_t);
    constexpr uint32_t thread_num = 128;
    constexpr uint32_t warp_size = 32;
    constexpr uint32_t wrap_num = thread_num / warp_size;

    const uint32_t head_idx = blockIdx.x;
    const uint32_t chunk_idx = blockIdx.y;
    const uint32_t thread_idx = threadIdx.x;

    const uint32_t warp_idx = thread_idx / warp_size;
    const uint32_t lane_idx = thread_idx % warp_size;

    const int seq_begin = begins[chunk_idx];
    const int seq_end = ends[chunk_idx];
    const int n = seq_end - seq_begin;
    assert(n == n_seqs);

    const uint32_t q_row_offset = seq_begin * n_heads * d_head + head_idx * d_head;
    const uint32_t kv_row_offset = head_idx * chunk_size * d_head;
    const int result_offset = offsets[chunk_idx];
    const uint32_t max_sum_offset = result_offset * n_heads + head_idx * n;
    const uint32_t attn_offset = max_sum_offset * d_head;
    
    const scalar_t* q = query + q_row_offset;
    auto* __restrict__ k = reinterpret_cast<scalar_t*>(keys[chunk_idx]) + kv_row_offset;
    auto* __restrict__ v = reinterpret_cast<scalar_t*>(values[chunk_idx]) + kv_row_offset;
    auto* __restrict__ attn_result = reinterpret_cast<scalar_t*>(attns) + attn_offset;
    float* weight_max = maxs + max_sum_offset;
    float* weight_sum = sums + max_sum_offset;

    extern __shared__ char smem[];
    scalar_t* shared_q = reinterpret_cast<scalar_t*>(smem);
    scalar_t* shared_k = shared_q + n_seqs * padded_head_dim;
    float* shared_weight = reinterpret_cast<float*>(shared_k + chunk_size * padded_head_dim);
    scalar_t* shared_v = shared_k;
    scalar_t* shared_half_score = shared_q;
    float* shared_output = reinterpret_cast<float*>(smem);
    // share v with k
    matrix_multiply_gAB_sC<scalar_t, n_seqs, chunk_size, d_head>(
      q, n_heads * d_head, shared_q, k, shared_k, shared_weight, padded_chunk_size);


    // compute
#pragma unroll
    for (int i = warp_idx; i < n_seqs; i += wrap_num) {
        warp_vector_scale<chunk_size>(shared_weight + i * padded_chunk_size, dim_scale);
        float seq_weight_max = warp_vector_max<chunk_size>(shared_weight + i * padded_chunk_size);
        warp_cal_exp<scalar_t, chunk_size>(shared_weight + i * padded_chunk_size,
                                           shared_half_score + i * padded_chunk_size,
                                           seq_weight_max);
        float seq_weight_sum = warp_vector_sum<chunk_size>(shared_weight + i * padded_chunk_size);
        if (lane_idx == 0) {
            weight_max[i] = seq_weight_max;
            weight_sum[i] = seq_weight_sum;
        }
    }

    __syncthreads();

    matrix_multiply_sA_gBC<scalar_t, n_seqs, d_head, chunk_size>(
      shared_half_score, v, shared_v, attn_result, padded_chunk_size, d_head);
}

template<typename scalar_t, int n_seqs, int chunk_size, int d_head>
__global__ void attn_chunk_first_kernel_v2(
  const scalar_t* __restrict__ query, // [n_heads, n_seqs, head_size]
  void** __restrict__ keys,           // chunk_num<[n_heads, chunk_size, d_head]>
  void** __restrict__ values,         // chunk_num<[n_heads, chunk_size, d_head]>
  scalar_t* __restrict__ attns,       // chunk_num<[n_heads, n_seqs , d_head]>
  float* __restrict__ maxs,           // chunk_num<[n_heads, n_seqs]>
  float* __restrict__ sums,           // chunk_num<[n_heads, n_seqs]>
  int* __restrict__ offsets,          // where each chunk's results(attns, maxs, sums) starts
  const int* __restrict__ begins,
  const int* __restrict__ ends,
  float dim_scale,
  int n_heads) {
    constexpr uint32_t padded_chunk_size = chunk_size + 16 / sizeof(scalar_t);
    constexpr uint32_t padded_head_dim = d_head + 16 / sizeof(scalar_t);
    constexpr uint32_t thread_num = 128;
    constexpr uint32_t warp_size = 32;
    constexpr uint32_t wrap_num = thread_num / warp_size;

    const uint32_t head_idx = blockIdx.x;
    const uint32_t chunk_idx = blockIdx.y;
    const uint32_t thread_idx = threadIdx.x;

    const uint32_t warp_idx = thread_idx / warp_size;
    const uint32_t lane_idx = thread_idx % warp_size;

    const int seq_begin = begins[chunk_idx];
    const int seq_end = ends[chunk_idx];
    const int n = seq_end - seq_begin;
//    assert(n == n_seqs);

    const uint32_t q_row_offset = seq_begin * n_heads * d_head + head_idx * d_head;
    const uint32_t kv_row_offset = head_idx * chunk_size * d_head;
    const int result_offset = offsets[chunk_idx];
    const uint32_t max_sum_offset = result_offset * n_heads + head_idx * n;
    const uint32_t attn_offset = max_sum_offset * d_head;

    const scalar_t* q = query + q_row_offset;
    auto* __restrict__ k = reinterpret_cast<scalar_t*>(keys[chunk_idx]) + kv_row_offset;
    auto* __restrict__ v = reinterpret_cast<scalar_t*>(values[chunk_idx]) + kv_row_offset;
    auto* __restrict__ attn_result = reinterpret_cast<scalar_t*>(attns) + attn_offset;
    float* weight_max = maxs + max_sum_offset;
    float* weight_sum = sums + max_sum_offset;

    extern __shared__ char smem[];
    scalar_t* shared_q = reinterpret_cast<scalar_t*>(smem);
    scalar_t* shared_k = shared_q + n_seqs * padded_head_dim;
    float* shared_weight = reinterpret_cast<float*>(shared_k + chunk_size * padded_head_dim);

    scalar_t* shared_half_score = shared_q;
    scalar_t* shared_v = shared_half_score + n_seqs * padded_chunk_size;
    scalar_t* shared_output = shared_v + chunk_size * padded_head_dim;
    // share v with k
    matrix_multiply_gAB_sC<scalar_t, n_seqs, chunk_size, d_head>(
      q, n_heads * d_head, shared_q, k, shared_k, shared_weight, padded_chunk_size);

    // compute
#pragma unroll
    for (int i = warp_idx; i < n; i += wrap_num) {
        warp_vector_scale<chunk_size>(shared_weight + i * padded_chunk_size, dim_scale);
        float seq_weight_max = warp_vector_max<chunk_size>(shared_weight + i * padded_chunk_size);
        warp_cal_exp<scalar_t, chunk_size>(shared_weight + i * padded_chunk_size,
                                           shared_half_score + i * padded_chunk_size,
                                           seq_weight_max);
        float seq_weight_sum = warp_vector_sum<chunk_size>(shared_weight + i * padded_chunk_size);
        if (lane_idx == 0) {
            weight_max[i] = seq_weight_max;
            weight_sum[i] = seq_weight_sum;
        }
    }

    __syncthreads();

    if (n == n_seqs) {
        matrix_multiply_sA_gBC<scalar_t, n_seqs, d_head, chunk_size>(
          shared_half_score, v, shared_v, attn_result, padded_chunk_size, d_head);
    }
    else {
        matrix_multiply_sA_gBC<scalar_t, n_seqs, d_head, chunk_size>(
          shared_half_score, v, shared_v, shared_output, padded_chunk_size, padded_head_dim);

        for (int i = 0; i < n; i++) {
            for (int j = thread_idx; j < d_head; j += thread_num) {
                attn_result[i * d_head + j] = shared_output[i * padded_head_dim + j];
            }
        }
    }
}

template<typename scalar_t, int chunk_size, int d_head>
__global__ void attn_seq_first_kernel(
  const scalar_t* __restrict__ query, // [n_heads, n_seqs, d_head]
  void** __restrict__ keys,           // chunk_num<[n_heads, chunk_size, d_head]>
  void** __restrict__ values,         // chunk_num<[n_heads, chunk_size, d_head]>
  scalar_t* __restrict__ output,      // [n_heads, n_seqs, d_head]
  scalar_t* __restrict__ attns,       // chunk_num<[n_heads, n_seqs, d_head]>
  float* __restrict__ maxs,           // chunk_num<[n_heads, n_seqs]>
  float* __restrict__ sums,           // chunk_num<[n_heads, n_seqs]>
  int* __restrict__ offsets,
  const int* __restrict__ begins,
  const int* __restrict__ ends,
  int n_shared_chunks,
  int* __restrict__ seq_chunk_map,
  int seq_chunk_map_stride,
  int* __restrict__ seq_n_tokens,
  int delta_tokens,
  float dim_scale,
  int n_heads,
  int n_seqs) {
    constexpr uint32_t thread_num = 128;
    constexpr uint32_t warp_size = 32;
    constexpr uint32_t warp_num = thread_num / warp_size;
    constexpr uint32_t tokens_per_thread = chunk_size / warp_size;
    constexpr uint32_t dim_per_thread = d_head / warp_size;
    constexpr uint32_t padding_head_dim = d_head + 16 / sizeof(scalar_t);

    // static_assert(chunk_size % warp_size == 0, "chunk_size must be divided by warp_size");
    // static_assert(d_head % warp_size == 0, "d_head must be divided by warp_size");

    const uint32_t head_idx = blockIdx.x;
    const uint32_t seq_idx = blockIdx.y;

    const uint32_t thread_idx = threadIdx.x;
    const uint32_t wrap_idx = thread_idx / warp_size;
    const uint32_t lane_idx = thread_idx % warp_size;

    const uint32_t seq_length = seq_n_tokens[seq_idx] + delta_tokens;
    const uint32_t last_chunk_unmask_token = seq_length % chunk_size;
    const uint32_t chunk_num = (seq_length + chunk_size - 1) / chunk_size;

    const uint32_t q_row_offset = seq_idx * n_heads * d_head + head_idx * d_head;
    const uint32_t kv_row_offset = head_idx * chunk_size * d_head;

    const scalar_t* q = query + q_row_offset;
    scalar_t* output_seq = output + q_row_offset;
    const int* seq_mapping = seq_chunk_map + seq_idx * seq_chunk_map_stride;

    __shared__ scalar_t shared_q[d_head];
    __shared__ scalar_t shared_output[warp_num * padding_head_dim];
    __shared__ scalar_t shared_score[warp_num * chunk_size];
    __shared__ float shared_score_max[warp_num];
    __shared__ float shared_score_sum[warp_num];
    extern __shared__ char smem[];
    scalar_t* shared_kv =
      reinterpret_cast<scalar_t*>(smem) + wrap_idx * chunk_size * padding_head_dim;
    // load shared q
#pragma unroll
    for (int i = thread_idx; i < d_head; i += thread_num) {
        shared_q[i] = q[i];
    }
#pragma unroll
    for (int i = 0; i < dim_per_thread; i++) {
        shared_output[wrap_idx * padding_head_dim + lane_idx + i * warp_size] = 0;
    }
    __syncthreads();

    // each warp compute one chunk
    // warp 0 -> chunk: 0 4 8 12...
    // warp 1 -> chunk: 1 5 9 13...
    float score_max = -FLT_MAX;
    float score_sum = 0;
    for (int i = wrap_idx; i < chunk_num; i += warp_num) {
        const int chunk_idx = seq_mapping[i];
        scalar_t* shared_output_chunk = shared_output + wrap_idx * padding_head_dim;
        scalar_t* shared_chunk_score = shared_score + wrap_idx * chunk_size;

        // merge existing result
        if (chunk_idx < n_shared_chunks) {
            int result_offset = offsets[chunk_idx];
            int seq_begin = begins[chunk_idx];
            int seq_end = ends[chunk_idx];
            int max_sum_offset =
              result_offset * n_heads + head_idx * (seq_end - seq_begin) + seq_idx - seq_begin;
            float cached_max = maxs[max_sum_offset];
            float cached_sum = sums[max_sum_offset];
            int attn_offset = max_sum_offset * d_head;
            scalar_t* cached_qkv_result = attns + attn_offset;

            float new_score_max = fmax(score_max, cached_max);
            float cached_scale =
              __shfl_sync(0xffffffff, lane_idx == 0 ? expf(cached_max - new_score_max) : 0, 0);
            float scale =
              __shfl_sync(0xffffffff, lane_idx == 0 ? expf(score_max - new_score_max) : 0, 0);
            score_max = new_score_max;
            score_sum = cached_sum * cached_scale + score_sum * scale;
            warp_vector_merge<scalar_t, d_head>(
              shared_output_chunk, cached_qkv_result, scale, cached_scale);
            continue;
        }

        const scalar_t* __restrict__ g_k =
          reinterpret_cast<scalar_t*>(keys[chunk_idx]) + kv_row_offset; // [chunk_size, d_head]
        const scalar_t* __restrict__ g_v =
          reinterpret_cast<scalar_t*>(values[chunk_idx]) + kv_row_offset; // [chunk_size, d_head]

        float chunk_score[tokens_per_thread] = { 0 };

        warp_vect_mul_raw_major_matrix_v2<scalar_t, chunk_size, d_head>(
          shared_q, g_k, shared_kv, chunk_score, dim_scale);
        if (i == (chunk_num - 1) && last_chunk_unmask_token != 0) {
            for (int j = 0; j < tokens_per_thread; j += 1) {
                if (j * warp_size + lane_idx >= last_chunk_unmask_token) {
                    chunk_score[j] = -FLT_MAX;
                }
            }
        }

        float chunk_score_max = score_max;
#pragma unroll
        for (int j = 0; j < tokens_per_thread; j++) {
            chunk_score_max = fmaxf(chunk_score_max, chunk_score[j]);
        }
        //        __syncwarp();

        // warp reduce max
#pragma unroll
        for (int mask = warp_size / 2; mask >= 1; mask /= 2) {
            chunk_score_max =
              fmaxf(chunk_score_max, __shfl_xor_sync(uint32_t(-1), chunk_score_max, mask));
        }

        // compute score and store to smem
        float chunk_score_sum = 0;
#pragma unroll
        for (int j = 0; j < tokens_per_thread; j++) {
            chunk_score[j] = expf(chunk_score[j] - chunk_score_max);
            shared_chunk_score[j * warp_size + lane_idx] = __float2half(chunk_score[j]);
            chunk_score_sum += chunk_score[j];
        }
        __syncwarp();

        // warp reduce sum
#pragma unroll
        for (int mask = warp_size / 2; mask >= 1; mask /= 2) {
            chunk_score_sum += __shfl_xor_sync(uint32_t(-1), chunk_score_sum, mask);
        }

        float score_scale =
          __shfl_sync(0xffffffff, lane_idx == 0 ? expf(score_max - chunk_score_max) : 0, 0);
        score_max = chunk_score_max;
        score_sum = score_sum * score_scale + chunk_score_sum;

        warp_vect_mul_col_major_matrix<scalar_t, d_head, chunk_size>(
          shared_chunk_score, g_v, shared_kv, shared_output_chunk, score_scale);
        //        __syncwarp();
    }
    if (lane_idx == 0) {
        shared_score_max[wrap_idx] = score_max;
        shared_score_sum[wrap_idx] = score_sum;
    }
    __syncthreads();

    float scale[warp_num];
    float div = 0;

    // only one thread compute the scale in the warp
    if (lane_idx == 0) {
        score_max = shared_score_max[0];
        score_sum = 0;
#pragma unroll
        for (int i = 1; i < warp_num; i++) {
            score_max = fmaxf(score_max, shared_score_max[i]);
        }
#pragma unroll
        for (int i = 0; i < warp_num; i++) {
            scale[i] = expf(shared_score_max[i] - score_max);
            score_sum += shared_score_sum[i] * scale[i];
        }
        div = __fdividef(1.f, score_sum + 1e-6f);
    }
    __syncwarp();

    // sync with other threads
#pragma unroll
    for (int i = 0; i < warp_num; i++) {
        scale[i] = __shfl_sync(0xffffffff, scale[i], 0);
    }
    div = __shfl_sync(0xffffffff, div, 0);

    // compute output
#pragma unroll
    for (int i = thread_idx; i < d_head; i += thread_num) {
        float output_dim = 0;
#pragma unroll
        for (int j = 0; j < warp_num; j++) {
            output_dim +=
              TypeTraits<scalar_t>::to_float(shared_output[j * padding_head_dim + i]) * scale[j];
        }
        output_seq[i] = TypeTraits<scalar_t>::float_to_scalar(output_dim * div);
    }
}

template<typename scalar_t, int chunk_size, int d_head>
__global__ void append_kv_kernel(
  const scalar_t* __restrict__ new_keys, // [n_seqs, n_heads, d_head]
  const scalar_t* __restrict__ new_values,
  scalar_t** __restrict__ cached_keys,   // [n_heads, chunk_size, d_head]
  scalar_t** __restrict__ cached_values, // [n_heads, chunk_size, d_head]
  int* __restrict__ seq_chunk_map,
  int seq_chunk_map_stride,
  int* __restrict__ seq_n_tokens,
  int delta_tokens,
  int n_heads) {
    const uint32_t head_idx = blockIdx.x;
    const uint32_t seq_idx = blockIdx.y;
    const uint32_t thread_idx = threadIdx.x;

    int seq_length = seq_n_tokens[seq_idx] + delta_tokens;
    const int* seq_mapping = seq_chunk_map + seq_idx * seq_chunk_map_stride;
    int last_chunk_idx = seq_mapping[seq_length / chunk_size];
    scalar_t* last_chunk_key = cached_keys[last_chunk_idx];
    scalar_t* last_chunk_value = cached_values[last_chunk_idx];

    int token_idx = seq_length % chunk_size;
    int src_offset = seq_idx * n_heads * d_head + head_idx * d_head + thread_idx;
    int dst_offset = head_idx * chunk_size * d_head + token_idx * d_head + thread_idx;
    // TODO: need to handle d_head is not 128
    last_chunk_key[dst_offset] = new_keys[src_offset];
    last_chunk_value[dst_offset] = new_values[src_offset];
}


__host__ void attn_chunks_first(KernelContext& context, torch::Tensor& query) {
    uint32_t batch_size = ((query.size(0) - 1) / 16 + 1) * 16;
    uint32_t elem_size = query.element_size();
    int chunk_size = context.chunk_size;
    float scale = 1.f / sqrtf(static_cast<float>(context.d_head));

    size_t query_shared_mem_size =
      batch_size * context.d_head * elem_size + 16 * batch_size; // 16 is for padding
    size_t keys_shared_mem_size = chunk_size * context.d_head * elem_size + 16 * chunk_size;
    size_t score_shared_mem_size = (chunk_size + 16) * batch_size * sizeof(float);
    size_t shared_mem_size = query_shared_mem_size + keys_shared_mem_size + score_shared_mem_size;

    dim3 grid(context.n_heads, context.n_shared_chunks);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define CALL_ATTN_CHUNK_KERNEL_FUNCTION(scalar_t, batch_size, chunk_size, d_head)                  \
    do {                                                                                           \
        if (shared_mem_size >= 48 * 1024) {                                                        \
            C10_CUDA_CHECK(cudaFuncSetAttribute(                                                   \
              attn_chunk_first_kernel_v2<scalar_t, batch_size, chunk_size, d_head>,                   \
              cudaFuncAttributeMaxDynamicSharedMemorySize,                                         \
              shared_mem_size));                                                                   \
        }                                                                                          \
        attn_chunk_first_kernel_v2<scalar_t, batch_size, chunk_size, d_head>                          \
          <<<grid, 128, shared_mem_size, stream>>>(query_ptr,                                      \
                                                   keys_ptr,                                       \
                                                   values_ptr,                                     \
                                                   attns_ptr,                                      \
                                                   maxs_ptr,                                       \
                                                   sums_ptr,                                       \
                                                   offsets_ptr,                                    \
                                                   begins_ptr,                                     \
                                                   ends_ptr,                                       \
                                                   scale,                                          \
                                                   context.n_heads);                               \
    } while (0)
//    std::cout << "v2 chunk first" << std::endl;
    if (query.dtype() == at::ScalarType::Half) {
        half* query_ptr = reinterpret_cast<half*>(query.data_ptr());
        void** keys_ptr = reinterpret_cast<void**>(context.keys_values.data_ptr());
        void** values_ptr = keys_ptr + context.keys_values.stride(0);
        half* attns_ptr = reinterpret_cast<half*>(context.attns.data_ptr());
        float* maxs_ptr = context.maxs_sums.data_ptr<float>();
        float* sums_ptr = context.maxs_sums.data_ptr<float>() + context.maxs_sums.stride(0);
        int* begins_ptr = context.begins_ends_offsets.data_ptr<int>();
        int* ends_ptr = begins_ptr + context.begins_ends_offsets.stride(0);
        int* offsets_ptr = ends_ptr + context.begins_ends_offsets.stride(0);

        if (batch_size == 16 && chunk_size == 64) {
            CALL_ATTN_CHUNK_KERNEL_FUNCTION(half, 16, 64, 128);
        } else if (batch_size == 32 && chunk_size == 32) {
            CALL_ATTN_CHUNK_KERNEL_FUNCTION(half, 32, 32, 128);
        } else if (batch_size == 32 && chunk_size == 64) {
            CALL_ATTN_CHUNK_KERNEL_FUNCTION(half, 32, 64, 128);
        } else if (batch_size == 48 && chunk_size == 64) {
            CALL_ATTN_CHUNK_KERNEL_FUNCTION(half, 48, 64, 128);
        } else if (batch_size == 64 && chunk_size == 64) {
            CALL_ATTN_CHUNK_KERNEL_FUNCTION(half, 64, 64, 128);
        } else if (batch_size == 96 && chunk_size == 64) {
            CALL_ATTN_CHUNK_KERNEL_FUNCTION(half, 96, 64, 128);
        } else if (batch_size == 128 && chunk_size == 64) {
            CALL_ATTN_CHUNK_KERNEL_FUNCTION(half, 128, 64, 128);
        } else if (batch_size == 64 && chunk_size == 32) {
            CALL_ATTN_CHUNK_KERNEL_FUNCTION(half, 64, 32, 128);
        } else if (batch_size == 64 && chunk_size == 128) {
            CALL_ATTN_CHUNK_KERNEL_FUNCTION(half, 64, 128, 128);
        } else {
            LOG_ERROR("unsupported chunk_size {} or batch_size {}", chunk_size, batch_size);
            TORCH_CHECK(false, "unsupported chunk_size ", chunk_size, " or batch_size ", batch_size);
        }
    } else {
        LOG_ERROR("unsupported data type {}", query.dtype());
        TORCH_CHECK(false, "unsupported data type ", query.dtype());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__host__ void attn_seqs_first(KernelContext& context,
                              const torch::Tensor& query,
                              torch::Tensor& output) {
    uint32_t n_seqs = query.size(0);
    if (context.d_head != 128) {
        LOG_ERROR("unsupported head dim {}", context.d_head);
        TORCH_CHECK(false, "unsupported head dim ", context.d_head);
    }

    float scale = 1.f / sqrtf(static_cast<float>(context.d_head));
    int shared_mem_size =
      4 * (context.d_head * context.chunk_size * sizeof(half) + context.chunk_size * 16);
    dim3 grid(context.n_heads, n_seqs);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define CALL_ATTN_SEQ_KERNEL_FUNCTION(scalar_t, chunk_size, d_head)                                \
    do {                                                                                           \
        if (shared_mem_size >= 48 * 1024) {                                                        \
            C10_CUDA_CHECK(                                                                        \
              cudaFuncSetAttribute(attn_seq_first_kernel<scalar_t, chunk_size, d_head>,            \
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,                    \
                                   shared_mem_size));                                              \
        }                                                                                          \
        attn_seq_first_kernel<scalar_t, chunk_size, d_head>                                        \
          <<<grid, 128, shared_mem_size, stream>>>(query_ptr,                                      \
                                                   keys_ptr,                                       \
                                                   values_ptr,                                     \
                                                   output_ptr,                                     \
                                                   attns_ptr,                                      \
                                                   maxs_ptr,                                       \
                                                   sums_ptr,                                       \
                                                   offsets_ptr,                                    \
                                                   begins_ptr,                                     \
                                                   ends_ptr,                                       \
                                                   n_shared_chunks,                                \
                                                   seq_chunk_map_ptr,                              \
                                                   seq_chunk_map_stride,                           \
                                                   seq_n_tokens_ptr,                               \
                                                   delta_tokens,                                   \
                                                   scale,                                          \
                                                   context.n_heads,                                 \
                                                   n_seqs);                                        \
    } while (0)

    if (query.dtype() == at::ScalarType::Half) {
        half* query_ptr = reinterpret_cast<half*>(query.data_ptr());
        void** keys_ptr = reinterpret_cast<void**>(context.keys_values.data_ptr());
        void** values_ptr = keys_ptr + context.keys_values.stride(0);
        half* output_ptr = reinterpret_cast<half*>(output.data_ptr());
        half* attns_ptr = reinterpret_cast<half*>(context.attns.data_ptr());
        float* maxs_ptr = context.maxs_sums.data_ptr<float>();
        float* sums_ptr = maxs_ptr + context.maxs_sums.stride(0);
        int* begins_ptr = context.begins_ends_offsets.data_ptr<int>();
        int* ends_ptr = begins_ptr + context.begins_ends_offsets.stride(0);
        int* offsets_ptr = ends_ptr + context.begins_ends_offsets.stride(0);
        int n_shared_chunks = context.n_shared_chunks;
        int* seq_chunk_map_ptr = context.seq_chunk_map.data_ptr<int>();
        int seq_chunk_map_stride = context.seq_chunk_map.stride(0);
        int* seq_n_tokens_ptr = context.seq_n_tokens.data_ptr<int>();
        int delta_tokens = context.delta_tokens;

        if (context.chunk_size == 32) {
            CALL_ATTN_SEQ_KERNEL_FUNCTION(half, 32, 128);
        } else if (context.chunk_size == 64) {
            CALL_ATTN_SEQ_KERNEL_FUNCTION(half, 64, 128);
        } else if (context.chunk_size == 128) {
            CALL_ATTN_SEQ_KERNEL_FUNCTION(half, 128, 128);
        } else {
            LOG_ERROR("unsupported chunk_size {}", context.chunk_size);
            TORCH_CHECK(false, "unsupported chunk_size ", context.chunk_size);
        }
    } else {
        LOG_ERROR("unsupported data type {}", query.dtype());
        TORCH_CHECK(false, "unsupported data type ", query.dtype());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__host__ void append_kv(KernelContext& context,
                        const torch::Tensor& keys,
                        const torch::Tensor& values) {
    uint32_t n_seqs = keys.size(0);
    int chunk_size = context.chunk_size;
    dim3 grid(context.n_heads, n_seqs);

#define CALL_APPEND_KV_KERNEL_FUNCTION(scalar_t, chunk_size, d_head)                               \
    do {                                                                                           \
        append_kv_kernel<scalar_t, chunk_size, d_head><<<grid, 128>>>(new_keys_ptr,                \
                                                                      new_values_ptr,              \
                                                                      cached_keys_ptr,             \
                                                                      cached_values_ptr,           \
                                                                      seq_chunk_map_ptr,           \
                                                                      seq_chunk_map_stride,        \
                                                                      seq_n_tokens_ptr,            \
                                                                      delta_tokens,                \
                                                                      context.n_heads);            \
    } while (0)

    if (keys.dtype() == at::ScalarType::Half) {
        half* new_keys_ptr = reinterpret_cast<half*>(keys.data_ptr());
        half* new_values_ptr = reinterpret_cast<half*>(values.data_ptr());
        half** cached_keys_ptr = reinterpret_cast<half**>(context.keys_values.data_ptr());
        half** cached_values_ptr = cached_keys_ptr + context.keys_values.stride(0);
        int* seq_chunk_map_ptr = context.seq_chunk_map.data_ptr<int>();
        int seq_chunk_map_stride = context.seq_chunk_map.stride(0);
        int* seq_n_tokens_ptr = context.seq_n_tokens.data_ptr<int>();
        int delta_tokens = context.delta_tokens;

        if (chunk_size == 32) {
            CALL_APPEND_KV_KERNEL_FUNCTION(half, 32, 128);
        } else if (chunk_size == 64) {
            CALL_APPEND_KV_KERNEL_FUNCTION(half, 64, 128);
        } else if (chunk_size == 128)
            CALL_APPEND_KV_KERNEL_FUNCTION(half, 128, 128);
        else {
            LOG_ERROR("unsupported chunk_size {}", chunk_size);
            TORCH_CHECK(false, "unsupported chunk_size ", chunk_size);
        }
    } else {
        LOG_ERROR("unsupported data type {}", keys.dtype());
        TORCH_CHECK(false, "unsupported data type ", keys.dtype());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void attention(KernelContext& context, torch::Tensor q, torch::Tensor& output, int partition) {
    int n_seqs = q.size(0);
    int n_shared_chunks = context.n_shared_chunks;
    // trunk first, then seq first
    if (n_shared_chunks > 0) {
        attn_chunks_first(context, q);
    }

    attn_seqs_first(context, q, output);
}

KernelContext& refresh_kernel_context(KernelContext& context,
                                      std::vector<ChunkInfo>& chunk_infos,
                                      int n_seqs) {
    NVTX3_FUNC_RANGE();

    torch::TensorOptions pinned_options =
      torch::TensorOptions().device(torch::kCPU).pinned_memory(true);
    auto device = context.options.device();

    nvtxRangePushA("create data on host");
    int n_chunks = chunk_infos.size();

    int n_shared_chunks = 0;
    for (auto& chunk_info : chunk_infos) {
        int n = chunk_info.seq_idx_end - chunk_info.seq_idx_begin;
        if (n > context.tpp_threshold) {
            n_shared_chunks++;
        }
    }

    torch::Tensor begins_ends_offsets_host =
      torch::empty({ 3, n_shared_chunks }, pinned_options.dtype(torch::kInt32));
    int* begins_host_ptr = begins_ends_offsets_host.data_ptr<int>();
    int* ends_host_ptr = begins_host_ptr + n_shared_chunks;
    int* offsets_host_ptr = ends_host_ptr + n_shared_chunks;
    int shared_chunks_visited = 0;
    int partial_attn_dim = 0;
    for (auto& chunk_info : chunk_infos) {
        int n = chunk_info.seq_idx_end - chunk_info.seq_idx_begin;
        if (n <= context.tpp_threshold) {
            continue;
        }

        begins_host_ptr[shared_chunks_visited] = chunk_info.seq_idx_begin;
        ends_host_ptr[shared_chunks_visited] = chunk_info.seq_idx_end;
        offsets_host_ptr[shared_chunks_visited] = partial_attn_dim;
        partial_attn_dim += n;
        shared_chunks_visited++;
    }

    torch::Tensor seq_n_tokens_host = torch::zeros({ n_seqs }, pinned_options.dtype(torch::kInt32));
    int* seq_n_tokens_host_ptr = seq_n_tokens_host.data_ptr<int>();
    for (auto& chunk_info : chunk_infos) {
        for (int j = chunk_info.seq_idx_begin; j < chunk_info.seq_idx_end; j++) {
            seq_n_tokens_host_ptr[j] += chunk_info.chunk->n_tokens();
        }
    }
    int max_seq_len = 0;
    for (int i = 0; i < n_seqs; i++) {
        max_seq_len = std::max(max_seq_len, seq_n_tokens_host_ptr[i]);
    }
    int max_seq_n_chunks = max_seq_len / context.chunk_size + 1;

    torch::Tensor keys_values_host =
      torch::empty({ 2, n_chunks }, pinned_options.dtype(torch::kInt64));
    void** keys_host_ptr = reinterpret_cast<void**>(keys_values_host.data_ptr());
    void** values_host_ptr = keys_host_ptr + n_chunks;
    torch::Tensor seq_chunk_map_host =
      torch::empty({ n_seqs, max_seq_n_chunks }, pinned_options.dtype(torch::kInt32));
    int* seq_chunk_map_host_ptr = seq_chunk_map_host.data_ptr<int>();
    int seq_chunk_map_stride = seq_chunk_map_host.stride(0);
    std::vector<int> seq_n_chunks(n_seqs, 0);
    int s1 = 0, s2 = n_shared_chunks;
    for (auto& chunk_info : chunk_infos) {
        int final_idx = -1;
        int n = chunk_info.seq_idx_end - chunk_info.seq_idx_begin;
        if (n <= context.tpp_threshold) {
            final_idx = s2;
            s2++;
        } else {
            final_idx = s1;
            s1++;
        }
        keys_host_ptr[final_idx] = chunk_info.chunk->key_ptr();
        values_host_ptr[final_idx] = chunk_info.chunk->value_ptr();
        for (int j = chunk_info.seq_idx_begin; j < chunk_info.seq_idx_end; j++) {
            seq_chunk_map_host_ptr[j * seq_chunk_map_stride + seq_n_chunks[j]] = final_idx;
            seq_n_chunks[j]++;
        }
    }
    nvtxRangePop();

    nvtxRangePushA("transfer begins ends offsets");
    torch::Tensor begins_ends_offsets_new =
      begins_ends_offsets_host.to(device, torch::kInt32, true);
    nvtxRangePop();

    nvtxRangePushA("transfer keys values");
    torch::Tensor keys_values_new = keys_values_host.to(device, torch::kInt64, true);
    nvtxRangePop();

    nvtxRangePushA("transfer seq_n_tokens_new");
    torch::Tensor seq_n_tokens_new = seq_n_tokens_host.to(device, torch::kInt32, true);
    nvtxRangePop();

    nvtxRangePushA("transfer seq_chunk_map");
    torch::Tensor seq_chunk_map_new = seq_chunk_map_host.to(device, torch::kInt, false);
    nvtxRangePop();

    nvtxRangePushA("resize partial attn results");
    context.attns.resize_({ partial_attn_dim, context.n_heads, context.d_head });
    context.maxs_sums.resize_({ 2, partial_attn_dim, context.n_heads });
    nvtxRangePop();

    context.keys_values = keys_values_new;
    context.begins_ends_offsets = begins_ends_offsets_new;
    context.seq_chunk_map = seq_chunk_map_new;
    context.seq_n_tokens = seq_n_tokens_new;

    context.n_chunks = n_chunks;
    context.n_shared_chunks = n_shared_chunks;
    context.valid = true;
    context.delta_tokens = 0;
    return context;
}

}
