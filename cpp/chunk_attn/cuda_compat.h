/*
 * Adapted from https://github.com/vllm-project/vllm/blob/main/csrc/cuda_compat.h
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifndef USE_ROCM
  #define VLLM_LDG(arg) __ldg(arg)
#else
  #define VLLM_LDG(arg) *(arg)
#endif

#ifndef USE_ROCM
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#else
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
#endif

#ifndef USE_ROCM
  #define VLLM_SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane)
#else
  #define VLLM_SHFL_SYNC(var, src_lane) __shfl(var, src_lane)
#endif

#ifndef USE_ROCM
  #define VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
    cudaFuncSetAttribute(FUNC, cudaFuncAttributeMaxDynamicSharedMemorySize, VAL)
#else
  #define VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
    hipFuncSetAttribute(FUNC, hipFuncAttributeMaxDynamicSharedMemorySize, VAL)
#endif

