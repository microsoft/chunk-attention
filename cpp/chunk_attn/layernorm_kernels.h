/*
 * Adapted from https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu
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

#include <torch/torch.h>
#include "cuda_compat.h"

namespace vllm {

void rms_norm(torch::Tensor& out,    // [..., hidden_size]
              torch::Tensor& input,  // [..., hidden_size]
              torch::Tensor& weight, // [hidden_size]
              float epsilon);

void fused_add_rms_norm(torch::Tensor& input,    // [..., hidden_size]
                        torch::Tensor& residual, // [..., hidden_size]
                        torch::Tensor& weight,   // [hidden_size]
                        float epsilon);

} // namespace vllm
