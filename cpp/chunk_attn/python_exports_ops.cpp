#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "pos_encoding_kernels.h"
#include "layernorm_kernels.h"

namespace py = pybind11;

namespace GPT {

void init_vllm_ops(py::module& m) {
    // rotary embedding
    m.def("rotary_embedding",
          &vllm::rotary_embedding,
          "Apply GPT-NeoX or GPT-J style rotary embedding to query and key");
    m.def(
      "rms_norm", &vllm::rms_norm, "Apply Root Mean Square (RMS) Normalization to the input tensor.");
    m.def("fused_add_rms_norm", &vllm::fused_add_rms_norm, "In-place fused Add and RMS Normalization");
}

} // namespace GPT
