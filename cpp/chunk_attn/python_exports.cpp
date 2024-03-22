#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "attention.h"

namespace GPT {
namespace py = pybind11;

void init_vllm_ops(py::module& m);

PYBIND11_MODULE(chunk_attn_c, m) {
    py::class_<std::future<int>, std::shared_ptr<std::future<int>>>(m, "IntFuture")
      .def(py::init<>())
      .def("get", &std::future<int>::get);

    py::class_<Chunk>(m, "Chunk")
      .def_readonly("children", &Chunk::children, py::return_value_policy::reference)
      .def_readonly("tokens", &Chunk::tokens)
      .def_property_readonly("capacity", &Chunk::capacity)
      .def_readonly("n_seqs", &Chunk::n_seqs)
      .def_property_readonly("n_tokens", &Chunk::n_tokens);

    py::class_<Attention, std::shared_ptr<Attention>>(m, "Attention")
      .def(py::init([](int n_heads,
                       int d_head,
                       int chunk_size,
                       int memory_mb,
                       bool share_prefix,
                       int tpp_threshold,
                       py::object dtype,
                       py::object device) {
               torch::Dtype dtype_c = torch::get_default_dtype_as_scalartype();
               if (!dtype.is_none()) {
                   dtype_c = torch::python::detail::py_object_to_dtype(dtype);
               }
               torch::Device device_c = torch::randn(1).device();
               if (!device.is_none()) {
                   device_c = torch::python::detail::py_object_to_device(device);
               }
               return new Attention(n_heads,
                                    d_head,
                                    chunk_size,
                                    memory_mb,
                                    share_prefix,
                                    tpp_threshold,
                                    dtype_c,
                                    device_c);
           }),
           py::arg("n_heads") = 12,
           py::arg("d_head") = 64,
           py::arg("chunk_size") = 64,
           py::arg("memory_mb") = 1024,
           py::arg("share_prefix") = true,
           py::arg("tpp_threshold") = 1,
           py::arg("dtype") = py::none(),
           py::arg("device") = py::none())
      .def(py::init<Attention&>())
      .def("forward", &GPT::Attention::forward, py::arg("q"), py::arg("partition") = 0)
      .def("add_seq", &GPT::Attention::add_seq, py::arg("tokens"), py::arg("k"), py::arg("v"))
      .def("add_seq_async",
           &GPT::Attention::add_seq_async,
           py::arg("tokens"),
           py::arg("k"),
           py::arg("v"))
      .def("append_token",
           &GPT::Attention::append_token,
           py::arg("tokens"),
           py::arg("k"),
           py::arg("v"),
           py::arg("fused") = true,
           py::return_value_policy::reference)
      .def(
        "refresh_kernel_context", &GPT::Attention::refresh_kernel_context, py::arg("force") = false)
      .def("reserve", &GPT::Attention::reserve, py::return_value_policy::reference)
      .def("duplicate", &GPT::Attention::duplicate, py::arg("seq_idx"), py::arg("copies"))
      .def("remove_seq", &GPT::Attention::remove_seq, py::arg("seq_idx"))
      .def("clear", &GPT::Attention::clear)
      .def("get_trie", &GPT::Attention::get_trie, py::return_value_policy::reference)
      .def_property_readonly("tails", &Attention::tails, py::return_value_policy::reference)
      .def("print", &GPT::Attention::print, py::arg("root") = nullptr, py::arg("level") = 0)
      .def("peak_memory_allocated", &GPT::Attention::peak_memory_allocated);

#ifdef USE_CUDA
    py::module ops = m.def_submodule("ops");
    init_vllm_ops(ops);
#endif
}

} // namespace GPT
