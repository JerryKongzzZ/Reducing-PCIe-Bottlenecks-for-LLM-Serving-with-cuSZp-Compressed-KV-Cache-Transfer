#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cuszp_wrapper.h"

namespace py = pybind11;

PYBIND11_MODULE(cuszp_wrapper_cpp, m) {
    m.doc() = "cuSZp wrapper for vLLM integration";
    
    // 👉 Step 1: Define all enum types first!
    py::enum_<cuszp_dim_t>(m, "CuszpDim")
        .value("DIM_1D", CUSZP_DIM_1D)
        .value("DIM_2D", CUSZP_DIM_2D)
        .value("DIM_3D", CUSZP_DIM_3D)
        .export_values();
    
    py::enum_<cuszp_mode_t>(m, "CuszpMode")
        .value("MODE_FIXED", CUSZP_MODE_FIXED)
        .value("MODE_PLAIN", CUSZP_MODE_PLAIN)
        .value("MODE_OUTLIER", CUSZP_MODE_OUTLIER)
        .export_values();
    
    py::enum_<cuszp_type_t>(m, "CuszpType")
        .value("TYPE_FLOAT", CUSZP_TYPE_FLOAT)
        .value("TYPE_DOUBLE", CUSZP_TYPE_DOUBLE)
        .export_values();

    // 👉 Step 2: Then define CompressionConfig class
    py::class_<CuSZpWrapper::CompressionConfig>(m, "CompressionConfig")
        .def(py::init<>())
        .def(py::init<float, bool, cuszp_dim_t, cuszp_mode_t, cuszp_type_t>(),
             py::arg("error_bound") = 1e-4f,
             py::arg("use_relative_error") = true,
             py::arg("processing_dim") = CUSZP_DIM_1D,  // PyBind11 recognizes it now!
             py::arg("encoding_mode") = CUSZP_MODE_PLAIN,
             py::arg("data_type") = CUSZP_TYPE_FLOAT)
        .def_readwrite("error_bound", &CuSZpWrapper::CompressionConfig::error_bound)
        .def_readwrite("use_relative_error", &CuSZpWrapper::CompressionConfig::use_relative_error)
        .def_readwrite("processing_dim", &CuSZpWrapper::CompressionConfig::processing_dim)
        .def_readwrite("encoding_mode", &CuSZpWrapper::CompressionConfig::encoding_mode)
        .def_readwrite("data_type", &CuSZpWrapper::CompressionConfig::data_type);
    
    // CuSZpWrapper class
    py::class_<CuSZpWrapper>(m, "CuSZpWrapper")
        .def(py::init<const CuSZpWrapper::CompressionConfig&, int>(),
             py::arg("config"), py::arg("device_id") = 0)
        .def("compress", 
             [](CuSZpWrapper& self, torch::Tensor input_tensor, torch::Tensor compressed_buffer, float eps_override) {
                 size_t compressed_size = 0;
                 float actual_error_bound = 0.0f;
                 // Automatically get the CUDA stream currently used by PyTorch to synchronize with vLLM
                 cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
                 
                 // Call the actual C++ method with per-call eps_override
                 bool success = self.compress(input_tensor, compressed_buffer, compressed_size, actual_error_bound, eps_override, stream);
                 
                 // Return Tuple to Python: (success flag, potentially reallocated buffer, actual compressed size, actual error bound)
                 return py::make_tuple(success, compressed_buffer, compressed_size, actual_error_bound);
             },
             py::arg("input_tensor"),
             py::arg("compressed_buffer"),
             py::arg("eps_override") = -1.0f)
        .def("compress_batch_fixed",
             [](CuSZpWrapper& self,
                const std::vector<torch::Tensor>& input_tensors,
                const std::vector<torch::Tensor>& compressed_buffers,
                const std::vector<float>& eps_overrides) {
                 std::vector<size_t> compressed_sizes;
                 std::vector<float> actual_error_bounds;
                 cudaStream_t stream =
                     c10::cuda::getCurrentCUDAStream().stream();
                 const bool success = self.compress_batch_fixed(
                     input_tensors, compressed_buffers, eps_overrides,
                     compressed_sizes, actual_error_bounds, stream);
                 return py::make_tuple(
                     success, compressed_sizes, actual_error_bounds);
             },
             py::arg("input_tensors"),
             py::arg("compressed_buffers"),
             py::arg("eps_overrides"))

        .def("compress_batch_fixed_bf16",
             [](CuSZpWrapper& self,
                const std::vector<torch::Tensor>& input_tensors,
                const std::vector<torch::Tensor>& compressed_buffers,
                const std::vector<float>& eps_overrides) {
                 std::vector<size_t> compressed_sizes;
                 std::vector<float> actual_error_bounds;
                 cudaStream_t stream =
                     c10::cuda::getCurrentCUDAStream().stream();
                 const bool success = self.compress_batch_fixed_bf16(
                     input_tensors, compressed_buffers, eps_overrides,
                     compressed_sizes, actual_error_bounds, stream);
                 return py::make_tuple(
                     success, compressed_sizes, actual_error_bounds);
             },
             py::arg("input_tensors"),
             py::arg("compressed_buffers"),
             py::arg("eps_overrides"))
             

        .def("compress_batch_fixed_bf16_indexed",
             [](CuSZpWrapper& self,
                const std::vector<torch::Tensor>& input_tensors,
                const std::vector<torch::Tensor>& compressed_buffers,
                torch::Tensor layer_indices,
                size_t prefix_count,
                size_t source_layers,
                size_t elements_per_layer,
                const std::vector<float>& eps_overrides) {
                 std::vector<size_t> compressed_sizes;
                 std::vector<float> actual_error_bounds;
                 cudaStream_t stream =
                     c10::cuda::getCurrentCUDAStream().stream();
                 const bool success =
                     self.compress_batch_fixed_bf16_indexed(
                         input_tensors, compressed_buffers, layer_indices,
                         prefix_count, source_layers, elements_per_layer,
                         eps_overrides, compressed_sizes,
                         actual_error_bounds, stream);
                 return py::make_tuple(
                     success, compressed_sizes, actual_error_bounds);
             },
             py::arg("input_tensors"),
             py::arg("compressed_buffers"),
             py::arg("layer_indices"),
             py::arg("prefix_count"),
             py::arg("source_layers"),
             py::arg("elements_per_layer"),
             py::arg("eps_overrides"))
        .def("decompress",
             [](CuSZpWrapper& self, torch::Tensor compressed_buffer, size_t compressed_size, torch::Tensor output_tensor, float actual_error_bound) {
                 cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
                 bool success = self.decompress(compressed_buffer, compressed_size, output_tensor, actual_error_bound, stream);
                 return success;
             },
             py::arg("compressed_buffer"),
             py::arg("compressed_size"),
             py::arg("output_tensor"),
             py::arg("actual_error_bound"))
        .def("decompress_batch",
             [](CuSZpWrapper& self,
                const std::vector<torch::Tensor>& compressed_buffers,
                const std::vector<size_t>& compressed_sizes,
                const std::vector<torch::Tensor>& output_tensors,
                const std::vector<float>& actual_error_bounds) {
                 cudaStream_t stream =
                     c10::cuda::getCurrentCUDAStream().stream();
                 return self.decompress_batch(
                     compressed_buffers, compressed_sizes, output_tensors,
                     actual_error_bounds, stream);
             },
             py::arg("compressed_buffers"),
             py::arg("compressed_sizes"),
             py::arg("output_tensors"),
             py::arg("actual_error_bounds"))
        .def("decompress_batch_fixed_bf16",
             [](CuSZpWrapper& self,
                const std::vector<torch::Tensor>& compressed_buffers,
                const std::vector<size_t>& compressed_sizes,
                torch::Tensor output_tensor,
                size_t num_elements_per_page,
                const std::vector<float>& actual_error_bounds) {
                 cudaStream_t stream =
                     c10::cuda::getCurrentCUDAStream().stream();
                 return self.decompress_batch_fixed_bf16(
                     compressed_buffers, compressed_sizes, output_tensor,
                     num_elements_per_page, actual_error_bounds, stream);
             },
             py::arg("compressed_buffers"),
             py::arg("compressed_sizes"),
             py::arg("output_tensor"),
             py::arg("num_elements_per_page"),
             py::arg("actual_error_bounds"))
        .def("decompress_batch_fixed_bf16_scatter",
             [](CuSZpWrapper& self,
                const std::vector<torch::Tensor>& compressed_buffers,
                const std::vector<size_t>& compressed_sizes,
                const std::vector<torch::Tensor>& output_tensors,
                size_t num_elements_per_page,
                size_t destinations_per_page,
                size_t elements_per_destination,
                const std::vector<float>& actual_error_bounds) {
                 cudaStream_t stream =
                     c10::cuda::getCurrentCUDAStream().stream();
                 return self.decompress_batch_fixed_bf16_scatter(
                     compressed_buffers, compressed_sizes, output_tensors,
                     num_elements_per_page, destinations_per_page,
                     elements_per_destination, actual_error_bounds, stream);
             },
             py::arg("compressed_buffers"),
             py::arg("compressed_sizes"),
             py::arg("output_tensors"),
             py::arg("num_elements_per_page"),
             py::arg("destinations_per_page"),
             py::arg("elements_per_destination"),
             py::arg("actual_error_bounds"))

        .def("decompress_batch_fixed_bf16_indexed_scatter",
             [](CuSZpWrapper& self,
                const std::vector<torch::Tensor>& compressed_buffers,
                const std::vector<size_t>& compressed_sizes,
                const std::vector<torch::Tensor>& output_tensors,
                torch::Tensor layer_indices,
                size_t prefix_count,
                size_t source_layers,
                size_t elements_per_layer,
                const std::vector<float>& actual_error_bounds) {
                 cudaStream_t stream =
                     c10::cuda::getCurrentCUDAStream().stream();
                 return self.decompress_batch_fixed_bf16_indexed_scatter(
                     compressed_buffers, compressed_sizes, output_tensors,
                     layer_indices, prefix_count, source_layers,
                     elements_per_layer, actual_error_bounds, stream);
             },
             py::arg("compressed_buffers"),
             py::arg("compressed_sizes"),
             py::arg("output_tensors"),
             py::arg("layer_indices"),
             py::arg("prefix_count"),
             py::arg("source_layers"),
             py::arg("elements_per_layer"),
             py::arg("actual_error_bounds"))


             
        .def_static("estimate_compressed_buffer_size", 
                    &CuSZpWrapper::estimate_compressed_buffer_size)
        .def("get_config", &CuSZpWrapper::get_config)
        .def("update_config", &CuSZpWrapper::update_config);
}
