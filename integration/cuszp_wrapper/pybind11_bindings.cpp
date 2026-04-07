#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
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
             [](CuSZpWrapper& self, torch::Tensor input_tensor, torch::Tensor compressed_buffer) {
                 size_t compressed_size = 0;
                 float actual_error_bound = 0.0f;
                 // Automatically get the CUDA stream currently used by PyTorch to synchronize with vLLM
                 cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
                 
                 // Call the actual C++ method
                 bool success = self.compress(input_tensor, compressed_buffer, compressed_size, actual_error_bound, stream);
                 
                 // Return Tuple to Python: (success flag, potentially reallocated buffer, actual compressed size, actual error bound)
                 return py::make_tuple(success, compressed_buffer, compressed_size, actual_error_bound);
             },
             py::arg("input_tensor"),
             py::arg("compressed_buffer"))
             
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
             
        .def_static("estimate_compressed_buffer_size", 
                    &CuSZpWrapper::estimate_compressed_buffer_size)
        .def("get_config", &CuSZpWrapper::get_config)
        .def("update_config", &CuSZpWrapper::update_config);
}
