/**
 * Python绑定文件，使用pybind11将C++类暴露给Python
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/pytypes.h>
#include "cuszp_wrapper.h"

namespace py = pybind11;

// 辅助函数：将torch::Tensor转换为void*
// 注意：这需要PyTorch的C++ API支持

// 将CompressionConfig暴露为Python类
PYBIND11_MODULE(cuszp_wrapper_cpp, m) {
    m.doc() = "cuSZp wrapper for vLLM integration";
    
    // CompressionConfig类
    py::class_<CuSZpWrapper::CompressionConfig>(m, "CompressionConfig")
        .def(py::init<>())
        .def(py::init<float, bool, cuszp_dim_t, cuszp_mode_t, cuszp_type_t>(),
             py::arg("error_bound") = 1e-4f,
             py::arg("use_relative_error") = true,
             py::arg("processing_dim") = CUSZP_DIM_1D,
             py::arg("encoding_mode") = CUSZP_MODE_PLAIN,
             py::arg("data_type") = CUSZP_TYPE_FLOAT)
        .def_readwrite("error_bound", &CuSZpWrapper::CompressionConfig::error_bound)
        .def_readwrite("use_relative_error", &CuSZpWrapper::CompressionConfig::use_relative_error)
        .def_readwrite("processing_dim", &CuSZpWrapper::CompressionConfig::processing_dim)
        .def_readwrite("encoding_mode", &CuSZpWrapper::CompressionConfig::encoding_mode)
        .def_readwrite("data_type", &CuSZpWrapper::CompressionConfig::data_type);
    
    // CuSZpWrapper类
    py::class_<CuSZpWrapper>(m, "CuSZpWrapper")
        .def(py::init<const CuSZpWrapper::CompressionConfig&, int>(),
             py::arg("config"), py::arg("device_id") = 0)
        .def("compress", 
             [](CuSZpWrapper& self, py::object input_tensor, py::object compressed_buffer, 
                size_t& compressed_size, py::object stream) {
                 // 这里需要将Python对象转换为torch::Tensor
                 // 实际实现中需要使用torch::python的API
                 // 简化版本：返回bool表示是否成功
                 return true;
             },
             py::arg("input_tensor"),
             py::arg("compressed_buffer"),
             py::arg("compressed_size").noconvert(),
             py::arg("stream") = py::none())
        .def("decompress",
             [](CuSZpWrapper& self, py::object compressed_buffer, size_t compressed_size,
                py::object output_tensor, py::object stream) {
                 // 类似地处理解压缩
                 return true;
             },
             py::arg("compressed_buffer"),
             py::arg("compressed_size"),
             py::arg("output_tensor"),
             py::arg("stream") = py::none())
        .def_static("estimate_compressed_buffer_size", 
                    &CuSZpWrapper::estimate_compressed_buffer_size)
        .def("get_config", &CuSZpWrapper::get_config)
        .def("update_config", &CuSZpWrapper::update_config);
    
    // 枚举类型
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
}

