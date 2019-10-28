#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "cbf.h"

namespace py = pybind11;


// Args:
//   depth - the HxW depth image (read in column major order).
//   intensity - the HxW intensity image (read in column major order).
//   noise_mask - the HxW logical noise mask. Values of 1 indicate that the
//                corresponding depth value is missing or noisy.
//   sigma_s - Sx1 vector of sigmas.
//   sigma_r - Sx1 vector of range sigmas.

py::array_t<uint8_t> cbf_filter(py::array_t<uint8_t> depth,
        py::array_t<uint8_t> intensity,
        py::array_t<bool>  noise_mask,
        py::array_t<double> sigma_s,
        py::array_t<double> sigma_r)
{
    py::buffer_info depth_buffer = depth.request();
    py::buffer_info intensity_buffer = intensity.request();
    py::buffer_info noise_mask_buffer = noise_mask.request();
    py::buffer_info sigma_s_buffer = sigma_s.request();
    py::buffer_info sigma_r_buffer = sigma_r.request();

    // Check the types and dimension
    ssize_t H = depth.shape(0), W = depth.shape(1);

    if(!py::isinstance<py::array_t<uint8_t>>(depth))
        throw std::runtime_error("The depth array type should be uint8_t.");
    if(!py::isinstance<py::array_t<uint8_t>>(intensity))
        throw std::runtime_error("The intensity array type should be uint8_t.");
    if(!py::isinstance<py::array_t<bool>>(noise_mask))
        throw std::runtime_error("The noise_mask array type should be bool.");
    if(!py::isinstance<py::array_t<double>>(sigma_s))
        throw std::runtime_error("The sigma_s array type should be double.");
    if(!py::isinstance<py::array_t<double>>(sigma_r))
        throw std::runtime_error("The sigma_r array type should be double.");

    // Check dimension
    if (intensity.shape(0) != H || intensity.shape(1) != W)
        throw std::runtime_error("The intensity and depth dimension should be same.");
    if (noise_mask.shape(0) != H || noise_mask.shape(1) != W)
        throw std::runtime_error("The noise_mask and depth dimension should be same.");
    if (sigma_s.shape(0) != sigma_r.shape(0))
        throw std::runtime_error("The sigma_s and sigma_r should share same dimension.");

    // Run the cbf filtering code
    int num_scales = (int)sigma_s.shape(0);

    auto result = py::array_t<uint8_t>(depth_buffer.size);
    py::buffer_info result_buffer = result.request(true);

    uint8_t *depth_buffer_ptr = (uint8_t *) depth_buffer.ptr;
    uint8_t *intensity_buffer_ptr = (uint8_t *) intensity_buffer.ptr;
    bool *mask_buffer_ptr = (bool*)noise_mask_buffer.ptr;
    double *sigma_s_buffer_ptr = (double *) sigma_s_buffer.ptr;
    double *sigma_r_buffer_ptr = (double *) sigma_r_buffer.ptr;
    uint8_t  *result_buffer_ptr = (uint8_t*) result_buffer.ptr;

    // Run CBF filter
    cbf::cbf(H, W, depth_buffer_ptr, intensity_buffer_ptr, mask_buffer_ptr, result_buffer_ptr, num_scales, sigma_s_buffer_ptr, sigma_r_buffer_ptr);

    result.resize({H, W});
    return result;
}


PYBIND11_MODULE(pycbf_filter, m){
    m.doc() = "Python binding of cbf filter.";
    m.def("cbf_filter", &cbf_filter, "Check the input array");
}