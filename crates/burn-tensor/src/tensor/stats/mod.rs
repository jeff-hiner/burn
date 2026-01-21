use crate::{FloatDType, Tensor, backend::Backend};

pub fn var<B: Backend, const D: usize>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let mean = tensor.clone().mean_dim(dim);
    var_with_mean(tensor, mean, dim)
}

pub fn var_with_mean<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    mean: Tensor<B, D>,
    dim: usize,
) -> Tensor<B, D> {
    let n = tensor.shape().dims[dim] - 1;
    var_with_mean_n(tensor, mean, dim, n)
}

pub fn var_bias<B: Backend, const D: usize>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let mean = tensor.clone().mean_dim(dim);
    var_with_mean_bias(tensor, mean, dim)
}

pub fn var_with_mean_bias<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    mean: Tensor<B, D>,
    dim: usize,
) -> Tensor<B, D> {
    let n = tensor.shape().dims[dim];
    var_with_mean_n(tensor, mean, dim, n)
}

pub fn var_with_mean_n<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    mean: Tensor<B, D>,
    dim: usize,
    n: usize,
) -> Tensor<B, D> {
    // Compute variance in f32 to avoid overflow when squaring large values in f16
    // (e.g., 800^2 = 640000 exceeds f16 max of 65504)
    let original_dtype: FloatDType = tensor.dtype().into();
    let diff = tensor.sub(mean).cast(FloatDType::F32);
    let var_f32 = diff.square().sum_dim(dim).div_scalar(n as f32);
    var_f32.cast(original_dtype)
}
