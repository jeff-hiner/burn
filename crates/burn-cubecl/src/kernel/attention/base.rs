use crate::{
    CubeBackend, CubeRuntime,
    kernel::index::{slice, slice_assign},
    ops::numeric::{empty_device_dtype, zeros_client},
    tensor::CubeTensor,
};
#[cfg(feature = "autotune")]
use crate::kernel::attention::attention_autotune;
use burn_backend::{
    DType, Shape, Slice,
    ops::{AttentionModuleOptions, attention::attention_fallback},
};
use cubek::attention::definition::{
    AccumulatorPrecision, AttentionGlobalTypes, AttentionOptions, AttentionSetupError,
};
use cubek::attention::launch;

#[derive(Debug)]
/// Strategy used to select which attention implementation to run.
pub enum AttentionStrategy {
    /// Flash Attention using accelerated inner matmuls.
    FlashBlackboxAccelerated,

    /// Flash Attention using unit inner matmuls.
    FlashUnit,

    /// Fallback implementation using multiple separate kernels.
    Fallback,

    /// Automatically benchmark and select the best strategy at runtime.
    #[cfg(feature = "autotune")]
    Autotune,
}

impl Default for AttentionStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "autotune")]
        return AttentionStrategy::Autotune;

        // if autotune is disabled, default to fallback to make sure it runs
        #[cfg(not(feature = "autotune"))]
        AttentionStrategy::Fallback
    }
}

#[allow(clippy::too_many_arguments)]
/// Launch an attention kernel with given strategy
pub fn attention<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    attn_bias: Option<CubeTensor<R>>,
    options: AttentionModuleOptions,
    strategy: &AttentionStrategy,
    out: Option<CubeTensor<R>>,
) -> Result<CubeTensor<R>, AttentionSetupError> {
    let mut out = out.unwrap_or_else(|| init_attention_output(&query, &value));
    match strategy {
        AttentionStrategy::FlashBlackboxAccelerated => flash_attention(
            query,
            key,
            value,
            mask,
            attn_bias,
            options,
            out,
            launch::Strategy::BlackboxAccelerated(
                cubek::attention::launch::BlueprintStrategy::Inferred(()),
            ),
        ),
        AttentionStrategy::FlashUnit => flash_attention(
            query,
            key,
            value,
            mask,
            attn_bias,
            options,
            out,
            launch::Strategy::Unit(cubek::attention::launch::BlueprintStrategy::Inferred(())),
        ),
        AttentionStrategy::Fallback => {
            out = attention_fallback::<CubeBackend<R, f32, i32, u8>>(
                query, key, value, mask, attn_bias, options,
            );
            Ok(out)
        }
        #[cfg(feature = "autotune")]
        AttentionStrategy::Autotune => {
            attention_autotune(query, key, value, mask, attn_bias, options, out)
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Launch a flash attention kernel, auto-padding head_dim to next multiple of 16 if needed.
pub fn flash_attention<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    _attn_bias: Option<CubeTensor<R>>,
    options: AttentionModuleOptions,
    out: CubeTensor<R>,
    strategy: launch::Strategy,
) -> Result<CubeTensor<R>, AttentionSetupError> {
    let head_dim = query.meta.shape[3];
    let padded_dim = head_dim.next_multiple_of(16);
    let needs_padding = padded_dim != head_dim;

    let (query, key, value, out, original_head_dim) = if needs_padding {
        // Q can be uninitialized — K's zero-padded columns kill those dot-product terms.
        // K and V must be zero-padded so the kernel sees zeros beyond original_head_dim.
        let q = pad_dim3_empty(&query, padded_dim);
        let k = pad_dim3_zeros(&key, padded_dim);
        let v = pad_dim3_zeros(&value, padded_dim);
        let o = init_attention_output(&q, &v);
        (q, k, v, o, Some(head_dim))
    } else {
        (query, key, value, out, None)
    };

    let client = &query.client;

    let dtypes = AttentionGlobalTypes {
        query: query.dtype.into(),
        key: key.dtype.into(),
        value: value.dtype.into(),
        mask: mask.as_ref().map(|m| m.dtype).unwrap_or(DType::U8).into(),
        out: out.dtype.into(),
    };

    cubek::attention::launch::launch_ref::<R>(
        strategy,
        client,
        &query.as_handle_ref(),
        &key.as_handle_ref(),
        &value.as_handle_ref(),
        &mask.as_ref().map(|mask| mask.as_handle_ref()),
        &out.as_handle_ref(),
        &dtypes,
        AttentionOptions {
            causal: options.is_causal,
            accumulator_precision: AccumulatorPrecision::Strict(cubecl::ir::StorageType::Scalar(
                cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32),
            )),
        },
        original_head_dim,
    )?;

    if needs_padding {
        Ok(narrow_dim3(&out, head_dim))
    } else {
        Ok(out)
    }
}

/// Pad dimension 3 with zeros (for K, V where padding must be zero).
fn pad_dim3_zeros<R: CubeRuntime>(tensor: &CubeTensor<R>, padded_dim: usize) -> CubeTensor<R> {
    let b = tensor.meta.shape[0];
    let h = tensor.meta.shape[1];
    let s = tensor.meta.shape[2];
    let padded = zeros_client::<R>(
        tensor.client.clone(),
        tensor.device.clone(),
        Shape::new([b, h, s, padded_dim]),
        tensor.dtype,
    );
    copy_into_padded(tensor, padded)
}

/// Pad dimension 3 without initializing (for Q where padding values are irrelevant).
fn pad_dim3_empty<R: CubeRuntime>(tensor: &CubeTensor<R>, padded_dim: usize) -> CubeTensor<R> {
    let b = tensor.meta.shape[0];
    let h = tensor.meta.shape[1];
    let s = tensor.meta.shape[2];
    let padded = empty_device_dtype::<R>(
        tensor.client.clone(),
        tensor.device.clone(),
        Shape::new([b, h, s, padded_dim]),
        tensor.dtype,
    );
    copy_into_padded(tensor, padded)
}

/// Copy `src` into the leading region of `dst` along dim 3 via slice_assign.
fn copy_into_padded<R: CubeRuntime>(
    src: &CubeTensor<R>,
    dst: CubeTensor<R>,
) -> CubeTensor<R> {
    let b = src.meta.shape[0];
    let h = src.meta.shape[1];
    let s = src.meta.shape[2];
    let d = src.meta.shape[3];
    let slices = [
        Slice { start: 0, end: Some(b as isize), step: 1 },
        Slice { start: 0, end: Some(h as isize), step: 1 },
        Slice { start: 0, end: Some(s as isize), step: 1 },
        Slice { start: 0, end: Some(d as isize), step: 1 },
    ];
    slice_assign(dst, &slices, src.clone())
}

/// Narrow dimension 3 back to the original head_dim (potentially zero-copy).
fn narrow_dim3<R: CubeRuntime>(tensor: &CubeTensor<R>, head_dim: usize) -> CubeTensor<R> {
    let b = tensor.meta.shape[0];
    let h = tensor.meta.shape[1];
    let s = tensor.meta.shape[2];
    slice(tensor.clone(), &[0..b, 0..h, 0..s, 0..head_dim])
}

pub(crate) fn init_attention_output<R: CubeRuntime>(
    query: &CubeTensor<R>,
    value: &CubeTensor<R>,
) -> CubeTensor<R> {
    let num_batches = query.meta.shape[0];
    let num_heads = query.meta.shape[1];
    let seq_q = query.meta.shape[2];
    let val_dim = value.meta.shape[3];
    let out_shape = Shape::new([num_batches, num_heads, seq_q, val_dim]);

    empty_device_dtype::<R>(
        query.client.clone(),
        query.device.clone(),
        out_shape,
        query.dtype,
    )
}
