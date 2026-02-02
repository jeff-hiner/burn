//! SageAttention kernel integration
//!
//! Provides INT8 CMMA-based attention matching the reference SageAttention implementation.
//! Uses hardware CMMA for Q·K^T with i8×i8→i32 accumulation, then converts to f32 for softmax.

use super::{pad_head_dim, padded_head_dim, unpad_head_dim};
use crate::{CubeRuntime, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use burn_backend::{DType, Shape};
use cubek::attention::{
    definition::{AccumulatorPrecision, AttentionGlobalTypes, AttentionOptions, AttentionSetupError},
    launch::Strategy,
};

/// Launch SageAttention kernel using INT8 CMMA
///
/// This uses the tiled attention infrastructure with INT8 CMMA for Q·K^T:
/// - Q and K are quantized to i8 tiles
/// - CMMA computes i8×i8→i32 for scores
/// - Scores are converted to f32 for softmax
/// - V stays f32 throughout
///
/// Head dimensions are automatically padded to 64 or 128 as required by INT8 CMMA,
/// matching the reference SageAttention implementation.
pub fn sage_attention<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    out_dtype: DType,
) -> Result<CubeTensor<R>, AttentionSetupError> {
    let num_batches = query.shape.dims[0];
    let num_heads = query.shape.dims[1];
    let seq_q = query.shape.dims[2];
    let original_head_dim = query.shape.dims[3];
    let original_val_dim = value.shape.dims[3];

    // Pad head_dim to 64 or 128 for INT8 CMMA (matching reference SageAttention)
    let target_head_dim = padded_head_dim(original_head_dim);
    let target_val_dim = padded_head_dim(original_val_dim);

    let query = pad_head_dim(query, target_head_dim);
    let key = pad_head_dim(key, target_head_dim);
    let value = pad_head_dim(value, target_val_dim);

    let out_shape = Shape::new([num_batches, num_heads, seq_q, target_val_dim]);
    let out = empty_device_dtype::<R>(query.client.clone(), query.device.clone(), out_shape, out_dtype);

    let dtypes = AttentionGlobalTypes {
        query: query.dtype.into(),
        key: key.dtype.into(),
        value: value.dtype.into(),
        mask: mask.as_ref().map(|m| m.dtype).unwrap_or(DType::U8).into(),
        out: out.dtype.into(),
    };

    cubek::attention::launch::launch_ref::<R>(
        Strategy::Int8Cmma(cubek::attention::launch::BlueprintStrategy::Inferred(())),
        &query.client,
        &query.as_handle_ref(),
        &key.as_handle_ref(),
        &value.as_handle_ref(),
        &mask.as_ref().map(|mask| mask.as_handle_ref()),
        &out.as_handle_ref(),
        &dtypes,
        AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::Strict(cubecl::ir::StorageType::Scalar(
                cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32),
            )),
            int8_cmma: true, // Enable INT8 CMMA for SageAttention
        },
    )?;

    // Slice output back to original val_dim
    let out = unpad_head_dim(out, original_val_dim);

    Ok(out)
}
