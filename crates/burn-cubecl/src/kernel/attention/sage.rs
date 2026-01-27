//! SageAttention kernel integration
//!
//! Provides INT8 CMMA-based attention matching the reference SageAttention implementation.
//! Uses hardware CMMA for Q·K^T with i8×i8→i32 accumulation, then converts to f32 for softmax.

use crate::{
    CubeRuntime,
    kernel::{slice, slice_assign},
    ops::numeric::{empty_device_dtype, zeros_client},
    tensor::CubeTensor,
};
use burn_backend::{DType, Shape, Slice};
use cubek::attention::{
    definition::{AccumulatorPrecision, AttentionGlobalTypes, AttentionOptions, AttentionSetupError},
    launch::Strategy,
};

/// Compute the padded head_dim for INT8 CMMA.
/// Following reference SageAttention: pad to 64 if <= 64, else pad to 128.
fn padded_head_dim(head_dim: usize) -> usize {
    if head_dim <= 64 {
        64
    } else if head_dim <= 128 {
        128
    } else {
        panic!("head_dim > 128 not supported by SageAttention");
    }
}

/// Pad a 4D tensor along the last dimension (head_dim) to target size.
fn pad_head_dim<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    target_head_dim: usize,
) -> CubeTensor<R> {
    let [batch, heads, seq, head_dim] = [
        tensor.shape.dims[0],
        tensor.shape.dims[1],
        tensor.shape.dims[2],
        tensor.shape.dims[3],
    ];

    if head_dim == target_head_dim {
        return tensor;
    }

    // Create zero tensor with padded shape
    let padded_shape = Shape::new([batch, heads, seq, target_head_dim]);
    let padded = zeros_client::<R>(
        tensor.client.clone(),
        tensor.device.clone(),
        padded_shape,
        tensor.dtype,
    );

    // Copy original data into padded tensor
    let slices = [
        Slice::new(0, Some(batch as isize), 1),
        Slice::new(0, Some(heads as isize), 1),
        Slice::new(0, Some(seq as isize), 1),
        Slice::new(0, Some(head_dim as isize), 1),
    ];
    slice_assign(padded, &slices, tensor)
}

/// Slice output back to original head_dim.
fn unpad_head_dim<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    original_head_dim: usize,
) -> CubeTensor<R> {
    let [batch, heads, seq, _padded_dim] = [
        tensor.shape.dims[0],
        tensor.shape.dims[1],
        tensor.shape.dims[2],
        tensor.shape.dims[3],
    ];

    slice(
        tensor,
        &[0..batch, 0..heads, 0..seq, 0..original_head_dim],
    )
}

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
        Strategy::BlackboxAccelerated(cubek::attention::launch::BlueprintStrategy::Inferred(())),
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
