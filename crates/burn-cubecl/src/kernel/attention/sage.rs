//! SageAttention kernel integration
//!
//! Provides a direct entry point for the standalone SageAttention kernel.

use crate::{CubeRuntime, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use burn_backend::{DType, Shape};
use cubek::attention::kernels::{SageAttentionConfig, launch_sage_attention};

/// Launch standalone SageAttention kernel
///
/// This bypasses cubek's tiling infrastructure and uses a simpler online softmax kernel.
/// Useful for debugging and performance comparison.
pub fn sage_attention<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    _mask: Option<CubeTensor<R>>,
    out_dtype: DType,
) -> Result<CubeTensor<R>, String> {
    let client = &query.client;
    let device = &query.device;

    // Validate shapes
    if query.shape.dims.len() != 4 {
        return Err(format!("Expected 4D query tensor, got {:?}", query.shape.dims));
    }

    let batch = query.shape.dims[0];
    let heads = query.shape.dims[1];
    let seq_q = query.shape.dims[2];
    let head_dim = query.shape.dims[3];
    let seq_kv = key.shape.dims[2];

    // Output shape matches query shape
    let out_shape = Shape::new([batch, heads, seq_q, head_dim]);
    let out = empty_device_dtype::<R>(client.clone(), device.clone(), out_shape, out_dtype);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let config = SageAttentionConfig {
        batch,
        heads,
        seq_q,
        seq_kv,
        head_dim,
        scale,
    };

    launch_sage_attention(
        client,
        &query.as_handle_ref(),
        &key.as_handle_ref(),
        &value.as_handle_ref(),
        &out.as_handle_ref(),
        config,
    )
    .map_err(|e| format!("SageAttention launch failed: {:?}", e))?;

    Ok(out)
}
