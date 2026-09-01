// SPDX-License-Identifier: AGPL-3.0-only

//! Small, allocation-free CPU oracle for one GLM KDA head and token.
//!
//! This is intentionally not a serving implementation. It is the numerical
//! contract used to validate CUDA prefill/decode kernels on exact tiny cases.

use anyhow::{Result, bail};

/// One head's already-projected inputs for a single token.
pub struct KdaStepInputs<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub beta_logit: f32,
    pub forget_projection: &'a [f32],
    pub a_log: f32,
    pub dt_bias: &'a [f32],
    pub gate_lower_bound: f32,
}

/// Apply one recurrent KDA update for one head.
///
/// `state` is row-major `[key_dim, value_dim]`; the official GLM geometry has
/// equal dimensions, but the oracle keeps them independent so transposition
/// mistakes are visible in tests. `q` and `k` must already be FP32-normalized
/// and `q` must already include its `1/sqrt(head_dim)` scale.
pub fn step(state: &mut [f32], inputs: KdaStepInputs<'_>, output: &mut [f32]) -> Result<()> {
    let KdaStepInputs {
        q,
        k,
        v,
        beta_logit,
        forget_projection,
        a_log,
        dt_bias,
        gate_lower_bound,
    } = inputs;
    let key_dim = k.len();
    let value_dim = v.len();
    if q.len() != key_dim
        || forget_projection.len() != key_dim
        || dt_bias.len() != key_dim
        || output.len() != value_dim
        || state.len() != key_dim.saturating_mul(value_dim)
    {
        bail!("inconsistent KDA oracle dimensions");
    }

    // The forget value is per key channel. Keep the expression spelled out
    // like the pinned Transformers contract to make CUDA parity failures easy
    // to localize.
    let decay_scale = a_log.exp();
    for key in 0..key_dim {
        let g = gate_lower_bound * sigmoid(decay_scale * (forget_projection[key] + dt_bias[key]));
        let decay = g.exp();
        let row = &mut state[key * value_dim..(key + 1) * value_dim];
        for element in row {
            *element *= decay;
        }
    }

    // memory = k^T S; S += outer(k, sigmoid(beta) * (v - memory)).
    output.fill(0.0);
    for key in 0..key_dim {
        let row = &state[key * value_dim..(key + 1) * value_dim];
        for value in 0..value_dim {
            output[value] += k[key] * row[value];
        }
    }
    let beta = sigmoid(beta_logit);
    for value in 0..value_dim {
        output[value] = beta * (v[value] - output[value]);
    }
    for key in 0..key_dim {
        let row = &mut state[key * value_dim..(key + 1) * value_dim];
        for value in 0..value_dim {
            row[value] += k[key] * output[value];
        }
    }

    // out = q^T S. Reuse the caller-owned output buffer after the delta has
    // been committed so there is no hidden allocation in the reference path.
    output.fill(0.0);
    for key in 0..key_dim {
        let row = &state[key * value_dim..(key + 1) * value_dim];
        for value in 0..value_dim {
            output[value] += q[key] * row[value];
        }
    }
    Ok(())
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

#[cfg(test)]
mod tests {
    use super::{KdaStepInputs, step};

    fn step_inputs<'a>(q: &'a [f32], k: &'a [f32], v: &'a [f32]) -> KdaStepInputs<'a> {
        KdaStepInputs {
            q,
            k,
            v,
            beta_logit: 0.25,
            forget_projection: &[0.1, -0.2],
            a_log: -0.3,
            dt_bias: &[0.05, 0.1],
            gate_lower_bound: -5.0,
        }
    }

    #[test]
    fn zero_state_matches_outer_product_closed_form() {
        let mut state = vec![0.0; 4];
        let mut output = vec![0.0; 2];
        step(
            &mut state,
            KdaStepInputs {
                q: &[0.5, -0.25],
                k: &[0.8, 0.6],
                v: &[2.0, -1.0],
                beta_logit: 0.0,
                forget_projection: &[0.0, 0.0],
                a_log: 0.0,
                dt_bias: &[0.0, 0.0],
                gate_lower_bound: -5.0,
            },
            &mut output,
        )
        .unwrap();

        // sigmoid(beta)=0.5, so delta=[1,-0.5]. q dot k = 0.25.
        assert!((output[0] - 0.25).abs() < 1e-6);
        assert!((output[1] + 0.125).abs() < 1e-6);
    }

    #[test]
    fn token_loop_and_chunk_loop_are_the_same_state_machine() {
        let inputs = [
            ([1.0, 0.0], [0.6, 0.8], [1.0, 2.0]),
            ([0.0, 1.0], [0.8, 0.6], [-1.0, 0.5]),
            ([0.5, 0.5], [1.0, 0.0], [0.25, -0.75]),
        ];
        let mut decode_state = vec![0.0; 4];
        let mut prefill_state = vec![0.0; 4];
        let mut decode_out = vec![0.0; 2];
        let mut prefill_out = vec![0.0; 2];

        for (q, k, v) in inputs {
            step(&mut decode_state, step_inputs(&q, &k, &v), &mut decode_out).unwrap();
        }
        for (q, k, v) in inputs {
            step(
                &mut prefill_state,
                step_inputs(&q, &k, &v),
                &mut prefill_out,
            )
            .unwrap();
        }
        assert_eq!(decode_state, prefill_state);
        assert_eq!(decode_out, prefill_out);
    }

    #[test]
    fn rejects_a_transposed_or_truncated_state() {
        let error = step(
            &mut [0.0; 3],
            KdaStepInputs {
                q: &[1.0, 0.0],
                k: &[1.0, 0.0],
                v: &[1.0, 0.0],
                beta_logit: 0.0,
                forget_projection: &[0.0, 0.0],
                a_log: 0.0,
                dt_bias: &[0.0, 0.0],
                gate_lower_bound: -5.0,
            },
            &mut [0.0; 2],
        )
        .unwrap_err();
        assert!(error.to_string().contains("dimensions"));
    }
}
