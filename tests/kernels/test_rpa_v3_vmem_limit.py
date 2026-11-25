# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RPA v3 compiles its three launches within the default VMEM limit.

A limit below what Mosaic needs fails at compile time, which stops a server at
warm-up, so every launch (decode, prefill, mixed) is compiled with the default
block sizes and limit for bf16 and fp8 KV.
"""

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel as K

pytestmark = pytest.mark.tpu_test


@pytest.mark.parametrize("kv_dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
@pytest.mark.parametrize(
    "num_q_heads, num_kv_heads, head_dim, sliding_window, soft_cap",
    [
        (8, 1, 128, None, None),
        (32, 8, 128, None, None),
        # Wide heads with a sliding window and soft cap, one KV head per device.
        (8, 1, 256, 4096, 50.0),
    ],
)
def test_default_vmem_limit_compiles(
    kv_dtype, num_q_heads, num_kv_heads, head_dim, sliding_window, soft_cap
):
    page_size, pages_per_seq, max_num_seqs = 128, 128, 8
    # 2048 tokens and 16k-token sequences reach the largest default block of
    # every v7x launch.
    max_num_tokens = 2048

    def spec(shape, dtype):
        return jax.ShapeDtypeStruct(shape, dtype)

    K.ragged_paged_attention.lower(
        spec((max_num_tokens, num_q_heads, head_dim), jnp.bfloat16),
        spec((max_num_tokens, num_kv_heads, head_dim), kv_dtype),
        spec((max_num_tokens, num_kv_heads, head_dim), kv_dtype),
        spec(
            K.get_kv_cache_shape(
                max_num_seqs * pages_per_seq,
                page_size,
                num_kv_heads,
                head_dim,
                kv_dtype,
            ),
            kv_dtype,
        ),
        spec((max_num_seqs,), jnp.int32),
        spec((max_num_seqs * pages_per_seq,), jnp.int32),
        spec((max_num_seqs + 1,), jnp.int32),
        spec((3,), jnp.int32),
        sm_scale=head_dim**-0.5,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        chunk_prefill_size=max_num_tokens // 2,
    ).compile()
