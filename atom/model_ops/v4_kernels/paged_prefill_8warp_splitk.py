"""TileLang sparse_attn_v4_paged_prefill — 8-warp split-K version.

Uses 512 threads (8 warps) with split-K across D for QK^T.

Architecture:
- QK^T: 8-warp split-K. Each warp handles D/8=64 columns (2 MFMA iters with k_pack=2).
  Reduce 8 partials in shared. Same total MFMA work as 4-warp but 2x lower latency.
- Online softmax: T.Parallel(BLOCK_H), one thread per row.
- PV: T.gemm with 8 warps FullRow. Each warp handles D/8=64 output columns (4 MFMA tiles).
"""

import functools
import torch
import tilelang
import tilelang.language as T
from tilelang.language.kernel import get_thread_bindings
from tilelang.rocm.intrinsics.mfma_macro_generator import MatrixCoreIntrinEmitter
from tilelang.rocm.intrinsics.mfma_layout import make_mfma_swizzle_layout
from tilelang.rocm.intrinsics.utils import mfma_store_index_map
from tilelang.utils import determine_target

_TORCH_TO_TL = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
}

_NEG_LARGE = -3.4028234663852886e38
LN2_E = 1.44269504


@functools.lru_cache(maxsize=None)
def _build_kernel_mfma_8warp_splitk(
    H: int,
    D: int,
    BLOCK_H: int,
    BLOCK_K: int,
    threads: int,
    in_dtype: str,
):
    N_WARPS = threads // 64  # 8
    D_PER_WARP = D // N_WARPS  # 64
    accum_dtype = "float32"

    micro_size_x = 16
    micro_size_y = 16
    micro_size_k = 16
    k_pack = 2
    block_row_warps = 1
    block_col_warps = N_WARPS
    warp_row_tiles = BLOCK_H
    warp_col_tiles = BLOCK_K
    chunk = BLOCK_K

    warp_rows = warp_row_tiles // micro_size_x  # 1
    warp_cols = warp_col_tiles // micro_size_y  # 1
    local_size_a = (micro_size_x * micro_size_k) // 64  # 4
    local_size_b = (micro_size_y * micro_size_k) // 64  # 4
    local_size_c = (micro_size_x * micro_size_y) // 64  # 4
    ki_iters = D_PER_WARP // (micro_size_k * k_pack)  # 64/32 = 2

    target_obj = determine_target("auto", return_object=True)
    mfma_emitter = MatrixCoreIntrinEmitter(
        a_dtype=in_dtype,
        b_dtype=in_dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
        k_pack=k_pack,
        target=target_obj,
    )

    _, reverse_index_map_a = mfma_emitter.get_ldmatrix_index_map(is_b=False)
    _, reverse_index_map_b = mfma_emitter.get_ldmatrix_index_map(is_b=True)

    T_TOK = T.dynamic("T_TOK")
    P_PAGES = T.dynamic("P_PAGES")
    E_TOKENS = T.dynamic("E_TOKENS")
    PI_DIM = T.dynamic("PI_DIM")
    EI_DIM = T.dynamic("EI_DIM")
    TP1_DIM = T.dynamic("TP1_DIM")

    @T.prim_func
    def main(
        Q: T.Tensor((T_TOK, H, D), in_dtype),
        PrefixKV: T.Tensor((P_PAGES, D), in_dtype),
        PrefixIdx: T.Tensor((PI_DIM,), "int32"),
        PrefixPtr: T.Tensor((TP1_DIM,), "int32"),
        ExtendKV: T.Tensor((E_TOKENS, D), in_dtype),
        ExtendIdx: T.Tensor((EI_DIM,), "int32"),
        ExtendPtr: T.Tensor((TP1_DIM,), "int32"),
        AttnSink: T.Tensor((H,), "float"),
        Out: T.Tensor((T_TOK, H, D), in_dtype),
        softmax_scale: T.float32,
    ):
        with T.Kernel(T_TOK, T.ceildiv(H, BLOCK_H), threads=threads) as (bx, by):
            Q_shared = T.alloc_shared([BLOCK_H, D], in_dtype)
            T.annotate_layout({Q_shared: make_mfma_swizzle_layout(Q_shared)})
            K_shared = T.alloc_shared([BLOCK_K, D], in_dtype)
            scores_partial = T.alloc_shared(
                [N_WARPS * micro_size_x, micro_size_y + 8], accum_dtype
            )
            scores_4d = T.alloc_shared(
                [BLOCK_H // micro_size_x, BLOCK_K // micro_size_y,
                 micro_size_x, micro_size_y], accum_dtype
            )
            scores_bf16 = T.alloc_shared([BLOCK_H, BLOCK_K], in_dtype)
            l_i_shared = T.alloc_shared([BLOCK_H], accum_dtype)
            m_i_shared = T.alloc_shared([BLOCK_H], accum_dtype)
            m_prev_shared = T.alloc_shared([BLOCK_H], accum_dtype)
            slots_shared = T.alloc_shared([BLOCK_K], "int32")

            A_persist = T.alloc_local(
                [ki_iters * warp_rows * k_pack * local_size_a], in_dtype
            )
            A_local = T.alloc_local([warp_rows * local_size_a * k_pack], in_dtype)
            B_local = T.alloc_local([warp_cols * local_size_b * k_pack], in_dtype)
            C_local = T.alloc_local([warp_rows * warp_cols * local_size_c], accum_dtype)
            acc_o = T.alloc_fragment([BLOCK_H, D], accum_dtype)

            tx, ty, tz = T.meta_var(get_thread_bindings())
            lane_id = tx % 64
            warp_id = tx // 64
            warp_k_offset = warp_id * D_PER_WARP

            scale_log2 = softmax_scale * LN2_E

            # Load Q (512 threads)
            for ij in T.Parallel(BLOCK_H * D):
                ii = ij // D
                jj = ij % D
                hh = by * BLOCK_H + ii
                Q_shared[ii, jj] = T.if_then_else(
                    (hh < H), Q[bx, hh, jj], T.Cast(in_dtype, 0.0)
                )

            T.tvm_storage_sync("shared")

            # Preload Q into A_persist registers (per-warp persistent fragment)
            for ki in T.serial(ki_iters):
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map_a(lane_id, local_id))
                        A_persist[
                            (ki * warp_rows + i) * k_pack * local_size_a + local_id
                        ] = Q_shared[
                            i * micro_size_x + row,
                            warp_k_offset + ki * (k_pack * micro_size_k) + col,
                        ]

            T.fill(acc_o, 0.0)
            for i in T.Parallel(BLOCK_H):
                l_i_shared[i] = T.float32(0.0)
                m_i_shared[i] = T.float32(_NEG_LARGE)

            # ===== Region 1: prefix =====
            p_start = PrefixPtr[bx]
            p_end = PrefixPtr[bx + 1]
            p_len = p_end - p_start
            n_p_iters = T.ceildiv(p_len, BLOCK_K)

            for kk in T.serial(n_p_iters):
                k_base = kk * BLOCK_K

                # Hoist slot lookups to LDS once per tile
                for ii in T.Parallel(BLOCK_K):
                    in_range = (k_base + ii) < p_len
                    slots_shared[ii] = T.if_then_else(
                        in_range, PrefixIdx[p_start + k_base + ii], -1
                    )

                T.tvm_storage_sync("shared")

                for ij in T.Parallel(BLOCK_K * D):
                    ii = ij // D
                    jj = ij % D
                    raw_slot = slots_shared[ii]
                    slot = T.if_then_else(raw_slot >= 0, raw_slot, 0)
                    K_shared[ii, jj] = T.if_then_else(
                        raw_slot >= 0,
                        PrefixKV[slot, jj],
                        T.Cast(in_dtype, 0.0),
                    )

                T.tvm_storage_sync("shared")

                # 8-warp split-K QK^T (2 MFMA iters per warp)
                T.clear(C_local)
                for ki in T.serial(ki_iters):
                    for i in T.serial(warp_rows):
                        for local_id in T.vectorized(k_pack * local_size_a):
                            A_local[i * k_pack * local_size_a + local_id] = (
                                A_persist[
                                    (ki * warp_rows + i) * k_pack * local_size_a + local_id
                                ]
                            )
                    for j in T.serial(warp_cols):
                        for local_id in T.vectorized(k_pack * local_size_b):
                            row, col = T.meta_var(reverse_index_map_b(lane_id, local_id))
                            B_local[j * k_pack * local_size_b + local_id] = (
                                K_shared[j * micro_size_y + row,
                                         warp_k_offset + ki * (k_pack * micro_size_k) + col]
                            )
                    mfma_emitter.mfma(A_local, B_local, C_local)

                for local_id in T.vectorized(local_size_c):
                    row, col = T.meta_var(mfma_store_index_map(lane_id, local_id))
                    scores_partial[warp_id * micro_size_x + row, col] = C_local[local_id]

                T.tvm_storage_sync("shared")

                # Reduce 8 partials + scale + mask
                for ij in T.Parallel(BLOCK_H * BLOCK_K):
                    ii = ij // BLOCK_K
                    jj = ij % BLOCK_K
                    hh = by * BLOCK_H + ii
                    s = (
                        scores_partial[0 * micro_size_x + ii % micro_size_x, jj % micro_size_y]
                        + scores_partial[1 * micro_size_x + ii % micro_size_x, jj % micro_size_y]
                        + scores_partial[2 * micro_size_x + ii % micro_size_x, jj % micro_size_y]
                        + scores_partial[3 * micro_size_x + ii % micro_size_x, jj % micro_size_y]
                        + scores_partial[4 * micro_size_x + ii % micro_size_x, jj % micro_size_y]
                        + scores_partial[5 * micro_size_x + ii % micro_size_x, jj % micro_size_y]
                        + scores_partial[6 * micro_size_x + ii % micro_size_x, jj % micro_size_y]
                        + scores_partial[7 * micro_size_x + ii % micro_size_x, jj % micro_size_y]
                    )
                    in_range_j = (k_base + jj) < p_len
                    raw_slot_j = slots_shared[jj]
                    valid_j = (hh < H) and in_range_j and (raw_slot_j >= 0)
                    scaled = T.if_then_else(
                        valid_j, s * scale_log2, T.float32(_NEG_LARGE)
                    )
                    scores_4d[
                        ii // micro_size_x, jj // micro_size_y,
                        ii % micro_size_x, jj % micro_size_y
                    ] = scaled

                T.tvm_storage_sync("shared")

                # Online softmax: split into max pass, parallel exp2+cast, then sum pass
                for i in T.Parallel(BLOCK_H):
                    m_prev_shared[i] = m_i_shared[i]
                    for j in T.serial(BLOCK_K):
                        s = scores_4d[
                            i // micro_size_x, j // micro_size_y,
                            i % micro_size_x, j % micro_size_y
                        ]
                        m_i_shared[i] = T.max(m_i_shared[i], s)

                T.tvm_storage_sync("shared")

                # Parallel exp2 + cast across BLOCK_H*BLOCK_K = 256 threads
                for ij in T.Parallel(BLOCK_H * BLOCK_K):
                    i = ij // BLOCK_K
                    j = ij % BLOCK_K
                    s = scores_4d[
                        i // micro_size_x, j // micro_size_y,
                        i % micro_size_x, j % micro_size_y
                    ]
                    e = T.exp2(s - m_i_shared[i])
                    scores_bf16[i, j] = T.Cast(in_dtype, e)
                    scores_4d[
                        i // micro_size_x, j // micro_size_y,
                        i % micro_size_x, j % micro_size_y
                    ] = e

                T.tvm_storage_sync("shared")

                # Sum reduction per row + l_i update
                for i in T.Parallel(BLOCK_H):
                    sv = T.exp2(m_prev_shared[i] - m_i_shared[i])
                    l_i_shared[i] = l_i_shared[i] * sv
                    for j in T.serial(BLOCK_K):
                        l_i_shared[i] = l_i_shared[i] + scores_4d[
                            i // micro_size_x, j // micro_size_y,
                            i % micro_size_x, j % micro_size_y
                        ]
                # Skip rescale on first iter (acc_o == 0)
                if kk > 0:
                    for i, j in T.Parallel(BLOCK_H, D):
                        acc_o[i, j] = acc_o[i, j] * T.exp2(m_prev_shared[i] - m_i_shared[i])

                T.gemm(scores_bf16, K_shared, acc_o,
                       policy=T.GemmWarpPolicy.FullRow)

            # ===== Region 2: extend =====
            e_start = ExtendPtr[bx]
            e_end = ExtendPtr[bx + 1]
            e_len = e_end - e_start
            n_e_iters = T.ceildiv(e_len, BLOCK_K)

            for kk in T.serial(n_e_iters):
                k_base = kk * BLOCK_K

                for ii in T.Parallel(BLOCK_K):
                    in_range = (k_base + ii) < e_len
                    slots_shared[ii] = T.if_then_else(
                        in_range, ExtendIdx[e_start + k_base + ii], -1
                    )

                T.tvm_storage_sync("shared")

                for ij in T.Parallel(BLOCK_K * D):
                    ii = ij // D
                    jj = ij % D
                    raw_slot = slots_shared[ii]
                    slot = T.if_then_else(raw_slot >= 0, raw_slot, 0)
                    K_shared[ii, jj] = T.if_then_else(
                        raw_slot >= 0,
                        ExtendKV[slot, jj],
                        T.Cast(in_dtype, 0.0),
                    )

                T.tvm_storage_sync("shared")

                T.clear(C_local)
                for ki in T.serial(ki_iters):
                    for i in T.serial(warp_rows):
                        for local_id in T.vectorized(k_pack * local_size_a):
                            A_local[i * k_pack * local_size_a + local_id] = (
                                A_persist[
                                    (ki * warp_rows + i) * k_pack * local_size_a + local_id
                                ]
                            )
                    for j in T.serial(warp_cols):
                        for local_id in T.vectorized(k_pack * local_size_b):
                            row, col = T.meta_var(reverse_index_map_b(lane_id, local_id))
                            B_local[j * k_pack * local_size_b + local_id] = (
                                K_shared[j * micro_size_y + row,
                                         warp_k_offset + ki * (k_pack * micro_size_k) + col]
                            )
                    mfma_emitter.mfma(A_local, B_local, C_local)

                for local_id in T.vectorized(local_size_c):
                    row, col = T.meta_var(mfma_store_index_map(lane_id, local_id))
                    scores_partial[warp_id * micro_size_x + row, col] = C_local[local_id]

                T.tvm_storage_sync("shared")

                for ij in T.Parallel(BLOCK_H * BLOCK_K):
                    ii = ij // BLOCK_K
                    jj = ij % BLOCK_K
                    hh = by * BLOCK_H + ii
                    s = (
                        scores_partial[0 * micro_size_x + ii % micro_size_x, jj % micro_size_y]
                        + scores_partial[1 * micro_size_x + ii % micro_size_x, jj % micro_size_y]
                        + scores_partial[2 * micro_size_x + ii % micro_size_x, jj % micro_size_y]
                        + scores_partial[3 * micro_size_x + ii % micro_size_x, jj % micro_size_y]
                        + scores_partial[4 * micro_size_x + ii % micro_size_x, jj % micro_size_y]
                        + scores_partial[5 * micro_size_x + ii % micro_size_x, jj % micro_size_y]
                        + scores_partial[6 * micro_size_x + ii % micro_size_x, jj % micro_size_y]
                        + scores_partial[7 * micro_size_x + ii % micro_size_x, jj % micro_size_y]
                    )
                    in_range_j = (k_base + jj) < e_len
                    raw_slot_j = slots_shared[jj]
                    valid_j = (hh < H) and in_range_j and (raw_slot_j >= 0)
                    scaled = T.if_then_else(
                        valid_j, s * scale_log2, T.float32(_NEG_LARGE)
                    )
                    scores_4d[
                        ii // micro_size_x, jj // micro_size_y,
                        ii % micro_size_x, jj % micro_size_y
                    ] = scaled

                T.tvm_storage_sync("shared")

                for i in T.Parallel(BLOCK_H):
                    m_prev_shared[i] = m_i_shared[i]
                    for j in T.serial(BLOCK_K):
                        s = scores_4d[
                            i // micro_size_x, j // micro_size_y,
                            i % micro_size_x, j % micro_size_y
                        ]
                        m_i_shared[i] = T.max(m_i_shared[i], s)

                T.tvm_storage_sync("shared")

                for ij in T.Parallel(BLOCK_H * BLOCK_K):
                    i = ij // BLOCK_K
                    j = ij % BLOCK_K
                    s = scores_4d[
                        i // micro_size_x, j // micro_size_y,
                        i % micro_size_x, j % micro_size_y
                    ]
                    e = T.exp2(s - m_i_shared[i])
                    scores_bf16[i, j] = T.Cast(in_dtype, e)
                    scores_4d[
                        i // micro_size_x, j // micro_size_y,
                        i % micro_size_x, j % micro_size_y
                    ] = e

                T.tvm_storage_sync("shared")

                for i in T.Parallel(BLOCK_H):
                    sv = T.exp2(m_prev_shared[i] - m_i_shared[i])
                    l_i_shared[i] = l_i_shared[i] * sv
                    for j in T.serial(BLOCK_K):
                        l_i_shared[i] = l_i_shared[i] + scores_4d[
                            i // micro_size_x, j // micro_size_y,
                            i % micro_size_x, j % micro_size_y
                        ]

                for i, j in T.Parallel(BLOCK_H, D):
                    acc_o[i, j] = acc_o[i, j] * T.exp2(m_prev_shared[i] - m_i_shared[i])

                T.gemm(scores_bf16, K_shared, acc_o,
                       policy=T.GemmWarpPolicy.FullRow)

            # ===== Sink finalization =====
            for i in T.Parallel(BLOCK_H):
                hh = by * BLOCK_H + i
                if hh < H:
                    sink_v = AttnSink[hh] * LN2_E
                    m_prev_shared[i] = m_i_shared[i]
                    m_i_shared[i] = T.max(m_i_shared[i], sink_v)
                    sv = T.exp2(m_prev_shared[i] - m_i_shared[i])
                    l_i_shared[i] = l_i_shared[i] * sv + T.exp2(sink_v - m_i_shared[i])

            T.tvm_storage_sync("shared")

            for i, j in T.Parallel(BLOCK_H, D):
                acc_o[i, j] = acc_o[i, j] * T.exp2(m_prev_shared[i] - m_i_shared[i])

            # ===== Write output =====
            for ij in T.Parallel(BLOCK_H * D):
                ii = ij // D
                jj = ij % D
                hh = by * BLOCK_H + ii
                if (hh < H) and (jj < D):
                    denom = T.max(l_i_shared[ii], T.float32(1.0e-30))
                    val = T.if_then_else(
                        l_i_shared[ii] > 0.0,
                        acc_o[ii, jj] / denom,
                        T.float32(0.0),
                    )
                    Out[bx, hh, jj] = T.Cast(in_dtype, val)

    return tilelang.compile(main, target="hip")


def sparse_attn_v4_paged_prefill_tilelang_8warp_splitk(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    assert q.is_cuda and q.dim() == 3
    T_tok, H, D = q.shape
    assert attn_sink.shape == (H,)
    if attn_sink.dtype != torch.float32:
        attn_sink = attn_sink.to(torch.float32)

    kv_indices_prefix = kv_indices_prefix.to(torch.int32).contiguous()
    kv_indptr_prefix = kv_indptr_prefix.to(torch.int32).contiguous()
    kv_indices_extend = kv_indices_extend.to(torch.int32).contiguous()
    kv_indptr_extend = kv_indptr_extend.to(torch.int32).contiguous()

    out = torch.empty_like(q)

    BLOCK_H = 16
    BLOCK_K = 16
    threads = 512
    in_dtype = _TORCH_TO_TL[q.dtype]
    kernel = _build_kernel_mfma_8warp_splitk(
        int(H), int(D), int(BLOCK_H), int(BLOCK_K), int(threads), in_dtype
    )
    kernel(
        q.contiguous(),
        unified_kv.contiguous(),
        kv_indices_prefix.contiguous(),
        kv_indptr_prefix.contiguous(),
        kv.contiguous(),
        kv_indices_extend.contiguous(),
        kv_indptr_extend.contiguous(),
        attn_sink.contiguous(),
        out,
        float(softmax_scale),
    )
    return out
