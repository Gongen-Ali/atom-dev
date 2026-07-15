"""TileLang sparse_attn_v4_paged_decode — split-K version.

Two-kernel split-K decode mirroring Triton's _paged_decode_split_kernel +
_paged_decode_reduce_kernel design:

  Split kernel : grid = (T_tok, ceil(H / BLOCK_H), KV_SPLITS)
                 Each CTA computes pre-sink (m, l, acc) for its tile range
                 and writes to fp32 partials [T, KV_SPLITS, H_padded, ...].

  Reduce kernel: grid = (T_tok, ceil(H / BLOCK_H))
                 Combines KV_SPLITS partials, folds attn_sink, writes out.

Reuses the v4 design (Q held in registers, no Q_shared LDS).
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

from atom_v4.tilelang_impl.paged_decode_8warp_v4 import (
    sparse_attn_v4_paged_decode_tilelang_8warp_v4,
)

_TORCH_TO_TL = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
}

_NEG_LARGE = -3.4028234663852886e38
LN2_E = 1.44269504


@functools.lru_cache(maxsize=None)
def _build_split_kernel(
    H: int,
    D: int,
    BLOCK_H: int,
    BLOCK_K: int,
    KV_SPLITS: int,
    threads: int,
    in_dtype: str,
):
    N_WARPS = threads // 64  # 8
    D_PER_WARP = D // N_WARPS  # 64 for D=512
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

    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y
    local_size_a = (micro_size_x * micro_size_k) // 64
    local_size_b = (micro_size_y * micro_size_k) // 64
    local_size_c = (micro_size_x * micro_size_y) // 64
    ki_iters = D_PER_WARP // (micro_size_k * k_pack)

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
    P_DIM = T.dynamic("P_DIM")
    N_DIM = T.dynamic("N_DIM")
    TP1_DIM = T.dynamic("TP1_DIM")

    @T.prim_func
    def main(
        Q: T.Tensor((T_TOK, H, D), in_dtype),
        UnifiedKV: T.Tensor((P_DIM, D), in_dtype),
        KvIdx: T.Tensor((N_DIM,), "int32"),
        KvPtr: T.Tensor((TP1_DIM,), "int32"),
        MPart: T.Tensor((T_TOK, KV_SPLITS, H), "float"),
        LPart: T.Tensor((T_TOK, KV_SPLITS, H), "float"),
        APart: T.Tensor((T_TOK, KV_SPLITS, H, D), "float"),
        softmax_scale: T.float32,
    ):
        with T.Kernel(
            T_TOK, T.ceildiv(H, BLOCK_H), KV_SPLITS, threads=threads
        ) as (bx, by, bz):
            K_shared = T.alloc_shared([BLOCK_K, D], in_dtype)

            scores_partial = T.alloc_shared(
                [N_WARPS * micro_size_x, micro_size_y], accum_dtype
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

            A_local = T.alloc_local([warp_rows * local_size_a * k_pack], in_dtype)
            B_local = T.alloc_local([warp_cols * local_size_b * k_pack], in_dtype)
            C_local = T.alloc_local([warp_rows * warp_cols * local_size_c], accum_dtype)
            acc_o = T.alloc_fragment([BLOCK_H, D], accum_dtype)

            tx, _, _ = T.meta_var(get_thread_bindings())
            lane_id = tx % 64
            warp_id = tx // 64
            warp_k_offset = warp_id * D_PER_WARP

            scale_log2 = softmax_scale * LN2_E

            kv_start = KvPtr[bx]
            kv_end = KvPtr[bx + 1]
            kv_len = kv_end - kv_start

            tiles_per_segment = T.ceildiv(kv_len, KV_SPLITS * BLOCK_K)
            seg_start_tile = bz * tiles_per_segment
            seg_end_tile = T.min((bz + 1) * tiles_per_segment, T.ceildiv(kv_len, BLOCK_K))
            active = (seg_start_tile * BLOCK_K) < kv_len

            T.fill(acc_o, 0.0)
            for i in T.Parallel(BLOCK_H):
                m_i_shared[i] = T.float32(_NEG_LARGE)
                l_i_shared[i] = T.float32(0.0)

            if active:
                for kk in T.serial(seg_end_tile - seg_start_tile):
                    k_base = (seg_start_tile + kk) * BLOCK_K

                    for ii in T.Parallel(BLOCK_K):
                        in_range = (k_base + ii) < kv_len
                        slots_shared[ii] = T.if_then_else(
                            in_range, KvIdx[kv_start + k_base + ii], 0
                        )
                    T.tvm_storage_sync("shared")

                    for ij in T.Parallel(BLOCK_K * D):
                        ii = ij // D
                        jj = ij % D
                        in_range = (k_base + ii) < kv_len
                        slot = slots_shared[ii]
                        K_shared[ii, jj] = T.if_then_else(
                            in_range, UnifiedKV[slot, jj], T.Cast(in_dtype, 0.0)
                        )

                    T.tvm_storage_sync("shared")

                    # QK^T: A_local from Q (GM/L1 cache), B_local from K_shared
                    T.clear(C_local)
                    for ki in T.serial(ki_iters):
                        for i in T.serial(warp_rows):
                            for local_id in T.vectorized(k_pack * local_size_a):
                                row, col = T.meta_var(reverse_index_map_a(lane_id, local_id))
                                hh = by * BLOCK_H + i * micro_size_x + row
                                d_idx = warp_k_offset + ki * (k_pack * micro_size_k) + col
                                A_local[i * k_pack * local_size_a + local_id] = (
                                    T.if_then_else(
                                        hh < H, Q[bx, hh, d_idx], T.Cast(in_dtype, 0.0)
                                    )
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
                        in_range_j = (k_base + jj) < kv_len
                        valid_j = (hh < H) and in_range_j
                        scaled = T.if_then_else(
                            valid_j, s * scale_log2, T.float32(_NEG_LARGE)
                        )
                        scores_4d[
                            ii // micro_size_x, jj // micro_size_y,
                            ii % micro_size_x, jj % micro_size_y
                        ] = scaled

                    T.tvm_storage_sync("shared")

                    # Online softmax — Phase 1: row-max (serial dependency chain)
                    for i in T.Parallel(BLOCK_H):
                        m_prev_shared[i] = m_i_shared[i]
                        for j in T.serial(BLOCK_K):
                            s = scores_4d[
                                i // micro_size_x, j // micro_size_y,
                                i % micro_size_x, j % micro_size_y
                            ]
                            m_i_shared[i] = T.max(m_i_shared[i], s)

                    T.tvm_storage_sync("shared")

                    # Online softmax — Phase 2: exp2 + cast (fully parallel)
                    for ij in T.Parallel(BLOCK_H * BLOCK_K):
                        ii = ij // BLOCK_K
                        jj = ij % BLOCK_K
                        s = scores_4d[
                            ii // micro_size_x, jj // micro_size_y,
                            ii % micro_size_x, jj % micro_size_y
                        ]
                        e = T.exp2(s - m_i_shared[ii])
                        scores_bf16[ii, jj] = T.Cast(in_dtype, e)
                        scores_4d[
                            ii // micro_size_x, jj // micro_size_y,
                            ii % micro_size_x, jj % micro_size_y
                        ] = e

                    T.tvm_storage_sync("shared")

                    # Online softmax — Phase 3: l_i update (serial sum per row)
                    for i in T.Parallel(BLOCK_H):
                        sv = T.exp2(m_prev_shared[i] - m_i_shared[i])
                        l_i_shared[i] = l_i_shared[i] * sv
                        for j in T.serial(BLOCK_K):
                            l_i_shared[i] = l_i_shared[i] + scores_4d[
                                i // micro_size_x, j // micro_size_y,
                                i % micro_size_x, j % micro_size_y
                            ]

                    T.tvm_storage_sync("shared")

                    for i, j in T.Parallel(BLOCK_H, D):
                        acc_o[i, j] = acc_o[i, j] * T.exp2(m_prev_shared[i] - m_i_shared[i])

                    T.gemm(scores_bf16, K_shared, acc_o,
                           policy=T.GemmWarpPolicy.FullRow)

            # Store partials. Inactive segments still write NEG_LARGE/0/0 so
            # the reduce kernel's act_num_segments mask is the only filter.
            for i in T.Parallel(BLOCK_H):
                hh = by * BLOCK_H + i
                if hh < H:
                    MPart[bx, bz, hh] = m_i_shared[i]
                    LPart[bx, bz, hh] = l_i_shared[i]
            for ij in T.Parallel(BLOCK_H * D):
                ii = ij // D
                jj = ij % D
                hh = by * BLOCK_H + ii
                if (hh < H) and (jj < D):
                    APart[bx, bz, hh, jj] = acc_o[ii, jj]

    return tilelang.compile(main, target="hip")


@functools.lru_cache(maxsize=None)
def _build_reduce_kernel(
    H: int,
    D: int,
    KV_SPLITS: int,
    BLOCK_K: int,
    threads: int,
    in_dtype: str,
):
    """Reduce kernel with BLOCK_H=1 (one CTA per head), matching Triton.

    Grid: (T_TOK, H). Each CTA combines KV_SPLITS partials for a single
    (token, head) pair, folds attn_sink, and writes one D-vector output.
    KV_SPLITS reduction uses T.serial — no thread-count constraint from exp2.
    """
    accum_dtype = "float32"

    T_TOK = T.dynamic("T_TOK")
    TP1_DIM = T.dynamic("TP1_DIM")

    @T.prim_func
    def reduce(
        MPart: T.Tensor((T_TOK, KV_SPLITS, H), "float"),
        LPart: T.Tensor((T_TOK, KV_SPLITS, H), "float"),
        APart: T.Tensor((T_TOK, KV_SPLITS, H, D), "float"),
        AttnSink: T.Tensor((H,), "float"),
        KvPtr: T.Tensor((TP1_DIM,), "int32"),
        Out: T.Tensor((T_TOK, H, D), in_dtype),
    ):
        with T.Kernel(T_TOK, H, threads=threads) as (bx, by):
            acc_combined = T.alloc_fragment([D], accum_dtype)
            m_local = T.alloc_local([KV_SPLITS], accum_dtype)
            l_local = T.alloc_local([KV_SPLITS], accum_dtype)
            alpha_local = T.alloc_local([KV_SPLITS], accum_dtype)
            m_max_l = T.alloc_local([1], accum_dtype)
            l_combined_l = T.alloc_local([1], accum_dtype)
            m_final_l = T.alloc_local([1], accum_dtype)
            alpha_kv_l = T.alloc_local([1], accum_dtype)
            l_final_l = T.alloc_local([1], accum_dtype)
            denom_l = T.alloc_local([1], accum_dtype)

            hh = by

            kv_start = KvPtr[bx]
            kv_end = KvPtr[bx + 1]
            kv_len = kv_end - kv_start
            tps = T.ceildiv(kv_len, KV_SPLITS * BLOCK_K)
            tps_safe = T.max(tps, 1)
            act_num_segments = T.ceildiv(kv_len, tps_safe * BLOCK_K)

            m_max_l[0] = T.float32(_NEG_LARGE)
            for k in T.serial(KV_SPLITS):
                seg_active = k < act_num_segments
                m_local[k] = T.if_then_else(
                    seg_active, MPart[bx, k, hh], T.float32(_NEG_LARGE)
                )
                l_local[k] = T.if_then_else(
                    seg_active, LPart[bx, k, hh], T.float32(0.0)
                )
                m_max_l[0] = T.max(m_max_l[0], m_local[k])

            l_combined_l[0] = T.float32(0.0)
            for k in T.serial(KV_SPLITS):
                alpha_local[k] = T.exp2(m_local[k] - m_max_l[0])
                l_combined_l[0] = l_combined_l[0] + l_local[k] * alpha_local[k]

            sink_val = AttnSink[hh] * LN2_E
            m_final_l[0] = T.max(m_max_l[0], sink_val)
            alpha_kv_l[0] = T.exp2(m_max_l[0] - m_final_l[0])
            l_final_l[0] = l_combined_l[0] * alpha_kv_l[0] + T.exp2(sink_val - m_final_l[0])
            denom_l[0] = T.max(l_final_l[0], T.float32(1.0e-30))

            for j in T.Parallel(D):
                acc_combined[j] = T.float32(0.0)
                for k in T.serial(KV_SPLITS):
                    acc_combined[j] = acc_combined[j] + APart[bx, k, hh, j] * alpha_local[k]

            for j in T.Parallel(D):
                s = acc_combined[j] * alpha_kv_l[0]
                val = T.if_then_else(
                    l_final_l[0] > 0.0,
                    s / denom_l[0],
                    T.float32(0.0),
                )
                Out[bx, hh, j] = T.Cast(in_dtype, val)

    return tilelang.compile(reduce, target="hip")


def _kv_splits_heuristic_local(T_tok: int, H: int, BLOCK_H: int) -> int:
    """Match Triton's heuristic. Hardcoded num_cu=80 for MI308X to avoid
    importing aiter; in practice this is queried once per process. We round
    DOWN to a power of two and cap at 64."""
    num_cu = 80
    target_wg = max(1, int(2.0 * num_cu))
    head_blocks = max(1, (H + BLOCK_H - 1) // BLOCK_H)
    base_ctas = max(1, T_tok * head_blocks)
    if base_ctas >= target_wg:
        return 1
    splits_to_fill = max(1, target_wg // base_ctas)
    capped = min(splits_to_fill, 64)
    if capped < 1:
        return 1
    return 1 << (capped.bit_length() - 1)


def sparse_attn_v4_paged_decode_tilelang_splitk(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    kv_scales: torch.Tensor | None = None,
    kv_splits: int | None = None,
) -> torch.Tensor:
    if kv_scales is not None:
        raise NotImplementedError("fp8 path not implemented")
    assert q.is_cuda and q.dim() == 3
    T_tok, H, D = q.shape
    assert unified_kv.dim() == 2 and unified_kv.size(-1) == D
    assert q.dtype in (torch.bfloat16, torch.float16)
    assert unified_kv.dtype == q.dtype
    assert attn_sink.shape == (H,)
    if attn_sink.dtype != torch.float32:
        attn_sink = attn_sink.to(torch.float32)

    BLOCK_H = 16
    BLOCK_K = 16
    threads = 512

    if kv_splits is None:
        kv_splits = _kv_splits_heuristic_local(T_tok, H, BLOCK_H)

    # Fast path: when split-K is unnecessary, fall back to the v4 fused kernel
    if kv_splits == 1:
        return sparse_attn_v4_paged_decode_tilelang_8warp_v4(
            q=q,
            unified_kv=unified_kv,
            kv_indices=kv_indices,
            kv_indptr=kv_indptr,
            attn_sink=attn_sink,
            softmax_scale=softmax_scale,
        )

    kv_indices = kv_indices.to(torch.int32).contiguous()
    kv_indptr = kv_indptr.to(torch.int32).contiguous()

    in_dtype = _TORCH_TO_TL[q.dtype]
    n_head_blocks = (H + BLOCK_H - 1) // BLOCK_H
    h_padded = n_head_blocks * BLOCK_H

    m_partial = torch.empty(T_tok, kv_splits, H, dtype=torch.float32, device=q.device)
    l_partial = torch.empty(T_tok, kv_splits, H, dtype=torch.float32, device=q.device)
    acc_partial = torch.empty(
        T_tok, kv_splits, H, D, dtype=torch.float32, device=q.device
    )
    out = torch.empty_like(q)

    split_kernel = _build_split_kernel(
        int(H), int(D), int(BLOCK_H), int(BLOCK_K),
        int(kv_splits), int(threads), in_dtype,
    )
    split_kernel(
        q.contiguous(),
        unified_kv.contiguous(),
        kv_indices,
        kv_indptr,
        m_partial,
        l_partial,
        acc_partial,
        float(softmax_scale),
    )

    reduce_threads = 256
    reduce_kernel = _build_reduce_kernel(
        int(H), int(D),
        int(kv_splits), int(BLOCK_K),
        int(reduce_threads), in_dtype,
    )
    reduce_kernel(
        m_partial,
        l_partial,
        acc_partial,
        attn_sink.contiguous(),
        kv_indptr,
        out,
    )
    return out