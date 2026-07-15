"""TileLang sparse_attn_v4_paged_prefill — 8-warp variant using T.gemm split_k.

Functionally equivalent to ``paged_prefill_8warp_splitk.py`` but the QK^T
phase delegates to ``T.gemm(..., split_k=8)`` instead of the hand-rolled
MFMA + LDS-tree-reduce. This lets us A/B compare the compiler-emitted
butterfly LDS reduction against the manual implementation under identical
softmax / PV / sink finalization code.

Same caller ABI as ``sparse_attn_v4_paged_prefill_tilelang_8warp_splitk``.
"""

import functools
import torch
import tilelang
import tilelang.language as T

_TORCH_TO_TL = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
}

_NEG_LARGE = -3.4028234663852886e38
LN2_E = 1.44269504


@functools.lru_cache(maxsize=None)
def _build_kernel_tgemm_splitk(
    H: int,
    D: int,
    BLOCK_H: int,
    BLOCK_K: int,
    threads: int,
    in_dtype: str,
):
    N_WARPS = threads // 64  # 8
    accum_dtype = "float32"

    # 8 warps, BLOCK_H = BLOCK_K = 16 → only 1 warp fits on the M-N plane,
    # so all 8 warps cooperate along K via in-kernel split-K.
    SPLIT_K = N_WARPS
    K_PACK = 2

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
            K_shared = T.alloc_shared([BLOCK_K, D], in_dtype)
            scores_frag = T.alloc_fragment([BLOCK_H, BLOCK_K], accum_dtype)
            scores_shared = T.alloc_shared([BLOCK_H, BLOCK_K], accum_dtype)
            scores_bf16 = T.alloc_shared([BLOCK_H, BLOCK_K], in_dtype)
            l_i_shared = T.alloc_shared([BLOCK_H], accum_dtype)
            m_i_shared = T.alloc_shared([BLOCK_H], accum_dtype)
            m_prev_shared = T.alloc_shared([BLOCK_H], accum_dtype)
            slots_shared = T.alloc_shared([BLOCK_K], "int32")

            acc_o = T.alloc_fragment([BLOCK_H, D], accum_dtype)

            scale_log2 = softmax_scale * LN2_E

            # Load Q
            for ij in T.Parallel(BLOCK_H * D):
                ii = ij // D
                jj = ij % D
                hh = by * BLOCK_H + ii
                Q_shared[ii, jj] = T.if_then_else(
                    (hh < H), Q[bx, hh, jj], T.Cast(in_dtype, 0.0)
                )

            T.tvm_storage_sync("shared")

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

                # 8-warp split-K QK^T via T.gemm
                T.clear(scores_frag)
                T.gemm(
                    Q_shared,
                    K_shared,
                    scores_frag,
                    transpose_B=True,
                    k_pack=K_PACK,
                    split_k=SPLIT_K,
                )

                # Scale + mask + stage in shared so the softmax pass (which
                # iterates with T.serial on BLOCK_K) can read across rows.
                for ij in T.Parallel(BLOCK_H * BLOCK_K):
                    ii = ij // BLOCK_K
                    jj = ij % BLOCK_K
                    hh = by * BLOCK_H + ii
                    in_range_j = (k_base + jj) < p_len
                    raw_slot_j = slots_shared[jj]
                    valid_j = (hh < H) and in_range_j and (raw_slot_j >= 0)
                    scores_shared[ii, jj] = T.if_then_else(
                        valid_j, scores_frag[ii, jj] * scale_log2, T.float32(_NEG_LARGE)
                    )

                T.tvm_storage_sync("shared")

                # Online softmax max pass
                for i in T.Parallel(BLOCK_H):
                    m_prev_shared[i] = m_i_shared[i]
                    for j in T.serial(BLOCK_K):
                        m_i_shared[i] = T.max(m_i_shared[i], scores_shared[i, j])

                T.tvm_storage_sync("shared")

                # Parallel exp2 + cast across BLOCK_H * BLOCK_K
                for ij in T.Parallel(BLOCK_H * BLOCK_K):
                    i = ij // BLOCK_K
                    j = ij % BLOCK_K
                    e = T.exp2(scores_shared[i, j] - m_i_shared[i])
                    scores_bf16[i, j] = T.Cast(in_dtype, e)
                    scores_shared[i, j] = e

                T.tvm_storage_sync("shared")

                # Sum reduction per row + l_i update
                for i in T.Parallel(BLOCK_H):
                    sv = T.exp2(m_prev_shared[i] - m_i_shared[i])
                    l_i_shared[i] = l_i_shared[i] * sv
                    for j in T.serial(BLOCK_K):
                        l_i_shared[i] = l_i_shared[i] + scores_shared[i, j]

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

                T.clear(scores_frag)
                T.gemm(
                    Q_shared,
                    K_shared,
                    scores_frag,
                    transpose_B=True,
                    k_pack=K_PACK,
                    split_k=SPLIT_K,
                )

                for ij in T.Parallel(BLOCK_H * BLOCK_K):
                    ii = ij // BLOCK_K
                    jj = ij % BLOCK_K
                    hh = by * BLOCK_H + ii
                    in_range_j = (k_base + jj) < e_len
                    raw_slot_j = slots_shared[jj]
                    valid_j = (hh < H) and in_range_j and (raw_slot_j >= 0)
                    scores_shared[ii, jj] = T.if_then_else(
                        valid_j, scores_frag[ii, jj] * scale_log2, T.float32(_NEG_LARGE)
                    )

                T.tvm_storage_sync("shared")

                for i in T.Parallel(BLOCK_H):
                    m_prev_shared[i] = m_i_shared[i]
                    for j in T.serial(BLOCK_K):
                        m_i_shared[i] = T.max(m_i_shared[i], scores_shared[i, j])

                T.tvm_storage_sync("shared")

                for ij in T.Parallel(BLOCK_H * BLOCK_K):
                    i = ij // BLOCK_K
                    j = ij % BLOCK_K
                    e = T.exp2(scores_shared[i, j] - m_i_shared[i])
                    scores_bf16[i, j] = T.Cast(in_dtype, e)
                    scores_shared[i, j] = e

                T.tvm_storage_sync("shared")

                for i in T.Parallel(BLOCK_H):
                    sv = T.exp2(m_prev_shared[i] - m_i_shared[i])
                    l_i_shared[i] = l_i_shared[i] * sv
                    for j in T.serial(BLOCK_K):
                        l_i_shared[i] = l_i_shared[i] + scores_shared[i, j]

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


def sparse_attn_v4_paged_prefill_tilelang_8warp_tgemm_splitk(
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
    kernel = _build_kernel_tgemm_splitk(
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
