"""Generate synthetic prefill inputs at larger T based on real dump data.

Usage:
    # Analyze real dump first
    python synth_prefill.py --analyze prefill_inputs/prefill_T712_H16_D512_pfx7744_ext32040.pt

    # Generate for target T values
    python synth_prefill.py --source prefill_inputs/prefill_T712_H16_D512_pfx7744_ext32040.pt \
        --target-T 2048 4096 8192 --output-dir prefill_inputs/synth/

    # Generate and benchmark in one go
    python synth_prefill.py --source prefill_inputs/prefill_T712_H16_D512_pfx7744_ext32040.pt \
        --target-T 2048 4096 --bench
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

import torch


# ─── V4 model constants ───
WIN = 128          # SWA window size
INDEX_TOPK = 512   # CSA topk per sequence
MTP_K = 3          # speculative tokens
CS = WIN + MTP_K   # 131, ring buffer stride


def analyze_dump(path: str):
    """Analyze a real dump to understand per-token distributions."""
    data = torch.load(path, map_location="cpu")
    q = data["q"]
    T, H, D = q.shape
    kv = data["kv"]
    unified_kv = data["unified_kv"]
    kv_indices_prefix = data["kv_indices_prefix"]
    kv_indptr_prefix = data["kv_indptr_prefix"]
    kv_indices_extend = data["kv_indices_extend"]
    kv_indptr_extend = data["kv_indptr_extend"]

    print(f"=== Dump Analysis: {os.path.basename(path)} ===")
    print(f"  q:              {list(q.shape)}  dtype={q.dtype}")
    print(f"  unified_kv:     {list(unified_kv.shape)}  dtype={unified_kv.dtype}")
    print(f"  kv:             {list(kv.shape)}  dtype={kv.dtype}")
    print(f"  prefix indices: {kv_indices_prefix.numel()}  (indptr len={len(kv_indptr_prefix)})")
    print(f"  extend indices: {kv_indices_extend.numel()}  (indptr len={len(kv_indptr_extend)})")
    print(f"  attn_sink:      {list(data['attn_sink'].shape)}  dtype={data['attn_sink'].dtype}")
    print(f"  softmax_scale:  {data['softmax_scale']}")

    # Per-token extend counts
    ext_counts = (kv_indptr_extend[1:] - kv_indptr_extend[:-1]).int()
    print(f"\n  --- Per-token extend distribution ---")
    print(f"    min={ext_counts.min().item()}, max={ext_counts.max().item()}, "
          f"mean={ext_counts.float().mean().item():.1f}, sum={ext_counts.sum().item()}")

    # Count tokens at each extend_count level
    unique, counts = torch.unique(ext_counts, return_counts=True)
    print(f"    extend_count histogram (count → num_tokens):")
    for c, n in zip(unique.tolist(), counts.tolist()):
        print(f"      {c:4d} → {n} tokens")

    # Infer per-sequence token counts from extend pattern
    # Tokens with extend_count < 128 are the first `extend_count` tokens of a sequence
    # Token with extend_count=1 is the first token of a sequence
    first_token_mask = ext_counts == 1
    n_seqs = first_token_mask.sum().item()
    print(f"\n  --- Inferred sequence structure ---")
    print(f"    sequences: {n_seqs}")

    if n_seqs > 0:
        first_token_indices = first_token_mask.nonzero(as_tuple=True)[0]
        # Compute per-seq token counts
        seq_starts = first_token_indices.tolist()
        seq_starts.append(T)  # sentinel
        seq_lens = [seq_starts[i + 1] - seq_starts[i] for i in range(n_seqs)]
        print(f"    seq lens: {seq_lens}")
        print(f"    total T:  {sum(seq_lens)}")

        # Per-seq extend sums
        for i, (start, slen) in enumerate(zip(seq_starts[:-1], seq_lens)):
            seq_ext_sum = ext_counts[start:start + slen].sum().item()
            print(f"    seq {i}: len={slen}, extend_sum={seq_ext_sum}")

    if kv_indices_prefix.numel() > 0:
        pfx_counts = (kv_indptr_prefix[1:] - kv_indptr_prefix[:-1]).int()
        print(f"\n  --- Per-token prefix distribution ---")
        print(f"    min={pfx_counts.min().item()}, max={pfx_counts.max().item()}, "
              f"mean={pfx_counts.float().mean().item():.1f}, sum={pfx_counts.sum().item()}")
        unique_p, counts_p = torch.unique(pfx_counts, return_counts=True)
        for c, n in zip(unique_p.tolist(), counts_p.tolist()):
            print(f"      {c:4d} → {n} tokens")

    print()
    return {
        "T": T, "H": H, "D": D,
        "n_seqs": n_seqs,
        "seq_lens": seq_lens if n_seqs > 0 else [],
        "dtype_q": q.dtype,
        "dtype_kv": kv.dtype,
        "softmax_scale": data["softmax_scale"],
    }


def synth_extend_indptr(seq_lens: List[int]) -> torch.Tensor:
    """Build extend indptr from per-sequence token counts."""
    counts = []
    for slen in seq_lens:
        for pos in range(slen):
            counts.append(min(pos + 1, WIN))
    counts = torch.tensor(counts, dtype=torch.int32)
    indptr = torch.zeros(len(counts) + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(counts, dim=0)
    return indptr


def synth_extend_indices(kv_indptr_extend: torch.Tensor,
                         cu_seqlens_q: List[int],
                         seq_lens: List[int]) -> torch.Tensor:
    """Build extend indices: row offsets into the flat kv tensor."""
    T = len(seq_lens) and sum(seq_lens)
    ext_total = kv_indptr_extend[-1].item()
    indices = torch.empty(ext_total, dtype=torch.int32)

    offset = 0
    for seq_i, (cu_q, slen) in enumerate(zip(cu_seqlens_q, seq_lens)):
        for pos in range(slen):
            ext_count = min(pos + 1, WIN)
            ext_start_row = cu_q + pos - ext_count + 1
            base = kv_indptr_extend[offset].item()
            indices[base:base + ext_count] = torch.arange(
                ext_start_row, ext_start_row + ext_count, dtype=torch.int32
            )
            offset += 1

    return indices


def synth_prefix_indices(seq_lens: List[int],
                         chunk_starts: List[int],
                         prefix_type: str = "csa",
                         n_committed_csa: Optional[int] = None,
                         n_committed_hca: Optional[int] = None,
                         positions: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Build prefix indices for SWA / CSA / HCA.

    Args:
        prefix_type: "swa" | "csa" | "hca"
        n_committed_csa: per-seq committed CSA entries (for csa type)
        n_committed_hca: per-seq committed HCA entries (for hca type)
    """
    if positions is None:
        positions = torch.cat([
            torch.arange(cs, cs + slen, dtype=torch.int32)
            for cs, slen in zip(chunk_starts, seq_lens)
        ])

    counts = []
    T = sum(seq_lens)
    offset = 0
    for seq_i, (cs, slen) in enumerate(zip(chunk_starts, seq_lens)):
        for pos_in_chunk in range(slen):
            pos = cs + pos_in_chunk
            swa_low = max(pos - WIN + 1, 0)
            prefix_swa_count = max(cs - swa_low, 0)

            if prefix_type == "swa":
                counts.append(prefix_swa_count)
            elif prefix_type == "csa":
                csa_valid_k = min(
                    min((pos + 1) // 4, n_committed_csa or 0),
                    INDEX_TOPK,
                )
                counts.append(prefix_swa_count + csa_valid_k)
            elif prefix_type == "hca":
                counts.append(prefix_swa_count + (n_committed_hca or 0))
            offset += 1

    counts_t = torch.tensor(counts, dtype=torch.int32)
    indptr = torch.zeros(T + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(counts_t, dim=0)

    # Build actual prefix index values (placeholder paged offsets)
    pfx_total = indptr[-1].item()
    indices = torch.arange(pfx_total, dtype=torch.int32)  # sequential placeholder

    return indices, indptr


def generate_synth_inputs(
    target_T: int,
    n_seqs: int,
    H: int = 16,
    D: int = 512,
    dtype: torch.dtype = torch.bfloat16,
    softmax_scale: float = 0.0625,
    prefix_type: str = "csa",
    chunk_starts: Optional[List[int]] = None,
    n_committed_csa: int = 2000,
    n_committed_hca: int = 0,
) -> Dict:
    """Generate synthetic prefill inputs for a target T.

    Distributes target_T tokens evenly across n_seqs sequences.
    """
    base_len = target_T // n_seqs
    remainder = target_T % n_seqs
    seq_lens = [base_len + (1 if i < remainder else 0) for i in range(n_seqs)]
    actual_T = sum(seq_lens)

    if chunk_starts is None:
        chunk_starts = [0] * n_seqs

    # cu_seqlens_q: starting row in kv tensor per sequence
    cu_seqlens_q = [0]
    for slen in seq_lens[:-1]:
        cu_seqlens_q.append(cu_seqlens_q[-1] + slen)

    # Positions: absolute positions for each token
    positions_list = []
    for cs, slen in zip(chunk_starts, seq_lens):
        positions_list.extend(range(cs, cs + slen))

    # Q: random [T, H, D]
    q = torch.randn(actual_T, H, D, dtype=dtype) * 0.1

    # KV: random [total_kv_tokens, D]  (one row per token)
    kv = torch.randn(actual_T, D, dtype=dtype) * 0.1

    # Extend indices
    ext_indptr = synth_extend_indptr(seq_lens)
    ext_indices = synth_extend_indices(ext_indptr, cu_seqlens_q, seq_lens)

    # Prefix indices
    pfx_indices, pfx_indptr = synth_prefix_indices(
        seq_lens, chunk_starts, prefix_type,
        n_committed_csa=n_committed_csa,
        n_committed_hca=n_committed_hca,
    )

    # Unified KV: enough pages to cover prefix indices
    pfx_max = pfx_indices.max().item() + 1 if pfx_indices.numel() > 0 else 1
    unified_kv = torch.randn(pfx_max, D, dtype=dtype) * 0.1

    # Attn sink
    attn_sink = torch.zeros(H, dtype=torch.float32)

    return {
        "q": q,
        "unified_kv": unified_kv,
        "kv_indices_prefix": pfx_indices,
        "kv_indptr_prefix": pfx_indptr,
        "kv": kv,
        "kv_indices_extend": ext_indices,
        "kv_indptr_extend": ext_indptr,
        "attn_sink": attn_sink,
        "softmax_scale": softmax_scale,
    }


def bench_backend(fn, name, inputs, warmup=3, iters=20):
    """Benchmark a backend function."""
    for _ in range(warmup):
        fn(**inputs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(**inputs)
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - start) / iters * 1e6
    return elapsed_us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", type=str, help="Analyze a dump file")
    parser.add_argument("--source", type=str, help="Source dump for generating")
    parser.add_argument("--target-T", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--n-seqs", type=int, default=8)
    parser.add_argument("--prefix-type", choices=["swa", "csa", "hca"], default="csa")
    parser.add_argument("--output-dir", type=str, default="prefill_inputs/synth")
    parser.add_argument("--bench", action="store_true", help="Run benchmark after gen")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if args.analyze:
        analyze_dump(args.analyze)
        return

    # Determine H, D, dtype from source or defaults
    H, D, dtype, softmax_scale = 16, 512, torch.bfloat16, 0.0625
    n_seqs = args.n_seqs

    if args.source and os.path.exists(args.source):
        info = analyze_dump(args.source)
        H, D = info["H"], info["D"]
        dtype = info["dtype_q"]
        softmax_scale = info["softmax_scale"]
        if info["n_seqs"] > 0:
            n_seqs = info["n_seqs"]

    os.makedirs(args.output_dir, exist_ok=True)

    # Load backends
    from atom.model_ops.v4_kernels.paged_prefill import (
        _sparse_attn_v4_paged_prefill_triton,
    )
    tilelang_fn = None
    try:
        from atom.model_ops.v4_kernels.paged_prefill_8warp_splitk import (
            sparse_attn_v4_paged_prefill_tilelang_8warp_splitk,
        )
        tilelang_fn = sparse_attn_v4_paged_prefill_tilelang_8warp_splitk
    except ImportError:
        pass

    print(f"[synth] H={H} D={D} dtype={dtype} n_seqs={n_seqs} prefix_type={args.prefix_type}")
    print()

    for target_T in args.target_T:
        inputs = generate_synth_inputs(
            target_T=target_T,
            n_seqs=n_seqs,
            H=H, D=D,
            dtype=dtype,
            softmax_scale=softmax_scale,
            prefix_type=args.prefix_type,
        )

        T = inputs["q"].shape[0]
        pfx_n = inputs["kv_indices_prefix"].numel()
        ext_n = inputs["kv_indices_extend"].numel()

        print(f"==========  T = {T}  ==========")
        print(f"[synth] {args.prefix_type}: prefix={pfx_n}  extend={ext_n}")
        print(f"[synth] q={list(inputs['q'].shape)}  "
              f"unified_kv={list(inputs['unified_kv'].shape)}  "
              f"kv={list(inputs['kv'].shape)}")

        # Move to GPU
        gpu_inputs = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Save
        out_path = os.path.join(
            args.output_dir,
            f"synth_T{T}_H{H}_D{D}_{args.prefix_type}_pfx{pfx_n}_ext{ext_n}.pt",
        )
        torch.save(inputs, out_path)
        print(f"[synth] saved → {out_path}")

        if args.bench:
            triton_us = bench_backend(
                _sparse_attn_v4_paged_prefill_triton,
                "triton", gpu_inputs,
                warmup=args.warmup, iters=args.iters,
            )
            print(f"  triton                          : {triton_us:10.1f} us")

            if tilelang_fn is not None:
                tl_us = bench_backend(
                    tilelang_fn, "tilelang", gpu_inputs,
                    warmup=args.warmup, iters=args.iters,
                )
                print(f"  tilelang 8warp_splitk           : {tl_us:10.1f} us  "
                      f"(vs triton: {triton_us / tl_us:.2f}x)")
        print()


if __name__ == "__main__":
    main()
