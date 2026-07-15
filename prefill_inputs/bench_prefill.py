"""Standalone benchmark for prefill attention backends.

Usage:
    # Step 1: dump inputs from a real run
    ATOM_DUMP_PREFILL=1 ATOM_DUMP_PREFILL_MIN_T=100 python -m atom.examples.profile_offline ...

    # Step 2: run benchmark (inside docker)
    python -m atom.model_ops.v4_kernels.bench_prefill
    python -m atom.model_ops.v4_kernels.bench_prefill --input prefill_inputs/prefill_T8000_H16_D512_pfx64000_ext16000.pt
    python -m atom.model_ops.v4_kernels.bench_prefill --input prefill_inputs/   # auto-pick largest .pt
"""

import argparse
import glob
import os
import sys
import time

import torch


def load_inputs(path):
    data = torch.load(path, map_location="cpu")
    return {
        k: v.cuda() if isinstance(v, torch.Tensor) else v
        for k, v in data.items()
    }


def find_input_file(path):
    """If path is a directory, pick the largest .pt file by T value."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        pts = sorted(glob.glob(os.path.join(path, "*.pt")))
        if pts:
            # pick largest T from filename prefill_T{N}_...
            best = max(pts, key=lambda f: int(f.split("_T")[1].split("_")[0]) if "_T" in f else 0)
            return best
    return None


def bench(fn, name, inputs, warmup=5, iters=100):
    for _ in range(warmup):
        fn(**inputs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(**inputs)
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - start) / iters * 1e6
    print(f"  {name:50s} : {elapsed_us:10.1f} us")
    return elapsed_us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/gengen.ge/ATOM/prefill_inputs",
        help="Path to .pt file or directory (picks largest)",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    input_file = find_input_file(args.input)
    if input_file is None:
        print(f"No .pt file found: {args.input}")
        print("Run with ATOM_DUMP_PREFILL=1 first to dump inputs.")
        sys.exit(1)
    print(f"[bench] loading: {input_file}")

    inputs = load_inputs(input_file)
    T, H, D = inputs["q"].shape
    print(f"[bench] T={T}  H={H}  D={D}  softmax_scale={inputs['softmax_scale']}")
    print(f"[bench] unified_kv: {list(inputs['unified_kv'].shape)}")
    print(
        f"[bench] kv(prefix): {inputs['kv_indices_prefix'].numel()}  "
        f"kv(extend): {inputs['kv_indices_extend'].numel()}"
    )
    print()

    # --- Triton ---
    from atom.model_ops.v4_kernels.paged_prefill import (
        _sparse_attn_v4_paged_prefill_triton,
    )

    triton_us = bench(
        _sparse_attn_v4_paged_prefill_triton,
        "triton baseline",
        inputs,
        warmup=args.warmup,
        iters=args.iters,
    )

    # --- TileLang ---
    try:
        from atom.model_ops.v4_kernels.paged_prefill_8warp_splitk import (
            sparse_attn_v4_paged_prefill_tilelang_8warp_splitk,
        )
        tilelang_us = bench(
            sparse_attn_v4_paged_prefill_tilelang_8warp_splitk,
            "tilelang 8warp_splitk",
            inputs,
            warmup=args.warmup,
            iters=args.iters,
        )
        print(f"\n  speedup vs triton: {triton_us / tilelang_us:.2f}x")
    except ImportError:
        print("  TileLang backend not available")


if __name__ == "__main__":
    main()
