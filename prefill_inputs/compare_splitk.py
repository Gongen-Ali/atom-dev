"""Compare hand-rolled split-K vs T.gemm(split_k=8) variants of
paged_prefill on real dumped inputs.

Usage:
    python prefill_inputs/compare_splitk.py
    python prefill_inputs/compare_splitk.py --input prefill_inputs/prefill_T712_H16_D512_pfx7744_ext32040.pt
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
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        pts = sorted(glob.glob(os.path.join(path, "*.pt")))
        if pts:
            best = max(
                pts,
                key=lambda f: int(f.split("_T")[1].split("_")[0]) if "_T" in f else 0,
            )
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
        default="/home/gongen.ge/ATOM/prefill_inputs",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()

    input_file = find_input_file(args.input)
    if input_file is None:
        print(f"No .pt file found: {args.input}")
        sys.exit(1)
    print(f"[bench] loading: {input_file}")

    inputs = load_inputs(input_file)
    Tt, H, D = inputs["q"].shape
    print(f"[bench] T={Tt}  H={H}  D={D}  softmax_scale={inputs['softmax_scale']}")
    print(
        f"[bench] kv(prefix)={inputs['kv_indices_prefix'].numel()}  "
        f"kv(extend)={inputs['kv_indices_extend'].numel()}"
    )
    print()

    from atom.model_ops.v4_kernels.paged_prefill_8warp_splitk import (
        sparse_attn_v4_paged_prefill_tilelang_8warp_splitk as fn_manual,
    )
    from atom.model_ops.v4_kernels.paged_prefill_8warp_tgemm_splitk import (
        sparse_attn_v4_paged_prefill_tilelang_8warp_tgemm_splitk as fn_tgemm,
    )

    if not args.skip_correctness:
        out_manual = fn_manual(**inputs)
        out_tgemm = fn_tgemm(**inputs)
        max_abs = (out_manual.float() - out_tgemm.float()).abs().max().item()
        print(f"[correctness] max |manual - T.gemm| = {max_abs:.4e}")
        try:
            torch.testing.assert_close(
                out_tgemm, out_manual, rtol=args.rtol, atol=args.atol
            )
            print("[correctness] PASS")
        except AssertionError as e:
            print(f"[correctness] FAIL: {e}")
        print()

    print("[bench] running...")
    us_manual = bench(
        fn_manual, "tilelang manual 8warp split-K", inputs,
        warmup=args.warmup, iters=args.iters,
    )
    us_tgemm = bench(
        fn_tgemm, "tilelang T.gemm(split_k=8)     ", inputs,
        warmup=args.warmup, iters=args.iters,
    )
    print()
    print(f"  speedup (manual / T.gemm) : {us_manual / us_tgemm:.3f}x")


if __name__ == "__main__":
    main()
