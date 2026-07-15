#!/usr/bin/env bash
# restore_large_inputs.sh
# 还原被 zstd 压缩的大输入张量(benchmark 用的合成占位输入)。
#
# 背景:
#   prefill_inputs/prefill_T712_H16_D512_pfx7744_ext32040.pt  (原始 ~4.0GB)
#   decode_inputs/decode_T32_H16_D512_kv3648.pt               (原始 ~4.0GB)
# 这两个文件几乎全是 padding 0,zstd 压缩后仅 ~9MB / ~1.5MB,
# 因此以 .zst 形式存入仓库;原始 .pt 已在 .gitignore 中忽略,不进版本库。
#
# 用法:
#   git clone 后在仓库根目录执行:  bash restore_large_inputs.sh
#
# 依赖: zstd  (Ubuntu: apt install zstd / CentOS: yum install zstd)

set -euo pipefail
cd "$(dirname "$0")"

if ! command -v zstd >/dev/null 2>&1; then
  echo "错误: 未找到 zstd,请先安装 (apt install zstd 或 yum install zstd)" >&2
  exit 1
fi

files=(
  "prefill_inputs/prefill_T712_H16_D512_pfx7744_ext32040.pt"
  "decode_inputs/decode_T32_H16_D512_kv3648.pt"
)

for f in "${files[@]}"; do
  if [[ -f "$f.zst" ]]; then
    echo "解压: $f.zst -> $f"
    zstd -d -f "$f.zst" -o "$f"
  else
    echo "警告: 未找到 $f.zst,跳过" >&2
  fi
done

echo "完成,已还原原始 .pt 输入文件。"
