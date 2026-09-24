#!/usr/bin/env python3
"""Generate fa-sweep.txt: FLASH_ATTN_EXT cases for `test-backend-ops --test-file`.

The lines follow the format written by tests/test-export-graph-ops.cpp. They were derived
from the FA node exported for Qwen3.8-27B (qwen35 full-attention layer: head 256, 24 Q heads,
4 KV heads -> GQA 6, scale 1/16, prec F32, mask f16) and checked field by field against the
exported q8_0 lines for kv = 102400, nb = 1 and nb = 512.

Line format:
  op type ne0..3 | n_params params... | n_src [type ne0..3 nb0..3]... | name
"""

import argparse

GGML_OP_FLASH_ATTN_EXT = 74  # same value in upstream b11040 and rdna-boosts r11/r13
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8

D = 256          # head size (K and V)
N_HEAD_Q = 24
N_HEAD_KV = 4

# op_params: [0] scale = 1/16 as float bits, [3] prec = GGML_PREC_F32 (10), rest 0
OP_PARAMS = "16 1031798784 0 0 10 " + " ".join(["0"] * 12)


def kv_src(kv_type: str, kv: int) -> str:
    if kv_type == "q8":
        t, nb0, row = GGML_TYPE_Q8_0, 34, 34 * D // 32  # 272 bytes per head row
    else:
        t, nb0, row = GGML_TYPE_F16, 2, 2 * D           # 512 bytes per head row
    nb1 = row * N_HEAD_KV  # one KV cell = all KV heads
    return f"{t} {D} {kv} {N_HEAD_KV} 1 {nb0} {nb1} {row} {nb1 * kv}"


def fa_line(kv: int, n_q: int, kv_type: str) -> str:
    q_row = D * 4 * N_HEAD_Q  # f32 Q, heads interleaved per token (permuted view)
    q = f"{GGML_TYPE_F32} {D} {n_q} {N_HEAD_Q} 1 4 {q_row} {D * 4} {q_row * n_q}"
    mask = f"{GGML_TYPE_F16} {kv} {n_q} 1 1 2 {2 * kv} {2 * kv * n_q} {2 * kv * n_q}"
    kvs = kv_src(kv_type, kv)
    return (f"{GGML_OP_FLASH_ATTN_EXT} {GGML_TYPE_F32} {D} {N_HEAD_Q} {n_q} 1 {OP_PARAMS} "
            f"4 {q} {kvs} {kvs} {mask} fa_{kv_type}_kv{kv}_nb{n_q}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", default="fa-sweep.txt")
    ap.add_argument("--kv", type=int, nargs="+", default=[20480, 51200, 102400, 204800],
                    help="KV lengths (multiples of 256)")
    ap.add_argument("--nb", type=int, nargs="+", default=[1, 2, 3],
                    help="q8_0 query widths (decode = 1, MTP verify = draft + 1)")
    args = ap.parse_args()

    lines = []
    for kv in args.kv:
        for n_q in args.nb:
            lines.append(fa_line(kv, n_q, "q8"))
        lines.append(fa_line(kv, 1, "f16"))  # MTP draft context (f16 KV, n_q = 1)

    with open(args.output, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
