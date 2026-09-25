# rdna4-fa-band-wmma

An opt-in patch for [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts)
that speeds up long-context **decode / MTP-verify flash attention** on RDNA4 (gfx1201) for models with
**head size 256, GQA ratio 5–8 and a q8_0 KV cache**, e.g. Qwen3.5/3.8-27B (GQA 6). It keeps the
rdna-boosts **greedy-purity** invariant: with MTP on or off, greedy decoding produces the same tokens.

The patch routes the whole decode/verify band (`n_q <= 8`) to the existing WMMA (MMA-f16) FA kernel,
with the full GQA group folded into one block. The KV is split round-robin over a fixed number of
blocks, so the split does not depend on the query width or the KV length.

Enable it with `GGML_HIP_FA_BAND_WMMA=4`. When the variable is unset, the patched build behaves exactly
like stock rdna-boosts.

## Results

### Real server, Linux

**Setup:**
- Hardware and software: R9700, Ubuntu, ROCm 7.2.4.
- Model: Qwen3.8-27B Q4_K_M, q8_0 K/V.
- Server flags: `--spec-type draft-mtp --spec-draft-n-max 2`, `-c 131072`.
- Run: one greedy 512-token generation per cell. The prompt is the same for every build, so draft
  acceptance is comparable (~313/395).
- Script: [`scripts/bench-server.py`](scripts/bench-server.py). Log: [`bench/results/linux/server-bench.log`](bench/results/linux/server-bench.log).

| prompt depth | upstream llama.cpp (2026-09-17) | rdna-boosts r11 | **r11 + this patch (`=4`)** | patch vs r11 (ms/step) |
|---|---|---|---|---|
| 20K | 43.63 t/s | 44.12 t/s | **46.40 t/s** | 57.9 → 56.2 (−3 %) |
| 60K | 34.32 t/s | 36.14 t/s | **41.31 t/s** | 71.1 → 62.2 (−13 %) |
| 110K | 26.96 t/s | 28.20 t/s | **36.03 t/s** | 88.8 → 70.2 (−21 %) |

Prefill is unchanged: `n_q > 8` does not take the new path, and the numbers are within 0.3 %.

### Real server, Windows

**Setup:**
- Hardware and software: R9700, Windows 11, ROCm 10.0.
- Base: rdna-boosts r11 rebased onto b11040.
- Model: Qwen3.8-27B Q5_K_M, q8_0 K/V, MTP n-max 2, `-c 262144`.
- Workload: one long local agent conversation (commentary + translation task), 93,225 prompt tokens.
  The same turn was rolled back and regenerated for every build.
- Sampling was not greedy, so output length and content differ a little between runs (about 17.7K
  generated tokens in the runs with full logs). The acceptance rate and ms/step columns come from
  those logs.

| build | decode | per verify step | draft acceptance |
|---|---|---|---|
| rdna-boosts r11 (stock) | 27.78 t/s | — | — |
| patch v3 (fixed-chunk split, superseded) | 28.87 t/s | 83.0 ms | 0.70 |
| LM Studio Vulkan runtime 2.43.0 (llama.cpp `f95b0d9`) | 32.94 t/s | — | — |
| patch v2 (contiguous split, **not greedy-pure**) | 33.26 t/s | — | — |
| **this patch (`=4`, P = 64)** | **36.41 t/s** | **67.2 ms** | 0.72 |

### Kernel microbenchmark (Linux)

- Command: `test-backend-ops perf --test-file bench/fa-sweep.txt`, in µs/run.
- Shape: Q f32 `[256, n_q, 24]`, K/V q8_0 `[256, kv, 4]`, f16 mask, prec F32.
- Patch column: `GGML_HIP_FA_BAND_WMMA=4`, P = 64 (the default on a 64-CU R9700).

| case | stock r11 | patch |
|---|---|---|
| kv 20480, n_q 3 | 370 | **211** |
| kv 102400, n_q 1 | 780 | 893 |
| kv 102400, n_q 3 (MTP verify) | 1921 | **893** |
| kv 204800, n_q 3 | 4121 | **1654** |
| f16 KV (MTP draft context) | — | unchanged path |

With `ncols1 = 4`, one block covers the whole verify batch (`n_q = draft + 1 = 3`), so K/V are read
once per step, and `n_q = 1` and `n_q = 3` cost the same.

## Diagnosis

1. **Measurement.** At `n_q = 1` and kv 102400, the stock tile kernel reads f16 K/V at ~615 GB/s, close
   to DRAM bandwidth, but q8_0 K/V at only ~276 GB/s. The q8_0 path is limited by instruction issue and
   latency, not by bandwidth.
2. **Narrow GQA fold.** With GQA 6, the tile kernel's `ncols2` falls back to 2, and block 08 pins
   `cols_per_block = 2` for the whole `n_q <= 8` band. Each block covers 2 of the 6 query heads, so
   every K/V element is fetched and dequantized three times per query token.
3. **Costly q8_0 staging.** The q8_0 chunk loader issues five 2-byte global loads plus about 25–30 VALU
   per 8 elements (q8_0 blocks are 34 bytes, so they are only 2-byte aligned).
4. **How Vulkan differs.** Vulkan folds all six heads into one workgroup, so it dequantizes each element
   only once.

## The patch

[`patches/0001-fa-band-wmma.patch`](patches/0001-fa-band-wmma.patch) touches `fattn.cu`, `fattn-common.cuh`
and `fattn-mma-f16.cuh` (about 170 lines). It applies cleanly to rdna-boosts r11 and r13. No new
kernel instance is added: it reuses the `(DKQ 256, ncols1 2|4, ncols2 8)` MMA instances and the
block-15 native q8_0 read.

- **Selection** (`ggml_cuda_fattn_band_wmma_applies`). Kernel selection and the ncols dispatch share
  this function, so they cannot disagree. The band path is taken only when all of the following hold:
  - `GGML_HIP_FA_BAND_WMMA` is 2 or 4
  - RDNA4 with WMMA
  - the GQA optimization applies
  - `n_q <= 8`, one sequence
  - head 256, GQA ratio in (4, 8]
  - no softcap
  - q8_0/q8_0 K/V

  f16 K/V stays on the tile kernel: it is already DRAM-bound there, and it is the MTP draft context's
  cache type.
- **Round-robin KV split** (`launch_fattn` and `flash_attn_ext_f16`). The band launches
  `gridDim = (output tiles, P)`.
  - Block `(t, i)` processes KV iterations `i, i+P, i+2P, …`, like the tile kernel's `parallel_blocks`.
  - `process_tile` gained a `kb0_step` argument and may own no iteration at short KV lengths. An empty
    block writes a neutral partial.
  - Block `P−1` writes the tile-finishing partial and the others write fixup partials. That is the
    layout `flash_attn_stream_k_fixup_uniform` already combines, in a fixed order.
- **Choice of P.** `P = nsm`, the CU count, overridable with `GGML_HIP_FA_BAND_WMMA_SPLIT`. It depends
  only on the device, never on `n_q` or on the KV length.
- **Why this stays pure.** A given KV iteration always lands in the same block, at the same position in
  its accumulation order. A longer KV only appends iterations that are fully masked for the earlier
  query rows, which are exact no-ops (`P = 0`, max unchanged). This covers a verify batch that crosses
  a 256-row padding boundary.

### History

| version | split | purity | speed |
|---|---|---|---|
| v1/v2 | contiguous stream-k, `k` blocks per tile | **broken**: split points moved when the 256-padded KV length grew (first greedy mismatch at KV position 18,944 = 74×256) | fast |
| v3 | contiguous, fixed-size chunks | pure | slow: block count grew with depth |
| v4 | round-robin, occupancy-derived P | pure | good for `=2`, P too small for `=4` |
| **v5** | round-robin, P = CU count | **pure** | **best with `=4`** |

### Trade-offs

- Plain decode without speculative drafting (`n_q = 1` only) is slower than stock:
  - kernel at kv 102400: +14 % (893 vs 780 µs)
  - end to end at 60K: −4.6 % t/s (21.73 vs 22.78)

  The band pays off with MTP, where verify batches have 3 tokens.
- The RDNA MMA configs for 16/32 columns were not tuned for decode. `n_q = 1` still trails the Windows
  Vulkan backend at the kernel level, so there is headroom left.

## Validation

- **Greedy purity** (draft-mtp vs plain decode, identical tokens), with the stock build as a harness
  control each time:
  - Linux, 60K prompt, `=4`, P = 64: PASS (512 tokens)
  - Windows, 18.7K prompt, v4 `=2`: PASS (512 tokens), see [`bench/results/windows/purity-v4-summary.txt`](bench/results/windows/purity-v4-summary.txt)
- **`test-backend-ops test`:** 53/53 FLASH_ATTN_EXT cases pass on the shape, with the variable unset,
  `=2` and `=4`. This includes 48 added eval cases (kv 512–16K × n_q 1,2,3,4,5,8 × two memory layouts) that use
  the realistic `init_tensor_kq_mask` (−inf blocks). The cases are in
  [`tests/band-eval-cases.diff`](tests/band-eval-cases.diff) and the logs in `bench/results/linux/`.
- **Builds:** every GPU result was measured with v4 plus `GGML_HIP_FA_BAND_WMMA_SPLIT=64`, which is exactly the v5 default on a 64-CU R9700 (v5 changes only that default). Linux used ROCm 7.2.4 and Windows ROCm 10.0. v5 itself was compile-checked with ROCm 7.2.4.
- **Purity script:** [`scripts/purity-check.ps1`](scripts/purity-check.ps1) (PowerShell 7).

## Usage

```sh
# in a llama.cpp tree with the rdna-boosts r11/r13 patch set applied
git apply /path/to/rdna4-fa-band-wmma/patches/0001-fa-band-wmma.patch
cmake --build <build-dir> --config Release --target ggml-hip
```

```sh
GGML_HIP_FA_BAND_WMMA=4 llama-server ...          # Linux
$env:GGML_HIP_FA_BAND_WMMA="4"; llama-server ...   # PowerShell
```

Reproduce the microbenchmark (`test-backend-ops` needs `-DLLAMA_BUILD_TESTS=ON`):

```sh
python3 bench/gen_fa_sweep.py -o fa-sweep.txt   # identical to bench/fa-sweep.txt
GGML_HIP_FA_BAND_WMMA=4 test-backend-ops perf --test-file fa-sweep.txt -b ROCm0
```

`bench/results/windows/` also keeps the v1/v2 microbenchmark logs from the development history.

## License

MIT, same as llama.cpp. See [LICENSE](LICENSE).
