# rdna4-fa-band-wmma

An opt-in patch for [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts)
that speeds up long-context **decode / MTP-verify flash attention** on RDNA4 (gfx1201) for models with
**head size 256, GQA ratio 5–8 and a q8_0 KV cache** (e.g. Qwen3.5/3.8-27B, GQA 6).

It routes the whole decode/verify band (`n_q <= 8`) to the existing WMMA (MMA-f16) FA kernel with the
full GQA group folded into one block, and pins the stream-k KV split so that plain decode and every
verify width still reduce identically (the rdna-boosts GREEDY-PURITY band invariant).

Status: **experimental, opt-in** (`GGML_HIP_FA_BAND_WMMA=2`). With the variable unset the patched
build behaves exactly like stock rdna-boosts.

## Results

Hardware: AMD Radeon AI PRO R9700 (gfx1201, 32 GB), Windows 11, ROCm 10.0 (TheRock, clang 23).
Base: rdna-boosts `v16-ebbb18522-r11` rebased onto llama.cpp b11040.

### Real server (llama-server)

Qwen3.8-27B Q5_K_M, q8_0 K/V, `--spec-type draft-mtp --spec-draft-n-max 2`, `-c 262144`, one long
local conversation (context depth: **TBD**):

| | stock r11 | + this patch (`=2`) | LM Studio Vulkan runtime 2.43.0 (llama.cpp `f95b0d9`) |
|---|---|---|---|
| decode (t/s) | 27.78 | **33.26** (+19.7 %) | 32.94 |
| prefill (t/s) | 874.12 | 854.95 | — |

Prefill does not take the changed path (`n_q > 8` is untouched); the difference is run-to-run noise.

### Kernel microbenchmark

`test-backend-ops perf --test-file bench/fa-sweep.txt`, µs/run. Shape: Q f32 `[256, n_q, 24]`, K/V
`[256, kv, 4]`, mask f16, prec F32. Raw logs are in [`bench/results/`](bench/results/).

| case | stock r11 | patch `=2` | Vulkan b11040 |
|---|---|---|---|
| q8_0 kv 20480, n_q 3 | 386 | **189** | 268 |
| q8_0 kv 51200, n_q 3 | 1004 | **465** | 662 |
| q8_0 kv 102400, n_q 1 | 805 | 887 | 466 |
| q8_0 kv 102400, n_q 2 | 1437 | 900 | 856 |
| q8_0 kv 102400, n_q 3 | 2053 | **930** | 1327 |
| q8_0 kv 204800, n_q 3 | 4351 | **1886** | 2700 |
| f16 kv 102400, n_q 1 (MTP draft ctx) | 672 | 674 (unchanged path) | 678 |

Per decode step with `--spec-draft-n-max 2` (16 full-attention layers × verify `n_q = 3`, plus two
f16 `n_q = 1` MTP draft passes), FA time at kv 102400 drops from ~34.4 ms to ~16.3 ms (Vulkan: ~22.6 ms).

## Diagnosis

Measured first: at `n_q = 1`, kv 102400, the stock tile kernel reads **f16** K/V at ~615 GB/s (close to
the R9700's DRAM limit) but **q8_0** K/V at only ~276 GB/s. Half the bytes take longer, so the q8_0 path
is bound by instruction issue and latency rather than by bandwidth.

From the source (r11):

1. **Narrow GQA fold.** For GQA 6, `ncols2` falls back to 2 (6 is not divisible by 4 or 8), and in the
   `n_q <= 8` band the tile kernel is pinned to `cols_per_block = 2` (block 08). Each block therefore
   covers 2 of the 6 query heads, and every K/V element is fetched and dequantized **three times** per
   query token. The re-reads mostly hit L2, so DRAM traffic stays near 1×, but the dequant and
   load instructions triple.
2. **q8_0 staging cost.** `ggml_cuda_fattn_dequantize_q8_0_chunk` issues five 2-byte global loads per
   8 elements (q8_0 blocks are 34 bytes, so they are only 2-byte aligned), plus ~25–30 VALU. Each
   32-row iteration has 8 load→barrier phases with no prefetch.
3. **Not the cause:** block count and occupancy (about one full wave, ~99 % wave efficiency) and the
   combine pass (< 1 %).

Vulkan folds all 6 heads into one workgroup (N := 6, coopmat1), so it dequantizes each element once.

Restoring the VEC kernel is not the answer: the rdna-boosts measurements already show VEC slower than
the native q8_0 tile path at depth (`wip/issue-30-mtp-decode-regression`).

## The patch

[`patches/0001-fa-band-wmma.patch`](patches/0001-fa-band-wmma.patch) changes `ggml/src/ggml-cuda/fattn.cu`
and `ggml/src/ggml-cuda/fattn-common.cuh` (~65 lines). No new kernel code: it uses the existing
`(DKQ 256, ncols1 2|4, ncols2 8)` MMA instances and the block-15 native q8_0 read.

- **Selection** (`ggml_cuda_fattn_band_wmma_applies`, shared by kernel selection and ncols dispatch so
  they cannot disagree): `GGML_HIP_FA_BAND_WMMA` is 2 or 4, RDNA4, WMMA available, GQA optimization
  applies, `n_q <= 8`, head 256, GQA ratio in (4, 8], no logit softcap, K and V both q8_0.
- **Dispatch:** the whole band uses one instance, `ncols1` = the variable's value, `ncols2 = 8` (GQA 6
  folded with 2 masked columns). One instance means one kernel config (`nbatch_fa` etc.) for every
  `n_q` in the band.
- **Purity-preserving stream-k split** (`launch_fattn`): inside the band every output tile gets exactly
  `k` blocks, with `k = min(ntiles_KV, max_blocks / ntiles_dst(n_q = 1))`. With
  `nblocks = k · ntiles_dst`, the kernel's `kbc = floor(b · iter_k · ntiles_dst / nblocks)` becomes
  `t · iter_k + floor(i · iter_k / k)` for block `b = t·k + i`. So each tile's KV partition depends only
  on `k` and the KV length, never on `n_q`. The combine uses the uniform fixup (`bpt = k`).
- **f16 K/V is excluded on purpose.** It is already DRAM-bound on the tile kernel, measured 11–25 %
  slower on this path, and it is the MTP draft context's cache type.

### Trade-offs

- `n_q = 1` alone (plain decode without speculative drafting) is ~10 % slower than stock at kv 102400
  (887 vs 805 µs), because an `ncols1 = 2` tile carries one empty query column. With MTP
  (`n_q = draft + 1 = 3`) the band is a large net win.
- `GGML_HIP_FA_BAND_WMMA=4` measured slower than `=2` at every depth (see `bench/results/`).
- The RDNA MMA config for `(256, 256, ncols 16)` (64 threads, occupancy 2) was not tuned for decode;
  `n_q = 1` still trails Vulkan (887 vs 466 µs), so there is headroom left.

## Validation

- `test-backend-ops test --test-file bench/fa-sweep.txt -b ROCm0`: 16/16 OK with the variable unset and
  set to 2 ([`bench/results/test-*.log`](bench/results/)).
- With the variable unset, the patched build's sweep matches stock r11 within the run-to-run spread.
  The stock q8_0 tile path shows occasional ~2× outliers across runs; `hip-r11-stock-run*.log` shows them.
- **Greedy purity (draft-mtp vs plain decode, same tokens):** [`scripts/purity-check.ps1`](scripts/purity-check.ps1)
  — result: **TBD**.
- The patch applies cleanly (`git apply --check`) to both r11 and r13 (`v16-ebbb18522-r13`) trees.
  Compile-checked with ROCm 7.2.4 `amdclang++ --offload-arch=gfx1201`. Runtime-tested only with ROCm 10.0 on Windows.

## Usage

```sh
# in a llama.cpp tree with the rdna-boosts r11/r13 patch set applied
git apply /path/to/rdna4-fa-band-wmma/patches/0001-fa-band-wmma.patch
# rebuild the HIP backend (only ggml-hip changes)
cmake --build <build-dir> --config Release --target ggml-hip
```

Enable it at run time:

```sh
GGML_HIP_FA_BAND_WMMA=2 llama-server ...        # Linux
$env:GGML_HIP_FA_BAND_WMMA="2"; llama-server ... # PowerShell
```

Reproduce the microbenchmark (`test-backend-ops` is built with `-DLLAMA_BUILD_TESTS=ON`):

```sh
python3 bench/gen_fa_sweep.py -o fa-sweep.txt   # identical to bench/fa-sweep.txt
GGML_HIP_FA_BAND_WMMA=2 test-backend-ops perf --test-file fa-sweep.txt -b ROCm0
```

## License

MIT, same as llama.cpp. See [LICENSE](LICENSE).
