# 建置指南（給 AI 代理）／ Build Guide (for AI agents)

> **給人類讀者：** 這份檔案是寫給 AI 代理（Claude Code、Codex、Cursor 等）讀的。把這句話交給你的代理：
> 「請閱讀 https://github.com/overdoingism/rdna4-fa-band-wmma/blob/main/AGENT-GUIDE.md ，在我的電腦上照著建置，每一步需要安裝或下載東西時先問我。」
>
> **For humans:** This file is written for an AI coding agent (Claude Code, Codex, Cursor, …). Give your agent this
> instruction: "Read https://github.com/overdoingism/rdna4-fa-band-wmma/blob/main/AGENT-GUIDE.md and build it on my
> machine. Ask me before installing or downloading anything."

**這是什麼／What this is**

- 本專案是一個 llama.cpp 的 patch，疊在 [rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) 之上。
- 效果：AMD RDNA4 顯卡（已實測 Radeon AI PRO R9700）跑 Qwen3.8-27B 這類模型，**長上下文 + MTP 推測解碼**時，decode 明顯變快。
  - 110K 上下文約 +28%。
  - 短上下文幾乎沒差。
- 啟用方式：設環境變數 `GGML_HIP_FA_BAND_WMMA=4`。不設就和原版 rdna-boosts 完全相同。

A llama.cpp patch on top of rdna-boosts. On AMD RDNA4 (tested on the Radeon AI PRO R9700), it makes decode
clearly faster for Qwen3.8-27B-class models at **long context with MTP speculative decoding**: about +28 % at 110K
context, and little difference at short context. Turn it on with `GGML_HIP_FA_BAND_WMMA=4`; with the variable
unset, the build is identical to stock rdna-boosts.

The patch and this guide were written by an AI assistant (Claude); the repository owner ran the tests.

---

## 0. 代理守則／Rules for the agent

1. **安裝軟體、下載大型檔案、變更系統設定之前，先徵求使用者同意。**
   - 大型檔案：ROCm 約數 GB，模型約 17–20 GB。
   - 不要關閉防毒、不要改安全性設定、不要用系統管理員權限做本指南沒要求的事。
   Ask the user before installing software, downloading large files (ROCm is several GB, the model 17–20 GB) or
   changing system settings. Never disable antivirus or security features.
2. **每一步都檢查結果**，失敗就停下來，把錯誤訊息原文回報給使用者，不要自行跳過或改寫 patch。
   Check the result of every step. On failure, stop and report the exact error. Do not skip steps or edit the patches.
3. **記錄實際使用的版本**：OS、驅動、ROCm、CMake、各 repo 的 commit。回報時會用到。
   Record every version you actually used (OS, driver, ROCm, CMake, the commit of each repo) for the report.
4. 建置很吃 CPU，`ggml-hip` 約需 30–90 分鐘。開始前先告訴使用者。
   The build is CPU-heavy (`ggml-hip` takes 30–90 min). Tell the user before starting.

## 1. 適用條件檢查／Eligibility check

先確認以下三項，任何一項不符就停下來告訴使用者。Check these first; stop and tell the user if any fails.

| 項目／Item | 需求／Requirement | 怎麼查／How to check |
|---|---|---|
| GPU | RDNA4 = `gfx1201`（R9700 / RX 9070 / 9070 XT；**只有 R9700 實測過**）。`gfx1200`（RX 9060 系列）未測。 | Windows：裝置管理員或 `rocm-sdk targets`；Linux：`rocminfo \| grep gfx` |
| VRAM | 27B 模型 Q4/Q5 量化約 17–20 GB，再加上長上下文的 KV。**建議 32 GB**；16 GB 卡放不下 27B + 長上下文。 | — |
| 模型／Model | head size 256、GQA 比例 5–8（Qwen3.5 / 3.8-27B：24 Q / 4 KV heads）。**GGUF 須含 MTP 層**，否則沒有推測解碼可加速。 | 載入 log：`n_embd_head_k = 256`、`n_head = 24`、`n_head_kv = 4`；`creating MTP draft context` |

**本 patch 只在以下情況生效：** KV cache 為 **q8_0**（`-ctk q8_0 -ctv q8_0`），一次處理 1–8 個 token（decode / MTP verify）。其他模型或設定不受影響，行為與原版相同。

It only takes effect with a **q8_0 KV cache** and 1–8 query tokens (decode / MTP verify). Everything else runs
unchanged.

## 2. 工具鏈／Toolchain

| 工具／Tool | Windows 10/11 | Linux (Ubuntu 24.04+) |
|---|---|---|
| GPU 驅動／driver | AMD Adrenalin（最新版） | 核心內建 amdgpu；使用者需在 `render`、`video` 群組 |
| Shell | **PowerShell 7（`pwsh`）**：`winget install Microsoft.PowerShell`。本指南所有 Windows 指令都在 `pwsh` 裡執行，內建的 Windows PowerShell 5.1 不適用 | bash |
| Git | `winget install Git.Git` | `sudo apt install git` |
| CMake ≥ 3.21 | `winget install Kitware.CMake` | `sudo apt install cmake` |
| Ninja | `winget install Ninja-build.Ninja`；若裝壞，改用 https://github.com/ninja-build/ninja/releases 的 `ninja-win.zip`，解壓後放進 `PATH` | `sudo apt install ninja-build`（可選） |
| Python 3.10+ | `winget install Python.Python.3.12` | `sudo apt install python3 python3-venv` |
| C/C++ 函式庫 | **Visual Studio 2022**（17.x）Build Tools，勾選「使用 C++ 的桌面開發」（提供 Windows SDK / MSVC）。**不要只裝 VS 2026（18.x）**：它的新 STL 會讓 HIP 編譯失敗，見 §7 | `sudo apt install build-essential` |
| ROCm SDK | **ROCm 10.0 pip 套件**（見下方，免安裝程式） | 已有 `/opt/rocm`（7.2+）就直接用；否則同樣用 pip 套件 |

Windows 的 VS Build Tools 可用：
`winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive"`

### ROCm SDK（pip 套件，Windows 與 Linux 通用）

AMD 官方的 ROCm 10.0 穩定版 pip 套件，裝在獨立的 Python 虛擬環境裡，不會動到系統。
AMD's official ROCm 10.0 stable wheels, installed into a separate virtual environment.

```powershell
# Windows (pwsh). Linux: use python3 and .venv/bin/ instead of Scripts\
py -3.12 -m venv C:\rocm-venv
C:\rocm-venv\Scripts\python.exe -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,devel,device-gfx1201]==10.0.0"
C:\rocm-venv\Scripts\rocm-sdk.exe init
$ROCM = (Resolve-Path (C:\rocm-venv\Scripts\rocm-sdk.exe path --root)).Path   # absolute path
$env:PATH = "$(C:\rocm-venv\Scripts\rocm-sdk.exe path --bin);$env:PATH"   # needed by rocm-sdk test, the build and the server
C:\rocm-venv\Scripts\rocm-sdk.exe test
```

- **版本固定為 `==10.0.0`**，這是實測過的版本。之後出新版時行為可能改變，不要自動升級。
  Pinned to the tested `==10.0.0`; do not upgrade automatically.
- `rocm-sdk test` 必須通過（有 skipped 無妨）。它需要 `rocm-sdk path --bin` 已在 `PATH` 裡，所以上面先設 `PATH` 再測。
  - `PATH` 的設定**只在目前的 shell 有效**。之後每開一個新視窗（建置、執行 server）都要重設一次 `$ROCM` 和 `PATH`。
  `rocm-sdk test` must pass (skips are fine). It needs `rocm-sdk path --bin` on `PATH`, hence the order above.
  The `PATH` change only lasts for the current shell: set `$ROCM` and `PATH` again in every new window.
- GPU 是 RX 9060 系列時，把 `device-gfx1201` 換成 `device-gfx1200`（未實測）。
  Use `device-gfx1200` for the RX 9060 series (untested).
- 參考：[TheRock RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)

## 3. 取得原始碼並套用 patch／Get the sources and apply the patches

依序套上三層：upstream llama.cpp → rdna-boosts → 本 patch。rdna-boosts 的基準 commit 寫在它的 `release.json` 裡。
Three layers: upstream llama.cpp → rdna-boosts → this patch. The rdna-boosts base commit is in its `release.json`.

```powershell
# Windows: disable CRLF conversion, or the patches will not apply
git clone -c core.autocrlf=false https://github.com/ggml-org/llama.cpp
git clone -c core.autocrlf=false https://github.com/stew675/llama-cpp-rdna-boosts
git clone -c core.autocrlf=false https://github.com/overdoingism/rdna4-fa-band-wmma

$BASE = (Get-Content llama-cpp-rdna-boosts\release.json | ConvertFrom-Json).base
cd llama.cpp
git checkout $BASE
git apply --check ..\llama-cpp-rdna-boosts\rdna-boosts-all.patch
git apply ..\llama-cpp-rdna-boosts\rdna-boosts-all.patch
git apply --check ..\rdna4-fa-band-wmma\patches\0001-fa-band-wmma.patch
git apply ..\rdna4-fa-band-wmma\patches\0001-fa-band-wmma.patch
```

```bash
# Linux
git clone https://github.com/ggml-org/llama.cpp
git clone https://github.com/stew675/llama-cpp-rdna-boosts
git clone https://github.com/overdoingism/rdna4-fa-band-wmma
BASE=$(python3 -c "import json;print(json.load(open('llama-cpp-rdna-boosts/release.json'))['base'])")
cd llama.cpp && git checkout "$BASE"
git apply ../llama-cpp-rdna-boosts/rdna-boosts-all.patch
git apply ../rdna4-fa-band-wmma/patches/0001-fa-band-wmma.patch
```

- 已驗證可乾淨套用：rdna-boosts `v16-ebbb18522-r11` 與 `r13`（base `ebbb18522`）。
  Verified to apply cleanly on rdna-boosts r11 and r13 (base `ebbb18522`).
- **如果 `git apply --check` 失敗**：多半是 rdna-boosts 已更新到新的 base。停下來回報，附上 `release.json` 的 `release` 與 `base`。不要硬套或手動改。
  If `git apply --check` fails, rdna-boosts has most likely moved to a new base. Stop and report the `release` and
  `base` fields of `release.json`. Do not force the patch or edit it by hand.

## 4. 建置／Build

只需要 `llama-server`。建置 log 請直接重導到檔案：
- Windows 用 `*>`，可同時收下 stdout 與 stderr，錯誤訊息才會進檔案。
- 不要用 `2>&1 > file`：順序錯了 stderr 會留在畫面上。
- 不要用 `ForEach-Object` 逐行寫 log：會拖垮建置。

Only `llama-server` is needed. Redirect the build log straight to a file. On Windows use `*>`, which captures stdout
and stderr alike. Do not use `2>&1 > file` (stderr stays on the console) or a per-line `ForEach-Object` logger (it
stalls the build).

**Windows 的 RC 編譯器**：CMake 自動偵測到的 `rc.exe` 位在 Windows SDK 的長路徑下，路徑含空格和括號，會讓 CMake 的 RC 偵測步驟失敗。所以要明確傳入它的 8.3 短路徑。
**Windows RC compiler:** CMake's auto-detected `rc.exe` sits under a Windows SDK path with spaces and parentheses,
which breaks CMake's RC detection. Pass its 8.3 short path explicitly:

```powershell
$rcLong = (Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\rc.exe" | Sort-Object FullName -Descending | Select-Object -First 1).FullName
$RC = (New-Object -ComObject Scripting.FileSystemObject).GetFile($rcLong).ShortPath -replace '\\','/'
$RC   # must contain no spaces (e.g. C:/PROGRA~2/WI3CF2~1/10/bin/...); if it still has spaces, 8.3 names are disabled on that drive: see §7
```

```powershell
# Windows (in llama.cpp\, with $ROCM from step 2)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  "-DCMAKE_C_COMPILER=$ROCM\lib\llvm\bin\clang.exe" `
  "-DCMAKE_CXX_COMPILER=$ROCM\lib\llvm\bin\clang++.exe" `
  "-DCMAKE_HIP_COMPILER=$ROCM\lib\llvm\bin\clang.exe" `
  "-DCMAKE_PREFIX_PATH=$ROCM" "-DHIP_PATH=$ROCM" `
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DGGML_BACKEND_DL=ON -DGGML_CPU=ON -DGGML_NATIVE=OFF `
  -DLLAMA_BUILD_WEBUI=OFF '-DCMAKE_C_FLAGS=-Wno-error=incompatible-pointer-types' "-DCMAKE_RC_COMPILER=$RC"
cmake --build build --config Release -j 16 --target llama-server ggml-hip ggml-cpu *> build.log
```

```bash
# Linux (ROCM=/opt/rocm, or the output of: .venv/bin/rocm-sdk path --root)
HIPCXX="$ROCM/lib/llvm/bin/clang" HIP_PATH="$ROCM" \
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DLLAMA_BUILD_WEBUI=OFF
cmake --build build -j"$(nproc)" --target llama-server > build.log 2>&1
```

- 成功的判斷：`build.log` 結尾沒有 error，而且 `build/bin/` 裡有 `llama-server`。Windows 使用 `GGML_BACKEND_DL=ON`，後端是獨立 DLL，所以還要有 `ggml-hip.dll` 與 `ggml-cpu.dll`。
  Success means no error at the end of `build.log` and `llama-server` in `build/bin/`. On Windows (`GGML_BACKEND_DL=ON`) the backends are separate DLLs, so `ggml-hip.dll` and `ggml-cpu.dll` must be there too.
- 建置途中可能出現「Provisioning UI assets」或 UI 下載失敗的訊息。
  - 只要最後判斷成功就沒關係。
  - 如果建置因此在連結 `llama-server` 之前就停了：先重跑一次 `cmake --build`；仍失敗就停下來回報。
  "Provisioning UI assets" or UI download failures may show up. They are harmless if the build succeeds. If they stop
  the build before `llama-server` is linked, re-run `cmake --build` once, and report if it still fails.

## 5. 執行／Run

```powershell
# Windows: the ROCm runtime DLLs must be on PATH
$env:PATH = "$(C:\rocm-venv\Scripts\rocm-sdk.exe path --bin);$env:PATH"
$env:GGML_HIP_FA_BAND_WMMA = "4"
.\build\bin\llama-server.exe -m <model.gguf> -ngl 99 -c 131072 -fa on -ctk q8_0 -ctv q8_0 --parallel 1 `
  --spec-type draft-mtp --spec-draft-n-max 2 --jinja --host 127.0.0.1 --port 8080
```

```bash
# Linux
GGML_HIP_FA_BAND_WMMA=4 ./build/bin/llama-server -m <model.gguf> -ngl 99 -c 131072 -fa on -ctk q8_0 -ctv q8_0 --parallel 1 \
  --spec-type draft-mtp --spec-draft-n-max 2 --jinja --host 127.0.0.1 --port 8080
```

- 參數說明：
  - `-ctk q8_0 -ctv q8_0`：**必要**，本 patch 只作用於 q8_0 KV。
  - `--spec-type draft-mtp`：開啟 MTP 推測解碼，本 patch 主要加速的就是它。
  - `-c`：上下文長度，依 VRAM 調整（32 GB 卡搭配 Q5_K_M 可到 262144）。
  - `--parallel 1`：**必要**。本 patch 只在單一序列時生效：多個 slot 又沒有共用 KV 時，attention 會分成多個序列，patch 就不會啟用。
- `-ctk/-ctv q8_0` is required; `draft-mtp` is what the patch speeds up; size `-c` to your VRAM.
- `--parallel 1` is required: the patch only engages for a single sequence, and several slots without a unified KV
  cache split attention into several sequences.
- 啟動後，OpenAI 相容 API 在 `http://127.0.0.1:8080/v1`，可以接任何前端使用。
  An OpenAI-compatible API is then served at `http://127.0.0.1:8080/v1`.

## 6. 驗證／Verify

1. **載入 log** 應出現：`ROCm0` 或 `gfx1201`、`creating MTP draft context`。
   The load log should show `ROCm0`/`gfx1201` and `creating MTP draft context`.
2. **A/B 比較**：用同一個長 prompt（≥ 50K token），`temperature` 設 0，分別在 `GGML_HIP_FA_BAND_WMMA` 未設定與設為 `4` 時各跑一次。比較 server log 的 `eval time`（ms per token），以及 `draft acceptance`。
   A/B: run the same long prompt (≥ 50K tokens, temperature 0) with the variable unset and with `=4`, then compare
   `eval time` (ms per token) and `draft acceptance` in the server log.
   - **預期：** 60K 快約 10–15%，110K 快約 20–30%；短上下文差異很小。
     Expected: about 10–15 % faster at 60K and 20–30 % at 110K; little change at short context.
   - 兩次的 draft acceptance 若差很多，代表生成內容不同，t/s 就不能直接比。這時改用「每步耗時」比較：`eval 毫秒 ÷ (生成 token 數 − 接受數)`。
     If acceptance differs a lot, compare per-step time instead: eval ms ÷ (generated tokens − accepted drafts).
3. **可選：greedy 一致性**。這項檢查 MTP 開與關時，greedy 輸出是否逐字相同。
   Optional greedy-purity check (same greedy tokens with MTP on and off):
   - Windows（需 `pwsh`）：`pwsh -File scripts/purity-check.ps1 -ServerExe <llama-server.exe> -Model <gguf> -PromptFile <long.txt> -Control`
   - Linux：`scripts/bench-server.py`，需先設 `MODEL`、`UPSTREAM_SERVER`、`PATCHED_SERVER`、`CORPUS_DIR` 環境變數（見檔頭說明）
   - 兩組都應該顯示 `PASS`。Both pairs should report PASS.

## 7. 疑難排解／Troubleshooting

| 症狀／Symptom | 處理／Fix |
|---|---|
| `git apply` 失敗 | Windows 確認 clone 時有 `core.autocrlf=false`；否則是 rdna-boosts 的 base 已變，停下回報 |
| CMake 找不到 HIP / hipBLAS | `CMAKE_PREFIX_PATH`、`HIP_PATH` 要指向 `rocm-sdk path --root`；先跑 `rocm-sdk init` |
| Windows 連結錯誤（找不到 `msvcrt`、`kernel32.lib`） | 安裝 VS 2022 Build Tools 的 C++ 工作負載 |
| configure 在 RC / `CMakeRCCompiler.cmake` 出錯 | 沒傳 `CMAKE_RC_COMPILER`，或傳入的路徑含空格。若 `$RC` 仍含空格，代表該磁碟停用了 8.3 短檔名（ROCm wheel 不含 `llvm-rc`，不能拿來替代）。**首選**（不改系統設定）：把 `rc.exe` 連同同目錄的 `rcdll.dll`、`ServicingCommon.dll` 複製到無空格的資料夾（少了任一個 DLL，`rc.exe` 都無法執行），例如 `New-Item -ItemType Directory C:\tools\winrc -Force; 'rc.exe','rcdll.dll','ServicingCommon.dll' \| ForEach-Object { Copy-Item (Join-Path (Split-Path $rcLong) $_) C:\tools\winrc\ }`，先用 `C:\tools\winrc\rc.exe /?` 確認能執行，再傳 `-DCMAKE_RC_COMPILER=C:/tools/winrc/rc.exe`，並刪掉 `build` 目錄重跑 configure。**次選**：開啟 8.3 短檔名。這需要系統管理員權限，也會改動系統設定，**必須先徵得使用者同意** |
| `cannot overload __host__ __device__ function 'isgreater'`（或其他 `<cmath>` 相關錯誤） | 編譯器抓到 VS 2026（18.x）的新 STL。**首選**：安裝 VS 2022 Build Tools，並在 configure 前設定 `$env:VCToolsInstallDir` 指向 VS 2022 的 `...\VC\Tools\MSVC\14.4x.xxxxx\`，然後刪掉 `build` 目錄重跑 configure。**備案**（只改虛擬環境內的檔案）：在 `$ROCM` 底下的 `__clang_hip_runtime_wrapper.h` 中，把 `#include <__clang_cuda_math_forward_declares.h>` 移到 `#include <cmath>` **之前**（原檔中 `<cmath>` 在前）。改之前先備份，並告知使用者 |
| configure 或編譯找不到 HIP 裝置函式庫／`hipconfig` 路徑錯誤 | 在 configure 前補設：`$env:HIP_PLATFORM="amd"`、`$env:ROCM_PATH=$ROCM`、`$env:HIP_PATH=$ROCM`、`$env:LLVM_PATH="$ROCM\lib\llvm"`、`$env:HIP_DEVICE_LIB_PATH="$ROCM\lib\llvm\amdgcn\bitcode"`（先確認該目錄存在） |
| `rocm-sdk test` 報 `WinError 2`（`hipconfig`） | `rocm-sdk path --bin` 沒在 `PATH` 裡，照 §2 的順序先設 `PATH` 再測 |
| 腳本報 `#Requires -Version 7.0` | 用 `pwsh` 而不是 `powershell` 執行 |
| 啟動時找不到 `amdhip64*.dll` | 把 `rocm-sdk path --bin` 加進 `PATH` |
| `no ROCm devices` | 更新驅動；Linux 確認使用者在 `render`、`video` 群組 |
| 顯存不足 OOM | 降低 `-c`，或改用較小的量化 |
| 看不出加速 | 確認：q8_0 KV、有開 MTP、`--parallel 1`、上下文夠長（≥ 50K）、環境變數設在**啟動 server 的同一個 shell** |
| 沒有 `creating MTP draft context` | 這個 GGUF 不含 MTP 層，換一個有 MTP 的版本 |

## 8. 回報／Reporting

- 問題或測試結果請開在 https://github.com/overdoingism/rdna4-fa-band-wmma/issues 。
- 附上：GPU 型號、OS、驅動版本、ROCm 版本、rdna-boosts release、模型與量化、啟動參數、A/B 的 `eval time` 與 `draft acceptance`。

Report problems or results at https://github.com/overdoingism/rdna4-fa-band-wmma/issues with: GPU, OS, driver,
ROCm version, rdna-boosts release, model and quantization, server flags, and the A/B `eval time` and
`draft acceptance`.

- 這是實驗性質的第三方 patch，**與 rdna-boosts、llama.cpp 官方無關**。有問題請不要回報到那兩個專案。
  This is an experimental third-party patch, not affiliated with rdna-boosts or llama.cpp. Please do not report its
  problems there.
