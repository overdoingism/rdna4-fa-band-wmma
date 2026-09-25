#Requires -Version 7.0
<#
.SYNOPSIS
  Greedy-purity check for the FA band WMMA patch: plain decode vs draft-mtp verify must
  produce the same tokens.

.DESCRIPTION
  Starts llama-server once per run, sends the same prompt through /completion with greedy
  sampling (temperature 0, top_k 1, fixed seed, no prompt cache), stores the full JSON
  response and compares the generated token ids.

  Runs (each on a fresh server process):
    band-mtp    GGML_HIP_FA_BAND_WMMA=<Band> (and _SPLIT=<Split> if given), --spec-type draft-mtp
    band-plain  same variables, no speculative decoding
  With -Control the same pair is also run with both variables unset (stock rdna-boosts path),
  which checks the harness itself: that pair is expected to match on a stock build.

  The prompt can be as long as the context allows; a long prompt (e.g. 50K+ tokens) exercises
  the deep-KV path the patch changes.  Prefill runs at normal speed, so each run costs roughly
  one prefill of the prompt plus -NPredict decode steps.

.EXAMPLE
  pwsh -File purity-check.ps1 -ServerExe C:\llama\bin\llama-server.exe `
       -Model D:\models\model.gguf -PromptFile .\long-prompt.txt -NPredict 512 -Control
#>
param(
    [Parameter(Mandatory)] [string] $ServerExe,
    [Parameter(Mandatory)] [string] $Model,
    [Parameter(Mandatory)] [string] $PromptFile,
    [int]    $NPredict   = 512,
    [int]    $CtxSize    = 262144,
    [int]    $Port       = 8199,
    [int]    $LoadTimeoutSec = 900,
    [string] $OutDir     = ".\purity-out",
    [switch] $Control,
    # GGML_HIP_FA_BAND_WMMA value for the patched runs (2 or 4; 4 is the recommended setting).
    [string] $Band       = "4",
    # Optional GGML_HIP_FA_BAND_WMMA_SPLIT override (empty = the patch's default, one block per CU).
    [string] $Split      = "",
    # Server arguments shared by every run (model, host, port and ctx-size are added separately).
    [string[]] $BaseArgs = @(
        "--parallel", "1", "--n-gpu-layers", "999", "--split-mode", "layer",
        "--batch-size", "2048", "--ubatch-size", "512", "--threads", "6",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--flash-attn", "on",
        "--kv-unified", "--no-webui"
    ),
    # Arguments that turn MTP self-drafting on (removed for the MTP-off runs).
    [string[]] $MtpArgs = @(
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
        "--spec-draft-n-min", "0", "--spec-draft-p-min", "0"
    )
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$prompt = Get-Content -Raw -Encoding UTF8 -Path $PromptFile
$base   = "http://127.0.0.1:$Port"

function Invoke-Run([string] $name, [string] $bandWmma, [string] $split, [bool] $mtp) {
    Write-Host "=== $name (GGML_HIP_FA_BAND_WMMA=$(if ($bandWmma) { $bandWmma } else { '<unset>' }), SPLIT=$(if ($split) { $split } else { '<unset>' }), MTP=$mtp)"

    # Both variables are set or cleared explicitly so nothing leaks in from the calling shell.
    if ($bandWmma) { $env:GGML_HIP_FA_BAND_WMMA = $bandWmma }
    else { Remove-Item Env:GGML_HIP_FA_BAND_WMMA -ErrorAction SilentlyContinue }
    if ($split) { $env:GGML_HIP_FA_BAND_WMMA_SPLIT = $split }
    else { Remove-Item Env:GGML_HIP_FA_BAND_WMMA_SPLIT -ErrorAction SilentlyContinue }

    $srvArgs = @("--model", $Model, "--host", "127.0.0.1", "--port", "$Port", "--ctx-size", "$CtxSize") + $BaseArgs
    if ($mtp) { $srvArgs += $MtpArgs }

    # Start-Process joins the list with spaces, so quote anything that contains one (model paths).
    $srvArgs = $srvArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }

    $log  = Join-Path $OutDir "$name.server.log"
    $proc = Start-Process -FilePath $ServerExe -ArgumentList $srvArgs -PassThru -NoNewWindow `
                          -RedirectStandardOutput "$log.out" -RedirectStandardError $log
    try {
        $deadline = (Get-Date).AddSeconds($LoadTimeoutSec)
        while ($true) {
            if ($proc.HasExited) { throw "llama-server exited during load (see $log)" }
            try {
                $h = Invoke-WebRequest -Uri "$base/health" -TimeoutSec 5 -SkipHttpErrorCheck
                if ($h.StatusCode -eq 200) { break }
            } catch { }
            if ((Get-Date) -gt $deadline) { throw "llama-server not healthy after $LoadTimeoutSec s" }
            Start-Sleep -Seconds 2
        }

        $body = @{
            prompt         = $prompt
            n_predict      = $NPredict
            temperature    = 0
            top_k          = 1
            seed           = 1
            cache_prompt   = $false
            ignore_eos     = $true
            return_tokens  = $true
        } | ConvertTo-Json -Depth 4

        $resp = Invoke-RestMethod -Uri "$base/completion" -Method Post -ContentType "application/json; charset=utf-8" `
                                  -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 7200
        $resp | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -Path (Join-Path $OutDir "$name.json")

        $t = $resp.timings
        Write-Host ("    prompt {0} tok @ {1:N1} t/s, gen {2} tok @ {3:N2} t/s, draft {4}/{5} accepted" -f `
            $t.prompt_n, $t.prompt_per_second, $t.predicted_n, $t.predicted_per_second, $t.draft_n_accepted, $t.draft_n)
        return $resp
    } finally {
        if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force; $proc.WaitForExit() }
        Start-Sleep -Seconds 3   # let the driver release VRAM before the next load
    }
}

function Compare-Runs([string] $label, $a, $b) {
    $ta = @($a.tokens); $tb = @($b.tokens)
    $n  = [Math]::Min($ta.Count, $tb.Count)
    for ($i = 0; $i -lt $n; $i++) {
        if ($ta[$i] -ne $tb[$i]) {
            Write-Host "FAIL  $label : first mismatch at generated token $i ($($ta[$i]) vs $($tb[$i]))" -ForegroundColor Red
            return $false
        }
    }
    if ($ta.Count -ne $tb.Count) {
        Write-Host "FAIL  $label : token counts differ ($($ta.Count) vs $($tb.Count))" -ForegroundColor Red
        return $false
    }
    if ($n -eq 0) {
        Write-Host "FAIL  $label : no tokens returned (server too old for return_tokens?)" -ForegroundColor Red
        return $false
    }
    Write-Host "PASS  $label : $n generated tokens identical" -ForegroundColor Green
    return $true
}

$results = @()

$dMtp   = Invoke-Run "band$Band-mtp"   $Band $Split $true
$dPlain = Invoke-Run "band$Band-plain" $Band $Split $false
$results += Compare-Runs "band=$($Band): draft-mtp vs plain decode" $dMtp $dPlain

if ($Control) {
    $sMtp   = Invoke-Run "stock-mtp"   "" "" $true
    $sPlain = Invoke-Run "stock-plain" "" "" $false
    $results += Compare-Runs "stock: draft-mtp vs plain decode" $sMtp $sPlain
}

Remove-Item Env:GGML_HIP_FA_BAND_WMMA -ErrorAction SilentlyContinue
Remove-Item Env:GGML_HIP_FA_BAND_WMMA_SPLIT -ErrorAction SilentlyContinue

if ($results -contains $false) { exit 1 }
exit 0
