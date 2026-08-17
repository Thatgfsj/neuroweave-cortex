@echo off
REM NeuroWeave Cortex — Reproduce All Benchmark Results (v1.4.1)
REM Usage: benchmarks\run_all.bat [--locomo-path PATH]
REM
REM Default LoCoMo-10 path: %USERPROFILE%\AppData\Local\Temp\locomo-10\data\locomo10.json

set LOCOMO_PATH=%1
if "%LOCOMO_PATH%"=="" set LOCOMO_PATH=%USERPROFILE%\AppData\Local\Temp\locomo-10\data\locomo10.json

echo === 1/4: NWC HybridFusion (full pipeline) ===
python benchmarks\run_locomo_full.py --locomo-path "%LOCOMO_PATH%"

echo === 2/4: NWC ANN+BM25 (standard) ===
python benchmarks\run_locomo_standard.py --locomo-path "%LOCOMO_PATH%"

echo === 3/4: DeepSeek LLM-assisted scoring ===
python benchmarks\run_locomo_standard.py --locomo-path "%LOCOMO_PATH%" --use-deepseek

echo === 4/4: Ablation analysis ===
python benchmarks\ablation.py --locomo-path "%LOCOMO_PATH%"

echo === Done ===
echo Results saved to benchmarks/locomo_results.json
pause
