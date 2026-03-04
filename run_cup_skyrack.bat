@echo off
REM ============================================================
REM  SubspaceAD - Cup_Skyrack Training & Evaluation
REM  32 Gutbilder (k_shot=32), 30 Augmentationen, DINOv2-Giant
REM ============================================================

REM --- In Projektverzeichnis wechseln ---
cd /d D:\PY2\SubspaceAD

REM --- Pfade anpassen ---
SET DATASET_ROOT=G:\OneDrive - Flex\3_ARBEIT ARBEIT ARBEIT\00_MAS Data Science - Master Thesis\Datasets\Cup_Skyrack\.raw
SET CATEGORY=Error_Images - Sky Rack - Sorted_crop

REM --- Umgebung synchronisieren (installiert CUDA-PyTorch beim ersten Mal) ---
uv sync

REM --- SubspaceAD ausfuehren ---
uv run main.py ^
    --dataset_name custom ^
    --dataset_path "%DATASET_ROOT%" ^
    --categories "%CATEGORY%" ^
    --model_ckpt facebook/dinov2-with-registers-large ^
    --image_res 672 ^
    --k_shot 32 ^
    --aug_count 30 ^
    --aug_list rotate ^
    --pca_ev 0.99 ^
    --score_method reconstruction ^
    --img_score_agg mtop1p ^
    --vis_count 10 ^
    --outdir results/cup_skyrack ^
    --batch_size 1

echo.
echo === Fertig! Ergebnisse unter results\cup_skyrack ===
pause