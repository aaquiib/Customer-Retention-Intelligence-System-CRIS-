# DVC 3.x Setup Script for CRIS Pipeline (Windows PowerShell)

$ErrorActionPreference = "Continue"

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "DVC 3.x Setup for CRIS Pipeline" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# Step 1: Check DVC installation
Write-Host ""
Write-Host "[STEP 1] Checking DVC installation..." -ForegroundColor Yellow

try {
    $dvcVersion = dvc --version
    Write-Host "DVC version: $dvcVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: DVC not installed. Run: pip install dvc" -ForegroundColor Red
    exit 1
}

# Step 2: Initialize DVC
Write-Host ""
Write-Host "[STEP 2] Initializing DVC..." -ForegroundColor Yellow

if (Test-Path ".dvc") {
    Write-Host "DVC already initialized" -ForegroundColor Green
} else {
    dvc init
    Write-Host "DVC initialized successfully" -ForegroundColor Green
}

# Step 3: Configure local remote
Write-Host ""
Write-Host "[STEP 3] Configuring local remote storage..." -ForegroundColor Yellow

$dvcRemotePath = Join-Path $env:TEMP "dvc-remote"

if (-not (Test-Path $dvcRemotePath)) {
    New-Item -ItemType Directory -Path $dvcRemotePath -Force | Out-Null
}

$remoteExists = dvc remote list 2>$null | Select-String "myremote"

if ($remoteExists) {
    dvc remote modify myremote url $dvcRemotePath
} else {
    dvc remote add -d myremote $dvcRemotePath
}

Write-Host "Local remote configured: $dvcRemotePath" -ForegroundColor Green

# Step 4: Add raw data to DVC
Write-Host ""
Write-Host "[STEP 4] Adding raw data to DVC tracking..." -ForegroundColor Yellow

$rawDataPath = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

if (Test-Path $rawDataPath) {
    if (Test-Path "$rawDataPath.dvc") {
        Write-Host "Raw data already tracked" -ForegroundColor Green
    } else {
        dvc add $rawDataPath
        Write-Host "Raw data tracked: $rawDataPath" -ForegroundColor Green
    }
} else {
    Write-Host "WARNING: Raw data file not found: $rawDataPath" -ForegroundColor Yellow
}

# Step 5: Display pipeline
Write-Host ""
Write-Host "[STEP 5] Pipeline structure:" -ForegroundColor Yellow
Write-Host ""
dvc dag
Write-Host ""

# Step 6: Summary
Write-Host "================================================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. View pipeline: dvc dag" -ForegroundColor Gray
Write-Host "  2. Run pipeline: dvc repro" -ForegroundColor Gray
Write-Host "  3. Show metrics: dvc metrics show" -ForegroundColor Gray
Write-Host "  4. See dvc_workflow.md for detailed instructions" -ForegroundColor Gray
Write-Host ""
