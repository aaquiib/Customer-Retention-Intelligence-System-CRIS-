#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════════
# DVC 3.x Setup Script for CRIS Pipeline
# 
# This script initializes DVC for the churn-segmentation decision system pipeline.
# Runs the following setup steps:
# 1. Initialize DVC in the project
# 2. Configure local remote storage at /tmp/dvc-remote
# 3. Add raw data and initial artifacts to DVC tracking
# 4. Display pipeline structure
# ════════════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════════════════════"
echo "DVC 3.x Setup for CRIS Pipeline"
echo "════════════════════════════════════════════════════════════════════════════════"

# Step 1: Check if DVC is installed
echo ""
echo "[STEP 1] Checking DVC installation..."
if ! command -v dvc &> /dev/null; then
    echo "❌ DVC is not installed. Please run:"
    echo "   pip install 'dvc>=3.0,<4.0'"
    exit 1
fi
echo "✓ DVC version: $(dvc --version)"

# Step 2: Initialize DVC (if not already initialized)
echo ""
echo "[STEP 2] Initializing DVC..."
if [ -d ".dvc" ]; then
    echo "✓ DVC already initialized"
else
    dvc init
    echo "✓ DVC initialized"
fi

# Step 3: Configure local remote
echo ""
echo "[STEP 3] Configuring local remote storage..."
DVC_REMOTE_PATH="/tmp/dvc-remote"
mkdir -p "$DVC_REMOTE_PATH"
dvc remote add -d myremote "$DVC_REMOTE_PATH"
echo "✓ Local remote configured at: $DVC_REMOTE_PATH"
echo "  (To change remote, use: dvc remote modify myremote url <new-path>)"

# Step 4: Add raw data to DVC tracking
echo ""
echo "[STEP 4] Adding raw data to DVC tracking..."
RAW_DATA="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
if [ -f "$RAW_DATA" ]; then
    if [ -f "$RAW_DATA.dvc" ]; then
        echo "✓ Raw data already tracked"
    else
        dvc add "$RAW_DATA"
        echo "✓ Raw data tracked: $RAW_DATA"
    fi
else
    echo "⚠ Raw data file not found: $RAW_DATA"
fi

# Step 5: Display pipeline structure
echo ""
echo "[STEP 5] Pipeline structure:"
echo ""
dvc dag
echo ""

# Step 6: Display next steps
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "Setup Complete! Next Steps:"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "1. Review pipeline configuration:"
echo "   $ dvc dag                    # View pipeline DAG"
echo "   $ dvc params diff            # Show all tracked parameters"
echo ""
echo "2. Run the pipeline:"
echo "   $ dvc repro                  # Run entire pipeline"
echo "   $ dvc repro preprocess       # Run specific stage"
echo "   $ dvc repro -s features      # Run from specific stage onwards"
echo ""
echo "3. Monitor metrics:"
echo "   $ dvc metrics show           # Display churn model metrics"
echo "   $ dvc metrics compare        # Compare metrics across runs"
echo ""
echo "4. Manage data:"
echo "   $ dvc push                   # Push cached data to remote"
echo "   $ dvc pull                   # Pull cached data from remote"
echo "   $ dvc status                 # Check pipeline status"
echo ""
echo "5. Documentation:"
echo "   See dvc_workflow.md for detailed DVC operations guide"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
