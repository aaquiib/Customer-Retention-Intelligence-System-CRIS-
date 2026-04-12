# DVC Workflow Guide: CRIS Pipeline

This guide covers all DVC operations for the CRIS (Customer Retention Intelligence System) pipeline.

---

## **Table of Contents**
1. [Setup & Initialization](#setup--initialization)
2. [Pipeline Execution](#pipeline-execution)
3. [Metrics & Monitoring](#metrics--monitoring)
4. [Data Management](#data-management)
5. [Troubleshooting](#troubleshooting)

---

## **Setup & Initialization**

### Windows (PowerShell)
```powershell
# Run the setup script
.\dvc_setup.ps1
```

### Mac/Linux (Bash)
```bash
# Make the script executable
chmod +x dvc_setup.sh

# Run the setup script
./dvc_setup.sh
```

### Manual Setup (All Platforms)
```bash
# Install DVC 3.x
pip install "dvc>=3.0,<4.0"

# Initialize DVC in the project root
dvc init

# Configure local remote (create directory first)
mkdir -p /tmp/dvc-remote  # Linux/Mac: or use Windows temp
dvc remote add -d myremote /tmp/dvc-remote

# Track raw data
dvc add data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## **Pipeline Execution**

### View Pipeline Structure
```bash
# Display the DAG (directed acyclic graph)
dvc dag

# Expected output:
# ┌─────────────────────┐
# │     preprocess      │
# └──────────┬──────────┘
#            │
# ┌──────────▼──────────┐
# │      features       │
# └──────────┬──────────┘
#            │
# ┌──────────▼─────────────────┐
# │  segmentation_train         │
# └──────────┬──────────────────┘
#            │
# ┌──────────▼─────────────────┐
# │  segmentation_assign        │
# └──────────┬──────────────────┘
#            │
# ┌──────────▼─────────────────┐
# │      churn_train            │
# └─────────────────────────────┘
```

### Run Full Pipeline
```bash
# Execute entire pipeline (all 5 stages in sequence)
dvc repro

# With verbose output
dvc repro -v

# Dry run (show what would execute, don't actually run)
dvc repro --dry
```

### Run Specific Stage
```bash
# Run only the preprocess stage
dvc repro preprocess

# Run only the features stage
dvc repro features

# Run from a specific stage onwards (includes dependent stages)
dvc repro -s features  # Runs features → segmentation_train → etc.

# Run churn_train stage only
dvc repro churn_train
```

### Force Re-run
```bash
# Force re-run of stage even if outputs exist
dvc repro --force preprocess

# Force re-run entire pipeline
dvc repro --force
```

---

## **Metrics & Monitoring**

### View Current Metrics
```bash
# Display churn model metrics (from latest run)
dvc metrics show

# Show in JSON format
dvc metrics show -R --json

# Show specific metrics file
dvc metrics show models/churn/metrics_latest.json
```

### Compare Metrics Across Experiments
```bash
# Compare metrics from current run vs Git branch (requires git)
dvc metrics diff

# Compare with specific baseline
dvc metrics diff main

# Show only metrics files
dvc metrics diff --all
```

### Inspect Metrics Files Directly
```bash
# View latest metrics (JSON)
cat models/churn/metrics_latest.json

# View metrics history (CSV)
cat models/churn/metrics_history.csv

# Show specific columns from history
head -20 models/churn/metrics_history.csv | cut -d, -f1,2,3,4,5
```

---

## **Data Management**

### Check Pipeline Status
```bash
# Show which outputs are missing or need re-computation
dvc status

# Show status with verbose details
dvc status -v

# Check status of specific stage
dvc status churn_train
```

### Push Data to Remote
```bash
# Push all cached data to remote storage
dvc push

# Push data from specific stage
dvc push churn_train

# Push with verbose output
dvc push -v
```

### Pull Data from Remote
```bash
# Pull cached data from remote
dvc pull

# Pull only outputs from specific stage
dvc pull churn_train

# Force pull (overwrite local cache)
dvc pull --force
```

### List Cached Data
```bash
# Show cache directory size
du -sh .dvc/cache/

# List all cached files
find .dvc/cache -type f | head -20
```

### Remove Cache
```bash
# Remove local cache (outputs still exist)
dvc gc

# Remove cache including remote
dvc gc --workspace

# Force remove (no confirmation)
dvc gc -f
```

---

## **Parameter Tracking**

### View Tracked Parameters
```bash
# Show all parameters tracked in config.yaml
dvc params diff

# Show parameters for specific stage
dvc params diff --targets churn_train

# Show in JSON format
dvc params diff --json
```

### Modify Parameters
Edit `config/config.yaml` directly:
```yaml
# Example: Change churn model learning rate
churn_modeling:
  lgbm_hyperparams:
    learning_rate: 0.01  # Changed from 0.008535633844517027
```

Then re-run the pipeline:
```bash
dvc repro churn_train  # Only re-trains churn model
```

---

## **Git Integration**

### Track Pipeline Files
DVC automatically integrates with Git. Commit these files:
```bash
# Add pipeline and configuration
git add dvc.yaml dvc.lock config/config.yaml .dvcignore .gitignore

# Commit changes
git commit -m "Add DVC pipeline definition"

# Push to remote (if using Git)
git push origin main
```

### .gitignore Auto-update
When you run `dvc add` or `dvc repro`, DVC automatically:
- Creates `.dvc` files for tracked data
- Updates `.gitignore` to exclude cached outputs

### View DVC Lock File
```bash
# dvc.lock tracks exact outputs of each stage run
cat dvc.lock

# Shows:
# schema: '2.0'
# stages:
#   preprocess:
#     cmd: python -m src.data.preprocess
#     deps:
#     - path: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
#       md5: abc123xyz789
#     outs:
#     - path: data/processed/processed_df.csv
#       md5: def456uvw012
```

---

## **Troubleshooting**

### Pipeline Won't Run
```bash
# Check if dependencies are installed
pip install -r requirements.txt

# Verify all input files exist
ls -la data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
ls -la config/config.yaml

# Run with verbose mode to see errors
dvc repro -v
```

### Stage Fails
```bash
# Check specific stage error
dvc repro <stage_name> -v

# Manually test the stage command
python -m src.data.preprocess

# Verify config.yaml is valid YAML
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"
```

### DVC Cache Issues
```bash
# Clear corrupted cache
dvc cache remove --not-in-remote

# Verify cache integrity
dvc cache status

# Rebuild cache from outputs
dvc commit -f
```

### Remote Issues
```bash
# List configured remotes
dvc remote list

# Check remote connectivity
dvc remote status

# Modify remote path
dvc remote modify myremote url /new/path/to/remote

# Remove remote
dvc remote remove myremote
```

### Restore Previous State
```bash
# Checkout specific version from dvc.lock
git checkout HEAD~1 dvc.lock

# Pull data for that version
dvc checkout
```

---

## **Quick Reference Commands**

| Task | Command |
|------|---------|
| **Initialize** | `dvc init` |
| **View pipeline** | `dvc dag` |
| **Run all** | `dvc repro` |
| **Run one stage** | `dvc repro <stage_name>` |
| **Check status** | `dvc status` |
| **View metrics** | `dvc metrics show` |
| **Push data** | `dvc push` |
| **Pull data** | `dvc pull` |
| **Show parameters** | `dvc params diff` |
| **Check remote** | `dvc remote list` |

---

## **Environment Setup**

### Conda Environment (For Clean Setup)
```bash
# Create isolated environment
conda create -n dvc-cris python=3.10

# Activate environment
conda activate dvc-cris

# Install requirements
pip install -r requirements.txt
pip install "dvc>=3.0,<4.0"

# Run setup
dvc_setup.ps1  # Windows PowerShell
# or
./dvc_setup.sh  # Linux/Mac
```

### Virtual Environment (venv)
```bash
# Create venv (separate from churn_venv)
python -m venv dvc_env

# Activate
source dvc_env/bin/activate  # Linux/Mac
# or
dvc_env\Scripts\activate.bat  # Windows

# Install
pip install -r requirements.txt
pip install "dvc>=3.0,<4.0"
```

---

## **Advanced: Custom Metrics**

To add additional metrics beyond churn model:

1. **Create custom metrics file** in any stage output:
   ```python
   # In src/segmentation/train_segments.py
   import json
   
   metrics = {
       "cluster_counts": {0: 1234, 1: 890, ...},
       "silhouette_score": 0.62,
       "davies_bouldin_index": 1.45
   }
   
   with open("models/segmentation/metrics.json", "w") as f:
       json.dump(metrics, f)
   ```

2. **Update dvc.yaml** to track metrics:
   ```yaml
   segmentation_train:
     metrics:
       - models/segmentation/metrics.json:
           cache: false
   ```

3. **View with DVC**:
   ```bash
   dvc metrics show models/segmentation/metrics.json
   ```

---

## **References**

- [DVC Official Docs](https://dvc.org/doc)
- [DVC Pipelines](https://dvc.org/doc/user-guide/pipelines)
- [DVC Metrics](https://dvc.org/doc/user-guide/metrics)
- [DVC Remote Storage](https://dvc.org/doc/user-guide/data-management/remote-storage)

---

**Last Updated:** April 12, 2026  
**DVC Version:** 3.x  
**Pipeline:** CRIS (churn + segmentation + decision system)
