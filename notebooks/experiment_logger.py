import pandas as pd
import os
from datetime import datetime

def log_experiment(
    model_name,
    X_train,
    classification_rep,
    roc_auc,
    features_desc,
    hyperparameters="default",
    imbalance_handling="None",
    notes="",
    file_path="experiment_log.csv"
):
    if os.path.exists(file_path):
        df_old = pd.read_csv(file_path)
        exp_id = f"exp_{len(df_old)+1:03d}"
    else:
        df_old = None
        exp_id = "exp_001"
    
    metrics = {
        "experiment_id": exp_id,
        "timestamp": datetime.now(),
        "model": model_name,
        "features": features_desc,
        "feature_count": X_train.shape[1],
        "imbalance_handling": imbalance_handling,
        "hyperparameters": hyperparameters,
        "roc_auc": roc_auc,
        "precision_0": classification_rep["0"]["precision"],
        "recall_0": classification_rep["0"]["recall"],
        "f1_0": classification_rep["0"]["f1-score"],
        "precision_1": classification_rep["1"]["precision"],
        "recall_1": classification_rep["1"]["recall"],
        "f1_1": classification_rep["1"]["f1-score"],
        "notes": notes
    }
    
    df_new = pd.DataFrame([metrics])
    
    if df_old is None:
        df_new.to_csv(file_path, index=False)
    else:
        df_final = pd.concat([df_old, df_new], ignore_index=True)
        df_final.to_csv(file_path, index=False)
    
    print(f"✅ Experiment {exp_id} logged successfully!")