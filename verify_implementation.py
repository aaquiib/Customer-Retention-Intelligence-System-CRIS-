#!/usr/bin/env python
"""
Verification script to validate implementation of refactored MLOps pipeline.

Checks:
1. All required files exist
2. Configuration loads without errors
3. All modules import correctly
4. Module exports are correct
5. Type hints are present
6. Logging works
7. I/O utilities work
"""

import sys
from pathlib import Path


def check_file_exists(filepath: str) -> bool:
    """Check if file exists."""
    exists = Path(filepath).exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {filepath}")
    return exists


def check_directory_exists(dirpath: str) -> bool:
    """Check if directory exists."""
    exists = Path(dirpath).exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {dirpath}/")
    return exists


def main():
    """Run all verification checks."""
    print("\n" + "=" * 80)
    print("VERIFICATION: Refactored MLOps Pipeline")
    print("=" * 80)

    all_pass = True

    # ──────────────────────────────────────────────────────────────────
    # 1. FILE STRUCTURE
    # ──────────────────────────────────────────────────────────────────
    print("\n[1] FILE STRUCTURE")
    print("-" * 80)

    required_files = [
        'config/config.yaml',
        'src/__init__.py',
        'src/config.py',
        'src/utils/__init__.py',
        'src/utils/io_utils.py',
        'src/utils/logging_config.py',
        'src/utils/feature_validation.py',
        'src/data/__init__.py',
        'src/data/ingest.py',
        'src/data/preprocess.py',
        'src/features/__init__.py',
        'src/features/engineering.py',
        'src/features/build_features.py',
        'src/segmentation/__init__.py',
        'src/segmentation/train_segments.py',
        'src/segmentation/assign_segments.py',
        'src/churn/__init__.py',
        'src/churn/train.py',
        'src/churn/evaluate.py',
        'tests/test_data_preprocess.py',
        'tests/test_features_engineering.py',
        'tests/test_segmentation_train.py',
        'tests/test_churn_evaluate.py',
        'run_pipeline.py',
        'README.md',
        'DEVELOPMENT.md',
    ]

    required_dirs = [
        'src',
        'src/utils',
        'src/data',
        'src/features',
        'src/segmentation',
        'src/churn',
        'config',
        'tests',
        'models',
        'models/segmentation',
        'models/churn',
        'data/processed',
    ]

    print("Required files:")
    for filepath in required_files:
        if not check_file_exists(filepath):
            all_pass = False

    print("\nRequired directories:")
    for dirpath in required_dirs:
        if not check_directory_exists(dirpath):
            all_pass = False

    # ──────────────────────────────────────────────────────────────────
    # 2. CONFIGURATION LOADING
    # ──────────────────────────────────────────────────────────────────
    print("\n[2] CONFIGURATION LOADING")
    print("-" * 80)

    try:
        from src.config import load_config
        cfg = load_config()
        print("✓ Config loaded successfully")

        required_keys = ['data', 'preprocessing', 'feature_engineering',
                        'segmentation', 'churn_modeling', 'logging']
        for key in required_keys:
            if key in cfg:
                print(f"  ✓ {key}")
            else:
                print(f"  ✗ {key} (MISSING)")
                all_pass = False
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        all_pass = False

    # ──────────────────────────────────────────────────────────────────
    # 3. MODULE IMPORTS
    # ──────────────────────────────────────────────────────────────────
    print("\n[3] MODULE IMPORTS")
    print("-" * 80)

    modules_to_import = [
        ('src.config', ['load_config', 'get_config']),
        ('src.utils', ['setup_logging', 'load_csv', 'save_csv',
                       'load_model', 'save_model',
                       'validate_feature_consistency']),
        ('src.data', ['load_raw_data', 'preprocess_data']),
        ('src.features', ['engineer_features']),
        ('src.segmentation', ['train_segmentation_model', 'assign_segments']),
        ('src.churn', ['train_churn_model', 'evaluate_model', 'compare_thresholds']),
    ]

    for module_name, expected_exports in modules_to_import:
        try:
            module = __import__(module_name, fromlist=expected_exports)
            print(f"✓ {module_name}")
            for export in expected_exports:
                if hasattr(module, export):
                    print(f"  ✓ {export}")
                else:
                    print(f"  ✗ {export} (NOT EXPORTED)")
                    all_pass = False
        except ImportError as e:
            print(f"✗ {module_name}: {e}")
            all_pass = False

    # ──────────────────────────────────────────────────────────────────
    # 4. FUNCTION SIGNATURES (TYPE HINTS)
    # ──────────────────────────────────────────────────────────────────
    print("\n[4] FUNCTION SIGNATURES (TYPE HINTS)")
    print("-" * 80)

    from src.data import load_raw_data, preprocess_data
    from src.features import engineer_features
    from src.segmentation import train_segmentation_model, assign_segments
    from src.churn import train_churn_model, evaluate_model

    functions_to_check = [
        ('load_raw_data', load_raw_data),
        ('preprocess_data', preprocess_data),
        ('engineer_features', engineer_features),
        ('train_segmentation_model', train_segmentation_model),
        ('assign_segments', assign_segments),
        ('train_churn_model', train_churn_model),
        ('evaluate_model', evaluate_model),
    ]

    for func_name, func in functions_to_check:
        has_annotations = bool(func.__annotations__)
        has_docstring = bool(func.__doc__)
        status_ann = "✓" if has_annotations else "✗"
        status_doc = "✓" if has_docstring else "✗"
        print(f"{status_ann} {func_name} - annotations: {status_ann}, docstring: {status_doc}")
        if not has_annotations or not has_docstring:
            all_pass = False

    # ──────────────────────────────────────────────────────────────────
    # 5. LOGGING SETUP
    # ──────────────────────────────────────────────────────────────────
    print("\n[5] LOGGING SETUP")
    print("-" * 80)

    try:
        from src.utils import setup_logging
        from src.config import get_config
        cfg = get_config()
        setup_logging(cfg['logging'])
        print("✓ Logging configured successfully")

        import logging
        logger = logging.getLogger(__name__)
        logger.info("✓ Test log message (INFO level)")
    except Exception as e:
        print(f"✗ Logging setup failed: {e}")
        all_pass = False

    # ──────────────────────────────────────────────────────────────────
    # 6. I/O UTILITIES
    # ──────────────────────────────────────────────────────────────────
    print("\n[6] I/O UTILITIES")
    print("-" * 80)

    try:
        import tempfile
        import os
        from src.utils import save_csv, load_csv, save_json, load_json, save_model, load_model
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test CSV I/O
            test_csv = os.path.join(tmpdir, 'test.csv')
            df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            save_csv(df, test_csv)
            df_loaded = load_csv(test_csv)
            assert df.equals(df_loaded), "CSV mismatch"
            print("✓ CSV I/O works")

            # Test JSON I/O
            test_json = os.path.join(tmpdir, 'test.json')
            data = {'key': 'value', 'number': 42}
            save_json(data, test_json)
            data_loaded = load_json(test_json)
            assert data == data_loaded, "JSON mismatch"
            print("✓ JSON I/O works")

            # Test model save/load
            import joblib
            test_model = os.path.join(tmpdir, 'test_model.pkl')
            model_obj = {'dummy': 'model'}
            save_model(model_obj, test_model)
            model_loaded = load_model(test_model)
            assert model_obj == model_loaded, "Model mismatch"
            print("✓ Model I/O works")

    except Exception as e:
        print(f"✗ I/O utilities test failed: {e}")
        all_pass = False

    # ──────────────────────────────────────────────────────────────────
    # 7. __init__.py EXPORTS
    # ──────────────────────────────────────────────────────────────────
    print("\n[7] __init__.py MODULE EXPORTS")
    print("-" * 80)

    init_files = {
        'src/__init__.py': ['load_config', 'get_config'],
        'src/utils/__init__.py': ['setup_logging', 'load_csv', 'save_csv'],
        'src/data/__init__.py': ['load_raw_data', 'preprocess_data'],
        'src/features/__init__.py': ['engineer_features'],
        'src/segmentation/__init__.py': ['train_segmentation_model', 'assign_segments'],
        'src/churn/__init__.py': ['train_churn_model', 'evaluate_model'],
    }

    for init_file, expected_exports in init_files.items():
        fp = Path(init_file)
        content = fp.read_text()
        all_exported = all(exp in content for exp in expected_exports)
        status = "✓" if all_exported else "✗"
        print(f"{status} {init_file}")
        if not all_exported:
            missing = [exp for exp in expected_exports if exp not in content]
            for exp in missing:
                print(f"  ✗ {exp} not in __all__")
            all_pass = False

    # ──────────────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    if all_pass:
        print("✓ ALL CHECKS PASSED")
        print("=" * 80)
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
