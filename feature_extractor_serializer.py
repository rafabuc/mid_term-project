"""
STEP 3: FEATURE EXTRACTION - COMPLETE EXECUTION & SAVE
=======================================================


Save all outputs needed for Step 4 (Model Training)

Outputs:
- data/features/X_train.npy
- data/features/y_train.npy
- data/features/X_test.npy
- data/features/y_test.npy
- data/features/feature_names.json
- data/features/class_names.json
- models/feature_extractor/ (fitted extractor)


"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from feature_extractor import FeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_features_and_metadata(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    extractor: FeatureExtractor,
    output_dir: str ,#= 'data/features',
    model_dir: str #= 'models/feature_extractor'
):
    """
    Save all feature extraction outputs.
    
    Args:
        X_train: Training feature matrix
        y_train: Training targets (encoded)
        X_test: Test feature matrix
        y_test: Test targets (encoded)
        feature_names: List of feature names
        extractor: Fitted FeatureExtractor instance
        output_dir: Directory to save features
        model_dir: Directory to save feature extractor
    """
    logger.info("=" * 80)
    logger.info("SAVING ALL OUTPUTS")
    logger.info("=" * 80)
    
    # Create directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # 1. SAVE NUMPY ARRAYS (Features and Targets)
    # ========================================================================
    logger.info("\n1. Saving numpy arrays...")
    
    np.save(output_path / 'X_train.npy', X_train)
    logger.info(f"   ✓ X_train saved: {X_train.shape}")
    
    np.save(output_path / 'y_train.npy', y_train)
    logger.info(f"   ✓ y_train saved: {y_train.shape}")
    
    np.save(output_path / 'X_test.npy', X_test)
    logger.info(f"   ✓ X_test saved: {X_test.shape}")
    
    np.save(output_path / 'y_test.npy', y_test)
    logger.info(f"   ✓ y_test saved: {y_test.shape}")
    
    # ========================================================================
    # 2. SAVE FEATURE NAMES
    # ========================================================================
    logger.info("\n2. Saving feature names...")
    
    feature_names_dict = {
        'total_features': len(feature_names),
        'feature_names': feature_names,
        'feature_breakdown': {
            'text': len(extractor.feature_names['text']),
            'numerical': len(extractor.feature_names['numerical']),
            'categorical': len(extractor.feature_names['categorical']),
            'engineered': len(extractor.feature_names['engineered'])
        },
        'numerical_features': extractor.feature_names['numerical'],
        'categorical_features': extractor.feature_names['categorical'],
        'engineered_features': extractor.feature_names['engineered']
    }
    
    with open(output_path / 'feature_names.json', 'w') as f:
        json.dump(feature_names_dict, f, indent=2)
    logger.info(f"   ✓ feature_names.json saved: {len(feature_names)} features")
    
    # ========================================================================
    # 3. SAVE CLASS NAMES AND MAPPING
    # ========================================================================
    logger.info("\n3. Saving class names and mapping...")
    
    class_names = extractor.target_label_encoder.classes_.tolist()
    
    class_info = {
        'n_classes': len(class_names),
        'class_names': class_names,
        'class_to_idx': {name: idx for idx, name in enumerate(class_names)},
        'idx_to_class': {idx: name for idx, name in enumerate(class_names)},
        'train_distribution': {
            class_names[i]: int((y_train == i).sum())
            for i in range(len(class_names))
        },
        'test_distribution': {
            class_names[i]: int((y_test == i).sum())
            for i in range(len(class_names))
        }
    }
    
    with open(output_path / 'class_names.json', 'w') as f:
        json.dump(class_info, f, indent=2)
    logger.info(f"   ✓ class_names.json saved: {len(class_names)} classes")
    
    # ========================================================================
    # 4. SAVE FEATURE EXTRACTOR
    # ========================================================================
    logger.info("\n4. Saving FeatureExtractor...")
    extractor.save(model_dir)
    logger.info(f"   ✓ FeatureExtractor saved to: {model_dir}")
    
    # ========================================================================
    # 5. SAVE SUMMARY STATISTICS
    # ========================================================================
    logger.info("\n5. Saving summary statistics...")
    
    summary = {
        'dataset': {
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'total_samples': int(len(X_train) + len(X_test)),
            'test_split_ratio': len(X_test) / (len(X_train) + len(X_test))
        },
        'features': {
            'total_features': int(X_train.shape[1]),
            'text_features': len(extractor.feature_names['text']),
            'numerical_features': len(extractor.feature_names['numerical']),
            'categorical_features': len(extractor.feature_names['categorical']),
            'engineered_features': len(extractor.feature_names['engineered'])
        },
        'targets': {
            'n_classes': len(class_names),
            'class_names': class_names,
            'encoding': 'LabelEncoder (0 to n_classes-1)'
        },
        'files': {
            'X_train': str(output_path / 'X_train.npy'),
            'y_train': str(output_path / 'y_train.npy'),
            'X_test': str(output_path / 'X_test.npy'),
            'y_test': str(output_path / 'y_test.npy'),
            'feature_names': str(output_path / 'feature_names.json'),
            'class_names': str(output_path / 'class_names.json'),
            'feature_extractor': str(model_dir)
        }
    }
    
    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"   ✓ summary.json saved")
    
    return summary

