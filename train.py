"""
Book Classification Pipeline 

"""

# ========================================
# ALL IMPORTS
# ========================================
import importlib
import pandas as pd
import json
from pathlib import Path
import numpy as np
# Custom modules
import data_loader
import data_preprocessor
import data_splitter
import feature_extractor
import model_trainer
import constants_v2
import feature_extractor_serializer
import book_classifier

# Reload modules
importlib.reload(data_loader)
importlib.reload(data_preprocessor)
importlib.reload(constants_v2)
importlib.reload(data_splitter)
importlib.reload(feature_extractor)
importlib.reload(model_trainer)
importlib.reload(constants_v2)
importlib.reload(feature_extractor_serializer)
importlib.reload(book_classifier)

# Import classes
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from data_splitter import DataSplitter
from feature_extractor import FeatureExtractor
from model_trainer import XGBoostTrainer
from book_classifier import example_single_prediction, BookClassifier, example_batch_prediction

# Import constants
from constants import PATH_DATASETS, TARGET_COLUMN_CATEGORY_LEVEL2
from constants_v2 import (
    PATH_DATASETS_V2,
    PATH_ARTIFACTS_V2,
    SAMPLE_METADATA_FILE_V2,
    CLASS_WEIGHTS_FILE_V2,
    TRAIN_FILE_V2,
    TEST_FILE_V2,
    INFO_FILE_V2,
    #FEATURE_EXTRACTOR_FILE_V2,
    REPORT_XGBOOST_CLASSIFICATION_REPORT_FILE_V2,
    REPORT_XGBOOST_FEATURE_IMPORTANCE_FILE_V2,
    REPORT_XGBOOST_CONFUSION_MATRIX_FILE_V2,
    X_TRAIN_FILE_V2, Y_TRAIN_FILE_V2, X_TEST_FILE_V2, Y_TEST_FILE_V2,
     CLASS_NAMES_FILE_V2, CLASS_WEIGHTS_FILE_V2,FEATURE_NAMES_FILE_V2,
     REPORT_XGBOOST_MODEL_FILE_V2, REPORT_XGBOOST_METRICS_FILE_V2

)

from feature_extractor_serializer import save_features_and_metadata
from constants_v2 import FEATURE_EXTRACTOR_FEATURES_OUTPUT_DIR_V2, MODEL_OUTPUT_DIR_V2



# ========================================
# DIRECTORY SETUP FUNCTION
# ========================================
def create_directory_structure():
    """
    Creates all necessary directories for the book classification pipeline.
    """
    directories = [
        './artifacts_v2/',
        './artifacts_v2/dataset/',
        './artifacts_v2/datasets/',
        './artifacts_v2/feature_extractor/',
        './artifacts_v2/feature_extractor/features/',
        './artifacts_v2/models/',
        './artifacts_v2/reports/',
    ]
    
    print("\n" + "="*70)
    print("CREATING DIRECTORY STRUCTURE")
    print("="*70)
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created: {directory}")
        else:
            print(f"✓ Already exists: {directory}")
    
    print("="*70 + "\n")
    
def main():
    """
    Main function that executes the complete book classification pipeline.
    
    """
    # ========================================
    # SETUP: CREATE DIRECTORY STRUCTURE
    # ========================================
    create_directory_structure()
    
    # ========================================
    #  LOAD DATA
    # ========================================
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)    
        
    loader = DataLoader(
            metadata_path=f'{PATH_DATASETS}amazon_books_metadata_sample_20k.csv',
            reviews_path=f'{PATH_DATASETS}amazon_books_reviews_sample_20k.csv'
        )

    # Load both datasets
    metadata_df = loader.load_metadata()#nrows=1000
    reviews_df = loader.load_reviews()#nrows=5000    

    # ========================================
    #  VALIDATE SCHEMA
    # ========================================
    #metadata_df.columns
    # Validate both
    validation = loader.validate_schema('both')

    # ========================================
    #  DATA SUMMARY
    # ========================================
    #loader.get_data_summary()
    #loader.get_reviews_summary()
    loader.print_summary()   

    # ========================================
    #  JOIN AND RATING DISTRIBUTION
    # ========================================
        
    # Join with minimum 5 reviews per book
    joined_df = loader.join_metadata_reviews(
            how='inner',
            min_reviews_per_book=5
    )
    print(f'\n{"="*60}')
    print(f"\nJoined data shape:\n {joined_df.shape}")
        
    # Get rating distribution by category
    rating_by_category = loader.get_rating_distribution_by_category(TARGET_COLUMN_CATEGORY_LEVEL2)
    print(f'\n{"="*60}')
    print(f"\nRating distribution by {TARGET_COLUMN_CATEGORY_LEVEL2} category:\n")
    print(rating_by_category)

    loader.export_summary_report(f'{PATH_DATASETS_V2}complete_data_report.txt')

    print(f"\nLoaded:")
    print(f"  Metadata: {len(metadata_df):,} books")
    print(f"  Reviews: {len(reviews_df):,} reviews")    

    # ========================================
    #  INITIALIZE PREPROCESSOR
    # ========================================
    print("\n" + "="*70)
    print("STEP 2: INITIALIZING PREPROCESSOR")
    print("="*70)

    #decrease categories choosen 
    #min_samples_per_category 
    # Step 2: Preprocess
    preprocessor = DataPreprocessor(
            min_samples_per_category=100,    # Minimum 100 books per category
            sampling_percentage=0.5,         # Use 20% of data (for faster training)
            balance_strategy='class_weight', # Calculate weights for imbalanced classes
            min_reviews_per_book=5          # Minimum 5 reviews per book
    )

    # ========================================
    # : CLEAN METADATA
    # ========================================
    print("\n" + "="*70)
    print("STEP 2.1: CLEANING METADATA")
    print("="*70)
        
    # Clean both datasets
    clean_metadata = preprocessor.clean_metadata(metadata_df)

    # ========================================
    #  CLEAN REVIEWS
    # ========================================
    print("\n" + "="*70)
    print("STEP 2.2: CLEANING REVIEWS")
    print("="*70)
        
    clean_reviews = preprocessor.clean_reviews(reviews_df)

    # ========================================
    #  ENRICH METADATA
    # ========================================
    print("\n" + "="*70)
    print("STEP 2.3: ENRICHING METADATA WITH REVIEWS")
    print("="*70)
        
    enriched_metadata = preprocessor.enrich_with_reviews(
            clean_metadata, 
            clean_reviews
        )

    # ========================================
    #  FILTER RARE CATEGORIES
    # ========================================    
    # Filter and sample
    filtered_metadata  = preprocessor.filter_rare_categories(enriched_metadata,
            category_column=TARGET_COLUMN_CATEGORY_LEVEL2)

    print(f"\nFiltered metadata:")
    print(f"  Books: {len(filtered_metadata):,}")
    print(f"  Categories: {filtered_metadata[TARGET_COLUMN_CATEGORY_LEVEL2].nunique()}")

    # ========================================
    #  FILTER BY REVIEW COUNT
    # ========================================
    print("\n" + "="*70)
    print("STEP 2.5: FILTERING BY REVIEW COUNT")
    print("="*70)
        
    filtered_metadata = preprocessor.filter_by_review_count(filtered_metadata)

    # ========================================
    #  STRATIFIED SAMPLING
    # ========================================
    print("\n" + "="*70)
    print("STEP 2.6: STRATIFIED SAMPLING")
    print("="*70)

    sampled_metadata  = preprocessor.sample_data(filtered_metadata,
            category_column=TARGET_COLUMN_CATEGORY_LEVEL2,
            random_state=42)

    print(f"\nSampled metadata:")
    print(f"  Books: {len(sampled_metadata):,}")
    print(f"  Categories: {sampled_metadata[TARGET_COLUMN_CATEGORY_LEVEL2].nunique()}")

    # ========================================
    #  CALCULATE CLASS WEIGHTS
    # ========================================
        
    # Calculate class weights
    class_weights = preprocessor.calculate_class_weights(sampled_metadata, 
                                                          category_column=TARGET_COLUMN_CATEGORY_LEVEL2)

    # ========================================
    #  SAVE PREPROCESSED DATA
    # ========================================
    preprocessor.save_preprocessed_data(sampled_metadata,
            output_path=SAMPLE_METADATA_FILE_V2,
            save_class_weights=True,
            class_weights=class_weights,
            weights_path=CLASS_WEIGHTS_FILE_V2)

    # ========================================
    #  PREPROCESSING SUMMARY
    # ========================================    
    # Print summary
    preprocessor.print_preprocessing_summary()
        
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
        
    print(f"\nFinal dataset statistics:")
    print(f"  Total books: {len(sampled_metadata):,}")
    print(f"  Total categories: {sampled_metadata[TARGET_COLUMN_CATEGORY_LEVEL2].nunique()}")
    print(f"  Total columns: {len(sampled_metadata.columns)}")
    print(f"  Avg reviews per book: {sampled_metadata['review_count'].mean():.1f}")
    print(f"  Avg rating: {sampled_metadata['avg_rating_from_reviews'].mean():.2f}")
        
    print("\nTop 5 categories by count:")
    top_categories = sampled_metadata[TARGET_COLUMN_CATEGORY_LEVEL2].value_counts().head(5)
    for cat, count in top_categories.items():
            pct = (count / len(sampled_metadata)) * 100
            print(f"  {cat}: {count:,} books ({pct:.1f}%)")

    # ========================================
    #  DATA SPLITTER INITIALIZATION
    # ========================================
    print("\n" + "=" * 80)
    print("DATA SPLITTER ")
    print("=" * 80)
        
    # Inicializar splitter
    splitter = DataSplitter(
            test_size=0.2,           # 20% para test
            random_state=42,         # Reproducibilidad
            target_column=TARGET_COLUMN_CATEGORY_LEVEL2,
            train_file_name=TRAIN_FILE_V2,
            test_file_name=TEST_FILE_V2,
            info_file_name=INFO_FILE_V2
    )

    # ========================================
    #  SPLIT DATA
    # ========================================
    # Run the complete pipeline
    results= splitter.run_pipeline(
            sampled_metadata_path=SAMPLE_METADATA_FILE_V2,
            output_dir=PATH_ARTIFACTS_V2
        )

    print("\n✅ Data splitting completed!")
    print(f"\nAccess files in: {PATH_ARTIFACTS_V2}/")
    print("  - train_metadata.csv")
    print("  - test_metadata.csv")
    print("  - split_info.json")
    
    
    # ========================================
    #  LOAD SPLIT DATA
    # ========================================
    df_train = pd.read_csv(f'{PATH_ARTIFACTS_V2}{TRAIN_FILE_V2}')
    df_test = pd.read_csv(f'{PATH_ARTIFACTS_V2}{TEST_FILE_V2}')

    print(f"\nTrain columns: {df_train.columns}")
    print(f"Test columns: {df_test.columns}")

    # ========================================
    #  FEATURE EXTRACTOR INITIALIZATION
    # ========================================
    # Apply FeatureExtractor
    extractor = FeatureExtractor(
            max_tfidf_features=8000,
            tfidf_ngram_range=(1, 3)
        )

    # ========================================
    #  FIT FEATURE EXTRACTOR
    # ========================================
    #. Fit  train (fits TF-IDF, Scaler, LabelEncoders and Target Encoder)
    # Select features numerical, categorical and engineered. also vectorized features
    # from title and description columns
    extractor.fit(df_train, target_column=TARGET_COLUMN_CATEGORY_LEVEL2)

    # ========================================
    # CELL 62: TRANSFORM TRAIN AND TEST
    # ========================================
    print("FeatureExtractor - Book Classification Pipeline")
    
    # Transform train and test (returns X and y)
    X_train, y_train, features_train, feature_names = extractor.transform(df_train, target_column=TARGET_COLUMN_CATEGORY_LEVEL2)
    X_test, y_test, features_test, _ = extractor.transform(df_test, target_column=TARGET_COLUMN_CATEGORY_LEVEL2)    

    # ========================================
    # SAVE FEATURE EXTRACTOR
    # ========================================
    #extractor.save(FEATURE_EXTRACTOR_FILE_V2)
    
    
    print(f"\nExtractor stats: \n")
    print("="*80)
    print(f'{extractor.get_stats()}')
    #Save for later use

    save_features_and_metadata(
        X_train, y_train,
        X_test, y_test,
        feature_names,
        extractor,
        output_dir=FEATURE_EXTRACTOR_FEATURES_OUTPUT_DIR_V2,
        model_dir=MODEL_OUTPUT_DIR_V2)




    X_train = np.load(X_TRAIN_FILE_V2)
    y_train = np.load(Y_TRAIN_FILE_V2)
    X_test = np.load(X_TEST_FILE_V2)
    y_test = np.load(Y_TEST_FILE_V2)

    # Load class names
    with open(CLASS_NAMES_FILE_V2, 'r') as f:
        class_info = json.load(f)
        class_names = class_info['class_names']

    # Load feature names (optional, for visualizations)
    with open(FEATURE_NAMES_FILE_V2, 'r') as f:
        feature_info = json.load(f)
        feature_names = feature_info['feature_names']

    # Load class weights (if available from Step 2)
    try:
        with open(CLASS_WEIGHTS_FILE_V2, 'r') as f:
            class_weights = json.load(f)
    except FileNotFoundError:
        class_weights = None
        print("No class_weights.json found, will use default balancing")


    print(f"Data loaded:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: {len(class_names)}")
        
    # ========================================
    # : MODEL TRAINER INITIALIZATION
    # ========================================
    # Load class weights
    #with open(CLASS_WEIGHTS_FILE_V2, 'r') as f:
    #    class_weights = json.load(f)

    # Initialize trainer
    xgb_trainer = XGBoostTrainer(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_weights=class_weights,
        output_dir='models',
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=10
    )

    # ========================================
    #  TRAIN MODEL
    # ========================================
    # Train
    xgb_trainer.train()

    # ========================================
    #  EVALUATE ON TEST
    # ========================================
    # Evaluate on test set
    xgb_test_metrics = xgb_trainer.evaluate(X_test, y_test, 'test')

    # ========================================
    #  CROSS-VALIDATION
    # ========================================
    # Cross-validation
    # Note: This task may take several minutes,  the screen may appear frozen with the message "Performing 2-fold cross-validation...", but the process is running.
    # Please wait for it to complete.
    xgb_cv_results = xgb_trainer.cross_validate(cv=2, scoring='f1_weighted')

    # ========================================
    #  GENERATE REPORTS
    # ========================================
    # Generate all reports
    # Visualizations
    xgb_trainer.plot_confusion_matrix(
        y_test,
        xgb_trainer.model.predict(X_test),
        normalize=True,
        save_path=REPORT_XGBOOST_CONFUSION_MATRIX_FILE_V2
    )
    
    xgb_trainer.plot_feature_importance(
        top_n=20,
        feature_names=feature_names,
        save_path=REPORT_XGBOOST_FEATURE_IMPORTANCE_FILE_V2
    )
    
    xgb_trainer.save_classification_report(
    y_test,
    xgb_trainer.model.predict(X_test),
    save_path=REPORT_XGBOOST_CLASSIFICATION_REPORT_FILE_V2
    )
    # Save model
    xgb_trainer.save_model(REPORT_XGBOOST_MODEL_FILE_V2)

    # Save metrics
    with open(REPORT_XGBOOST_METRICS_FILE_V2, 'w') as f:
        json.dump({
            'test_metrics': xgb_test_metrics,
            'cv_results': xgb_cv_results
        }, f, indent=2) 

    example_single_prediction(REPORT_XGBOOST_MODEL_FILE_V2, MODEL_OUTPUT_DIR_V2)

    example_batch_prediction(REPORT_XGBOOST_MODEL_FILE_V2, MODEL_OUTPUT_DIR_V2)
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)



if __name__ == "__main__":
    main()