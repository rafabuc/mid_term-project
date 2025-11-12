"""
STEP 4: MODEL TRAINING
======================
Book Classification Pipeline - Model Training Module

This module provides three classes for training and evaluating classification models:
1. BaseModelTrainer: Abstract base class with common methods
2. XGBoostTrainer: XGBoost classifier with class weight support
3. LogisticRegressionTrainer: Logistic Regression with balanced weights


"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import joblib
from datetime import datetime

# ML Libraries
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseModelTrainer(ABC):
    """
    Abstract base class for model training.
    
    Provides common functionality for model evaluation, visualization,
    and saving/loading models.
    """
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[List[str]] = None,
        class_weights: Optional[Dict[int, float]] = None,
        output_dir: str = 'models'
    ):
        """
        Initialize base trainer.
        
        Args:
            X_train: Training features (N_train, n_features)
            y_train: Training labels (N_train,)
            X_test: Test features (N_test, n_features)
            y_test: Test labels (N_test,)
            class_names: List of class names for visualization
            class_weights: Dictionary mapping class indices to weights
            output_dir: Directory to save models and reports
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names or [f"Class_{i}" for i in range(len(np.unique(y_train)))]
        self.class_weights = class_weights
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.model_name = self.__class__.__name__.replace('Trainer', '')
        
        logger.info(f"Initialized {self.model_name} Trainer")
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"Number of features: {X_train.shape[1]}")
        logger.info(f"Number of classes: {len(self.class_names)}")
        
    @abstractmethod
    def train(self):
        """Train the model. Must be implemented by subclasses."""
        pass
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str = 'test'
    ) -> Dict:
        """
        Evaluate model on given dataset.
        
        Args:
            X: Features to evaluate
            y: True labels
            dataset_name: Name of dataset for logging
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        logger.info(f"Evaluating {self.model_name} on {dataset_name} set...")
        
        # Predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision_weighted': precision_score(y, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0)
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y, y_pred, average=None, zero_division=0)
        
        metrics['per_class'] = {}
        for i, class_name in enumerate(self.class_names):
            metrics['per_class'][class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i])
            }
        
        # Log summary
        logger.info(f"{dataset_name.upper()} METRICS:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
        logger.info(f"  Recall (weighted): {metrics['recall_weighted']:.4f}")
        logger.info(f"  F1-Score (weighted): {metrics['f1_weighted']:.4f}")
        
        return metrics
    
    def cross_validate(
        self,
        cv: int = 5,
        scoring: str = 'f1_weighted'
    ) -> Dict:
        """
        Perform stratified k-fold cross-validation.
        
        Args:
            cv: Number of folds
            scoring: Scoring metric to use
            
        Returns:
            Dictionary with CV scores
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        cv_scores = cross_val_score(
            self.model,
            self.X_train,
            self.y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=-1
        )
        
        cv_results = {
            'scores': cv_scores.tolist(),
            'mean': float(cv_scores.mean()),
            'std': float(cv_scores.std()),
            'min': float(cv_scores.min()),
            'max': float(cv_scores.max())
        }
        
        logger.info(f"CV {scoring}: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")
        
        return cv_results
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize the confusion matrix
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = f'{self.model_name} - Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = f'{self.model_name} - Confusion Matrix'
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Generate and save classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the report
        """
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            digits=4
        )
        
        logger.info(f"\n{self.model_name} Classification Report:\n{report}")
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(f"{self.model_name} Classification Report\n")
                f.write("=" * 80 + "\n\n")
                f.write(report)
            logger.info(f"Classification report saved to {save_path}")
    
    def save_model(self, filepath: Optional[str] = None):
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model. If None, uses default naming
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.output_dir / f"{self.model_name.lower()}_{timestamp}.pkl"
        
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'n_features': self.X_train.shape[1],
            'n_classes': len(self.class_names),
            'class_names': self.class_names,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = str(filepath).replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Model metadata saved to {metadata_path}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to saved model
        """
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")


class XGBoostTrainer(BaseModelTrainer):
    """
    XGBoost classifier trainer with class weight support.
    """
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[List[str]] = None,
        class_weights: Optional[Dict[int, float]] = None,
        output_dir: str = 'models',
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 10
    ):
        """
        Initialize XGBoost trainer.
        
        Args:
            X_train, y_train, X_test, y_test: Train/test data
            class_names: List of class names
            class_weights: Dictionary of class weights
            output_dir: Output directory
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio
            colsample_bytree: Column subsample ratio
            early_stopping_rounds: Early stopping rounds
        """
        super().__init__(X_train, y_train, X_test, y_test, class_names, class_weights, output_dir)
        
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'early_stopping_rounds': early_stopping_rounds
        }
        
        self.feature_importance_ = None
    
    def train(self):
        """Train XGBoost model with class weights."""
        logger.info(f"Training {self.model_name} with parameters: {self.params}")
        
        # Calculate sample weights if class_weights provided
        sample_weights = None
        if self.class_weights:
            try:
                # Verify that class_weights has all classes
                unique_classes = np.unique(self.y_train)
                
                # Convert class_weights keys to int if they're strings
                if isinstance(list(self.class_weights.keys())[0], str):
                    self.class_weights = {int(k): v for k, v in self.class_weights.items()}
                
                # Check if all classes are present
                missing_classes = set(unique_classes) - set(self.class_weights.keys())
                
                if missing_classes:
                    logger.warning(f"Missing classes in class_weights: {missing_classes}")
                    logger.info("Computing class weights automatically using 'balanced' strategy")
                    # Compute class weights automatically
                    from sklearn.utils.class_weight import compute_class_weight
                    class_weights_array = compute_class_weight(
                        'balanced',
                        classes=unique_classes,
                        y=self.y_train
                    )
                    self.class_weights = dict(zip(unique_classes, class_weights_array))
                    logger.info(f"Computed class weights: {self.class_weights}")
                
                # Now compute sample weights
                sample_weights = compute_sample_weight(
                    class_weight=self.class_weights,
                    y=self.y_train
                )
                logger.info("Using sample weights for class imbalance")
                
            except Exception as e:
                logger.warning(f"Error computing sample weights: {e}")
                logger.info("Falling back to no sample weights")
                sample_weights = None
        
        # Initialize model
        # Note: In newer XGBoost versions, early_stopping_rounds is passed to fit(), not the constructor
        self.model = XGBClassifier(
            n_estimators=self.params['n_estimators'],
            max_depth=self.params['max_depth'],
            learning_rate=self.params['learning_rate'],
            subsample=self.params['subsample'],
            colsample_bytree=self.params['colsample_bytree'],
            objective='multi:softmax',
            num_class=len(self.class_names),
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
            verbosity=1
        )
        
        # Train with optional early stopping
        fit_params = {
            'sample_weight': sample_weights,
            'eval_set': [(self.X_test, self.y_test)],
            'verbose': True
        }
        
        # Add early_stopping_rounds if configured
        use_early_stopping = False
        if self.params.get('early_stopping_rounds') and self.params['early_stopping_rounds'] > 0:
            fit_params['early_stopping_rounds'] = self.params['early_stopping_rounds']
            use_early_stopping = True
            logger.info(f"Attempting to use early stopping with {self.params['early_stopping_rounds']} rounds")
        
        # Train - with fallback if early_stopping_rounds not supported
        try:
            self.model.fit(self.X_train, self.y_train, **fit_params)
        except TypeError as e:
            if 'early_stopping_rounds' in str(e) and use_early_stopping:
                # This XGBoost version doesn't support early_stopping_rounds
                logger.warning(f"XGBoost version doesn't support early_stopping_rounds in fit(): {e}")
                logger.info("Training without early stopping...")
                fit_params.pop('early_stopping_rounds', None)
                self.model.fit(self.X_train, self.y_train, **fit_params)
            else:
                raise
        
        # Store feature importance
        self.feature_importance_ = self.model.feature_importances_
        
        logger.info(f"{self.model_name} training completed")
        
        # Log best iteration info if early stopping was used
        try:
            if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
                logger.info(f"Best iteration: {self.model.best_iteration}")
            if hasattr(self.model, 'best_score') and self.model.best_score is not None:
                logger.info(f"Best score: {self.model.best_score:.4f}")
        except AttributeError:
            logger.info("Early stopping was not triggered (model trained for all n_estimators rounds)")
    
    def plot_feature_importance(
        self,
        top_n: int = 20,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot top N most important features.
        
        Args:
            top_n: Number of top features to plot
            feature_names: List of feature names
            save_path: Path to save the plot
        """
        if self.feature_importance_ is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Get top features
        indices = np.argsort(self.feature_importance_)[-top_n:]
        
        if feature_names:
            names = [feature_names[i] for i in indices]
        else:
            names = [f"Feature {i}" for i in indices]
        
        importances = self.feature_importance_[indices]
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances, align='center')
        plt.yticks(range(top_n), names)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Most Important Features - XGBoost', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Log top features
        logger.info(f"\nTop {top_n} Features:")
        for i, (name, importance) in enumerate(zip(names, importances), 1):
            logger.info(f"  {i}. {name}: {importance:.4f}")


class LogisticRegressionTrainer(BaseModelTrainer):
    """
    Logistic Regression classifier trainer with balanced class weights.
    """
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[List[str]] = None,
        class_weights: Optional[Dict[int, float]] = None,
        output_dir: str = 'models',
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = 'lbfgs'
    ):
        """
        Initialize Logistic Regression trainer.
        
        Args:
            X_train, y_train, X_test, y_test: Train/test data
            class_names: List of class names
            class_weights: Dictionary of class weights (or 'balanced')
            output_dir: Output directory
            C: Inverse of regularization strength
            max_iter: Maximum iterations
            solver: Optimization algorithm
        """
        super().__init__(X_train, y_train, X_test, y_test, class_names, class_weights, output_dir)
        
        self.params = {
            'C': C,
            'max_iter': max_iter,
            'solver': solver
        }
        
        self.coefficients_ = None
    
    def train(self):
        """Train Logistic Regression model with balanced class weights."""
        logger.info(f"Training {self.model_name} with parameters: {self.params}")
        
        # Determine class_weight parameter
        class_weight = 'balanced'  # Default
        
        if self.class_weights:
            try:
                # Verify that class_weights has all classes
                unique_classes = np.unique(self.y_train)
                
                # Convert class_weights keys to int if they're strings
                if isinstance(list(self.class_weights.keys())[0], str):
                    self.class_weights = {int(k): v for k, v in self.class_weights.items()}
                
                # Check if all classes are present
                missing_classes = set(unique_classes) - set(self.class_weights.keys())
                
                if missing_classes:
                    logger.warning(f"Missing classes in class_weights: {missing_classes}")
                    logger.info("Using automatic 'balanced' class weights instead")
                    class_weight = 'balanced'
                else:
                    class_weight = self.class_weights
                    logger.info("Using provided class weights")
                    
            except Exception as e:
                logger.warning(f"Error processing class weights: {e}")
                logger.info("Falling back to automatic 'balanced' class weights")
                class_weight = 'balanced'
        else:
            logger.info("Using automatic 'balanced' class weights")
        
        # Initialize model
        self.model = LogisticRegression(
            C=self.params['C'],
            max_iter=self.params['max_iter'],
            solver=self.params['solver'],
            multi_class='multinomial',
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        # Train
        self.model.fit(self.X_train, self.y_train)
        
        # Store coefficients
        self.coefficients_ = self.model.coef_
        
        logger.info(f"{self.model_name} training completed")
        logger.info(f"Number of iterations: {self.model.n_iter_}")
    
    def plot_coefficient_importance(
        self,
        class_idx: int = 0,
        top_n: int = 20,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot top N coefficients for a specific class.
        
        Args:
            class_idx: Index of class to analyze
            top_n: Number of top coefficients to plot
            feature_names: List of feature names
            save_path: Path to save the plot
        """
        if self.coefficients_ is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        
        # Get coefficients for specified class
        coef = self.coefficients_[class_idx]
        
        # Get top positive and negative coefficients
        top_positive_idx = np.argsort(coef)[-top_n:]
        top_negative_idx = np.argsort(coef)[:top_n]
        
        indices = np.concatenate([top_negative_idx, top_positive_idx])
        
        if feature_names:
            names = [feature_names[i] for i in indices]
        else:
            names = [f"Feature {i}" for i in indices]
        
        coefficients = coef[indices]
        
        # Plot
        colors = ['red' if c < 0 else 'green' for c in coefficients]
        
        plt.figure(figsize=(10, 12))
        plt.barh(range(len(coefficients)), coefficients, align='center', color=colors, alpha=0.7)
        plt.yticks(range(len(coefficients)), names, fontsize=8)
        plt.xlabel('Coefficient Value', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(
            f'Top {top_n} Positive/Negative Coefficients - {self.class_names[class_idx]}',
            fontsize=14,
            fontweight='bold'
        )
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Coefficient plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


