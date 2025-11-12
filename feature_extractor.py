"""
STEP 3: FEATURE EXTRACTOR
=========================

Extracts advanced features from preprocessed data to feed
classification models (XGBoost and Logistic Regression).

Generated features:
- Text Features: TF-IDF of title + description
- Numerical Features: price, average_rating, review_count, etc.
- Categorical Features: format, binned_rating, price_category
- Engineered Features: title_length, desc_length, has_series, etc.

"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Pipeline Step 3: Feature extraction and transformation.
    
    This class takes preprocessed data and generates optimized features
    for book category classification models.
    
    Attributes:
        text_vectorizer (TfidfVectorizer): TF-IDF vectorizer for text
        scaler (StandardScaler): Scaler for numerical features
        label_encoders (Dict[str, LabelEncoder]): Encoders for categorical features
        feature_names (Dict[str, List[str]]): Feature names by type
        config (Dict): Feature extraction configuration
    """
    
    def __init__(
        self,
        max_tfidf_features: int, # = 5000,
        tfidf_ngram_range: Tuple[int, int], # = (1, 2),
        tfidf_min_df: int = 2,
        tfidf_max_df: float = 0.95
    ):
        """
        Initialize the Feature Extractor.
        
        Args:
            max_tfidf_features: Maximum number of TF-IDF features
            tfidf_ngram_range: N-gram range for TF-IDF
            tfidf_min_df: Minimum document frequency for TF-IDF
            tfidf_max_df: Maximum document frequency for TF-IDF
        """
        self.config = {
            'max_tfidf_features': max_tfidf_features,
            'tfidf_ngram_range': tfidf_ngram_range,
            'tfidf_min_df': tfidf_min_df,
            'tfidf_max_df': tfidf_max_df
        }
        
        # Initialize transformers
        # TF-IDF Vectorization
        #convert a collection of raw text documents into a numerical matrix
        #Term Frequency-Inverse Document Frequency. It's a statistical measure used to evaluate how important a word is to a document in a collection or corpus
        #Use Cases:
        #   Document Search & Information Retrieval: To find the most relevant documents for a search query.
        #   Text Classification: Like spam detection, sentiment analysis, or topic labeling.
        #   Document Clustering: Grouping similar documents together.
        #TfidfVectorizer  is a feature extraction technique that transforms text into meaningful numerical features, emphasizing words that are important in a specific
        #document while being relatively rare in the entire collection.
          
        self.text_vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=tfidf_ngram_range,
            min_df=tfidf_min_df,
            max_df=tfidf_max_df,
            stop_words='english',
            strip_accents='unicode',
            lowercase=True
        )
        
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.target_label_encoder = LabelEncoder()  # For target (y)
        self.feature_names: Dict[str, List[str]] = {
            'text': [],
            'numerical': [],
            'categorical': [],
            'engineered': []
        }
        
        self._is_fitted = False
        
        logger.info("FeatureExtractor initialized successfully")
        logger.info(f"Configuration: {self.config}")
    
    def fit(
        self,
        df_metadata: pd.DataFrame,
        target_column: str #= 'main_category'
    ) -> 'FeatureExtractor':
        """
        Fit transformers with training data.
        
        Args:
            df_metadata: DataFrame with preprocessed metadata
            target_column: Name of target column
            
        Returns:
            self: Fitted instance
        """
        logger.info("=" * 80)
        logger.info("FITTING FEATURE EXTRACTOR")
        logger.info("=" * 80)
        
        if df_metadata.empty:
            raise ValueError("DataFrame of metadata is empty")
        
        logger.info(f"Input shape: {df_metadata.shape}")
        
        # Verify target column exists
        if target_column not in df_metadata.columns:
            raise ValueError(
                f"Target column '{target_column}' not found. "
                f"Available columns: {list(df_metadata.columns)}"
            )
        
        # 0. Fit LabelEncoder for target
        logger.info("\n0. Fitting LabelEncoder for target...")
        y_raw = df_metadata[target_column].dropna().values
        self.target_label_encoder.fit(y_raw)
        logger.info(f"   ✓ Target classes: {len(self.target_label_encoder.classes_)}")
        for i, cls in enumerate(self.target_label_encoder.classes_):
            count = (y_raw == cls).sum()
            logger.info(f"      {i}: {cls} ({count} samples)")
        
        # 1. Prepare text data
        logger.info("\n1. Preparing text features...")
        text_data = self._prepare_text_data(df_metadata)
        
        # Fit TF-IDF vectorizer
        self.text_vectorizer.fit(text_data)
        self.feature_names['text'] = [
            f'tfidf_{i}' for i in range(len(self.text_vectorizer.get_feature_names_out()))
        ]
        logger.info(f"   ✓ TF-IDF fitted: {len(self.feature_names['text'])} features")
        
        # 2. Prepare numerical features
        logger.info("\n2. Preparing numerical features...")
        numerical_features = self._extract_numerical_features(df_metadata)
        
        # Fit scaler
        self.scaler.fit(numerical_features)
        self.feature_names['numerical'] = list(numerical_features.columns)
        logger.info(f"   ✓ Scaler fitted: {len(self.feature_names['numerical'])} features")
        logger.info(f"   Features: {self.feature_names['numerical']}")
        
        # 3. Prepare categorical features
        logger.info("\n3. Preparing categorical features...")
        categorical_features = self._extract_categorical_features(df_metadata)
        
        # Fit label encoders
        for col in categorical_features.columns:
            self.label_encoders[col] = LabelEncoder()
            self.label_encoders[col].fit(categorical_features[col])
            self.feature_names['categorical'].append(col)
        
        logger.info(f"   ✓ Label encoders fitted: {len(self.feature_names['categorical'])} features")
        logger.info(f"   Features: {self.feature_names['categorical']}")
        
        # 4. Engineered features
        logger.info("\n4. Identifying engineered features...")
        engineered_cols = [
            'title_length', 'desc_length', 'has_series',
            'price_rating_ratio', 'engagement_score'
        ]
        self.feature_names['engineered'] = engineered_cols
        logger.info(f"   ✓ Engineered features: {len(self.feature_names['engineered'])}")
        logger.info(f"   Features: {self.feature_names['engineered']}")
        
        self._is_fitted = True
        
        # Final summary
        total_features = sum(len(v) for v in self.feature_names.values())
        logger.info("\n" + "=" * 80)
        logger.info("FEATURE EXTRACTION FIT COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total features: {total_features}")
        logger.info(f"  - Text (TF-IDF): {len(self.feature_names['text'])}")
        logger.info(f"  - Numerical: {len(self.feature_names['numerical'])}")
        logger.info(f"  - Categorical: {len(self.feature_names['categorical'])}")
        logger.info(f"  - Engineered: {len(self.feature_names['engineered'])}")
        logger.info("=" * 80)
        
        return self
    
    def transform(
        self,
        df_metadata: pd.DataFrame,
        target_column: str #= 'main_category'
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
        """
        Transform data using fitted transformers.
        
        Args:
            df_metadata: DataFrame with preprocessed metadata
            target_column: Name of target column
            
        Returns:
            X: Feature matrix (numpy array)
            y: Encoded target (numpy array)
            feature_df: DataFrame with all features
            feature_names: List with all feature names
        """
        if not self._is_fitted:
            raise ValueError("FeatureExtractor must be fitted first with fit()")
        
        logger.info("=" * 80)
        logger.info("TRANSFORMING FEATURES")
        logger.info("=" * 80)
        logger.info(f"Input shape: {df_metadata.shape}")
        
        # 0. Extract and encode target
        logger.info("\n0. Extracting and encoding target...")
        if target_column not in df_metadata.columns:
            raise ValueError(
                f"Target column '{target_column}' not found. "
                f"Available columns: {list(df_metadata.columns)}"
            )
        
        y_raw = df_metadata[target_column].fillna('Unknown').values
        
        # Handle unknown values in target
        known_classes = set(self.target_label_encoder.classes_)
        unknown_mask = ~pd.Series(y_raw).isin(known_classes)
        if unknown_mask.any():
            fallback_class = self.target_label_encoder.classes_[0]
            logger.warning(
                f"   ⚠ Found {unknown_mask.sum()} unknown target values, "
                f"replacing with '{fallback_class}'"
            )
            y_raw = y_raw.copy()
            y_raw[unknown_mask] = fallback_class
        
        y = self.target_label_encoder.transform(y_raw)
        logger.info(f"   ✓ Target shape: {y.shape}")
        logger.info(f"   Target distribution:")
        for i, cls in enumerate(self.target_label_encoder.classes_):
            count = (y == i).sum()
            pct = (count / len(y)) * 100
            logger.info(f"      {i}: {cls:30s} - {count:5d} ({pct:5.1f}%)")
        
        # 1. Text features
        logger.info("\n1. Transforming text features...")
        text_data = self._prepare_text_data(df_metadata)
        X_text = self.text_vectorizer.transform(text_data).toarray()
        logger.info(f"   ✓ Text features shape: {X_text.shape}")
        
        # 2. Numerical features
        logger.info("\n2. Transforming numerical features...")
        numerical_features = self._extract_numerical_features(df_metadata)
        X_numerical = self.scaler.transform(numerical_features)
        logger.info(f"   ✓ Numerical features shape: {X_numerical.shape}")
        
        # 3. Categorical features
        logger.info("\n3. Transforming categorical features...")
        categorical_features = self._extract_categorical_features(df_metadata)
        
        # Transform with handling of values desconocidos
        X_categorical_list = []
        for col in categorical_features.columns:
            # Get known classes from encoder
            known_classes = set(self.label_encoders[col].classes_)
            
            # Replace unknown values with first known value (as fallback)
            col_data = categorical_features[col].copy()
            unknown_mask = ~col_data.isin(known_classes)
            if unknown_mask.any():
                fallback_value = self.label_encoders[col].classes_[0]
                logger.warning(
                    f"   ⚠ Found {unknown_mask.sum()} unknown values in '{col}', "
                    f"replacing with '{fallback_value}'"
                )
                col_data[unknown_mask] = fallback_value
            
            # Transformar
            encoded = self.label_encoders[col].transform(col_data)
            X_categorical_list.append(encoded)
        
        X_categorical = np.column_stack(X_categorical_list)
        logger.info(f"   ✓ Categorical features shape: {X_categorical.shape}")
        
        # 4. Engineered features
        logger.info("\n4. Extracting engineered features...")
        engineered_features = self._extract_engineered_features(df_metadata)
        X_engineered = engineered_features.values
        logger.info(f"   ✓ Engineered features shape: {X_engineered.shape}")
        
        # Concatenate all features
        logger.info("\n5. Concatenating all features...")
        X = np.hstack([X_text, X_numerical, X_categorical, X_engineered])
        logger.info(f"   ✓ Final feature matrix shape: {X.shape}")
        
        # Create DataFrame with all features
        all_feature_names = (
            self.feature_names['text'] +
            self.feature_names['numerical'] +
            self.feature_names['categorical'] +
            self.feature_names['engineered']
        )
        
        feature_df = pd.DataFrame(
            X,
            columns=all_feature_names,
            index=df_metadata.index
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("FEATURE TRANSFORMATION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total samples: {X.shape[0]}")
        logger.info(f"Total features: {X.shape[1]}")
        logger.info(f"Target shape: {y.shape}")
        logger.info("=" * 80)
        
        return X, y, feature_df, all_feature_names
    
    
    def _prepare_text_data(self, df: pd.DataFrame) -> pd.Series:
        """
        Prepare text data for TF-IDF.
        
        Args:
            df: DataFrame with metadata
            
        Returns:
            Series with combined text
        """
        # Combine title + description for TF-IDF
        text_data = (
            df['title'].fillna('') + ' ' +
            df['description'].fillna('')
        ).str.strip()
        
        # Replace empty strings with placeholder
        text_data = text_data.replace('', 'no_text')
        
        return text_data
    
    def _extract_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract direct numerical features.
        
        Args:
            df: DataFrame with metadata
            
        Returns:
            DataFrame with features numéricas
        """
        numerical_cols = [
            'price',
            'average_rating',
            'review_count',
            'rating_std',
            'verified_purchase_pct',
            'helpful_votes_total',
            'avg_review_length'
        ]
        
        # If already fitted, use columns defined in fit
        if self._is_fitted and hasattr(self, '_numerical_cols_fitted'):
            numerical_cols = self._numerical_cols_fitted
        else:
            # In fit, save available columns
            available_cols = [col for col in numerical_cols if col in df.columns]
            if not self._is_fitted:
                self._numerical_cols_fitted = available_cols
            numerical_cols = available_cols
        
        # Create DataFrame with all needed columns
        numerical_df = pd.DataFrame(index=df.index)
        
        for col in numerical_cols:
            if col in df.columns:
                numerical_df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                # If column doesn't exist, fill with 0
                numerical_df[col] = 0
        
        # Fill NaNs with 0
        numerical_df = numerical_df.fillna(0)
        
        return numerical_df
    
    def _extract_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract categorical features for encoding.
        
        Args:
            df: DataFrame with metadata
            
        Returns:
            DataFrame with features categóricas
        """
        categorical_data = pd.DataFrame(index=df.index)
        
        # Book format
        if 'format' in df.columns:
            categorical_data['format'] = df['format'].fillna('Unknown')
        else:
            categorical_data['format'] = 'Unknown'
        
        # Binned rating category
        if 'average_rating' in df.columns:
            categorical_data['binned_rating'] = pd.cut(
                df['average_rating'],
                bins=[0, 3.0, 4.0, 4.5, 5.0],
                labels=['low', 'medium', 'high', 'very_high']
            ).astype(str)
            categorical_data['binned_rating'] = categorical_data['binned_rating'].fillna('unknown')
        else:
            categorical_data['binned_rating'] = 'unknown'
        
        # Price category
        if 'price' in df.columns:
            categorical_data['price_category'] = pd.cut(
                df['price'],
                bins=[0, 10, 20, 50, float('inf')],
                labels=['budget', 'moderate', 'premium', 'luxury']
            ).astype(str)
            categorical_data['price_category'] = categorical_data['price_category'].fillna('unknown')
        else:
            categorical_data['price_category'] = 'unknown'
        
        return categorical_data
    
    def _extract_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract advanced engineered features.
        
        Args:
            df: DataFrame with metadata
            
        Returns:
            DataFrame with features ingenieriles
        """
        engineered = pd.DataFrame(index=df.index)
        
        # 1. Title length
        if 'title' in df.columns:
            engineered['title_length'] = df['title'].fillna('').str.len()
        else:
            engineered['title_length'] = 0
        
        # 2. Description length
        if 'description' in df.columns:
            engineered['desc_length'] = df['description'].fillna('').str.len()
        else:
            engineered['desc_length'] = 0
        
        # 3. Indicates if part of a series (based on patterns in title)
        if 'title' in df.columns:
            series_patterns = r'(?:book \d+|volume \d+|part \d+|#\d+|vol\. \d+)'
            engineered['has_series'] = (
                df['title'].fillna('').str.lower().str.contains(series_patterns, regex=True)
            ).astype(int)
        else:
            engineered['has_series'] = 0
        
        # 4. Price/rating ratio
        if 'price' in df.columns and 'average_rating' in df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = df['price'] / df['average_rating']
                ratio = np.where(np.isfinite(ratio), ratio, 0)
                engineered['price_rating_ratio'] = ratio
        else:
            engineered['price_rating_ratio'] = 0
        
        # 5. Engagement score (combination of reviews and ratings)
        if 'review_count' in df.columns and 'average_rating' in df.columns:
            # Score = log(reviews + 1) * rating
            engineered['engagement_score'] = (
                np.log1p(df['review_count'].fillna(0)) *
                df['average_rating'].fillna(0)
            )
        else:
            engineered['engagement_score'] = 0
        
        # Fill any remaining NaNs with 0
        engineered = engineered.fillna(0)
        
        return engineered
    
    
    def _get_feature_type(self, feature_name: str) -> str:
        """Determine feature type by its name."""
        if feature_name.startswith('tfidf_'):
            return 'text'
        elif feature_name in self.feature_names['numerical']:
            return 'numerical'
        elif feature_name in self.feature_names['categorical']:
            return 'categorical'
        elif feature_name in self.feature_names['engineered']:
            return 'engineered'
        else:
            return 'unknown'


    def save(self, output_dir: str) -> None:
        """
        Save feature extractor and its transformers.
        
        Args:
            output_dir: Directory to save files
        """
        if not self._is_fitted:
            raise ValueError("FeatureExtractor must be fitted before saving")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving FeatureExtractor to {output_dir}...")
        
        # Save transformers
        joblib.dump(
            self.text_vectorizer,
            output_path / 'text_vectorizer.joblib'
        )
        joblib.dump(
            self.scaler,
            output_path / 'scaler.joblib'
        )
        joblib.dump(
            self.label_encoders,
            output_path / 'label_encoders.joblib'
        )
        joblib.dump(
            self.target_label_encoder,
            output_path / 'target_label_encoder.joblib'
        )
        
        # Save metadata
        metadata = {
            'config': self.config,
            'feature_names': self.feature_names,
            'is_fitted': self._is_fitted,
            'target_classes': self.target_label_encoder.classes_.tolist(),
            'n_classes': len(self.target_label_encoder.classes_)
        }
        
        with open(output_path / 'feature_extractor_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("   ✓ FeatureExtractor saved successfully")
        logger.info(f"   Files: text_vectorizer.joblib, scaler.joblib, label_encoders.joblib")
        logger.info(f"   Files: target_label_encoder.joblib")
        logger.info(f"   Metadata: feature_extractor_metadata.json")
    
    @classmethod
    def load(cls, input_dir: str) -> 'FeatureExtractor':
        """
        Load saved feature extractor.
        
        Args:
            input_dir: Directory with saved files
            
        Returns:
            Loaded FeatureExtractor
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        
        logger.info(f"\nLoading FeatureExtractor from {input_dir}...")
        
        # Load metadata
        with open(input_path / 'feature_extractor_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        extractor = cls(**metadata['config'])
        
        # Load transformers
        extractor.text_vectorizer = joblib.load(
            input_path / 'text_vectorizer.joblib'
        )
        extractor.scaler = joblib.load(
            input_path / 'scaler.joblib'
        )
        extractor.label_encoders = joblib.load(
            input_path / 'label_encoders.joblib'
        )
        extractor.target_label_encoder = joblib.load(
            input_path / 'target_label_encoder.joblib'
        )
        
        extractor.feature_names = metadata['feature_names']
        extractor._is_fitted = metadata['is_fitted']
        
        logger.info("   ✓ FeatureExtractor loaded successfully")
        logger.info(f"   Target classes: {len(extractor.target_label_encoder.classes_)}")
        
        return extractor

    
    
    def get_stats(self) -> Dict:
        """
        Return feature extractor statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self._is_fitted:
            return {'status': 'not_fitted'}
        
        total_features = sum(len(v) for v in self.feature_names.values())
        
        stats = {
            'status': 'fitted',
            'total_features': total_features,
            'feature_breakdown': {
                'text': len(self.feature_names['text']),
                'numerical': len(self.feature_names['numerical']),
                'categorical': len(self.feature_names['categorical']),
                'engineered': len(self.feature_names['engineered'])
            },
            'target_info': {
                'n_classes': len(self.target_label_encoder.classes_),
                'classes': self.target_label_encoder.classes_.tolist()
            },
            'config': self.config,
            'feature_names': self.feature_names
        }
        
        return stats
    

    def transform_inference(
        self,
        df_metadata: pd.DataFrame
    ) -> np.ndarray:
        """
        Transform data for inference (without target).
        
        Args:
            df_metadata: DataFrame with preprocessed metadata
            
        Returns:
            X: Feature matrix (numpy array)
        """
        if not self._is_fitted:
            raise ValueError("FeatureExtractor must be fitted first")
        
        # 1. Text features
        text_data = self._prepare_text_data(df_metadata)
        X_text = self.text_vectorizer.transform(text_data).toarray()
        
        # 2. Numerical features
        numerical_features = self._extract_numerical_features(df_metadata)
        X_numerical = self.scaler.transform(numerical_features)
        
        # 3. Categorical features
        categorical_features = self._extract_categorical_features(df_metadata)
        
        X_categorical_list = []
        for col in categorical_features.columns:
            known_classes = set(self.label_encoders[col].classes_)
            col_data = categorical_features[col].copy()
            unknown_mask = ~col_data.isin(known_classes)
            if unknown_mask.any():
                fallback_value = self.label_encoders[col].classes_[0]
                col_data[unknown_mask] = fallback_value
            
            encoded = self.label_encoders[col].transform(col_data)
            X_categorical_list.append(encoded)
        
        X_categorical = np.column_stack(X_categorical_list)
        
        # 4. Engineered features
        engineered_features = self._extract_engineered_features(df_metadata)
        X_engineered = engineered_features.values
        
        # Concatenate all
        X = np.hstack([X_text, X_numerical, X_categorical, X_engineered])
        
        logger.info(f"✓ Features for inference: {X.shape}")
        
        return X    

