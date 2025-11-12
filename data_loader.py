"""
Data Loader Module for Book Category Classification System

This module handles loading and initial validation of the book metadata dataset.
It provides methods for data exploration, schema validation, and category distribution analysis.


"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    DataLoader class for loading and validating book metadata and reviews.
    
    This class handles:
    - Loading METADATA and REVIEWS datasets
    - Schema validation for both datasets
    - Initial data exploration
    - Category distribution analysis
    - Reviews analysis (ratings, sentiment, helpful votes)
    - JOIN between metadata and reviews
    - Data quality checks
    
    Attributes:
        metadata_path (str): Path to the metadata file
        reviews_path (str): Path to the reviews file
        metadata_df (pd.DataFrame): Loaded metadata DataFrame
        reviews_df (pd.DataFrame): Loaded reviews DataFrame
        required_columns (Dict): Required columns for each dataset
    """
    
    # Required columns for METADATA
    REQUIRED_METADATA_COLUMNS = [
        'parent_asin',  # Unique identifier (JOIN KEY)
        'title',  # Book title
        #'main_category',  # Primary target for classification
        'category_level_2_sub',# Primary target for classification
        'description'  # Main text feature
    ]
    
    # Required columns for REVIEWS
    REQUIRED_REVIEWS_COLUMNS = [
        'parent_asin',  # Book identifier (JOIN KEY)
        'user_id',  # User identifier
        'rating',  # User rating (1-5)
        'text'  # Review text
    ]
    
    # Optional but useful columns for METADATA
    OPTIONAL_METADATA_COLUMNS = [
        'author_name',
        'price_numeric',
        'page_count',
        'average_rating',
        'rating_number',
        'category_level_1_main',
         #'category_level_2_sub',
        'features_text',
        'publisher',
        'format',
        'language'
    ]
    
    # Optional but useful columns for REVIEWS
    OPTIONAL_REVIEWS_COLUMNS = [
        'title',  # Review title
        'images',  # Review images
        'asin',  # Product ASIN
        'timestamp',  # Review timestamp
        'helpful_vote',  # Helpful votes count
        'verified_purchase',  # Verified purchase flag
        'date',  # Review date
        'year'  # Review year
    ]
    
    def __init__(self, metadata_path: str, reviews_path: Optional[str] = None):
        """
        Initialize DataLoader with paths to metadata and reviews files.
        
        Args:
            metadata_path (str): Path to the metadata CSV or Parquet file
            reviews_path (Optional[str]): Path to the reviews CSV or Parquet file
                                         If None, only metadata will be loaded
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
        """
        self.metadata_path = Path(metadata_path)
        self.reviews_path = Path(reviews_path) if reviews_path else None
        self.metadata_df: Optional[pd.DataFrame] = None
        self.reviews_df: Optional[pd.DataFrame] = None
        self._validation_results: Dict[str, Any] = {}
        
        # Validate metadata file exists
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Validate reviews file exists (if provided)
        if self.reviews_path and not self.reviews_path.exists():
            raise FileNotFoundError(f"Reviews file not found: {reviews_path}")
        
        # Validate file formats
        valid_extensions = ['.csv', '.pkl']
        if self.metadata_path.suffix.lower() not in valid_extensions:
            raise ValueError(
                f"Unsupported metadata file format: {self.metadata_path.suffix}. "
                f"Supported formats: {valid_extensions}"
            )
        
        if self.reviews_path and self.reviews_path.suffix.lower() not in valid_extensions:
            raise ValueError(
                f"Unsupported reviews file format: {self.reviews_path.suffix}. "
                f"Supported formats: {valid_extensions}"
            )
        
        logger.info(f"DataLoader initialized")
        logger.info(f"  Metadata: {self.metadata_path}")
        if self.reviews_path:
            logger.info(f"  Reviews: {self.reviews_path}")
        else:
            logger.info(f"  Reviews: Not provided (metadata only mode)")
    
    def load_metadata(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load metadata from file into pandas DataFrame.
        
        Args:
            nrows (Optional[int]): Number of rows to load (for testing). 
                                   If None, loads all data.
        
        Returns:
            pd.DataFrame: Loaded metadata DataFrame
            
        Raises:
            Exception: If loading fails
        """
        try:
            logger.info(f"Loading metadata from: {self.metadata_path}")
            start_time = datetime.now()
            
            # Load based on file extension
            if self.metadata_path.suffix.lower() == '.csv':
                self.metadata_df = pd.read_csv(
                    self.metadata_path,
                    nrows=nrows,
                    low_memory=False
                )
           
            elif self.metadata_path.suffix.lower() == '.pkl':
                self.metadata_df = pd.read_pickle(self.metadata_path)
                if nrows is not None:
                    self.metadata_df = self.metadata_df.head(nrows)
            
            load_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(
                f"Successfully loaded {len(self.metadata_df):,} rows "
                f"and {len(self.metadata_df.columns)} columns in {load_time:.2f}s"
            )
            
            return self.metadata_df
            
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            raise
    
    def load_reviews(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load reviews from file into pandas DataFrame.
        
        Args:
            nrows (Optional[int]): Number of rows to load (for testing). 
                                   If None, loads all data.
        
        Returns:
            pd.DataFrame: Loaded reviews DataFrame
            
        Raises:
            ValueError: If reviews_path was not provided
            Exception: If loading fails
        """
        if self.reviews_path is None:
            raise ValueError(
                "Reviews path not provided. Initialize DataLoader with reviews_path parameter."
            )
        
        try:
            logger.info(f"Loading reviews from: {self.reviews_path}")
            start_time = datetime.now()
            
            # Load based on file extension
            if self.reviews_path.suffix.lower() == '.csv':
                self.reviews_df = pd.read_csv(
                    self.reviews_path,
                    nrows=nrows,
                    low_memory=False
                )
            elif self.reviews_path.suffix.lower() == '.parquet':
                self.reviews_df = pd.read_parquet(self.reviews_path)
                if nrows is not None:
                    self.reviews_df = self.reviews_df.head(nrows)
            elif self.reviews_path.suffix.lower() == '.pkl':
                self.reviews_df = pd.read_pickle(self.reviews_path)
                if nrows is not None:
                    self.reviews_df = self.reviews_df.head(nrows)
            
            load_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(
                f"Successfully loaded {len(self.reviews_df):,} reviews "
                f"and {len(self.reviews_df.columns)} columns in {load_time:.2f}s"
            )
            
            return self.reviews_df
            
        except Exception as e:
            logger.error(f"Error loading reviews: {str(e)}")
            raise
    
    def validate_schema(self, dataset: str = 'both') -> Dict[str, Any]:
        """
        Validate that required columns exist in the dataset(s).
        
        Args:
            dataset (str): Which dataset to validate: 'metadata', 'reviews', or 'both'
        
        Returns:
            Dict[str, Any]: Dictionary with validation results for each dataset
        
        Raises:
            ValueError: If metadata/reviews haven't been loaded yet
        """
        logger.info(f"Validating schema for: {dataset}")
        
        validation_results = {}
        
        # Validate METADATA
        if dataset in ['metadata', 'both']:
            if self.metadata_df is None:
                raise ValueError("Metadata not loaded. Call load_metadata() first.")
            
            missing_required = [
                col for col in self.REQUIRED_METADATA_COLUMNS 
                if col not in self.metadata_df.columns
            ]
            
            optional_present = {
                col: col in self.metadata_df.columns 
                for col in self.OPTIONAL_METADATA_COLUMNS
            }
            
            validation_results['metadata'] = {
                'has_required_columns': len(missing_required) == 0,
                'missing_required_columns': missing_required,
                'optional_columns_present': optional_present,
                'total_columns': len(self.metadata_df.columns),
                'column_list': list(self.metadata_df.columns)
            }
            
            if validation_results['metadata']['has_required_columns']:
                logger.info("✓ Metadata schema validation passed")
            else:
                logger.warning(
                    f"✗ Metadata schema validation failed: Missing {missing_required}"
                )
            
            present_optional = sum(optional_present.values())
            logger.info(
                f"Metadata optional columns: {present_optional}/{len(self.OPTIONAL_METADATA_COLUMNS)}"
            )
        
        # Validate REVIEWS
        if dataset in ['reviews', 'both']:
            if self.reviews_path is None:
                logger.info("Reviews path not provided - skipping reviews validation")
            elif self.reviews_df is None:
                if dataset == 'reviews':
                    raise ValueError("Reviews not loaded. Call load_reviews() first.")
                else:
                    logger.info("Reviews not loaded yet - skipping reviews validation")
            else:
                missing_required = [
                    col for col in self.REQUIRED_REVIEWS_COLUMNS 
                    if col not in self.reviews_df.columns
                ]
                
                optional_present = {
                    col: col in self.reviews_df.columns 
                    for col in self.OPTIONAL_REVIEWS_COLUMNS
                }
                
                validation_results['reviews'] = {
                    'has_required_columns': len(missing_required) == 0,
                    'missing_required_columns': missing_required,
                    'optional_columns_present': optional_present,
                    'total_columns': len(self.reviews_df.columns),
                    'column_list': list(self.reviews_df.columns)
                }
                
                if validation_results['reviews']['has_required_columns']:
                    logger.info("✓ Reviews schema validation passed")
                else:
                    logger.warning(
                        f"✗ Reviews schema validation failed: Missing {missing_required}"
                    )
                
                present_optional = sum(optional_present.values())
                logger.info(
                    f"Reviews optional columns: {present_optional}/{len(self.OPTIONAL_REVIEWS_COLUMNS)}"
                )
        
        self._validation_results = validation_results
        return validation_results
    
    def get_category_distribution(
        self, 
        level: str , #= 'main_category',
        top_n: Optional[int] = None
    ) -> pd.Series:
        """
        Get distribution of categories at specified level.
        
        Args:
            level (str): Category level to analyze. Options:
                        - 'main_category' (default)
                        - 'category_level_1_main'
                        - 'category_level_2_sub'
            top_n (Optional[int]): Return only top N categories. If None, returns all.
        
        Returns:
            pd.Series: Category counts sorted in descending order
            
        Raises:
            ValueError: If metadata not loaded or column doesn't exist
        """
        if self.metadata_df is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")
        
        if level not in self.metadata_df.columns:
            raise ValueError(
                f"Column '{level}' not found in dataset. "
                f"Available category columns: {[c for c in self.metadata_df.columns if 'category' in c.lower()]}"
            )
        
        logger.info(f"Analyzing category distribution for: {level}")
        
        # Get value counts
        distribution = self.metadata_df[level].value_counts()
        
        if top_n is not None:
            distribution = distribution.head(top_n)
        
        logger.info(
            f"Found {len(distribution)} unique categories at level '{level}'"
        )
        
        return distribution
    
    def check_missing_values(self) -> pd.DataFrame:
        """
        Generate report of missing values for all columns.
        
        Returns:
            pd.DataFrame: DataFrame with columns:
                - column: Column name
                - missing_count: Number of missing values
                - missing_percentage: Percentage of missing values
                - dtype: Data type of column
        
        Raises:
            ValueError: If metadata not loaded
        """
        if self.metadata_df is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")
        
        logger.info("Checking missing values...")
        
        missing_info = []
        total_rows = len(self.metadata_df)
        
        for col in self.metadata_df.columns:
            missing_count = self.metadata_df[col].isna().sum()
            missing_percentage = (missing_count / total_rows) * 100
            
            missing_info.append({
                'column': col,
                'missing_count': missing_count,
                'missing_percentage': round(missing_percentage, 2),
                'dtype': str(self.metadata_df[col].dtype)
            })
        
        missing_df = pd.DataFrame(missing_info)
        missing_df = missing_df.sort_values('missing_percentage', ascending=False)
        
        # Log critical missing values
        critical_missing = missing_df[
            (missing_df['column'].isin(self.REQUIRED_METADATA_COLUMNS)) & 
            (missing_df['missing_percentage'] > 0)
        ]
        
        if len(critical_missing) > 0:
            logger.warning(
                f"Found missing values in required columns:\n{critical_missing}"
            )
        else:
            logger.info("✓ No missing values in required columns")
        
        return missing_df
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary of the loaded dataset.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - basic_info: DataFrame shape, memory usage
                - category_stats: Statistics for each category level
                - missing_values: Missing value summary
                - text_features: Statistics about text columns
                - numerical_features: Statistics about numerical columns
        
        Raises:
            ValueError: If metadata not loaded
        """
        if self.metadata_df is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")
        
        logger.info("Generating data summary...")
        
        summary = {}
        
        # Basic information
        summary['basic_info'] = {
            'total_rows': len(self.metadata_df),
            'total_columns': len(self.metadata_df.columns),
            'memory_usage_mb': round(
                self.metadata_df.memory_usage(deep=True).sum() / 1024**2, 2
            ),
            'duplicates': self.metadata_df.duplicated(subset=['parent_asin']).sum()
        }
        
        # Category statistics
        summary['category_stats'] = {}
        category_columns = [ 'category_level_2_sub']#'main_category', 'category_level_1_main',
        
        for cat_col in category_columns:
            if cat_col in self.metadata_df.columns:
                summary['category_stats'][cat_col] = {
                    'unique_categories': self.metadata_df[cat_col].nunique(),
                    'missing_values': self.metadata_df[cat_col].isna().sum(),
                    'most_common': self.metadata_df[cat_col].value_counts().head(5).to_dict()
                }
        
        # Text features analysis
        text_columns = ['title', 'description', 'features_text']
        summary['text_features'] = {}
        
        for col in text_columns:
            if col in self.metadata_df.columns:
                non_null = self.metadata_df[col].dropna()
                if len(non_null) > 0:
                    text_lengths = non_null.astype(str).str.len()
                    summary['text_features'][col] = {
                        'missing_count': self.metadata_df[col].isna().sum(),
                        'avg_length': round(text_lengths.mean(), 2),
                        'median_length': text_lengths.median(),
                        'max_length': text_lengths.max(),
                        'min_length': text_lengths.min()
                    }
        
        # Numerical features analysis
        numerical_columns = ['price_numeric', 'page_count', 'average_rating', 'rating_number']
        summary['numerical_features'] = {}
        
        for col in numerical_columns:
            if col in self.metadata_df.columns:
                summary['numerical_features'][col] = {
                    'count': self.metadata_df[col].count(),
                    'mean': round(self.metadata_df[col].mean(), 2),
                    'median': self.metadata_df[col].median(),
                    'std': round(self.metadata_df[col].std(), 2),
                    'min': self.metadata_df[col].min(),
                    'max': self.metadata_df[col].max(),
                    'missing': self.metadata_df[col].isna().sum()
                }
        
        logger.info("Data summary generated successfully")
        
        return summary
    
    def filter_by_category(
        self, 
        categories: List[str],
        level: str #= 'main_category'
    ) -> pd.DataFrame:
        """
        Filter dataset to include only specific categories.
        
        Args:
            categories (List[str]): List of category names to keep
            level (str): Category level to filter on
        
        Returns:
            pd.DataFrame: Filtered DataFrame
            
        Raises:
            ValueError: If metadata not loaded or column doesn't exist
        """
        if self.metadata_df is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")
        
        if level not in self.metadata_df.columns:
            raise ValueError(f"Column '{level}' not found in dataset")
        
        logger.info(f"Filtering by {len(categories)} categories at level '{level}'")
        
        original_count = len(self.metadata_df)
        filtered_df = self.metadata_df[self.metadata_df[level].isin(categories)].copy()
        filtered_count = len(filtered_df)
        
        logger.info(
            f"Filtered from {original_count:,} to {filtered_count:,} rows "
            f"({filtered_count/original_count*100:.1f}%)"
        )
        
        return filtered_df
    
    def get_category_analysis(self, level: str ) -> pd.DataFrame:#= 'main_category'
        """
        Get detailed analysis of categories including distribution and statistics.
        
        Args:
            level (str): Category level to analyze
        
        Returns:
            pd.DataFrame: DataFrame with category analysis including:
                - category: Category name
                - count: Number of books
                - percentage: Percentage of total
                - avg_rating: Average rating for books in category
                - avg_price: Average price
                - avg_pages: Average page count
        
        Raises:
            ValueError: If metadata not loaded
        """
        if self.metadata_df is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")
        
        if level not in self.metadata_df.columns:
            raise ValueError(f"Column '{level}' not found in dataset")
        
        logger.info(f"Performing detailed analysis for: {level}")
        
        analysis = []
        total_books = len(self.metadata_df)
        
        for category in self.metadata_df[level].unique():
            if pd.isna(category):
                continue
                
            cat_data = self.metadata_df[self.metadata_df[level] == category]
            
            analysis.append({
                'category': category,
                'count': len(cat_data),
                'percentage': round(len(cat_data) / total_books * 100, 2),
                'avg_rating': round(cat_data['average_rating'].mean(), 2) 
                    if 'average_rating' in cat_data.columns else None,
                'avg_price': round(cat_data['price_numeric'].mean(), 2) 
                    if 'price_numeric' in cat_data.columns else None,
                'avg_pages': round(cat_data['page_count'].mean(), 0) 
                    if 'page_count' in cat_data.columns else None
            })
        
        analysis_df = pd.DataFrame(analysis)
        analysis_df = analysis_df.sort_values('count', ascending=False)
        
        logger.info(f"Analysis complete for {len(analysis_df)} categories")
        
        return analysis_df
    
    def get_reviews_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of reviews data.
        
        Returns:
            Dict[str, Any]: Dictionary with reviews statistics
        
        Raises:
            ValueError: If reviews not loaded
        """
        if self.reviews_df is None:
            raise ValueError("Reviews not loaded. Call load_reviews() first.")
        
        logger.info("Generating reviews summary...")
        
        summary = {
            'total_reviews': len(self.reviews_df),
            'unique_users': self.reviews_df['user_id'].nunique(),
            'unique_books': self.reviews_df['parent_asin'].nunique(),
            'rating_distribution': self.reviews_df['rating'].value_counts().sort_index().to_dict(),
            'avg_rating': round(self.reviews_df['rating'].mean(), 2),
            'median_rating': self.reviews_df['rating'].median(),
        }
        
        # Text length analysis
        if 'text' in self.reviews_df.columns:
            review_texts = self.reviews_df['text'].dropna().astype(str)
            text_lengths = review_texts.str.len()
            summary['review_text_stats'] = {
                'avg_length': round(text_lengths.mean(), 0),
                'median_length': text_lengths.median(),
                'max_length': text_lengths.max(),
                'min_length': text_lengths.min(),
                'missing_text': self.reviews_df['text'].isna().sum()
            }
        
        # Verified purchase analysis
        if 'verified_purchase' in self.reviews_df.columns:
            verified_counts = self.reviews_df['verified_purchase'].value_counts()
            summary['verified_purchase'] = {
                'verified': int(verified_counts.get(True, 0)),
                'not_verified': int(verified_counts.get(False, 0)),
                'percentage_verified': round(
                    (verified_counts.get(True, 0) / len(self.reviews_df)) * 100, 2
                )
            }
        
        # Helpful votes analysis
        if 'helpful_vote' in self.reviews_df.columns:
            helpful_votes = self.reviews_df['helpful_vote'].fillna(0)
            summary['helpful_votes'] = {
                'total': int(helpful_votes.sum()),
                'avg_per_review': round(helpful_votes.mean(), 2),
                'max': int(helpful_votes.max()),
                'reviews_with_votes': int((helpful_votes > 0).sum())
            }
        
        # Temporal analysis
        if 'year' in self.reviews_df.columns:
            year_counts = self.reviews_df['year'].value_counts().sort_index()
            summary['temporal'] = {
                'earliest_year': int(year_counts.index.min()),
                'latest_year': int(year_counts.index.max()),
                'top_5_years': year_counts.head(5).to_dict()
            }
        
        logger.info("Reviews summary generated successfully")
        return summary
    
    def join_metadata_reviews(
        self, 
        how: str = 'inner',
        min_reviews_per_book: int = 1
    ) -> pd.DataFrame:
        """
        Join metadata and reviews datasets on parent_asin.
        
        Args:
            how (str): Type of join: 'inner', 'left', 'right', 'outer'
            min_reviews_per_book (int): Minimum reviews required per book to include
        
        Returns:
            pd.DataFrame: Joined DataFrame
            
        Raises:
            ValueError: If metadata or reviews not loaded
        """
        if self.metadata_df is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")
        if self.reviews_df is None:
            raise ValueError("Reviews not loaded. Call load_reviews() first.")
        
        logger.info(f"Joining metadata and reviews with '{how}' join...")
        
        # Count reviews per book
        reviews_per_book = self.reviews_df['parent_asin'].value_counts()
        
        # Filter reviews to include only books with minimum reviews
        if min_reviews_per_book > 1:
            valid_books = reviews_per_book[reviews_per_book >= min_reviews_per_book].index
            filtered_reviews = self.reviews_df[
                self.reviews_df['parent_asin'].isin(valid_books)
            ].copy()
            logger.info(
                f"Filtered to books with >={min_reviews_per_book} reviews: "
                f"{len(valid_books):,} books, {len(filtered_reviews):,} reviews"
            )
        else:
            filtered_reviews = self.reviews_df.copy()
        
        # Perform join
        joined_df = pd.merge(
            self.metadata_df,
            filtered_reviews,
            on='parent_asin',
            how=how,
            suffixes=('_book', '_review')
        )
        
        logger.info(
            f"Join complete: {len(joined_df):,} rows "
            f"({len(self.metadata_df):,} books × {len(filtered_reviews):,} reviews)"
        )
        
        # Show join statistics
        unique_books_in_join = joined_df['parent_asin'].nunique()
        unique_users_in_join = joined_df['user_id'].nunique()
        
        logger.info(
            f"Join statistics: {unique_books_in_join:,} unique books, "
            f"{unique_users_in_join:,} unique users"
        )
        
        return joined_df
    
    def get_reviews_per_book_stats(self) -> pd.DataFrame:
        """
        Get statistics about number of reviews per book.
        
        Returns:
            pd.DataFrame: Statistics about reviews per book
        
        Raises:
            ValueError: If reviews not loaded
        """
        if self.reviews_df is None:
            raise ValueError("Reviews not loaded. Call load_reviews() first.")
        
        logger.info("Analyzing reviews per book...")
        
        reviews_per_book = self.reviews_df.groupby('parent_asin').agg({
            'rating': ['count', 'mean', 'std'],
            'user_id': 'nunique'
        }).reset_index()
        
        reviews_per_book.columns = [
            'parent_asin', 'num_reviews', 'avg_rating', 'rating_std', 'unique_users'
        ]
        
        reviews_per_book = reviews_per_book.sort_values('num_reviews', ascending=False)
        
        logger.info(f"Analyzed {len(reviews_per_book):,} books")
        
        return reviews_per_book
    
    def get_rating_distribution_by_category(self, level: str ) -> pd.DataFrame:#= 'main_category'
        """
        Get rating distribution by category (requires joined data).
        
        Args:
            level (str): Category level to analyze
        
        Returns:
            pd.DataFrame: Rating distribution by category
            
        Raises:
            ValueError: If metadata or reviews not loaded
        """
        if self.metadata_df is None or self.reviews_df is None:
            raise ValueError("Both metadata and reviews must be loaded.")
        
        logger.info(f"Analyzing rating distribution by {level}...")
        
        # Join data
        joined = self.join_metadata_reviews(how='inner')
        
        # Calculate stats per category
        category_stats = joined.groupby(level).agg({
            'rating': ['count', 'mean', 'std', 'median'],
            'parent_asin': 'nunique'
        }).reset_index()
        
        category_stats.columns = [
            level, 'num_reviews', 'avg_rating', 'rating_std', 'median_rating', 'num_books'
        ]
        
        category_stats = category_stats.sort_values('num_reviews', ascending=False)
        
        logger.info(f"Analyzed {len(category_stats)} categories")
        
        return category_stats
    
    def print_summary(self):
        """
        Print a formatted summary of the loaded data to console.
        Includes both metadata and reviews if available.
        """
        if self.metadata_df is None:
            print("No metadata loaded yet. Call load_metadata() first.")
            return
        
        summary = self.get_data_summary()
        
        print("\n" + "="*70)
        print("📚 BOOK METADATA & REVIEWS SUMMARY")
        print("="*70)
        
        print("\n📊 METADATA - BASIC INFORMATION:")
        print(f"  • Total Books: {summary['basic_info']['total_rows']:,}")
        print(f"  • Total Columns: {summary['basic_info']['total_columns']}")
        print(f"  • Memory Usage: {summary['basic_info']['memory_usage_mb']} MB")
        print(f"  • Duplicates: {summary['basic_info']['duplicates']}")
        
        print("\n🏷️  CATEGORY STATISTICS:")
        for cat_level, stats in summary['category_stats'].items():
            print(f"\n  {cat_level}:")
            print(f"    - Unique Categories: {stats['unique_categories']}")
            print(f"    - Missing Values: {stats['missing_values']}")
            print(f"    - Top 3 Categories:")
            for i, (cat, count) in enumerate(list(stats['most_common'].items())[:3], 1):
                print(f"      {i}. {cat}: {count:,} books")
        
        print("\n📝 TEXT FEATURES:")
        for col, stats in summary['text_features'].items():
            print(f"\n  {col}:")
            print(f"    - Missing: {stats['missing_count']:,}")
            print(f"    - Avg Length: {stats['avg_length']:.0f} characters")
            print(f"    - Range: {stats['min_length']} - {stats['max_length']} chars")
        
        print("\n🔢 NUMERICAL FEATURES:")
        for col, stats in summary['numerical_features'].items():
            print(f"\n  {col}:")
            print(f"    - Count: {stats['count']:,}")
            print(f"    - Mean: {stats['mean']}")
            print(f"    - Range: {stats['min']} - {stats['max']}")
            print(f"    - Missing: {stats['missing']:,}")
        
        # Reviews summary if available
        if self.reviews_df is not None:
            reviews_summary = self.get_reviews_summary()
            
            print("\n" + "="*70)
            print("💬 REVIEWS SUMMARY")
            print("="*70)
            
            print("\n📊 BASIC STATISTICS:")
            print(f"  • Total Reviews: {reviews_summary['total_reviews']:,}")
            print(f"  • Unique Users: {reviews_summary['unique_users']:,}")
            print(f"  • Unique Books Reviewed: {reviews_summary['unique_books']:,}")
            print(f"  • Average Rating: {reviews_summary['avg_rating']}")
            print(f"  • Median Rating: {reviews_summary['median_rating']}")
            
            print("\n⭐ RATING DISTRIBUTION:")
            for rating in sorted(reviews_summary['rating_distribution'].keys()):
                count = reviews_summary['rating_distribution'][rating]
                percentage = (count / reviews_summary['total_reviews']) * 100
                print(f"  {int(rating)}★ : {count:,} reviews ({percentage:.1f}%)")
            
            if 'review_text_stats' in reviews_summary:
                print("\n📝 REVIEW TEXT:")
                text_stats = reviews_summary['review_text_stats']
                print(f"  • Avg Length: {text_stats['avg_length']:.0f} characters")
                print(f"  • Median Length: {text_stats['median_length']:.0f} characters")
                print(f"  • Range: {text_stats['min_length']} - {text_stats['max_length']} chars")
                print(f"  • Missing Text: {text_stats['missing_text']:,}")
            
            if 'verified_purchase' in reviews_summary:
                print("\n✅ VERIFIED PURCHASES:")
                vp = reviews_summary['verified_purchase']
                print(f"  • Verified: {vp['verified']:,} ({vp['percentage_verified']}%)")
                print(f"  • Not Verified: {vp['not_verified']:,}")
            
            if 'helpful_votes' in reviews_summary:
                print("\n👍 HELPFUL VOTES:")
                hv = reviews_summary['helpful_votes']
                print(f"  • Total Votes: {hv['total']:,}")
                print(f"  • Avg per Review: {hv['avg_per_review']}")
                print(f"  • Reviews with Votes: {hv['reviews_with_votes']:,}")
            
            if 'temporal' in reviews_summary:
                print("\n📅 TEMPORAL DISTRIBUTION:")
                temporal = reviews_summary['temporal']
                print(f"  • Date Range: {temporal['earliest_year']} - {temporal['latest_year']}")
                print(f"  • Top 3 Years:")
                for i, (year, count) in enumerate(list(temporal['top_5_years'].items())[:3], 1):
                    print(f"    {i}. {year}: {count:,} reviews")
        
        print("\n" + "="*70 + "\n")
    
    def export_summary_report(self, output_path: str):
        """
        Export comprehensive data summary to a text file.
        Includes both metadata and reviews if available.
        
        Args:
            output_path (str): Path where to save the report
        """
        if self.metadata_df is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")
        
        logger.info(f"Exporting summary report to: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("BOOK CATEGORY CLASSIFICATION - DATA LOADING REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            # METADATA SECTION
            f.write("="*70 + "\n")
            f.write("METADATA INFORMATION\n")
            f.write("="*70 + "\n\n")
            
            # Basic info
            summary = self.get_data_summary()
            f.write("BASIC INFORMATION:\n")
            f.write("-"*70 + "\n")
            for key, value in summary['basic_info'].items():
                f.write(f"{key}: {value}\n")
            
            # Category stats
            f.write("\n\nCATEGORY STATISTICS:\n")
            f.write("-"*70 + "\n")
            for cat_level, stats in summary['category_stats'].items():
                f.write(f"\n{cat_level}:\n")
                for key, value in stats.items():
                    if key != 'most_common':
                        f.write(f"  {key}: {value}\n")
            
            # Missing values
            f.write("\n\nMISSING VALUES REPORT (METADATA):\n")
            f.write("-"*70 + "\n")
            missing_df = self.check_missing_values()
            f.write(missing_df.to_string())
            
            # Category distribution
            #f.write("\n\n\nCATEGORY DISTRIBUTION (main_category):\n")
            #f.write("-"*70 + "\n")
            #if 'main_category' in self.metadata_df.columns:
            #    dist = self.get_category_distribution('main_category', top_n=20)
            #    f.write(dist.to_string())

            f.write("\n\n\nCATEGORY DISTRIBUTION (category_level_2_sub):\n")
            f.write("-"*70 + "\n")
            if 'category_level_2_sub' in self.metadata_df.columns:
                dist = self.get_category_distribution('category_level_2_sub', top_n=20)
                f.write(dist.to_string())

            
            # REVIEWS SECTION (if available)
            if self.reviews_df is not None:
                f.write("\n\n\n")
                f.write("="*70 + "\n")
                f.write("REVIEWS INFORMATION\n")
                f.write("="*70 + "\n\n")
                
                reviews_summary = self.get_reviews_summary()
                
                f.write("BASIC STATISTICS:\n")
                f.write("-"*70 + "\n")
                f.write(f"Total Reviews: {reviews_summary['total_reviews']:,}\n")
                f.write(f"Unique Users: {reviews_summary['unique_users']:,}\n")
                f.write(f"Unique Books: {reviews_summary['unique_books']:,}\n")
                f.write(f"Average Rating: {reviews_summary['avg_rating']}\n")
                f.write(f"Median Rating: {reviews_summary['median_rating']}\n")
                
                f.write("\n\nRATING DISTRIBUTION:\n")
                f.write("-"*70 + "\n")
                for rating in sorted(reviews_summary['rating_distribution'].keys()):
                    count = reviews_summary['rating_distribution'][rating]
                    percentage = (count / reviews_summary['total_reviews']) * 100
                    f.write(f"{int(rating)} stars: {count:,} reviews ({percentage:.2f}%)\n")
                
                if 'review_text_stats' in reviews_summary:
                    f.write("\n\nREVIEW TEXT STATISTICS:\n")
                    f.write("-"*70 + "\n")
                    text_stats = reviews_summary['review_text_stats']
                    for key, value in text_stats.items():
                        f.write(f"{key}: {value}\n")
                
                if 'verified_purchase' in reviews_summary:
                    f.write("\n\nVERIFIED PURCHASE STATISTICS:\n")
                    f.write("-"*70 + "\n")
                    vp = reviews_summary['verified_purchase']
                    for key, value in vp.items():
                        f.write(f"{key}: {value}\n")
                
                # Reviews per book stats
                f.write("\n\nREVIEWS PER BOOK STATISTICS:\n")
                f.write("-"*70 + "\n")
                reviews_per_book = self.get_reviews_per_book_stats()
                f.write(f"\nTop 10 most reviewed books:\n")
                f.write(reviews_per_book.head(10).to_string())
                
                f.write(f"\n\nReviews per book distribution:\n")
                review_counts = reviews_per_book['num_reviews'].describe()
                f.write(review_counts.to_string())
        
        logger.info(f"Report exported successfully to: {output_path}")

    