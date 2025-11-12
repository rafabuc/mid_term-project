"""
Data Preprocessor Module for Book Category Classification System

This module handles cleaning and preprocessing of both metadata and reviews datasets,
and enriches metadata with aggregated features from reviews.


"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    DataPreprocessor class for cleaning and preparing book metadata and reviews.
    
    This class handles:
    - Cleaning metadata (duplicates, missing values, text cleaning)
    - Cleaning reviews (duplicates, validation, text cleaning)
    - Enriching metadata with aggregated features from reviews
    - Filtering rare categories
    - Stratified sampling
    - Class balancing
    
    Workflow:
    1. Clean metadata
    2. Clean reviews
    3. Enrich metadata with review features
    4. Filter rare categories
    5. Sample data
    6. Calculate class weights
    
    Attributes:
        min_samples_per_category (int): Minimum samples to keep a category
        sampling_percentage (float): Percentage of data to sample (0.0-1.0)
        balance_strategy (str): Strategy for class balancing
        min_reviews_per_book (int): Minimum reviews to keep a book (optional filter)
    """
    
    def __init__(
        self,
        min_samples_per_category: int = 50,
        sampling_percentage: float = 1.0,
        balance_strategy: str = 'class_weight',
        min_reviews_per_book: int = 0
    ):
        """
        Initialize DataPreprocessor.
        
        Args:
            min_samples_per_category (int): Minimum books per category to keep
            sampling_percentage (float): Percentage of data to use (0.0-1.0)
            balance_strategy (str): 'class_weight', 'undersample', or 'none'
            min_reviews_per_book (int): Minimum reviews per book (0 = no filter)
        """
        self.min_samples_per_category = min_samples_per_category
        self.sampling_percentage = sampling_percentage
        self.balance_strategy = balance_strategy
        self.min_reviews_per_book = min_reviews_per_book
        
        # Statistics tracking
        self.preprocessing_stats = {
            'metadata': {},
            'reviews': {},
            'enrichment': {},
            'filtering': {},
            'sampling': {}
        }
        
        logger.info("DataPreprocessor initialized")
        logger.info(f"  Min samples per category: {min_samples_per_category}")
        logger.info(f"  Sampling percentage: {sampling_percentage*100}%")
        logger.info(f"  Balance strategy: {balance_strategy}")
        logger.info(f"  Min reviews per book: {min_reviews_per_book}")
    
    def clean_text_field(self, text_series: pd.Series) -> pd.Series:
        """
        Clean text field: lowercase, remove HTML, special characters, extra spaces.
        
        Args:
            text_series (pd.Series): Series with text to clean
            
        Returns:
            pd.Series: Cleaned text series
        """
        logger.info(f"Cleaning text field with {len(text_series)} entries...")
        
        def clean_text(text):
            if pd.isna(text):
                return ""
            
            text = str(text)
            
            # Lowercase
            text = text.lower()
            
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Remove URLs
            text = re.sub(r'http\S+|www.\S+', '', text)
            
            # Remove special characters but keep basic punctuation
            text = re.sub(r'[^a-z0-9\s.,!?\'"-]', '', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            return text.strip()
        
        cleaned = text_series.apply(clean_text)
        
        # Log statistics
        non_empty_before = text_series.notna().sum()
        non_empty_after = (cleaned != "").sum()
        
        logger.info(f"Text cleaning complete:")
        logger.info(f"  Non-empty before: {non_empty_before}")
        logger.info(f"  Non-empty after: {non_empty_after}")
        
        return cleaned
    
    def clean_metadata(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean metadata dataset.
        
        Steps:
        1. Remove duplicates by parent_asin
        2. Fill missing descriptions with title + features_text
        3. Clean all text fields
        4. Remove books without category        
        5. Clean numerical fields
        
        Args:
            metadata_df (pd.DataFrame): Raw metadata DataFrame
            
        Returns:
            pd.DataFrame: Cleaned metadata DataFrame
        """
        logger.info("="*70)
        logger.info("CLEANING METADATA")
        logger.info("="*70)
        
        df = metadata_df.copy()
        original_count = len(df)
        
        # Track statistics
        self.preprocessing_stats['metadata']['original_count'] = original_count
        
        # Step 1: Remove duplicates
        logger.info("\n[1/5] Removing duplicates...")
        duplicates_count = df.duplicated(subset=['parent_asin']).sum()
        df = df.drop_duplicates(subset=['parent_asin'], keep='first')
        logger.info(f"  Removed {duplicates_count} duplicates")
        self.preprocessing_stats['metadata']['duplicates_removed'] = duplicates_count
        
        # Step 2: Handle missing descriptions
        logger.info("\n[2/5] Handling missing descriptions...")
        missing_desc_before = df['description'].isna().sum()
        
        # Fill missing descriptions with title + features_text
        def create_fallback_description(row):
            if pd.notna(row['description']) and str(row['description']).strip():
                return row['description']
            
            parts = []
            if pd.notna(row.get('title')):
                parts.append(str(row['title']))
            if pd.notna(row.get('features_text')):
                parts.append(str(row['features_text']))
            if pd.notna(row.get('author_name')):
                parts.append(f"by {row['author_name']}")
            
            return ' '.join(parts) if parts else "No description available"
        
        df['description'] = df.apply(create_fallback_description, axis=1)
        missing_desc_after = df['description'].isna().sum()
        
        logger.info(f"  Missing descriptions before: {missing_desc_before}")
        logger.info(f"  Missing descriptions after: {missing_desc_after}")
        logger.info(f"  Filled: {missing_desc_before - missing_desc_after}")
        self.preprocessing_stats['metadata']['descriptions_filled'] = missing_desc_before - missing_desc_after
        
        # Step 3: Clean text fields
        logger.info("\n[3/5] Cleaning text fields...")
        
        text_columns = ['title', 'description', 'features_text', 'author_name']
        for col in text_columns:
            if col in df.columns:
                df[col] = self.clean_text_field(df[col])
        
        # Step 4: Remove books without category
        #ogger.info("\n[4/5] Removing books without category...")
        #missing_category = df['main_category'].isna().sum()
        #df = df[df['main_category'].notna()]
        #logger.info(f"  Removed {missing_category} books without category")
        #self.preprocessing_stats['metadata']['no_category_removed'] = missing_category

        logger.info("\n[4/5] Removing books without category...")
        missing_category = df['category_level_2_sub'].isna().sum()
        df = df[df['category_level_2_sub'].notna()]
        logger.info(f"  Removed {missing_category} books without category")
        self.preprocessing_stats['metadata']['no_category_removed'] = missing_category

        #logger.info("\n[4/5] grouping small categories...")
        #category_counts = df['category_level_2_sub'].value_counts()
        #small_categories = category_counts[category_counts < 100].index
        
        #df['category_level_2_sub'] = df['category_level_2_sub'].replace(small_categories, 'Other')
        #self.preprocessing_stats['metadata']['small_categories_count'] = len(small_categories)
        #self.preprocessing_stats['metadata']['small_categories_threshold'] = 100
        
        # Step 5: Clean numerical fields
        logger.info("\n[6/5] Cleaning numerical fields...")
        
        # Price: ensure positive
        if 'price_numeric' in df.columns:
            df.loc[df['price_numeric'] < 0, 'price_numeric'] = np.nan
        
        # Page count: ensure positive and reasonable
        if 'page_count' in df.columns:
            df.loc[df['page_count'] < 10, 'page_count'] = np.nan
            df.loc[df['page_count'] > 5000, 'page_count'] = np.nan
        
        # Rating: ensure in valid range [1, 5]
        if 'average_rating' in df.columns:
            df.loc[df['average_rating'] < 1, 'average_rating'] = np.nan
            df.loc[df['average_rating'] > 5, 'average_rating'] = np.nan
        
        # Final statistics
        final_count = len(df)
        self.preprocessing_stats['metadata']['final_count'] = final_count
        self.preprocessing_stats['metadata']['total_removed'] = original_count - final_count
        
        logger.info("\n✓ Metadata cleaning complete")
        logger.info(f"  Original: {original_count:,} books")
        logger.info(f"  Final: {final_count:,} books")
        logger.info(f"  Removed: {original_count - final_count:,} books")
        
        return df
    
    def clean_reviews(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean reviews dataset.
        
        Steps:
        1. Remove duplicates
        2. Validate ratings (1-5)
        3. Clean review text
        4. Handle verified_purchase field
        5. Clean helpful_vote field
        
        Args:
            reviews_df (pd.DataFrame): Raw reviews DataFrame
            
        Returns:
            pd.DataFrame: Cleaned reviews DataFrame
        """
        logger.info("\n" + "="*70)
        logger.info("CLEANING REVIEWS")
        logger.info("="*70)
        
        df = reviews_df.copy()
        original_count = len(df)
        
        # Track statistics
        self.preprocessing_stats['reviews']['original_count'] = original_count
        
        # Step 1: Remove duplicates
        logger.info("\n[1/5] Removing duplicates...")
        duplicates_count = df.duplicated(subset=['parent_asin', 'user_id', 'rating', 'text']).sum()
        df = df.drop_duplicates(subset=['parent_asin', 'user_id', 'rating', 'text'], keep='first')
        logger.info(f"  Removed {duplicates_count} duplicate reviews")
        self.preprocessing_stats['reviews']['duplicates_removed'] = duplicates_count
        
        # Step 2: Validate ratings
        logger.info("\n[2/5] Validating ratings...")
        invalid_ratings = ((df['rating'] < 1) | (df['rating'] > 5)).sum()
        df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]
        logger.info(f"  Removed {invalid_ratings} reviews with invalid ratings")
        self.preprocessing_stats['reviews']['invalid_ratings_removed'] = invalid_ratings
        
        # Step 3: Clean review text
        logger.info("\n[3/5] Cleaning review text...")
        if 'text' in df.columns:
            df['text'] = self.clean_text_field(df['text'])
            # Remove reviews with empty text after cleaning
            empty_text = (df['text'] == "").sum()
            df = df[df['text'] != ""]
            logger.info(f"  Removed {empty_text} reviews with empty text")
            self.preprocessing_stats['reviews']['empty_text_removed'] = empty_text
        
        # Step 4: Handle verified_purchase
        logger.info("\n[4/5] Processing verified_purchase field...")
        if 'verified_purchase' in df.columns:
            # Convert to boolean if needed
            df['verified_purchase'] = df['verified_purchase'].fillna(False)
            df['verified_purchase'] = df['verified_purchase'].astype(bool)
            verified_count = df['verified_purchase'].sum()
            logger.info(f"  Verified purchases: {verified_count:,} ({verified_count/len(df)*100:.1f}%)")
        else:
            logger.info("  verified_purchase column not found, skipping")
        
        # Step 5: Clean helpful_vote
        logger.info("\n[5/5] Cleaning helpful_vote field...")
        if 'helpful_vote' in df.columns:
            # Ensure non-negative
            df['helpful_vote'] = df['helpful_vote'].fillna(0)
            df.loc[df['helpful_vote'] < 0, 'helpful_vote'] = 0
            df['helpful_vote'] = df['helpful_vote'].astype(int)
        else:
            logger.info("  helpful_vote column not found, skipping")
        
        # Final statistics
        final_count = len(df)
        self.preprocessing_stats['reviews']['final_count'] = final_count
        self.preprocessing_stats['reviews']['total_removed'] = original_count - final_count
        
        logger.info("\n✓ Reviews cleaning complete")
        logger.info(f"  Original: {original_count:,} reviews")
        logger.info(f"  Final: {final_count:,} reviews")
        logger.info(f"  Removed: {original_count - final_count:,} reviews")
        
        return df
    
    def enrich_with_reviews(
        self,
        clean_metadata: pd.DataFrame,
        clean_reviews: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Enrich metadata with aggregated features from reviews.
        
        Features added:
        - review_count: Number of reviews per book
        - avg_rating_from_reviews: Average rating from reviews
        - rating_std: Standard deviation of ratings
        - verified_purchase_pct: Percentage of verified purchases
        - helpful_votes_total: Total helpful votes
        - review_text_avg_length: Average review text length
        
        Args:
            clean_metadata (pd.DataFrame): Cleaned metadata
            clean_reviews (pd.DataFrame): Cleaned reviews
            
        Returns:
            pd.DataFrame: Enriched metadata with review features
        """
        logger.info("\n" + "="*70)
        logger.info("ENRICHING METADATA WITH REVIEW FEATURES")
        logger.info("="*70)
        
        logger.info(f"\nInput data:")
        logger.info(f"  Metadata books: {len(clean_metadata):,}")
        logger.info(f"  Total reviews: {len(clean_reviews):,}")
        
        # Calculate text length
        if 'text' in clean_reviews.columns:
            clean_reviews['text_length'] = clean_reviews['text'].str.len()
        
        # Aggregate reviews by book
        logger.info("\nAggregating review features...")
        
        agg_dict = {
            'rating': ['count', 'mean', 'std']
        }
        
        if 'verified_purchase' in clean_reviews.columns:
            agg_dict['verified_purchase'] = lambda x: (x == True).sum() / len(x) * 100
        
        if 'helpful_vote' in clean_reviews.columns:
            agg_dict['helpful_vote'] = 'sum'
        
        if 'text_length' in clean_reviews.columns:
            agg_dict['text_length'] = 'mean'
        
        reviews_agg = clean_reviews.groupby('parent_asin').agg(agg_dict).reset_index()
        
        # Flatten column names
        reviews_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                               for col in reviews_agg.columns.values]
        
        # Rename columns for clarity
        column_mapping = {
            'rating_count': 'review_count',
            'rating_mean': 'avg_rating_from_reviews',
            'rating_std': 'rating_std',
            'verified_purchase_<lambda>': 'verified_purchase_pct',
            'helpful_vote_sum': 'helpful_votes_total',
            'text_length_mean': 'review_text_avg_length'
        }
        
        reviews_agg = reviews_agg.rename(columns=column_mapping)
        
        logger.info(f"\nAggregated features for {len(reviews_agg):,} books")
        logger.info(f"Features created: {list(reviews_agg.columns)}")
        
        # Merge with metadata
        logger.info("\nMerging with metadata...")
        enriched = clean_metadata.merge(
            reviews_agg,
            on='parent_asin',
            how='left'
        )
        
        # Fill NaN for books without reviews
        logger.info("\nHandling books without reviews...")
        books_without_reviews = enriched['review_count'].isna().sum()
        
        fill_values = {
            'review_count': 0,
            'avg_rating_from_reviews': 3.0,  # Neutral rating
            'rating_std': 0.0,
            'verified_purchase_pct': 0.0,
            'helpful_votes_total': 0,
            'review_text_avg_length': 0.0
        }
        
        for col, val in fill_values.items():
            if col in enriched.columns:
                enriched[col].fillna(val, inplace=True)
        
        logger.info(f"  Books without reviews: {books_without_reviews:,}")
        logger.info(f"  Filled with default values")
        
        # Statistics
        self.preprocessing_stats['enrichment']['books_with_reviews'] = len(reviews_agg)
        self.preprocessing_stats['enrichment']['books_without_reviews'] = books_without_reviews
        self.preprocessing_stats['enrichment']['features_added'] = len(reviews_agg.columns) - 1
        
        # Summary statistics
        logger.info("\n✓ Enrichment complete")
        logger.info(f"\nReview features summary:")
        if 'review_count' in enriched.columns:
            logger.info(f"  Avg reviews per book: {enriched['review_count'].mean():.1f}")
            logger.info(f"  Max reviews per book: {enriched['review_count'].max():.0f}")
        if 'avg_rating_from_reviews' in enriched.columns:
            logger.info(f"  Avg rating: {enriched['avg_rating_from_reviews'].mean():.2f}")
        
        return enriched
    
    def filter_rare_categories(
        self,
        df: pd.DataFrame,
        category_column: str #= 'main_category'
    ) -> pd.DataFrame:
        """
        Filter out categories with too few samples.
        
        Args:
            df (pd.DataFrame): DataFrame to filter
            category_column (str): Column name with categories
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        logger.info("\n" + "="*70)
        logger.info("FILTERING RARE CATEGORIES")
        logger.info("="*70)
        
        original_count = len(df)
        original_categories = df[category_column].nunique()
        
        logger.info(f"\nBefore filtering:")
        logger.info(f"  Total books: {original_count:,}")
        logger.info(f"  Total categories: {original_categories}")
        
        # Count samples per category
        category_counts = df[category_column].value_counts()
        
        # Identify categories to keep
        valid_categories = category_counts[
            category_counts >= self.min_samples_per_category
        ].index
        
        # Filter DataFrame
        df_filtered = df[df[category_column].isin(valid_categories)].copy()
        
        final_count = len(df_filtered)
        final_categories = df_filtered[category_column].nunique()
        removed_categories = original_categories - final_categories
        
        logger.info(f"\nAfter filtering (min {self.min_samples_per_category} samples):")
        logger.info(f"  Total books: {final_count:,}")
        logger.info(f"  Total categories: {final_categories}")
        logger.info(f"  Removed categories: {removed_categories}")
        logger.info(f"  Removed books: {original_count - final_count:,}")
        
        # Show removed categories
        if removed_categories > 0:
            removed_cats = set(df[category_column].unique()) - set(valid_categories)
            logger.info(f"\nRemoved categories: {list(removed_cats)[:10]}")  # Show first 10
        
        # Statistics
        self.preprocessing_stats['filtering']['original_count'] = original_count
        self.preprocessing_stats['filtering']['final_count'] = final_count
        self.preprocessing_stats['filtering']['original_categories'] = original_categories
        self.preprocessing_stats['filtering']['final_categories'] = final_categories
        self.preprocessing_stats['filtering']['removed_categories'] = removed_categories
        
        return df_filtered
    
    def filter_by_review_count(
        self,
        df: pd.DataFrame,
        min_reviews: int = None
    ) -> pd.DataFrame:
        """
        Filter books by minimum number of reviews.
        
        Args:
            df (pd.DataFrame): DataFrame with 'review_count' column
            min_reviews (int): Minimum reviews required (uses self.min_reviews_per_book if None)
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        
        logger.info("\n" + "="*70)
        logger.info("FILTERING BY REVIEW COUNT")
        logger.info("="*70)
        
        if min_reviews is None:
            min_reviews = self.min_reviews_per_book
        
        if min_reviews == 0:
            logger.info("\nSkipping review count filter (min_reviews = 0)")
            return df
        
        if 'review_count' not in df.columns:
            logger.warning("\nreview_count column not found, skipping filter")
            return df
        
        
        logger.info(f"\nFiltering books with less than {min_reviews} reviews...")
        original_count = len(df)
        
        df_filtered = df[df['review_count'] >= min_reviews].copy()
        
        final_count = len(df_filtered)
        removed_count = original_count - final_count
        
        logger.info(f"  Books before: {original_count:,}")
        logger.info(f"  Books after: {final_count:,}")
        logger.info(f"  Removed: {removed_count:,} books")
        
        return df_filtered
    
    def sample_data(
        self,
        df: pd.DataFrame,
        category_column: str , #= 'main_category',
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Perform stratified sampling to reduce dataset size while maintaining category distribution.
        
        Stratified sampling is a probability sampling technique used to obtain a representative sample 
        from a heterogeneous population. It involves dividing the total population into non-overlapping, 
        homogeneous subgroups called strata and then selecting a random sample from each stratum. 
        
        The process for stratified sampling includes several key steps: 
            Define the population and strata: Identify the relevant characteristics (such as age, gender, education level, or income) that will be used to divide the population into mutually exclusive strata.
            Separate the population: Assign every member of the population to exactly one stratum.
            Determine the sample size: Decide on the total sample size needed for the study.
            Sample from each stratum: Use a random sampling method (like simple random sampling or systematic sampling) to select members from each stratum until the required sample size is met. 

        Args:
            df (pd.DataFrame): DataFrame to sample
            category_column (str): Column to stratify by
            random_state (int): Random seed for reproducibility
            
        Returns:
            pd.DataFrame: Sampled DataFrame
        """
        if self.sampling_percentage >= 1.0:
            logger.info("\nSkipping sampling (sampling_percentage = 100%)")
            return df
        
        logger.info("\n" + "="*70)
        logger.info("STRATIFIED SAMPLING")
        logger.info("="*70)
        
        original_count = len(df)
        target_count = int(original_count * self.sampling_percentage)
        
        logger.info(f"\nSampling {self.sampling_percentage*100}% of data")
        logger.info(f"  Original: {original_count:,} books")
        logger.info(f"  Target: {target_count:,} books")
        
        # Stratified sampling
        df_sampled = df.groupby(category_column, group_keys=False).apply(
            lambda x: x.sample(
                frac=self.sampling_percentage,
                random_state=random_state
            )
        ).reset_index(drop=True)
        
        final_count = len(df_sampled)
        
        logger.info(f"  Sampled: {final_count:,} books")
        
        # Verify distribution is maintained
        logger.info("\nVerifying category distribution:")
        original_dist = df[category_column].value_counts(normalize=True)
        sampled_dist = df_sampled[category_column].value_counts(normalize=True)
        
        # Show top 5 categories
        for cat in original_dist.head(5).index:
            orig_pct = original_dist[cat] * 100
            samp_pct = sampled_dist[cat] * 100
            logger.info(f"  {cat}: {orig_pct:.1f}% → {samp_pct:.1f}%")
        
        # Statistics
        self.preprocessing_stats['sampling']['original_count'] = original_count
        self.preprocessing_stats['sampling']['sampled_count'] = final_count
        self.preprocessing_stats['sampling']['sampling_percentage'] = self.sampling_percentage
        
        return df_sampled
    
    def calculate_class_weights(
        self,
        df: pd.DataFrame,
        category_column: str , #= 'main_category'
    ) -> Dict[str, float]:
        """
        Calculate class weights for handling imbalanced classes.
        
        Args:
            df (pd.DataFrame): DataFrame with categories
            category_column (str): Column with categories
            
        Returns:
            Dict[str, float]: Dictionary mapping category to weight
        """
        logger.info("\n" + "="*70)
        logger.info("CALCULATING CLASS WEIGHTS")
        logger.info("="*70)
        
        if self.balance_strategy == 'none':
            logger.info("\nBalance strategy: none - Skipping class weights")
            return {}
        
        from sklearn.utils.class_weight import compute_class_weight
        
        categories = df[category_column].values
        unique_categories = np.unique(categories)
        
        # Compute weights
        weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_categories,
            y=categories
        )
        
        class_weights = dict(zip(unique_categories, weights))
        
        logger.info(f"\nBalance strategy: {self.balance_strategy}")
        logger.info(f"Computed weights for {len(class_weights)} categories")
        
        # Show top 5 categories with highest weights (minority classes)
        sorted_weights = sorted(class_weights.items(), key=lambda x: x[1], reverse=True)
        logger.info("\nTop 5 minority classes (highest weights):")
        for cat, weight in sorted_weights[:5]:
            count = (categories == cat).sum()
            logger.info(f"  {cat}: weight={weight:.2f}, samples={count}")
        
        return class_weights
    
    def get_preprocessing_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive preprocessing report.
        
        Returns:
            Dict[str, Any]: Dictionary with preprocessing statistics
        """
        return self.preprocessing_stats.copy()
    
    def print_preprocessing_summary(self):
        """Print formatted preprocessing summary."""
        stats = self.preprocessing_stats
        
        print("\n" + "="*70)
        print("PREPROCESSING SUMMARY")
        print("="*70)
        
        # Metadata
        if stats['metadata']:
            print("\n📚 METADATA CLEANING:")
            print(f"  Original books: {stats['metadata'].get('original_count', 0):,}")
            print(f"  Final books: {stats['metadata'].get('final_count', 0):,}")
            print(f"  Duplicates removed: {stats['metadata'].get('duplicates_removed', 0)}")
            print(f"  Descriptions filled: {stats['metadata'].get('descriptions_filled', 0)}")
            print(f"  No category removed: {stats['metadata'].get('no_category_removed', 0)}")
        
        # Reviews
        if stats['reviews']:
            print("\n💬 REVIEWS CLEANING:")
            print(f"  Original reviews: {stats['reviews'].get('original_count', 0):,}")
            print(f"  Final reviews: {stats['reviews'].get('final_count', 0):,}")
            print(f"  Duplicates removed: {stats['reviews'].get('duplicates_removed', 0)}")
            print(f"  Invalid ratings removed: {stats['reviews'].get('invalid_ratings_removed', 0)}")
            print(f"  Empty text removed: {stats['reviews'].get('empty_text_removed', 0)}")
        
        # Enrichment
        if stats['enrichment']:
            print("\n⭐ ENRICHMENT:")
            print(f"  Books with reviews: {stats['enrichment'].get('books_with_reviews', 0):,}")
            print(f"  Books without reviews: {stats['enrichment'].get('books_without_reviews', 0):,}")
            print(f"  Features added: {stats['enrichment'].get('features_added', 0)}")
        
        # Filtering
        if stats['filtering']:
            print("\n🔍 FILTERING:")
            print(f"  Original books: {stats['filtering'].get('original_count', 0):,}")
            print(f"  Final books: {stats['filtering'].get('final_count', 0):,}")
            print(f"  Original categories: {stats['filtering'].get('original_categories', 0)}")
            print(f"  Final categories: {stats['filtering'].get('final_categories', 0)}")
            print(f"  Removed categories: {stats['filtering'].get('removed_categories', 0)}")
        
        # Sampling
        if stats['sampling']:
            print("\n📊 SAMPLING:")
            print(f"  Original: {stats['sampling'].get('original_count', 0):,}")
            print(f"  Sampled: {stats['sampling'].get('sampled_count', 0):,}")
            print(f"  Percentage: {stats['sampling'].get('sampling_percentage', 0)*100:.0f}%")
        
        print("\n" + "="*70)



    def save_preprocessed_data(
        self,
        df: pd.DataFrame,
        output_path: str,
        save_class_weights: bool = True,
        class_weights: Dict[str, float] = None,
        weights_path: str = None
    ):
        """
        Save preprocessed data to CSV file.
        
        Args:
            df: Preprocessed DataFrame to save
            output_path: Path where to save the CSV file
            save_class_weights: Whether to save class weights separately
            class_weights: Class weights dictionary (optional)
        """
        #df.to_csv(output_path, index=False)
        
        if self.ensure_directory(output_path):
            df.to_csv(output_path, index=False)
            print(f"✓ CSV saved: {output_path}")
        else:
            print(f"✗ Failed to save CSV: {output_path}")
            return False
       
    
        #if save_class_weights and class_weights:
        #    with open(weights_path, "w") as f:
        #        json.dump(class_weights, f)
        
        if self.ensure_directory(weights_path):
                with open(weights_path, "w") as f:
                    json.dump(class_weights, f, indent=4)
                print(f"✓ Class weights saved: {weights_path}")
        else:
                print(f"✗ Failed to save class weights: {weights_path}")
                
    
    def ensure_directory(self, file_path):
        """
        Ensures that the directory for a file exists.
        If it doesn't exist, creates it.
        """
        try:
            directory = Path(file_path).parent
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ Directory ensured: {directory}")
            return True
        except Exception as e:
            print(f"✗ Error creating directory: {e}")
            return False