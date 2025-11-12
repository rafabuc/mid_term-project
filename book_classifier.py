"""
SIMPLE BOOK CLASSIFICATION INFERENCE 
=========================================================
Works with FeatureExtractor class.


"""
import importlib


import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from feature_extractor import FeatureExtractor


class BookClassifier:
    """Book classification pipeline."""
    
    def __init__(self, model_path, feature_extractor_dir):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to trained model
        """
        print("Loading model...")
        
        # Load model
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}\n"
                "Make sure you have trained the model first!"
            )
        
        self.model = joblib.load(model_path)
        print(f"✓ Model loaded from {model_path}")
        
        # Load feature extractor (uses custom load method)
        #feature_extractor_dir = 'models/feature_extractor'
        if not Path(feature_extractor_dir).exists():
            raise FileNotFoundError(
                f"Feature extractor not found at {feature_extractor_dir}\n"            
            )
        
        self.feature_extractor = FeatureExtractor.load(feature_extractor_dir)
        print(f"✓ Feature extractor loaded")
        
        # Load class names from feature extractor
        self.class_names = self.feature_extractor.target_label_encoder.classes_
        print(f"✓ Loaded {len(self.class_names)} classes")
        
        print("\n✓ Classifier ready!\n")
    
    def predict(self, title, description, price, rating, rating_count, book_format):
        """
        Predict the category of a book.
        
        Args:
            title: Book title (str)
            description: Book description (str)
            price: Price in USD (float)
            rating: Average rating 1-5 (float)
            rating_count: Number of ratings (int)
            book_format: Format like "Paperback", "Kindle", etc (str)
        
        Returns:
            dict with:
                - predicted_class: The predicted category
                - confidence: Confidence score 0-1
                - top_3: Top 3 predictions with probabilities
        """
        # Create DataFrame with column names
        # (matching what FeatureExtractor expects)
        book_data = pd.DataFrame([{
            'title': title,
            'description': description,
            'price': price,
            'average_rating': rating,
            'review_count': rating_count,
            'format': book_format,
              # Columnas que estaban en training (con defaults razonables)
            'rating_std': 0.5,
            'verified_purchase_pct': 0.75,
            'helpful_votes_total': 0
        }])
        
        # Extract features using inference method
        # (doesn't need target column)
        features = self.feature_extractor.transform_inference(book_data)
        
        # Predict
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Get predicted class and confidence
        predicted_class = self.class_names[prediction]
        confidence = probabilities[prediction]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        top_3 = [
            {
                'class': self.class_names[i],
                'probability': float(probabilities[i])
            }
            for i in top_3_idx
        ]
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'top_3': top_3
        }
    
    def predict_batch(self, books_list):
        """
        Predict categories for multiple books.
        
        Args:
            books_list: List of dicts, each with keys:
                        title, description, price, rating, rating_count, format
        
        Returns:
            List of predictions
        """
        # Convert to DataFrame with correct column names
        books_df = pd.DataFrame([
            {
                'title': book['title'],
                'description': book['description'],
                'price': book['price'],
                'average_rating': book['rating'],
                'review_count': book['rating_count'],
                'format': book['format'],
                # Columnas que estaban en training (con defaults razonables)
                'rating_std': 0.5,
                'verified_purchase_pct': 0.8,
                'helpful_votes_total': 10,             
            }
            for book in books_list
        ])
        
        # Extract features
        features = self.feature_extractor.transform_inference(books_df)
        
        # Predict
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        # Format results
        results = []
        for pred, probs in zip(predictions, probabilities):
            results.append({
                'predicted_class': self.class_names[pred],
                'confidence': float(probs[pred])
            })
        
        return results


def example_single_prediction(model_path, feature_extractor_dir):
    """Example: Predict a single book."""
    print("="*80)
    print("EXAMPLE 1: Single Book Prediction")
    print("="*80)
    
    # Load classifier
    classifier = BookClassifier(model_path, feature_extractor_dir)
    
    # Predict one book
    result = classifier.predict(
        title="The Da Vinci Code",
        description="A murder in the Louvre Museum leads Harvard symbologist Robert Langdon on a trail of clues.",
        price=14.99,
        rating=4.2,
        rating_count=8432,
        book_format="Paperback"
    )
    
    # Show results
    print(f"\nPredicted Category: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print("\nTop 3 Predictions:")
    for i, pred in enumerate(result['top_3'], 1):
        print(f"  {i}. {pred['class']}: {pred['probability']:.1%}")


def example_batch_prediction(model_path, feature_extractor_dir):
    """Example: Predict multiple books."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Prediction")
    print("="*80)
    
    # Load classifier
    classifier = BookClassifier(model_path, feature_extractor_dir)
    
    # Multiple books
    books = [
        {
            'title': 'Python Crash Course',
            'description': 'A hands-on introduction to programming with Python',
            'price': 29.99,
            'rating': 4.6,
            'rating_count': 2341,
            'format': 'Paperback'
        },
        {
            'title': 'The Great Gatsby',
            'description': 'A classic American novel set in the Jazz Age',
            'price': 10.99,
            'rating': 3.9,
            'rating_count': 15234,
            'format': 'Paperback'
        },
        {
            'title': 'Healthy Cooking Made Easy',
            'description': '200 delicious and nutritious recipes',
            'price': 24.99,
            'rating': 4.4,
            'rating_count': 543,
            'format': 'Hardcover'
        }
    ]
    
    # Predict all
    results = classifier.predict_batch(books)
    
    # Show results
    print("\nResults:")
    for book, result in zip(books, results):
        print(f"\n  {book['title']}")
        print(f"    → {result['predicted_class']} ({result['confidence']:.1%})")
