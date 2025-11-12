"""
CLASS IMBALANCE ANALYSIS AND CORRECTION
========================================
This script analyzes class distribution and provides
strategies to improve F1 score.

Author: Rafael
Date: 2025-11-09
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

from constants import Y_TEST_FILE, Y_TRAIN_FILE, CLASS_NAMES_FILE, CLASS_WEIGHTS_FILE, CLASS_AGRESSIVE_WEIGHTS_FILE
def analyze_class_distribution():
    """Analyzes class distribution in the dataset."""
    print("="*80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Load data
    y_train = np.load(Y_TRAIN_FILE)
    y_test = np.load(Y_TEST_FILE)
    
    with open(CLASS_NAMES_FILE) as f:
        class_info = json.load(f)
        class_names = class_info['class_names']
    
    # Count samples per class
    unique, counts = np.unique(y_train, return_counts=True)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Class': [class_names[i] for i in unique],
        'Samples': counts,
        'Percentage': (counts / len(y_train) * 100).round(2)
    })
    df = df.sort_values('Samples', ascending=False)
    
    # Analysis
    total = len(y_train)
    print(f"\nTotal training samples: {total:,}")
    print(f"Number of classes: {len(unique)}")
    print(f"\nDistribution:")
    print(df.to_string(index=False))
    
    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Mean samples per class: {counts.mean():.0f}")
    print(f"Median samples per class: {np.median(counts):.0f}")
    print(f"Std deviation: {counts.std():.0f}")
    print(f"Min samples: {counts.min()}")
    print(f"Max samples: {counts.max()}")
    print(f"Ratio (max/min): {counts.max()/counts.min():.1f}x")
    
    # Identify problems
    print("\n" + "="*80)
    print("IDENTIFIED PROBLEMS")
    print("="*80)
    
    # Very small classes (< 1% of dataset)
    small_threshold = total * 0.01
    small_classes = df[df['Samples'] < small_threshold]
    
    if len(small_classes) > 0:
        print(f"\n⚠️  VERY SMALL CLASSES (< 1% of dataset = < {small_threshold:.0f} samples):")
        for _, row in small_classes.iterrows():
            print(f"  - {row['Class']}: {row['Samples']} samples ({row['Percentage']:.2f}%)")
    
    # Large classes (> 10% of dataset)
    large_threshold = total * 0.10
    large_classes = df[df['Samples'] > large_threshold]
    
    if len(large_classes) > 0:
        print(f"\n⚠️  DOMINANT CLASSES (> 10% of dataset = > {large_threshold:.0f} samples):")
        for _, row in large_classes.iterrows():
            print(f"  - {row['Class']}: {row['Samples']} samples ({row['Percentage']:.2f}%)")
    
    # Balance ratio
    imbalance_ratio = counts.max() / counts.min()
    print(f"\n📊 Imbalance Ratio: {imbalance_ratio:.1f}x")
    
    if imbalance_ratio > 50:
        print("   🚨 SEVERE (>50x) - Critical problem")
    elif imbalance_ratio > 20:
        print("   ⚠️  HIGH (20-50x) - Important problem")
    elif imbalance_ratio > 10:
        print("   ⚠️  MODERATE (10-20x) - Needs attention")
    else:
        print("   ✅ ACCEPTABLE (<10x)")
    
    return df, small_classes, large_classes


def calculate_recommended_weights(df):
    """Calculates more aggressive recommended class_weights."""
    print("\n" + "="*80)
    print("RECOMMENDED CLASS WEIGHTS")
    print("="*80)
    
    # Load current class_weights if they exist
    try:
        with open(CLASS_WEIGHTS_FILE) as f:
            current_weights = json.load(f)
            current_weights = {int(k): v for k, v in current_weights.items()}
        print("\n✓ Current class weights loaded")
    except:
        current_weights = None
        print("\n⚠️ No current class_weights found")
    
    # Calculate new weights (more aggressive)
    total = df['Samples'].sum()
    n_classes = len(df)
    
    # Strategy: Inversely proportional to the square of sample count
    # This gives much more weight to minority classes
    df['Recommended_Weight'] = (total / (n_classes * df['Samples'])) ** 1.2
    
    # Normalize so average weight is 1.0
    df['Recommended_Weight'] = df['Recommended_Weight'] / df['Recommended_Weight'].mean()
    
    print("\nTop 10 classes that need MORE weight:")
    top_weight = df.nlargest(10, 'Recommended_Weight')[['Class', 'Samples', 'Recommended_Weight']]
    print(top_weight.to_string(index=False))
    
    print("\nTop 10 classes that need LESS weight:")
    bottom_weight = df.nsmallest(10, 'Recommended_Weight')[['Class', 'Samples', 'Recommended_Weight']]
    print(bottom_weight.to_string(index=False))
    
    # Create weights dictionary
    with open(CLASS_NAMES_FILE) as f:
        class_info = json.load(f)
        class_names = class_info['class_names']
    
    new_weights = {}
    for idx, row in df.iterrows():
        class_idx = class_names.index(row['Class'])
        new_weights[class_idx] = float(row['Recommended_Weight'])
    
    # Save to current directory (avoid permission issues)
    output_path = CLASS_AGRESSIVE_WEIGHTS_FILE
    
    #try:
    with open(output_path, 'w') as f:
            json.dump(new_weights, f, indent=2)
            print(f"\n✓ New class weights saved to: {output_path}")
    #except PermissionError:
        # If fails, print to console
    #    print("\n⚠️ Could not save file due to permissions")
    #    print("\n📋 COPY THIS JSON (use it in your code):")
    #    print("="*80)
    #    print(json.dumps(new_weights, indent=2))
    #    print("="*80)
    
    print("\nTo use these weights:")
    print("  1. Copy the generated file to your project directory")
    print("  2. In run_step4_training.py, load this file instead of the original")
    print("  3. Or copy and paste the JSON content directly into your code")
    
    return new_weights


def generate_recommendations(df, small_classes, large_classes):
    """Generates specific recommendations."""
    print("\n" + "="*80)
    print("SPECIFIC RECOMMENDATIONS")
    print("="*80)
    
    print("\n🎯 IMMEDIATE ACTION:")
    print("─" * 80)
    
    print("\n1️⃣  USE AGGRESSIVE CLASS WEIGHTS")
    print("   Current weights are not sufficient for very small classes.")
    print("   → Use: class_weights_aggressive.json (generated above)")
    print("   → Expected impact: +5-10% F1 on minority classes")
    
    if len(small_classes) > 0:
        print("\n2️⃣  VERY SMALL CLASSES - Consider:")
        for _, row in small_classes.iterrows():
            print(f"\n   {row['Class']} ({row['Samples']} samples):")
            if row['Samples'] < 20:
                print(f"      ❌ CRITICAL - Too few samples to learn")
                print(f"      → Option A: Remove this class from model")
                print(f"      → Option B: Merge with similar class")
                print(f"      → Option C: Get more data")
            else:
                print(f"      ⚠️  Few samples")
                print(f"      → Use aggressive class_weights")
                print(f"      → Or apply SMOTE oversampling")
    
    if len(large_classes) > 0:
        print("\n3️⃣  DOMINANT CLASSES - Consider:")
        for _, row in large_classes.iterrows():
            print(f"\n   {row['Class']} ({row['Samples']} samples, {row['Percentage']:.1f}%):")
            print(f"      ⚠️  Dominates the dataset")
            print(f"      → Undersample to max 10% of dataset")
            print(f"      → Or use class_weights to reduce influence")
    


