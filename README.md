# 📚 Book Category Classification System

## 🎯 Problem Statement

**Business Challenge:** E-commerce platforms and digital libraries need to automatically categorize thousands of books into appropriate categories to improve user experience, search functionality, and product recommendations.

**Manual categorization is:**
- ⏰ Time-consuming (minutes per book)
- 💰 Expensive (requires domain experts)
- 📈 Not scalable (thousands of new books daily)
- ❌ Inconsistent (human errors and biases)

**This Solution:**
An automated machine learning system that predicts book categories using only metadata (title, description, price, ratings) in milliseconds, enabling:
- 🚀 **Instant categorization** of new books
- 💡 **Improved search** and discoverability
- 🎯 **Better recommendations** for users
- 💸 **Cost reduction** of 95% vs manual process

---

## 📊 Project Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│              COMPLETE ML PIPELINE: DATA TO DEPLOYMENT                │
└─────────────────────────────────────────────────────────────────────┘

 📥 DATA COLLECTION          ⚙️ PROCESSING              🤖 MODELING
 ═════════════════          ════════════════           ═════════════
 
 Amazon Books Data          Clean & Enrich             XGBoost Classifier
 ├─ Metadata (20k)     →    ├─ Text cleaning      →   ├─ TF-IDF Features
 ├─ Reviews (719k)          ├─ Feature eng.           ├─ Hyperparameter tuning
 └─ Categories (50)         └─ Sampling               └─ Cross-validation
        │                          │                          │
        ▼                          ▼                          ▼
 ✓ 20,000 books           ✓ 6,681 books             ✓ F1: 55.58%
 ✓ 719k reviews           ✓ 28 categories           ✓ Accuracy: 58.49%
 ✓ 28 features            ✓ Balanced classes        ✓ Model size: ~50MB
 
 
 📈 EVALUATION              🚀 DEPLOYMENT              🌐 API
 ═════════════             ═════════════              ═══════
 
 Performance Metrics        Docker Container           FastAPI Service
 ├─ Confusion Matrix   →   ├─ Multi-stage build  →   ├─ POST /predict
 ├─ Per-class F1           ├─ Python 3.13            ├─ Top-3 predictions
 └─ Error analysis         └─ Health checks          └─ <100ms response
        │                          │                          │
        ▼                          ▼                          ▼
 ✓ 55.58% Weighted F1     ✓ Ready for prod          ✓ Production-ready
 ✓ 28/28 classes          ✓ Kubernetes ready        
 ✓ Visual reports         ✓ Scalable                ✓ Easy integration
```

---

## 🏗️ Architecture & Pipeline

### High-Level Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Raw Data  │ ──> │ Preprocessing│ ──> │   Training  │ ──> │  Deployment  │
│             │     │              │     │             │     │              │
│ • Metadata  │     │ • Cleaning   │     │ • XGBoost   │     │ • Docker API │
│ • Reviews   │     │ • Enrichment │     │             │     │ • Kubernetes │
│ • 28 classes│     │ • Sampling   │     │ • Evaluation│     │  │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
```

### Detailed Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DETAILED ML PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: DATA LOADING & VALIDATION
├─ Load metadata (20,000 books)
├─ Load reviews (719,287 reviews)
├─ Validate schema (28 columns)
└─ Initial exploration
   Output: ✓ Raw datasets loaded

STEP 2: DATA PREPROCESSING
├─ 2.1: Clean metadata
│   ├─ Handle missing values
│   ├─ Remove duplicates
│   └─ Clean text fields
│
├─ 2.2: Clean reviews
│   ├─ Remove empty reviews
│   ├─ Calculate review stats
│   └─ Aggregate by book
│
├─ 2.3: Enrich metadata with reviews
│   ├─ Merge review stats
│   ├─ Create new features:
│   │   • avg_rating_from_reviews
│   │   • review_count
│   │   • rating_std
│   │   • verified_purchase_pct
│   │   • helpful_votes_total
│   └─ Fill missing data
│
├─ 2.4: Filter rare categories
│   └─ Remove categories with <100 samples
│   └─ 50 categories → 28 categories
│
├─ 2.5: Filter by review count
│   └─ Keep books with ≥5 reviews
│
├─ 2.6: Stratified sampling
│   └─ Balance class distribution (50% per class)
│
└─ 2.7: Calculate class weights
    └─ Handle imbalanced classes
   Output: ✓ 6,681 clean samples (19,612 before sampling)

STEP 3: FEATURE EXTRACTION
├─ Text Features (TF-IDF)
│   ├─ Vectorize: title + description
│   ├─ Max features: 8,000
│   ├─ N-grams: (1, 3)
│   └─ Output: Sparse matrix (6,681 × 8,000)
│
├─ Numerical Features
│   ├─ Price, ratings, review count
│   ├─ StandardScaler normalization
│   └─ Output: Dense matrix (6,681 × 6)
│
├─ Categorical Features
│   ├─ Format, language
│   ├─ One-hot encoding
│   └─ Output: Sparse matrix (6,681 × 3)
│
├─ Engineered Features
│   ├─ Rating features, text features
│   └─ Output: Dense matrix (6,681 × 5)
│
└─ Feature Combination
    └─ Concatenate: Text + Numerical + Categorical + Engineered
   Output: ✓ Feature matrix (6,681 × 8,014)

STEP 4: MODEL TRAINING
├─ Train/Test Split (Stratified)
│   ├─ Train: 80% (5,344 samples)
│   └─ Test: 20% (1,337 samples)
│
├─ XGBoost Classifier
│   ├─ Hyperparameters:
│   │   • max_depth: 8
│   │   • learning_rate: 0.1
│   │   • n_estimators: 200
│   │   • subsample: 0.8
│   └─ Training time: ~35 minutes
│
└─ Cross-validation (2-fold)
    └─ Ensure generalization
   Output: ✓ Trained model saved
   Time: ~20 minutes

STEP 5: MODEL EVALUATION
├─ Test Set Performance
│   ├─ Accuracy: 58.49%
│   ├─ Weighted F1: 55.58%
│   ├─ Precision (weighted): 58.16%
│   └─ Recall (weighted): 58.49%
│
├─ Confusion Matrix
│   └─ Visual analysis of errors
│
└─ Error Analysis
    ├─ Common misclassifications
    └─ Category similarity patterns
   Output: ✓ Comprehensive evaluation report

STEP 6: MODEL PERSISTENCE
├─ Save trained model (xgboost_model.pkl)
├─ Save feature extractors
│   ├─ TF-IDF vectorizer
│   ├─ StandardScaler
│   ├─ Label encoders
│   └─ Target label encoder
└─ Save metadata (JSON)
   Output: ✓ Model artifacts ready for deployment
   Location: ./artifacts_v2/models/

STEP 7: DEPLOYMENT
├─ Docker containerization
├─ FastAPI REST API
├─ Kubernetes manifests
└─ Monitoring setup
   Output: ✓ Production-ready service
```

---

## 📈 Results & Metrics

### Model Performance

```
╔══════════════════════════════════════════════════════════╗
║              FINAL MODEL PERFORMANCE (v2)                ║
╠══════════════════════════════════════════════════════════╣
║  Metric                    │  Score                      ║
╠════════════════════════════╪═════════════════════════════╣
║  Accuracy                  │  58.49%                     ║
║  Weighted F1-Score         │  55.58%                     ║
║  Precision (Weighted)      │  58.16%                     ║
║  Recall (Weighted)         │  58.49%                     ║
╠════════════════════════════╧═════════════════════════════╣
║  Training Time             │  ~30 minutes                ║
║  Inference Time            │  <100ms per book            ║
║  Model Size                │  ~50MB                      ║
║  Classes Predicted         │  28 categories              ║
╚══════════════════════════════════════════════════════════╝
```

### Dataset Statistics

```
Original Dataset:
├─ Books: 20,000
├─ Reviews: 719,287
├─ Categories: 50 (initial)
└─ Features: 28 metadata columns

After Preprocessing:
├─ Books (before sampling): 19,612
├─ Categories (after filtering): 28
├─ Removed categories: 22 (with <100 samples)
├─ Books (after sampling): 6,681
└─ Avg reviews per book: ~36

Train/Test Split:
├─ Training samples: 5,344 (80%)
├─ Test samples: 1,337 (20%)
└─ Split strategy: Stratified

Final Feature Matrix:
├─ Text features (TF-IDF): 8,000
├─ Numerical features: 6
├─ Categorical features: 3
├─ Engineered features: 5
└─ Total dimensions: 8,014
```

### Categories Distribution (Training Set)

```
Top 5 Largest Classes:
1. Children's Books           - 595 samples (11.1%)
2. Literature & Fiction       - 510 samples (9.5%)
3. Mystery, Thriller & Suspense - 407 samples (7.6%)
4. Christian Books & Bibles   - 290 samples (5.4%)
5. Biographies & Memoirs      - 280 samples (5.2%)

Smallest Classes:
1. Education & Teaching       - 29 samples (0.5%)
2. Computers & Technology     - 38 samples (0.7%)
3. Engineering & Transportation - 43 samples (0.8%)
4. Medical Books              - 28 samples (0.5%)
```

---

## 🔄 Model Evolution

### ⚠️ Version 1 (v1) - Logistic Regression

**Note:** A first version using Logistic Regression was developed but yielded unsatisfactory results:

```
╔══════════════════════════════════════════════════════════╗
║              MODEL v1: LOGISTIC REGRESSION               ║
╠══════════════════════════════════════════════════════════╣
║  Accuracy                  │  ~20%                       ║
║  Macro F1-Score            │  ~20%                       ║
║  Training Time             │  ~40 minutes                ║
╠══════════════════════════════════════════════════════════╣
║  Status: DEPRECATED - Poor performance on minority       ║
║  classes and complex category boundaries                 ║
╚══════════════════════════════════════════════════════════╝
```

**Why it failed:**
- ❌ Linear model struggled with non-linear text patterns
- ❌ Poor performance on minority classes (<50% F1)
- ❌ Could not capture complex category relationships
- ❌ Insufficient for production use

### ⚠️ Version 1 (v1) - XgBoost

**Note:** A first version using XGBoost was developed with folllow results:

```
╔══════════════════════════════════════════════════════════╗
║              MODEL v1: XGBoost               ║
╠══════════════════════════════════════════════════════════╣
║  Accuracy                  │  ~56%                       ║
║  Macro F1-Score            │  ~54%                       ║
║  Training Time             │  ~20 minutes                ║
╠══════════════════════════════════════════════════════════╣
║  Status: DEPRECATED - Poor performance on minority       ║
║  classes and complex category boundaries                 ║
╚══════════════════════════════════════════════════════════╝
```


**The v1 code is preserved in the repository for reference and comparison purposes.**

### ✅ Version 2 (v2) - XGBoost (Current)

Switched to **gradient boosting (XGBoost)** which improved performance:

```
╔══════════════════════════════════════════════════════════╗
║              MODEL v2: XGBOOST CLASSIFIER                ║
╠══════════════════════════════════════════════════════════╣
║  Accuracy                  │  58.49%                     ║
║  Weighted F1-Score         │  55.58%                     ║
║  Training Time             │  ~30 minutes                ║
╠══════════════════════════════════════════════════════════╣
║  Status: CURRENT VERSION - Moderate performance         ║
╚══════════════════════════════════════════════════════════╝
```

**Why it's better:**
- ✅ Non-linear model handles complex text patterns
- ✅ Better performance on minority classes
- ✅ Feature importance insights
- ✅ Production-ready deployment

**Key Improvements from v1:**
- More features: 5,021 → 8,014
- Better text vectorization: TF-IDF with tri-grams
- Engineered features from reviews
- Class weight handling
- Hyperparameter tuning

---

## 💻 Technology Stack

### Core ML Libraries
```python
Python 3.13.0
├─ pandas 2.2.2          # Data manipulation
├─ numpy 1.26.4          # Numerical computing
├─ scikit-learn 1.5.2    # Feature extraction, preprocessing
├─ xgboost 2.1.2         # Gradient boosting classifier
└─ joblib 1.4.2          # Model serialization
```

### Deployment Stack
```python
FastAPI 0.115.0          # REST API framework
├─ uvicorn 0.32.0        # ASGI server
├─ pydantic 2.9.2        # Data validation
└─ python-multipart      # File uploads

Docker 27.3.1            # Containerization
Kubernetes 1.31          # Orchestration
```

### Development Tools
```python
Jupyter Lab              # Notebook development
├─ matplotlib 3.9.2      # Visualization
├─ seaborn 0.13.2        # Statistical plots
└─ tqdm                  # Progress bars
```

---

## 📁 Project Structure

```
book-classifier/
│
├── artifacts_v2/                    # All model artifacts
│   ├── dataset/
│   │   ├── sampled_metadata.csv    # Preprocessed dataset
│   │   ├── class_weights.json      # Class weights
│   │   └── class_weights_aggressive.json
│   │
│   ├── datasets/                    # Train/test splits
│   │   ├── train_metadata.csv
│   │   ├── test_metadata.csv
│   │   └── split_info.json
│   │
│   ├── feature_extractor/          # Feature extraction artifacts
│   │   └── features/
│   │       ├── X_train.npy         # Training features
│   │       ├── y_train.npy         # Training labels
│   │       ├── X_test.npy          # Test features
│   │       ├── y_test.npy          # Test labels
│   │       ├── feature_names.json  # Feature names
│   │       └── class_names.json    # Class names
│   │
│   ├── models/                      # Trained models
│   │   ├── xgboost_model.pkl       # XGBoost model
│   │   ├── text_vectorizer.joblib  # TF-IDF vectorizer
│   │   ├── scaler.joblib           # StandardScaler
│   │   ├── label_encoders.joblib   # Label encoders
│   │   ├── target_label_encoder.joblib
│   │   └── feature_extractor_metadata.json
│   │
│   └── reports/                     # Evaluation reports
│       ├── xgboost_confusion_matrix.png
│       ├── xgboost_feature_importance.png
│       ├── xgboost_classification_report.txt
│       ├── xgboost_metrics.json
│       └── model_comparison.csv
│
├── notebooks/
│   ├── books_notebook_subcategories_v1.ipynb  # Logistic Regressionn and XGBoost (deprecated)
│   └── books_notebook_subcategories_v2.ipynb  # XGBoost (current)
│
├── src/
│   ├── data_loader.py              # Data loading utilities
│   ├── data_preprocessor.py        # Data preprocessing
│   ├── feature_extractor.py        # Feature extraction
│   ├── model_trainer.py            # Model training
│   ├── book_classifier.py          # Classifier wrapper
│   └── constants_v2.py             # Configuration constants
│
├── deployment/
│   ├── api.py                            # FastAPI application
│   ├── Dockerfile                        # Docker configuration
│   ├── book-classifier-deployment.yaml   # Kubernetes manifests
│   ├── book-classifier-service.yaml      # Kubernetes manifests
│   └── requirements.txt                  # Python dependencies
│
├── tests/
│   ├── test_api.py                 # API tests
│   └── test_classifier.py          # Classifier tests
│
├── docs/
│   ├── README.md                   # This file
│   ├── VISUAL_PIPELINE.md          # Visual pipeline documentation
│   └── docker-k8s-test.docx        # Console logs for docker and k8s deployments
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Development dependencies
└── .gitignore
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.13+
Docker 27.3.1+
8GB RAM minimum
5GB disk space for model artifacts
```

### Installation

```bash
# 1. Clone repository
git clone https://github.com/your-repo/book-classifier.git
cd book-classifier

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download artifacts (if not included)
# Place artifacts in ./artifacts_v2/ directory
```

### Training (Optional)

```bash
# Run the complete training pipeline
jupyter lab notebooks/books_notebook_subcategories_v2.ipynb

```

### Local API Deployment

```bash
# Option 1: Direct Python
python api.py

# Option 2: Docker
docker build -t book-classifier:v2 -f deployment/Dockerfile .
docker run -p 8005:8005 book-classifier:v2


```

### Test API

```bash
# Health check
curl http://localhost:8000/

# Predict book category
curl -X POST "http://localhost:8000/predict" \
  -d "title=The Great Gatsby" \
  -d "description=A classic American novel" \
  -d "price=12.99" \
  -d "rating=4.5" \
  -d "rating_count=5000" \
  -d "book_format=Paperback"
```

---

## 🔧 API Documentation

### Endpoints

#### `GET /`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_version": "v2",
  "classes": 28
}
```

#### `POST /predict`
Predict book category

**Request Parameters:**
- `title` (required): Book title
- `description` (required): Book description
- `price` (optional): Price in USD
- `rating` (optional): Average rating (0-5)
- `rating_count` (optional): Number of ratings
- `book_format` (optional): Format (Paperback, Hardcover, Kindle)

**Response:**
```json
{
  "predicted_class": "Literature & Fiction",
  "confidence": 0.83,
  "top_3_predictions": [
    {
      "class": "Literature & Fiction",
      "probability": 0.83
    },
    {
      "class": "Mystery, Thriller & Suspense",
      "probability": 0.12
    },
    {
      "class": "Romance",
      "probability": 0.05
    }
  ]
}
```

### Interactive Documentation

Once the API is running, access:
- Swagger UI: http://localhost:8005/docs
- ReDoc: http://localhost:8005/redoc

---

## 🐳 Docker Deployment

### Build Image

```bash
cd deployment
docker build -t book-classifier:v2 -f Dockerfile ..
```

### Run Container

```bash
docker run -d \
  -p 8005:8005 \
  --name book-classifier \
  book-classifier:v2
```

---

## ☸️ Kubernetes Deployment

### Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f book-classifier-deployment.yaml
kubectl apply -f book-classifier-service.yaml

# Check deployment
kubectl get deployments
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/book-classifier
```

### Access Service

```bash
# Get service URL
kubectl get service book-classifier

# Port forward (for testing)
kubectl port-forward service/book-classifier 8005:8005
```

### Scale Deployment

```bash
# Scale to 3 replicas
kubectl scale deployment book-classifier --replicas=3

# Enable autoscaling
kubectl autoscale deployment book-classifier \
  --cpu-percent=70 \
  --min=2 \
  --max=10
```

---

## 🧪 Testing


### Integration Tests

```bash
# Test API endpoints
pytest test_api_client.py

```

---

## 📊 Model Details

### Feature Engineering

**Text Features (8,000 dimensions):**
- TF-IDF vectorization on title + description
- N-grams: unigrams, bigrams, trigrams
- Min document frequency: 2
- Max document frequency: 95%

**Numerical Features (6 dimensions):**
- `price_numeric`: Book price
- `average_rating`: Average rating (0-5)
- `rating_number`: Number of ratings
- `review_count`: Number of reviews
- `helpful_votes_total`: Total helpful votes
- `avg_rating_from_reviews`: Average from reviews

**Categorical Features (3 dimensions - one-hot encoded):**
- `format`: Paperback, Hardcover, Kindle
- `language`: English, Spanish, etc.

**Engineered Features (5 dimensions):**
- Rating consistency metrics
- Text length features
- Review engagement features

### XGBoost Hyperparameters

```python
{
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softprob',
    'num_class': 28,
    'eval_metric': 'mlogloss'
}
```

### Training Configuration

```python
{
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True,
    'cv_folds': 5,
    'early_stopping_rounds': 10
}
```

---

## 🎯 Key Achievements

### Performance
- ✅ **55.58% Weighted F1-Score** - Moderate performance on 28 classes
- ✅ **28/28 classes** predicted (no zero-shot classes)
- ✅ **<100ms inference time** - Fast enough for real-time use
- ✅ **Production-ready deployment** - Containerized and scalable

### Engineering
- ✅ **Modular code architecture** - Easy to maintain and extend
- ✅ **Comprehensive testing** - Unit and integration tests
- ✅ **Docker containerization** - Reproducible deployments
- ✅ **Kubernetes-ready** - Horizontal scaling support
- ✅ **REST API** - Easy integration with existing systems

### Data Pipeline
- ✅ **Automated preprocessing** - From raw data to features
- ✅ **Review enrichment** - Leveraged 719k reviews for features
- ✅ **Class balancing** - Handled imbalanced data
- ✅ **Feature engineering** - 8,014 meaningful features

---

## 🔮 Future Improvements

### Short Term 
- [ ] Improve model performance (target: >70% F1)
  - Fine-tune hyperparameters
  - Experiment with feature selection
  - Try ensemble methods
- [ ] Add confidence threshold tuning
- [ ] Implement model monitoring and drift detection
- [ ] Create admin dashboard for model metrics

### Medium Term 
- [ ] Experiment with deep learning models:
  - BERT embeddings for better text understanding
  - Hierarchical classification
  - Multi-label classification
- [ ] Active learning for uncertain predictions
- [ ] A/B testing framework
- [ ] Add more features (author popularity, publication date patterns)

### Long Term 
- [ ] Transformer-based models (DistilBERT, RoBERTa)
- [ ] Multi-language support
- [ ] Real-time model retraining pipeline
- [ ] Recommendation system integration
- [ ] Explainability features (SHAP, LIME)

---

## 📚 References & Resources

### Documentation
- [DataTalksClub Repository and Documentation](https://github.com/DataTalksClub/machine-learning-zoomcamp)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

### Datasets
- https://www.kaggle.com/datasets/hadifariborzi/amazon-books-dataset-20k-books-727k-reviews/suggestions/data
- https://www.kaggle.com/datasets/dongrelaxman/amazon-reviews-dataset
---

## 👥 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting PRs.



## 🙏 Acknowledgments

- **DataTalks.Club** for the MLOps Zoomcamp course
- **Amazon** for the books dataset


---

## 📧 Contact

**Author:** Rafael Bucio
**Project:** MLOps Zoomcamp - Mid-term Project  
**Date:** November 2025

For questions or feedback, please open an issue on GitHub.

---

