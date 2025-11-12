from fastapi import FastAPI
from book_classifier import BookClassifier
from constants_v2 import REPORT_XGBOOST_MODEL_FILE_V2, MODEL_OUTPUT_DIR_V2

app = FastAPI()
classifier = BookClassifier(REPORT_XGBOOST_MODEL_FILE_V2, MODEL_OUTPUT_DIR_V2)

@app.post("/predict")  
def predict_book(
    title: str,
    description: str,
    price: float,
    rating: float,
    rating_count: int,
    book_format: str
):
    """Predice la categoría de un libro."""
    result = classifier.predict(
        title, description, price, rating, rating_count, book_format
    )
    return result

@app.get("/")  
def home():
    return {"message": "Book Classifier API", "status": "running"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)