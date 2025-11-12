"""
PYTHON CLIENT FOR API TESTING
==============================
Client to test the book classification API.

Usage:
    python test_api_client_english.py

"""

import requests
import json
from typing import Dict, List

# Configuration
API_BASE_URL = "http://192.168.49.2:30080"#"http://localhost:8000"


class BookClassifierClient:
    """Client to interact with the book classification API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API (default: http://localhost:8000)
        """
        self.base_url = base_url
    
    def health_check(self) -> Dict:
        """
        Check if the API is running.
        
        Returns:
            Dict with API status
        """
        try:
            response = requests.get(f"{self.base_url}/")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "down"}
    
    def predict(
        self,
        title: str,
        description: str,
        price: float,
        rating: float,
        rating_count: int,
        book_format: str
    ) -> Dict:
        """
        Predict the category of a book.
        
        Args:
            title: Book title
            description: Book description
            price: Price in USD
            rating: Average rating (1-5)
            rating_count: Number of ratings
            book_format: Format (Paperback, Kindle, etc.)
        
        Returns:
            Dict with prediction results
        """
        url = f"{self.base_url}/predict"
        
        params = {
            "title": title,
            "description": description,
            "price": price,
            "rating": rating,
            "rating_count": rating_count,
            "book_format": book_format
        }
        
        try:
            response = requests.post(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}


def print_result(result: Dict, book_title: str):
    """
    Print the prediction result in a nice format.
    
    Args:
        result: Prediction result from API
        book_title: Title of the book
    """
    print(f"\n{'='*80}")
    print(f"📚 Book: {book_title}")
    print('='*80)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"\n🎯 Predicted Category: {result['predicted_class']}")
    print(f"📊 Confidence: {result['confidence']:.1%}")
    
    if 'top_3' in result:
        print("\n🏆 Top 3 Predictions:")
        for i, pred in enumerate(result['top_3'], 1):
            bar = '█' * int(pred['probability'] * 50)
            print(f"  {i}. {pred['class']:35s} {bar} {pred['probability']:.1%}")


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_single_book():
    """Test: Predict a single book."""
    print("\n" + "="*80)
    print("TEST 1: Single Book Prediction")
    print("="*80)
    
    client = BookClassifierClient()
    
    # Check API status
    health = client.health_check()
    print(f"\n✓ API Status: {health.get('status', 'unknown')}")
    
    # Predict a book
    result = client.predict(
        title="The Da Vinci Code",
        description="A murder in the Louvre Museum leads Harvard symbologist Robert Langdon on a trail of clues hidden in the works of Leonardo da Vinci.",
        price=14.99,
        rating=4.2,
        rating_count=8432,
        book_format="Paperback"
    )
    
    print_result(result, "The Da Vinci Code")


def test_multiple_books():
    """Test: Predict multiple books."""
    print("\n" + "="*80)
    print("TEST 2: Multiple Books Prediction")
    print("="*80)
    
    client = BookClassifierClient()
    
    books = [
        {
            "title": "Python Crash Course",
            "description": "A hands-on, project-based introduction to programming with Python",
            "price": 34.99,
            "rating": 4.6,
            "rating_count": 2341,
            "book_format": "Paperback"
        },
        {
            "title": "The Great Gatsby",
            "description": "A classic American novel set in the Jazz Age about wealth, love, and the American Dream",
            "price": 10.99,
            "rating": 3.9,
            "rating_count": 15234,
            "book_format": "Paperback"
        },
        {
            "title": "Healthy Cooking Made Easy",
            "description": "200 delicious and nutritious recipes for everyday meals and special occasions",
            "price": 24.99,
            "rating": 4.4,
            "rating_count": 543,
            "book_format": "Hardcover"
        },
        {
            "title": "Harry Potter and the Sorcerer's Stone",
            "description": "A young wizard discovers his magical heritage and attends Hogwarts School of Witchcraft and Wizardry",
            "price": 12.99,
            "rating": 4.8,
            "rating_count": 34521,
            "book_format": "Paperback"
        },
        {
            "title": "Becoming",
            "description": "The intimate memoir of former First Lady Michelle Obama",
            "price": 18.99,
            "rating": 4.7,
            "rating_count": 12834,
            "book_format": "Hardcover"
        }
    ]
    
    for book in books:
        result = client.predict(**book)
        print_result(result, book['title'])


def test_edge_cases():
    """Test: Edge cases with extreme values."""
    print("\n" + "="*80)
    print("TEST 3: Edge Cases")
    print("="*80)
    
    client = BookClassifierClient()
    
    # Very expensive book
    print("\n📌 Test: Very Expensive Book")
    result = client.predict(
        title="Limited Edition Collector's Item",
        description="A rare collectible book",
        price=299.99,
        rating=4.5,
        rating_count=12,
        book_format="Hardcover"
    )
    print_result(result, "Expensive Book")
    
    # Very cheap book
    print("\n📌 Test: Very Cheap Book")
    result = client.predict(
        title="Budget Friendly Read",
        description="An affordable book for everyone",
        price=0.99,
        rating=3.5,
        rating_count=234,
        book_format="Kindle"
    )
    print_result(result, "Cheap Book")
    
    # Very short description
    print("\n📌 Test: Short Description")
    result = client.predict(
        title="Short",
        description="Short book",
        price=15.99,
        rating=4.0,
        rating_count=100,
        book_format="Paperback"
    )
    print_result(result, "Short Description")


def interactive_mode():
    """Interactive mode: Enter your own books."""
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    
    client = BookClassifierClient()
    
    # Check API
    health = client.health_check()
    if health.get('status') != 'running':
        print("❌ Error: API is not running")
        print("Run first: python api.py")
        return
    
    print("\n✓ API connected")
    print("\nEnter book details:")
    
    try:
        title = input("\nTitle: ")
        description = input("Description: ")
        price = float(input("Price (USD): "))
        rating = float(input("Rating (1-5): "))
        rating_count = int(input("Number of ratings: "))
        book_format = input("Format (Paperback/Kindle/Hardcover): ")
        
        result = client.predict(
            title=title,
            description=description,
            price=price,
            rating=rating,
            rating_count=rating_count,
            book_format=book_format
        )
        
        print_result(result, title)
        
    except ValueError as e:
        print(f"\n❌ Error in input data: {e}")
    except KeyboardInterrupt:
        print("\n\n👋 Cancelled by user")


def benchmark_api():
    """Benchmark: Measure API speed."""
    print("\n" + "="*80)
    print("API SPEED BENCHMARK")
    print("="*80)
    
    import time
    
    client = BookClassifierClient()
    
    # Test book
    test_book = {
        "title": "Test Book",
        "description": "This is a test book for benchmarking",
        "price": 19.99,
        "rating": 4.0,
        "rating_count": 100,
        "book_format": "Paperback"
    }
    
    n_requests = 10
    print(f"\nMaking {n_requests} requests...")
    
    times = []
    for i in range(n_requests):
        start = time.time()
        result = client.predict(**test_book)
        end = time.time()
        
        elapsed = end - start
        times.append(elapsed)
        print(f"  Request {i+1}: {elapsed*1000:.1f}ms")
    
    # Statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"Total requests:   {n_requests}")
    print(f"Average time:     {avg_time*1000:.1f}ms")
    print(f"Minimum time:     {min_time*1000:.1f}ms")
    print(f"Maximum time:     {max_time*1000:.1f}ms")
    print(f"Requests/second:  {1/avg_time:.1f}")


def run_all_tests():
    """Run all tests sequentially."""
    print("\n" + "="*80)
    print("RUNNING ALL TESTS")
    print("="*80)
    
    test_single_book()
    test_multiple_books()
    test_edge_cases()
    benchmark_api()
    
    print("\n" + "="*80)
    print("✓ ALL TESTS COMPLETED")
    print("="*80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function with interactive menu."""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    TEST CLIENT - BOOK CLASSIFIER API                         ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"\nMake sure the API is running at {API_BASE_URL}")
   
    
    # Menu loop
    while True:
        print("\n" + "="*80)
        print("TEST MENU")
        print("="*80)
        print("1. Test: Single book")
        print("2. Test: Multiple books")
        print("3. Test: Edge cases")
        print("4. Interactive mode")
        print("5. Speed benchmark")
        print("6. Run all tests")
        print("7. Exit")
        
        try:
            option = input("\nChoose an option (1-7): ").strip()
            
            if option == '1':
                test_single_book()
            elif option == '2':
                test_multiple_books()
            elif option == '3':
                test_edge_cases()
            elif option == '4':
                interactive_mode()
            elif option == '5':
                benchmark_api()
            elif option == '6':
                run_all_tests()
            elif option == '7':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Choose 1-7.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == '__main__':
    main()