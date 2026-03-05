import os
import sys
from dotenv import load_dotenv

# Add app to path
sys.path.append(os.getcwd())

from app.utils.chunking import get_chunks

def test_semantic_chunking():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("GOOGLE_API_KEY not found in .env. Falling back to recursive or simple implementation.")
    
    text = """
    The Apple iPhone 15 Pro features a stunning titanium design, the innovative A17 Pro chip, and a versatile Pro camera system. 
    It is the first iPhone to feature an aerospace‑grade titanium design, using the same alloy that spacecraft use for missions to Mars.
    
    The new A17 Pro chip brings a huge leap forward in performance and efficiency. 
    The GPU is up to 20% faster, and the CPU is up to 10% faster, making it perfect for demanding mobile games and creative apps.
    
    The advanced Pro camera system includes a 48MP Main camera, a Ultra Wide camera, and a Telephoto camera. 
    With up to 5x optical zoom on the iPhone 15 Pro Max, you can take sharp close-ups from even further away.
    """
    
    print("Testing 'none' strategy:")
    chunks = get_chunks(text, strategy="none")
    print(f"Count: {len(chunks)}")
    
    print("\nTesting 'semantic' strategy (threshold=0.5):")
    chunks = get_chunks(text, strategy="semantic", google_api_key=api_key, threshold=0.5)
    print(f"Count: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---\n{chunk.strip()}")

if __name__ == "__main__":
    test_semantic_chunking()
