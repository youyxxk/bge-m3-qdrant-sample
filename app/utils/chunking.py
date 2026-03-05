"""
Utility functions for text chunking.
"""

import os
import numpy as np
from typing import List, Optional
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    # Ensure punkt and punkt_tab are available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
except ImportError:
    # Fallback to simple split if nltk is not installed
    def sent_tokenize(text):
        return [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]

try:
    import google.generativeai as genai
except ImportError:
    genai = None


def list_google_models(api_key: str) -> List[str]:
    """List available Google AI models that support embedding."""
    if genai is None:
        return []
    try:
        genai.configure(api_key=api_key)
        models = []
        for m in genai.list_models():
            if 'embedContent' in m.supported_generation_methods:
                models.append(m.name)
        return models
    except Exception as e:
        print(f"Error listing Google models: {e}")
        return []


def chunk_none(text: str, **kwargs) -> List[str]:
    """Return the entire text as a single chunk."""
    if not text:
        return []
    return [text]


def chunk_character(text: str, chunk_size: int = 500, chunk_overlap: int = 50, **kwargs) -> List[str]:
    """Split text into chunks of specified character length with overlap."""
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        
        step = chunk_size - chunk_overlap
        if step <= 0:
            step = 1
        start += step
        
    return chunks


def chunk_word(text: str, chunk_size: int = 100, chunk_overlap: int = 10, **kwargs) -> List[str]:
    """Split text into chunks of specified word count with overlap."""
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
        
    words = text.split()
    if not words:
        return []
        
    chunks = []
    start = 0
    words_len = len(words)
    
    while start < words_len:
        end = min(start + chunk_size, words_len)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        
        step = chunk_size - chunk_overlap
        if step <= 0:
            step = 1
        start += step
        
    return chunks


def chunk_recursive(
    text: str, 
    chunk_size: int = 500, 
    chunk_overlap: int = 50,
    separators: Optional[List[str]] = None,
    **kwargs
) -> List[str]:
    """
    Recursively split text using separators to keep related content together.
    """
    if not text:
        return []
        
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
        
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]
        
    # Get the best separator
    separator = separators[-1]
    new_separators = []
    for i, _s in enumerate(separators):
        if _s == "":
            separator = _s
            break
        if _s in text:
            separator = _s
            new_separators = separators[i + 1:]
            break
 
    # Split the text
    if separator:
        splits = text.split(separator)
    else:
        splits = list(text)

    # Merge splits
    good_splits = []
    for s in splits:
        if len(s) < chunk_size:
            good_splits.append(s)
        else:
            if new_separators:
                good_splits.extend(chunk_recursive(s, chunk_size, chunk_overlap, new_separators))
            else:
                # Fallback if no more separators (character level)
                good_splits.append(s)

    # Now merge good_splits into chunks with overlap
    return _merge_splits(good_splits, separator, chunk_size, chunk_overlap)


def chunk_semantic(
    text: str, 
    google_api_key: Optional[str] = None, 
    threshold: float = 0.5, 
    buffer_size: int = 1,
    model_name: str = "models/gemini-embedding-001",
    **kwargs
) -> List[str]:
    """
    Split text into semantically cohesive chunks using Google AI embeddings.
    
    Args:
        text: Text to chunk
        google_api_key: API key for Google Generative AI
        threshold: Similarity threshold (0.0 to 1.0). Lower means more chunks.
        buffer_size: Number of sentences to look ahead/behind for context.
        model_name: Google embedding model to use.
    """
    if not text:
        return []
    
    if not google_api_key:
        # Fallback to recursive if no API key
        return chunk_recursive(text, **kwargs)
    
    if genai is None:
        return chunk_recursive(text, **kwargs)

    try:
        genai.configure(api_key=google_api_key)
        
        # 1. Split into sentences
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return [text]

        # 2. Get embeddings for each sentence
        # Fetch the model name from args
        model = model_name
        
        # Batch embed sentences
        result = genai.embed_content(
            model=model,
            content=sentences,
            task_type="retrieval_document"
        )
        embeddings = np.array(result['embedding'])

        # 3. Calculate similarities between adjacent sentences
        # We can use a sliding window/buffer to smooth the similarities
        def get_weighted_embedding(idx):
            start = max(0, idx - buffer_size)
            end = min(len(embeddings), idx + buffer_size + 1)
            return np.mean(embeddings[start:end], axis=0)

        similarities = []
        for i in range(len(embeddings) - 1):
            emb1 = get_weighted_embedding(i)
            emb2 = get_weighted_embedding(i + 1)
            
            # Cosine similarity
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarities.append(sim)

        # 4. Create chunks based on similarity drops
        chunks = []
        current_sentences = [sentences[0]]
        
        for i, sim in enumerate(similarities):
            if sim < threshold:
                chunks.append(" ".join(current_sentences))
                current_sentences = [sentences[i + 1]]
            else:
                current_sentences.append(sentences[i + 1])
        
        if current_sentences:
            chunks.append(" ".join(current_sentences))
            
        return chunks
    except Exception as e:
        print(f"Error in semantic chunking: {e}. Falling back to recursive.")
        return chunk_recursive(text, **kwargs)


def _merge_splits(splits: List[str], separator: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Combine small splits into chunks up to chunk_size, allowing for chunk_overlap.
    """
    chunks = []
    current_chunk = []
    current_length = 0
    separator_len = len(separator)

    for split in splits:
        split_len = len(split)
        
        if current_length + split_len + (separator_len if current_chunk else 0) > chunk_size and current_chunk:
            chunks.append(separator.join(current_chunk))
            
            overlap_length = 0
            new_chunk = []
            for item in reversed(current_chunk):
                item_len = len(item)
                if overlap_length + item_len + (separator_len if new_chunk else 0) > chunk_overlap:
                    break
                new_chunk.insert(0, item)
                overlap_length += item_len + (separator_len if new_chunk else 0)
                
            current_chunk = new_chunk
            current_length = overlap_length
            
        current_chunk.append(split)
        current_length += split_len + (separator_len if len(current_chunk) > 1 else 0)

    if current_chunk:
        chunks.append(separator.join(current_chunk))

    return chunks


def get_chunks(
    text: str, 
    strategy: str = "none", 
    chunk_size: int = 500, 
    chunk_overlap: int = 50,
    google_api_key: Optional[str] = None,
    threshold: float = 0.5,
    model_name: str = "models/text-embedding-004",
    **kwargs
) -> List[str]:
    """
    Route to the appropriate chunking strategy.
    
    Args:
        text: Text to chunk
        strategy: 'none', 'character', 'word', 'recursive', or 'semantic'
        chunk_size: Maximum size of chunk (chars or words depending on strategy)
        chunk_overlap: Overlap between chunks (chars or words depending on strategy)
        google_api_key: Key for semantic chunking
        threshold: Threshold for semantic chunking
        
    Returns:
        List of text chunks
    """
    if strategy == "character":
        return chunk_character(text, chunk_size, chunk_overlap)
    elif strategy == "word":
        return chunk_word(text, chunk_size, chunk_overlap)
    elif strategy == "recursive":
        return chunk_recursive(text, chunk_size, chunk_overlap)
    elif strategy == "semantic":
        return chunk_semantic(
            text, 
            google_api_key=google_api_key, 
            threshold=threshold, 
            model_name=model_name,
            **kwargs
        )
    else:  # 'none' or fallback
        return chunk_none(text)
