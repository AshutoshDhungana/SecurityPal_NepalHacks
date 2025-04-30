import os
import sys
import numpy as np
import pandas as pd
import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Define directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
INDEX_DIR = os.path.join(MODELS_DIR, "indices")

# Global caches to avoid reloading
INDEX_CACHE = {}  # product_id -> faiss index
EMBEDDING_CACHE = {}  # product_id -> embeddings
MAPPING_CACHE = {}  # product_id -> mapping
DF_CACHE = {}  # product_id -> dataframe
MODEL_CACHE = None  # sentence transformer model

def get_model():
    """Get the sentence transformer model, loading it once and caching it"""
    global MODEL_CACHE
    if MODEL_CACHE is None:
        MODEL_CACHE = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return MODEL_CACHE

def get_index_path(product_id=None):
    """Get the path to the FAISS index file for a specific product or all products"""
    if product_id and product_id != "all":
        return os.path.join(INDEX_DIR, f"faiss_index_{product_id}.bin")
    return os.path.join(INDEX_DIR, "faiss_index.bin")

def get_mapping_path(product_id=None):
    """Get the path to the index mapping file for a specific product or all products"""
    if product_id and product_id != "all":
        return os.path.join(INDEX_DIR, f"index_mapping_{product_id}.json")
    return os.path.join(INDEX_DIR, "index_mapping.json")

def get_embeddings_path(product_id=None):
    """Get the path to the embeddings file for a specific product or all products"""
    if product_id and product_id != "all":
        return os.path.join(MODELS_DIR, f"qna_embeddings_{product_id}.npy")
    return os.path.join(MODELS_DIR, "qna_embeddings.npy")

def get_data_path(product_id=None):
    """Get the path to the data file for a specific product or all products"""
    if product_id and product_id != "all":
        return os.path.join(OUTPUT_DIR, f"data_{product_id}.csv")
    return os.path.join(OUTPUT_DIR, "combined_data.csv")

def load_index(product_id=None, force_reload=False):
    """
    Load the FAISS index for the specified product
    
    Args:
        product_id: Optional product ID to load specific index
        force_reload: Whether to force reload even if cached
        
    Returns:
        FAISS index object
    """
    # Check cache first
    cache_key = product_id if product_id else "all"
    if not force_reload and cache_key in INDEX_CACHE:
        return INDEX_CACHE[cache_key]
    
    # Get the index path
    index_path = get_index_path(product_id)
    
    # Check if index exists
    if not os.path.exists(index_path):
        print(f"Warning: Index file {index_path} not found")
        return None
    
    # Load the index
    try:
        index = faiss.read_index(index_path)
        INDEX_CACHE[cache_key] = index
        return index
    except Exception as e:
        print(f"Error loading index: {e}")
        return None

def load_embeddings(product_id=None, force_reload=False):
    """
    Load embeddings for the specified product
    
    Args:
        product_id: Optional product ID to load specific embeddings
        force_reload: Whether to force reload even if cached
        
    Returns:
        Numpy array of embeddings
    """
    # Check cache first
    cache_key = product_id if product_id else "all"
    if not force_reload and cache_key in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[cache_key]
    
    # Get the embeddings path
    embeddings_path = get_embeddings_path(product_id)
    
    # Check if embeddings exist
    if not os.path.exists(embeddings_path):
        print(f"Warning: Embeddings file {embeddings_path} not found")
        return None
    
    # Load the embeddings
    try:
        embeddings = np.load(embeddings_path)
        EMBEDDING_CACHE[cache_key] = embeddings
        return embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None

def load_mapping(product_id=None, force_reload=False):
    """
    Load the index-to-ID mapping for the specified product
    
    Args:
        product_id: Optional product ID to load specific mapping
        force_reload: Whether to force reload even if cached
        
    Returns:
        Dictionary mapping index positions to document IDs
    """
    # Check cache first
    cache_key = product_id if product_id else "all"
    if not force_reload and cache_key in MAPPING_CACHE:
        return MAPPING_CACHE[cache_key]
    
    # Get the mapping path
    mapping_path = get_mapping_path(product_id)
    
    # Check if mapping exists
    if not os.path.exists(mapping_path):
        print(f"Warning: Mapping file {mapping_path} not found")
        return None
    
    # Load the mapping
    try:
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        MAPPING_CACHE[cache_key] = mapping
        return mapping
    except Exception as e:
        print(f"Error loading mapping: {e}")
        return None

def load_data(product_id=None, force_reload=False):
    """
    Load the data for the specified product
    
    Args:
        product_id: Optional product ID to load specific data
        force_reload: Whether to force reload even if cached
        
    Returns:
        Pandas DataFrame with the data
    """
    # Check cache first
    cache_key = product_id if product_id else "all"
    if not force_reload and cache_key in DF_CACHE:
        return DF_CACHE[cache_key]
    
    # Get the data path
    data_path = get_data_path(product_id)
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Warning: Data file {data_path} not found")
        
        # Try loading merged_data instead if data_product_id.csv doesn't exist
        if product_id and product_id != "all":
            alt_path = os.path.join(OUTPUT_DIR, f"merged_data_{product_id}.csv")
            if os.path.exists(alt_path):
                try:
                    df = pd.read_csv(alt_path)
                    DF_CACHE[cache_key] = df
                    return df
                except Exception as e:
                    print(f"Error loading alternative data: {e}")
        
        return None
    
    # Load the data
    try:
        df = pd.read_csv(data_path)
        DF_CACHE[cache_key] = df
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def perform_similarity_search(query_text, product_id=None, top_k=5, threshold=0.0):
    """
    Perform similarity search for a query text
    
    Args:
        query_text: Text to search for
        product_id: Optional product ID to search in specific product
        top_k: Number of results to return
        threshold: Minimum similarity threshold
        
    Returns:
        List of dictionaries containing similar items with their similarity scores
    """
    # Load the index and data
    index = load_index(product_id)
    df = load_data(product_id)
    
    if index is None or df is None:
        print("Error: Could not load index or data for similarity search")
        return []
    
    # Get the model
    model = get_model()
    
    # Encode the query with safety check for None
    if query_text is None:
        query_text = ""
    query_embedding = model.encode([query_text])
    
    # Normalize the query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Perform search
    similarities, indices = index.search(query_embedding, top_k)
    
    # Prepare results
    results = []
    for i in range(min(top_k, len(indices[0]))):
        idx = indices[0][i]
        similarity = similarities[0][i]
        
        # Skip results below threshold
        if similarity < threshold:
            continue
        
        # Get the corresponding row from the dataframe
        if idx < len(df):
            row = df.iloc[idx]
            
            results.append({
                "question": row['question'],
                "similarity": similarity,
                "answer": row.get('answer', None),
                "details": row.get('details', None),
                "category": row.get('category', None),
                "deleted_at": row.get('deleted_at', None),
                "is_archived": pd.notna(row.get('deleted_at', None)),
                "cqid": row.get('cqid', None),
                "product_id": row.get('product_id', None)
            })
    
    # Sort by similarity score (highest first)
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results

def clear_cache():
    """Clear all caches to free memory"""
    global INDEX_CACHE, EMBEDDING_CACHE, MAPPING_CACHE, DF_CACHE
    INDEX_CACHE = {}
    EMBEDDING_CACHE = {}
    MAPPING_CACHE = {}
    DF_CACHE = {}
    print("Cache cleared")

def get_available_products():
    """
    Get a list of products that have indices available
    
    Returns:
        List of product IDs
    """
    products = []
    
    # Add "all" products option if available
    if os.path.exists(get_index_path()):
        products.append("all")
    
    # Check for product-specific indices
    for filename in os.listdir(INDEX_DIR):
        if filename.startswith("faiss_index_") and filename.endswith(".bin"):
            product_id = filename.replace("faiss_index_", "").replace(".bin", "")
            products.append(product_id)
    
    return products 