import os
import sys
import numpy as np
import pandas as pd
import faiss
import pickle
from pathlib import Path
import json

# Define directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
INDEX_DIR = os.path.join(MODELS_DIR, "indices")

# Create necessary directories
os.makedirs(INDEX_DIR, exist_ok=True)

def load_embeddings_and_data(product_id=None):
    """
    Load embeddings and corresponding dataset for a specific product or all products
    
    Args:
        product_id: Optional product ID to load specific embeddings
        
    Returns:
        tuple of (embeddings, dataframe)
    """
    try:
        # Determine file paths based on product_id
        if product_id and product_id != "all":
            embeddings_file = os.path.join(MODELS_DIR, f"qna_embeddings_{product_id}.npy")
            data_file = os.path.join(OUTPUT_DIR, f"data_{product_id}.csv")
        else:
            embeddings_file = os.path.join(MODELS_DIR, "qna_embeddings.npy")
            data_file = os.path.join(OUTPUT_DIR, "combined_data.csv")
        
        # Check if files exist
        if not os.path.exists(embeddings_file):
            print(f"Error: Embeddings file {embeddings_file} not found")
            return None, None
            
        if not os.path.exists(data_file):
            print(f"Error: Data file {data_file} not found")
            return None, None
        
        # Load embeddings and data
        embeddings = np.load(embeddings_file)
        df = pd.read_csv(data_file)
        
        # Ensure the number of embeddings matches the number of rows in the dataframe
        if len(embeddings) != len(df):
            print(f"Warning: Number of embeddings ({len(embeddings)}) doesn't match number of data rows ({len(df)})")
        
        print(f"Loaded {len(embeddings)} embeddings and {len(df)} data rows for {'product ' + product_id if product_id else 'all products'}")
        return embeddings, df
    
    except Exception as e:
        print(f"Error loading embeddings and data: {e}")
        return None, None

def create_faiss_index(embeddings, df, product_id=None):
    """
    Create a FAISS index for the embeddings and save it
    
    Args:
        embeddings: Numpy array of embeddings
        df: Dataframe containing the corresponding data
        product_id: Optional product ID to identify the index
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Get embedding dimension
        dimension = embeddings.shape[1]
        
        # Create index
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
        index.add(embeddings)
        
        # Determine index file path
        if product_id and product_id != "all":
            index_file = os.path.join(INDEX_DIR, f"faiss_index_{product_id}.bin")
            mapping_file = os.path.join(INDEX_DIR, f"index_mapping_{product_id}.json")
        else:
            index_file = os.path.join(INDEX_DIR, "faiss_index.bin")
            mapping_file = os.path.join(INDEX_DIR, "index_mapping.json")
        
        # Save the index
        faiss.write_index(index, index_file)
        
        # Create and save mapping between index positions and document IDs
        mapping = {}
        for i, cqid in enumerate(df['cqid']):
            mapping[i] = {
                'cqid': str(cqid),
                'question': str(df.iloc[i]['question']) if 'question' in df.columns and not pd.isna(df.iloc[i]['question']) else ""
            }
        
        # Save mapping as JSON
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"Created FAISS index with {len(embeddings)} vectors and saved to {index_file}")
        print(f"Created index-to-ID mapping and saved to {mapping_file}")
        
        return True
    
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        return False

def similarity_search_test(product_id=None, top_k=5):
    """
    Test the created index with a simple search
    
    Args:
        product_id: Optional product ID to test specific index
        top_k: Number of results to return
    """
    try:
        # Load embeddings and data
        embeddings, df = load_embeddings_and_data(product_id)
        if embeddings is None or df is None:
            return
        
        # Load the index
        if product_id and product_id != "all":
            index_file = os.path.join(INDEX_DIR, f"faiss_index_{product_id}.bin")
            mapping_file = os.path.join(INDEX_DIR, f"index_mapping_{product_id}.json")
        else:
            index_file = os.path.join(INDEX_DIR, "faiss_index.bin")
            mapping_file = os.path.join(INDEX_DIR, "index_mapping.json")
        
        index = faiss.read_index(index_file)
        
        # Load the mapping
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        
        # Select a random query vector for testing
        query_idx = np.random.randint(0, len(embeddings))
        query_vector = embeddings[query_idx:query_idx+1]
        
        # Perform search
        distances, indices = index.search(query_vector, top_k)
        
        print(f"\nTest search results for {'product ' + product_id if product_id else 'all products'}:")
        print(f"Query: {df.iloc[query_idx]['question']}")
        print("\nTop results:")
        
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            idx = int(idx)
            cqid = mapping[str(idx)]['cqid']
            question = mapping[str(idx)]['question']
            print(f"{i+1}. [Score: {distance:.4f}] {question}")
        
    except Exception as e:
        print(f"Error testing similarity search: {e}")

def create_all_indices():
    """Create indices for all products and the combined dataset"""
    try:
        # Get list of product IDs from the embedding files
        product_ids = []
        for filename in os.listdir(MODELS_DIR):
            if filename.startswith("qna_embeddings_") and filename.endswith(".npy"):
                product_id = filename.replace("qna_embeddings_", "").replace(".npy", "")
                product_ids.append(product_id)
        
        # Create index for all products combined
        print("\nCreating index for all products combined...")
        embeddings, df = load_embeddings_and_data()
        if embeddings is not None and df is not None:
            create_faiss_index(embeddings, df)
            similarity_search_test()
        
        # Create indices for each product
        for product_id in product_ids:
            print(f"\nCreating index for product {product_id}...")
            embeddings, df = load_embeddings_and_data(product_id)
            if embeddings is not None and df is not None:
                create_faiss_index(embeddings, df, product_id)
                similarity_search_test(product_id)
        
        print("\nAll indices created successfully")
        
    except Exception as e:
        print(f"Error creating indices: {e}")

if __name__ == "__main__":
    print("Starting index creation for similarity search...")
    create_all_indices()
    print("Index creation completed!")
