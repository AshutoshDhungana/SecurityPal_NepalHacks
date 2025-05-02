#!/usr/bin/env python3
"""
Similarity Check Pipeline

This module provides functions to:
1. Generate embeddings for user questions using a real embedding model (all-MiniLM-L6-v2)
2. Perform cosine similarity checks with existing questions in the database
3. Return the most similar question IDs from the same product and category
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename="similarity_pipeline.log"
)
logger = logging.getLogger("similarity_pipeline")

# Path constants
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_CLUSTERS_PATH = BASE_DIR / "processed_clusters"
DATA_PATH = BASE_DIR / "data"
EMBEDDING_CACHE_PATH = BASE_DIR / "embedding_cache"
MODELS_DIR = BASE_DIR / "models"  # Add models directory path

# Ensure embedding cache directory exists
EMBEDDING_CACHE_PATH.mkdir(parents=True, exist_ok=True)

class SimilarityCheck:
    """
    Class for finding similar questions based on semantic similarity.
    Uses pre-computed FAISS indices for efficient similarity search.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_cache: bool = True, use_precomputed: bool = True):
        """
        Initialize the SimilarityCheck class.
        
        Args:
            model_name: The name of the embedding model to use
            embedding_cache: Whether to cache embeddings for faster retrieval
            use_precomputed: Whether to use pre-computed embeddings from models directory
        """
        self.model_name = model_name
        self.embedding_cache = embedding_cache
        self.use_precomputed = use_precomputed
        self.cache_file = EMBEDDING_CACHE_PATH / f"{model_name.replace('/', '_')}_cache.json"
        
        # Load pre-computed embeddings metadata if available
        self.precomputed_metadata = {}
        self.precomputed_available = False
        metadata_file = MODELS_DIR / "embeddings_metadata.json"
        
        if use_precomputed and metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.precomputed_metadata = json.load(f)
                self.precomputed_available = True
                logger.info(f"Loaded pre-computed embeddings metadata")
            except Exception as e:
                logger.warning(f"Could not load pre-computed embeddings metadata: {str(e)}")
        
        # Load embedding model (only if pre-computed embeddings are not available or specifically requested)
        if not self.precomputed_available or not use_precomputed:
            logger.info(f"Loading embedding model: {model_name}")
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Successfully loaded {model_name}")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
                raise RuntimeError(f"Failed to load embedding model: {str(e)}")
        else:
            logger.info("Using pre-computed embeddings, skipping model loading")
            self.model = None
        
        # Load embedding cache if it exists
        self.embedding_dict = {}
        if embedding_cache and self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.embedding_dict = json.load(f)
                logger.info(f"Loaded {len(self.embedding_dict)} cached embeddings")
            except Exception as e:
                logger.warning(f"Could not load embedding cache: {str(e)}")
                
        # Import faiss if we're using pre-computed embeddings
        if self.precomputed_available and use_precomputed:
            try:
                global faiss
                import faiss
                logger.info("Imported FAISS successfully")
            except ImportError:
                logger.error("Failed to import FAISS. Make sure it's installed: pip install faiss-cpu")
                raise ImportError("FAISS library not found but required for pre-computed embeddings")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            A list of floats representing the embedding
        """
        # Check if embedding is in cache
        if self.embedding_cache and text in self.embedding_dict:
            return self.embedding_dict[text]
        
        # Generate embedding
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded, but trying to generate embedding")
                
            embedding = self.model.encode(text, convert_to_numpy=True).tolist()
            
            # Store in cache
            if self.embedding_cache:
                self.embedding_dict[text] = embedding
                # Save cache periodically (every 100 new embeddings)
                if len(self.embedding_dict) % 100 == 0:
                    self._save_cache()
                    
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")
    
    def _save_cache(self) -> None:
        """Save the embedding cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.embedding_dict, f)
            logger.info(f"Saved {len(self.embedding_dict)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Could not save embedding cache: {str(e)}")
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (float between -1 and 1)
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot = np.dot(vec1, vec2)
        norm_a = norm(vec1)
        norm_b = norm(vec2)
        
        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0
            
        return dot / (norm_a * norm_b)
    
    def _search_category_faiss(self, product_id: str, category: str, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """
        Search for similar questions in a specific product category using FAISS.
        
        Args:
            product_id: Product ID to search in
            category: Category to search in
            query_embedding: The embedding vector of the query
            top_k: Maximum number of results to return
            
        Returns:
            List of dictionaries with question, cq_id, and similarity
        """
        try:
            # Format product ID for directory name
            formatted_product = product_id.replace(" ", "_")
            
            # Get file paths
            product_dir = MODELS_DIR / formatted_product
            index_path = product_dir / f"{category}_index.faiss"
            embeddings_path = product_dir / f"{category}_embeddings.npy"
            metadata_path = product_dir / "product_metadata.json"
            
            # Ensure files exist
            if not index_path.exists() or not embeddings_path.exists() or not metadata_path.exists():
                logger.warning(f"Missing files for {formatted_product}/{category}")
                return []
            
            # Load product metadata
            with open(metadata_path, 'r') as f:
                product_metadata = json.load(f)
            
            # Check if category exists in metadata
            if "categories" not in product_metadata or category not in product_metadata["categories"]:
                logger.warning(f"Category {category} not found in product metadata for {formatted_product}")
                return []
            
            category_data = product_metadata["categories"][category]
            question_ids = category_data.get("question_ids", [])
            questions = category_data.get("questions", [])
            
            # Load FAISS index
            index = faiss.read_index(str(index_path))
            
            # Search the index
            query_embedding_np = np.array([query_embedding]).astype('float32')
            distances, indices = index.search(query_embedding_np, top_k)
            
            # Prepare results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(questions):  # Ensure index is valid
                    # Convert distance to similarity score (1 - normalized_distance)
                    similarity = 1 - (distance / 100)  # Normalize the distance
                    
                    results.append({
                        "cq_id": question_ids[idx] if idx < len(question_ids) else "",
                        "question": questions[idx],
                        "similarity": float(similarity)
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error in FAISS search for {product_id}/{category}: {str(e)}")
            return []
    
    def find_similar_questions(
        self, 
        query: str, 
        product_id: str, 
        category: Optional[str] = None, 
        threshold: float = 0.7, 
        top_k: int = 5
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Find questions similar to the query within the same product and category.
        
        Args:
            query: The question to find similar questions for
            product_id: The product ID to search within
            category: Optional category to search within
            threshold: Minimum similarity threshold
            top_k: Maximum number of results to return
            
        Returns:
            List of dictionaries with question, cq_id, and similarity
        """
        logger.info(f"Finding similar questions for query: {query[:50]}...")
        formatted_product = product_id.replace(" ", "_")
        
        # Generate embedding for query
        try:
            if self.model is not None:
                query_embedding = self.model.encode(query, convert_to_numpy=True)
            else:
                # If no model loaded, we need to load it temporarily
                logger.info("Loading embedding model temporarily for query encoding")
                temp_model = SentenceTransformer(self.model_name)
                query_embedding = temp_model.encode(query, convert_to_numpy=True)
                del temp_model  # Free up memory
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            return []
        
        # Check if product exists in models directory
        product_dir = MODELS_DIR / formatted_product
        if not product_dir.exists() or not product_dir.is_dir():
            logger.warning(f"Product directory not found: {product_dir}")
            return []
        
        # Get available categories
        try:
            with open(product_dir / "product_metadata.json", 'r') as f:
                product_metadata = json.load(f)
                
            available_categories = []
            if "categories" in product_metadata:
                available_categories = list(product_metadata["categories"].keys())
        except Exception as e:
            logger.error(f"Error loading product metadata: {str(e)}")
            return []
        
        # If category is specified, search only that category
        # Otherwise, search all categories
        categories_to_search = [category] if category and category in available_categories else available_categories
        
        # Search each category
        all_results = []
        for cat in categories_to_search:
            logger.info(f"Searching category: {cat}")
            cat_results = self._search_category_faiss(
                product_id=product_id,
                category=cat,
                query_embedding=query_embedding,
                top_k=top_k
            )
            all_results.extend(cat_results)
        
        # Filter by threshold
        all_results = [r for r in all_results if r["similarity"] >= threshold]
        
        # Sort by similarity (highest first) and get top_k
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return all_results[:top_k]

# Main function to run as a pipeline
def similarity_pipeline(
    query: str, 
    product_id: str, 
    category: Optional[str] = None, 
    threshold: float = 0.7,
    top_k: int = 5
) -> List[Dict[str, Union[str, float]]]:
    """
    Run the similarity check pipeline.
    
    Args:
        query: The user's question
        product_id: The product ID
        category: Optional category
        threshold: Minimum similarity threshold
        top_k: Maximum number of results to return
        
    Returns:
        List of similar questions with cq_id and similarity scores
    """
    try:
        # Initialize similarity checker
        checker = SimilarityCheck()
        
        # Find similar questions
        similar_questions = checker.find_similar_questions(
            query=query,
            product_id=product_id,
            category=category,
            threshold=threshold,
            top_k=top_k
        )
        
        return similar_questions
    except Exception as e:
        logger.error(f"Error in similarity pipeline: {str(e)}")
        return []

# Sample usage
if __name__ == "__main__":
    # Example query
    test_query = "How do I reset my password?"
    test_product = "Danfe_Corp_Product_1"
    
    # Run similarity pipeline
    results = similarity_pipeline(
        query=test_query,
        product_id=test_product,
        threshold=0.6,
        top_k=3
    )
    
    # Print results
    print(f"Query: {test_query}")
    print(f"Product: {test_product}")
    print(f"Results: {len(results)}")
    
    for i, result in enumerate(results):
        print(f"{i+1}. {result['question']} (ID: {result['cq_id']}, Similarity: {result['similarity']:.2f})")
