#!/usr/bin/env python
"""
Script to generate embeddings for all products and save them using FAISS indices.
This creates optimized similarity search indices that can be used by the API.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("generate_embeddings.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("generate_embeddings")

def main():
    """
    Main function to generate embeddings for all products and categories.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate embeddings for all products")
    parser.add_argument("--output-dir", default="output", help="Directory containing organized data (default: output)")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2", help="Name of the sentence transformer model (default: all-MiniLM-L6-v2)")
    parser.add_argument("--models-dir", default="models", help="Directory to save embeddings and indices (default: models)")
    parser.add_argument("--test-query", default=None, help="Optional query to test search functionality after generation")
    args = parser.parse_args()
    
    logger.info("Starting embedding generation process")
    
    try:
        # Dynamically import the EmbeddingPipeline from periodic_script
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from periodic_script.embedding import EmbeddingPipeline, normalize_path
        
        # Normalize paths
        output_dir = normalize_path(args.output_dir)
        models_dir = normalize_path(args.models_dir)
        
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Models directory: {models_dir}")
        
        if not os.path.exists(output_dir):
            logger.error(f"Output directory does not exist: {output_dir}")
            return 1
        
        # Create the models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize the embedding pipeline
        pipeline = EmbeddingPipeline(
            output_dir=output_dir,
            model_name=args.model_name,
            models_dir=models_dir
        )
        
        # Generate embeddings for all products and categories
        logger.info("Starting embeddings generation")
        embeddings_data = pipeline.create_embeddings()
        logger.info("Embeddings generation complete")
        
        # Log summary of generated embeddings
        for product, product_data in embeddings_data.get("products", {}).items():
            categories = product_data.get("categories", {})
            logger.info(f"Generated embeddings for product {product}: {len(categories)} categories")
            
            # Log detailed category information
            for category in categories:
                category_data = categories[category]
                question_count = len(category_data.get("questions", []))
                logger.info(f"  - {category}: {question_count} questions")
        
        # Test the search functionality if a test query was provided
        if args.test_query:
            logger.info(f"Testing search with query: '{args.test_query}'")
            
            # Search across all products
            results = pipeline.search_similar_questions(args.test_query, top_k=5)
            
            logger.info(f"Top 5 search results for '{args.test_query}':")
            for product, category, question, question_id, similarity in results:
                logger.info(f"Product: {product}")
                logger.info(f"Category: {category}")
                logger.info(f"Question: {question}")
                logger.info(f"Question ID: {question_id}")
                logger.info(f"Similarity: {similarity:.4f}")
                logger.info("-" * 40)
                
        logger.info("Embedding generation process completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error in embedding generation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 