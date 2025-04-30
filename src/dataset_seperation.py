import pandas as pd
import os
import sys
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch

# Define directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_datasets():
    """Load all necessary datasets"""
    try:
        # Load Product data
        product_path = os.path.join(DATA_DIR, "Product.csv")
        products_df = pd.read_csv(product_path)
        print(f"Loaded {len(products_df)} products from {product_path}")
        
        # Load AnswerLibraryEntry data
        answer_path = os.path.join(DATA_DIR, "AnswerLibraryEntry.csv")
        answer_df = pd.read_csv(answer_path)
        # Rename 'id' column to 'cqid' to match with CanonicalQuestionProducts
        answer_df = answer_df.rename(columns={'id': 'cqid'})
        print(f"Loaded {len(answer_df)} answer entries from {answer_path}")
        
        # Load CanonicalQuestionProducts data
        question_path = os.path.join(DATA_DIR, "CanonicalQuestionProducts.csv")
        question_df = pd.read_csv(question_path)
        print(f"Loaded {len(question_df)} question entries from {question_path}")
        
        return products_df, answer_df, question_df
    
    except FileNotFoundError as e:
        print(f"Error: Could not find the required data file: {e}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: One of the data files is empty")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"Error: Could not parse CSV data - check file format")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        sys.exit(1)

def merge_datasets(answer_df, question_df):
    """Merge answer and question datasets"""
    try:
        # Merge on 'cqid' since we renamed the column in answer_df
        merged_df = pd.merge(question_df, answer_df, on='cqid', how='left')
        print(f"Created merged dataset with {len(merged_df)} rows")
        return merged_df
    except Exception as e:
        print(f"Error merging datasets: {e}")
        sys.exit(1)

def generate_embeddings(df, product_id=None, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Generate embeddings for questions and answers in the dataframe
    
    Args:
        df: Dataframe containing questions and answers
        product_id: Product ID to identify embeddings
        model_name: Name of the sentence-transformer model to use
        
    Returns:
        numpy array of embeddings
    """
    try:
        # Determine the embeddings file path based on product_id
        if product_id and product_id != "all":
            embeddings_file = os.path.join(MODELS_DIR, f"qna_embeddings_{product_id}.npy")
        else:
            embeddings_file = os.path.join(MODELS_DIR, "qna_embeddings.npy")

        # Check if embeddings already exist
        if os.path.exists(embeddings_file):
            print(f"Loading existing embeddings from {embeddings_file}")
            embeddings = np.load(embeddings_file)
            return embeddings
        
        # Initialize model
        print(f"Generating new embeddings for {'product ' + product_id if product_id else 'all products'}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        model = SentenceTransformer(model_name, device=device)
        
        # Prepare text to embed (combine question and answer)
        texts = []
        for _, row in df.iterrows():
            question = str(row['question']) if 'question' in row and not pd.isna(row['question']) else ""
            answer = str(row['answer']) if 'answer' in row and not pd.isna(row['answer']) else ""
            # Combine question and answer with a separator
            combined_text = question + " [SEP] " + answer
            texts.append(combined_text)
        
        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            batch_embeddings = model.encode(batch_texts, show_progress_bar=True)
            all_embeddings.append(batch_embeddings)
        
        # Combine all batches
        embeddings = np.vstack(all_embeddings)
        
        # Save embeddings
        np.save(embeddings_file, embeddings)
        print(f"Saved embeddings to {embeddings_file}")
        
        return embeddings
    
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def separate_by_product_id(merged_df, products_df):
    """Separate the merged dataset by product_id and generate embeddings for each"""
    try:
        # Get list of all product IDs
        product_ids = products_df['product_id'].tolist()
        
        # Create a combined dataset with all data for reference
        combined_output_path = os.path.join(OUTPUT_DIR, "combined_data.csv")
        merged_df.to_csv(combined_output_path, index=False)
        print(f"Saved combined dataset with {len(merged_df)} rows to {combined_output_path}")
        
        # Generate embeddings for all data combined
        print("Generating embeddings for all data combined...")
        generate_embeddings(merged_df)
        
        # Separate for each product ID
        for product_id in product_ids:
            if not pd.isna(product_id) and product_id.strip():
                # Filter data for this product ID
                product_data = merged_df[merged_df['product_id'] == product_id]
                
                if len(product_data) > 0:
                    # Save to a separate CSV file
                    output_path = os.path.join(OUTPUT_DIR, f"data_{product_id}.csv")
                    product_data.to_csv(output_path, index=False)
                    print(f"Saved {len(product_data)} rows for product {product_id} to {output_path}")
                    
                    # Generate embeddings for this product
                    print(f"Generating embeddings for product {product_id}...")
                    generate_embeddings(product_data, product_id)
                else:
                    print(f"No data found for product {product_id}")
    
    except Exception as e:
        print(f"Error separating datasets: {e}")
        sys.exit(1)

def main():
    print("Starting dataset separation and embedding generation process...")
    
    # Load datasets
    products_df, answer_df, question_df = load_datasets()
    
    # Merge question and answer datasets
    merged_df = merge_datasets(answer_df, question_df)
    
    # Separate by product ID and generate embeddings
    separate_by_product_id(merged_df, products_df)
    
    print("Dataset separation and embedding generation completed successfully!")

if __name__ == "__main__":
    main()
