import pandas as pd
import numpy as np
import io
import requests
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import hdbscan
import os

# Define paths
DATA_DIR = "../data"
OUTPUT_DIR = "../output"
MODELS_DIR = "../models"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    """Load data from local files in the data directory"""
    local_file1 = os.path.join(DATA_DIR, "AnswerLibraryEntry.csv")
    local_file2 = os.path.join(DATA_DIR, "CanonicalQuestionProducts.csv")
    local_file3 = os.path.join(DATA_DIR, "Product.csv")
    
    # Check if all required files exist
    if not os.path.exists(local_file1) or not os.path.exists(local_file2):
        raise FileNotFoundError(f"Required data files not found in {DATA_DIR}. Please ensure AnswerLibraryEntry.csv and CanonicalQuestionProducts.csv exist.")
    
    print("Loading data from local files...")
    ans = pd.read_csv(local_file1)
    can = pd.read_csv(local_file2)
    
    # Load products if available
    products = None
    if os.path.exists(local_file3):
        products = pd.read_csv(local_file3)
    else:
        print(f"Warning: Product data file {local_file3} not found.")
    
    return ans, can, products

def filter_by_product(merged_df, product_id=None):
    """
    Filter merged dataframe by product_id if specified
    
    Args:
        merged_df: DataFrame with merged data
        product_id: The product_id to filter by, or None for all data
        
    Returns:
        Filtered DataFrame
    """
    if product_id is None or product_id == "all":
        return merged_df
    
    # Filter by product_id
    filtered_df = merged_df[merged_df['product_id'] == product_id]
    
    if len(filtered_df) == 0:
        print(f"Warning: No data found for product_id {product_id}")
        return merged_df  # Return original data if no matching rows
    
    print(f"Filtered data from {len(merged_df)} to {len(filtered_df)} rows for product_id {product_id}")
    return filtered_df

def get_product_list():
    """
    Load and return the list of available products
    
    Returns:
        DataFrame with product data or None if not available
    """
    product_path = os.path.join(DATA_DIR, "Product.csv")
    if os.path.exists(product_path):
        products_df = pd.read_csv(product_path)
        return products_df
    return None

def preprocess_data(ans, can, product_id=None):
    """
    Preprocess and merge the dataframes, optionally filtering by product_id
    
    Args:
        ans: Answer library dataframe
        can: Canonical questions dataframe
        product_id: Optional product ID to filter by
        
    Returns:
        Merged and filtered dataframe
    """
    # Rearranging columns in canonical questions dataframe
    cols = ['product_id', 'cqid'] + [col for col in can.columns if col not in ['product_id', 'cqid']]
    can = can[cols]
    
    # Renaming id with cqid in answer library dataframe
    ans = ans.rename(columns={"id": "cqid"})
    
    # Merge dataframes
    merged = pd.merge(can, ans, on="cqid", how="right")
    
    # Filter by product if specified
    if product_id and product_id != "all":
        merged = filter_by_product(merged, product_id)
    
    return merged

def generate_embeddings(merged_df, product_id=None, model_name='sentence-transformers/all-MiniLM-L6-v2', save=True):
    """
    Generate embeddings for questions using sentence transformer model
    
    Args:
        merged_df: DataFrame with question data
        product_id: Optional product ID to identify the embeddings
        model_name: Name of the transformer model to use
        save: Whether to save the embeddings to file
        
    Returns:
        Tuple of (embeddings, model)
    """
    print("Generating embeddings...")

    model = SentenceTransformer(model_name)
    
    # Determine the embeddings file path based on product_id
    embeddings_file = os.path.join(MODELS_DIR, "qna_embeddings.npy")
    if product_id and product_id != "all":
        # Create product-specific embeddings file
        embeddings_file = os.path.join(MODELS_DIR, f"qna_embeddings_{product_id}.npy")
    
    print(f"Embeddings file: {embeddings_file}")
    # Check if embeddings already exist
    if os.path.exists(embeddings_file):
        print(f"Loading embeddings from file {embeddings_file}...")
        return np.load(embeddings_file), model
    
    print(f"Generating new embeddings for {'product ' + product_id if product_id and product_id != 'all' else 'all products'}...")
    
    # Generate embeddings for questions
    questions = merged_df['question'].tolist()
    embeddings = model.encode(questions)
    
    if save:
        np.save(embeddings_file, embeddings)
        print(f"Saved embeddings to {embeddings_file}")
    
    return embeddings, model

# For backward compatibility
generate_embeddings_for_product = generate_embeddings

def create_similarity_index(embeddings):
    """Create FAISS index for similarity search"""
    # Normalize embeddings for cosine similarity
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product = Cosine similarity for normalized vectors
    index.add(norm_embeddings)
    
    return index, norm_embeddings

def find_similar_pairs(index, embeddings, df, k=2, similarity_threshold=0.6):

    print("this funstion was activated")
    """Find similar question pairs based on embeddings"""
    # Search for top k most similar items for each entry
    similarities, indices = index.search(embeddings, k)

    print("its lagging here")
    
    
    # Store pairs with high similarity (> threshold) but exclude self-matches
    similar_pairs = []
    for i in range(len(indices)):
        for j in range(1, k):  # skip j=0 (self match)
            if similarities[i][j] > similarity_threshold:
                similar_pairs.append({
                    "index_1": i,
                    "index_2": indices[i][j],
                    "similarity": similarities[i][j],
                    "cqid_1": df.iloc[i]['cqid'],
                    "cqid_2": df.iloc[indices[i][j]]['cqid'],
                    "question_1": df.iloc[i]['question'],
                    "question_2": df.iloc[indices[i][j]]['question'],
                    "product_id": df.iloc[i]['product_id']
                })
    
    similar_df = pd.DataFrame(similar_pairs)
    return similar_df

def find_potential_duplicates(index, embeddings, df, similarity_threshold=0.85):
    """
    Find potential duplicate questions based on high semantic similarity
    
    Args:
        index: FAISS index built from embeddings
        embeddings: The embeddings matrix
        df: DataFrame containing the questions
        similarity_threshold: Threshold above which questions are considered potential duplicates
        
    Returns:
        DataFrame containing pairs of potential duplicate questions with their similarity scores
    """
    # Check if dataframe is too small for duplicate detection
    if len(df) <= 1:
        print("DataFrame has too few rows for duplicate detection")
        return pd.DataFrame()  # Return empty dataframe
    
    # Ensure index and embeddings match dataframe size
    if len(embeddings) != len(df):
        print(f"Warning: Embeddings size ({len(embeddings)}) doesn't match dataframe size ({len(df)}). Regenerating embeddings.")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(df['question'].tolist())
        # Recreate index
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(norm_embeddings)
    
    # Search for top k most similar items for each entry
    k = min(10, len(df))  # Don't try to get more neighbors than there are rows
    similarities, indices = index.search(embeddings, k)
    
    # Store pairs with very high similarity (> threshold) excluding self-matches
    duplicate_pairs = []
    
    # Track pairs we've already identified to avoid duplicates
    seen_pairs = set()
    
    for i in range(len(indices)):
        for j in range(1, min(k, len(indices[i]))):  # skip j=0 (self match) and ensure j is within bounds
            # Check if the index is valid
            if j >= len(similarities[i]) or indices[i][j] >= len(df):
                continue  # Skip invalid indices
                
            if similarities[i][j] > similarity_threshold:
                # Ensure both indices are valid
                if i < len(df) and indices[i][j] < len(df):
                    # Create a unique key for this pair (using the smaller index first)
                    try:
                        pair_id = tuple(sorted([df.iloc[i]['cqid'], df.iloc[indices[i][j]]['cqid']]))
                        
                        # Only add if we haven't seen this pair before
                        if pair_id not in seen_pairs:
                            seen_pairs.add(pair_id)
                            
                            # Check if one of the questions is archived
                            is_archived_1 = pd.notna(df.iloc[i]['deleted_at'])
                            is_archived_2 = pd.notna(df.iloc[indices[i][j]]['deleted_at'])
                            
                            # Get the creation dates to determine which is newer
                            created_at_1 = pd.to_datetime(df.iloc[i]['created_at'])
                            created_at_2 = pd.to_datetime(df.iloc[indices[i][j]]['created_at'])
                            newer_question_idx = i if created_at_1 > created_at_2 else indices[i][j]
                            
                            duplicate_pairs.append({
                                "cqid_1": df.iloc[i]['cqid'],
                                "cqid_2": df.iloc[indices[i][j]]['cqid'],
                                "question_1": df.iloc[i]['question'],
                                "question_2": df.iloc[indices[i][j]]['question'],
                                "similarity": similarities[i][j],
                                "category_1": df.iloc[i]['category'],
                                "category_2": df.iloc[indices[i][j]]['category'],
                                "is_archived_1": is_archived_1,
                                "is_archived_2": is_archived_2,
                                "newer_question": df.iloc[newer_question_idx]['question'],
                                "newer_cqid": df.iloc[newer_question_idx]['cqid']
                            })
                    except (KeyError, IndexError) as e:
                        # If any required column is missing or index is out of bounds, skip this pair
                        print(f"Error processing pair {i} and {indices[i][j]}: {str(e)}")
                        continue
    
    duplicate_df = pd.DataFrame(duplicate_pairs)
    
    # Sort by similarity score descending
    if not duplicate_df.empty:
        duplicate_df = duplicate_df.sort_values('similarity', ascending=False)
    
    return duplicate_df

def find_similar_to_new_question(question_text, model, index, embeddings, df, top_k=5, similarity_threshold=0.7):
    """
    Find semantically similar questions to a new input question
    
    Args:
        question_text: The new question to compare against the database
        model: The sentence transformer model
        index: FAISS index built from embeddings
        embeddings: The embeddings matrix
        df: DataFrame containing the questions
        top_k: Number of similar questions to return
        similarity_threshold: Threshold above which to consider questions similar
        
    Returns:
        List of dictionaries containing similar questions and their similarity scores
    """
    # Encode the query
    query_embedding = model.encode([question_text])
    
    # Normalize the query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Search for similar questions
    similarities, indices = index.search(query_embedding, top_k)
    
    # Get the results
    similar_questions = []
    for i in range(top_k):
        if similarities[0][i] >= similarity_threshold:
            similar_questions.append({
                "question": df.iloc[indices[0][i]]['question'],
                "similarity": similarities[0][i],
                "cqid": df.iloc[indices[0][i]]['cqid'],
                "category": df.iloc[indices[0][i]]['category'],
                "answer": df.iloc[indices[0][i]]['answer'],
                "details": df.iloc[indices[0][i]]['details'],
                "is_archived": pd.notna(df.iloc[indices[0][i]]['deleted_at'])
            })
    
    return similar_questions

def perform_clustering(embeddings, min_cluster_size=10, min_samples=1):
    """Perform dimensionality reduction and clustering on embeddings"""
    # Check if reduced embeddings already exist
    reduced_embeddings_file = os.path.join(MODELS_DIR, "qna_embeddings_reduced.npy")
    if os.path.exists(reduced_embeddings_file):
        print("Loading reduced embeddings from file...")
        X_reduced = np.load(reduced_embeddings_file)
    else:
        print("Performing dimensionality reduction...")
        # Reduce dimensionality with PCA
        X_reduced = PCA(n_components=50).fit_transform(embeddings)
        np.save(reduced_embeddings_file, X_reduced)
    
    print("Clustering data...")
    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=0.0
    )
    
    labels = clusterer.fit_predict(X_reduced)
    return labels, X_reduced

def separate_deleted_data(merged_df):
    """Separate data into deleted and non-deleted dataframes"""
    deleted_df = merged_df[merged_df['deleted_at'].notna()]
    not_deleted_df = merged_df[merged_df['deleted_at'].isna()]
    
    return deleted_df, not_deleted_df

def search_similar_questions(query_text, model, index, embeddings, df, top_k=5):
    """Search for similar questions to a given query"""
    # Encode the query
    query_embedding = model.encode([query_text])
    
    # Normalize the query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Search for similar questions
    similarities, indices = index.search(query_embedding, top_k)
    
    # Get the results
    results = []
    for i in range(top_k):
        results.append({
            "question": df.iloc[indices[0][i]]['question'],
            "similarity": similarities[0][i],
            "answer": df.iloc[indices[0][i]]['answer'],
            "details": df.iloc[indices[0][i]]['details'],
            "category": df.iloc[indices[0][i]]['category']
        })
    
    return results

def generate_duplicate_report(duplicate_df, output_path=None):
    """
    Generate a report summarizing potential duplicate questions
    
    Args:
        duplicate_df: DataFrame of duplicate pairs from find_potential_duplicates
        output_path: Path to save the report, if None, report is not saved
        
    Returns:
        DataFrame with statistics on potential duplicates
    """
    # If the dataframe is empty, return empty report
    if duplicate_df.empty:
        print("No potential duplicates found.")
        return pd.DataFrame()
    
    # Count total unique questions involved in duplicates
    unique_questions = set()
    for _, row in duplicate_df.iterrows():
        unique_questions.add(row['cqid_1'])
        unique_questions.add(row['cqid_2'])
    
    # Get category pairs that have duplicates
    category_pairs = duplicate_df.groupby(['category_1', 'category_2']).size().reset_index(name='count')
    category_pairs = category_pairs.sort_values('count', ascending=False)
    
    # Count pairs where one is archived
    archived_pairs = duplicate_df[
        (duplicate_df['is_archived_1'] & ~duplicate_df['is_archived_2']) | 
        (~duplicate_df['is_archived_1'] & duplicate_df['is_archived_2'])
    ]
    
    # Count pairs where both are active
    active_pairs = duplicate_df[
        (~duplicate_df['is_archived_1'] & ~duplicate_df['is_archived_2'])
    ]
    
    # Summary statistics
    summary = {
        'Total Potential Duplicate Pairs': len(duplicate_df),
        'Unique Questions Involved': len(unique_questions),
        'Pairs with One Archived Question': len(archived_pairs),
        'Pairs with Both Active Questions': len(active_pairs),
        'Average Similarity Score': duplicate_df['similarity'].mean()
    }
    
    summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
    
    # Save report if output path is provided
    if output_path:
        # Save the full duplicate pairs dataframe
        duplicate_df.to_csv(output_path, index=False)
        print(f"Duplicate report saved to {output_path}")
    
    return summary_df

def main(product_id=None):
    """
    Main pipeline function that processes data, optionally filtered by product_id
    
    Args:
        product_id: Optional product ID to filter by
        
    Returns:
        Tuple of (merged_df, embeddings, model, index, norm_embeddings)
    """
    # Load products data (for reference)
    products_df = get_product_list()
    product_name = "All Products"
    
    if products_df is not None and product_id and product_id != "all":
        product_row = products_df[products_df['product_id'] == product_id]
        if not product_row.empty:
            product_name = product_row.iloc[0]['product_name']
    
    # Load data
    print("Loading data...")
    ans_df, can_df, _ = load_data()
    
    # Preprocess data with product filtering as the first step
    print(f"Preprocessing data for {product_name}...")
    merged_df = preprocess_data(ans_df, can_df, product_id)
    
    # Save merged data
    output_path = os.path.join(OUTPUT_DIR, "merged_data.csv")
    if product_id and product_id != "all":
        output_path = os.path.join(OUTPUT_DIR, f"merged_data_{product_id}.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"Merged data saved to {output_path}")
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings, model = generate_embeddings(merged_df, product_id)
    
    # Create similarity index
    print("Creating similarity index...")
    index, norm_embeddings = create_similarity_index(embeddings)
    
    # Find similar pairs
    print("Finding similar pairs...")
    similar_pairs_df = find_similar_pairs(index, norm_embeddings, merged_df)
    similar_pairs_path = os.path.join(OUTPUT_DIR, "similar_qna_pairs.csv")
    if product_id and product_id != "all":
        similar_pairs_path = os.path.join(OUTPUT_DIR, f"similar_qna_pairs_{product_id}.csv")
    similar_pairs_df.to_csv(similar_pairs_path, index=False)
    print(f"Similar pairs saved to {similar_pairs_path}")
    
    # Find potential duplicates
    print("Finding potential duplicate questions...")
    duplicate_df = find_potential_duplicates(index, norm_embeddings, merged_df, similarity_threshold=0.85)
    duplicates_path = os.path.join(OUTPUT_DIR, "potential_duplicates.csv")
    if product_id and product_id != "all":
        duplicates_path = os.path.join(OUTPUT_DIR, f"potential_duplicates_{product_id}.csv")
    summary_df = generate_duplicate_report(duplicate_df, duplicates_path)
    print("Duplicate Detection Summary:")
    print(summary_df)
    
    # Perform clustering
    print("Performing clustering...")
    labels, reduced_embeddings = perform_clustering(embeddings)
    
    # Add cluster labels to merged dataframe
    merged_df['cluster'] = labels
    clustered_data_path = os.path.join(OUTPUT_DIR, "clustered_data.csv")
    if product_id and product_id != "all":
        clustered_data_path = os.path.join(OUTPUT_DIR, f"clustered_data_{product_id}.csv")
    merged_df.to_csv(clustered_data_path, index=False)
    print(f"Clustered data saved to {clustered_data_path}")
    
    # Separate deleted and non-deleted data
    print("Separating deleted and non-deleted data...")
    deleted_df, not_deleted_df = separate_deleted_data(merged_df)
    deleted_data_path = os.path.join(OUTPUT_DIR, "deleted_data.csv")
    not_deleted_data_path = os.path.join(OUTPUT_DIR, "not_deleted_data.csv")
    if product_id and product_id != "all":
        deleted_data_path = os.path.join(OUTPUT_DIR, f"deleted_data_{product_id}.csv")
        not_deleted_data_path = os.path.join(OUTPUT_DIR, f"not_deleted_data_{product_id}.csv")
    deleted_df.to_csv(deleted_data_path, index=False)
    not_deleted_df.to_csv(not_deleted_data_path, index=False)
    print(f"Deleted data saved to {deleted_data_path}")
    print(f"Non-deleted data saved to {not_deleted_data_path}")
    
    print("Pipeline completed successfully!")
    
    # Example of how to use the search function
    print("\nExample search:")
    query = "What is the information security policy?"
    results = search_similar_questions(query, model, index, norm_embeddings, merged_df)
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result['question']} (Similarity: {result['similarity']:.4f})")
        if result['answer'] is not None:
            print(f"Answer: {result['answer']}")
    
    return merged_df, embeddings, model, index, norm_embeddings

if __name__ == "__main__":
    main() 