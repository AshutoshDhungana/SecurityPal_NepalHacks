from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from rapidfuzz import fuzz
import pandas as pd
import os
import json
from datetime import datetime
import nltk
import re
import sys
import uuid
from pathlib import Path


# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Path for audit logs and merged data storage
AUDIT_DIR = "../audit_logs"
OUTPUT_DIR = "../output" 
MERGED_DATA_DIR = "../output/merged_data"

# Create necessary directories
os.makedirs(AUDIT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MERGED_DATA_DIR, exist_ok=True)

# Path to merged questions database
MERGED_DB_PATH = os.path.join(MERGED_DATA_DIR, "merged_questions.csv")

def initialize_nltk():
    """Initialize NLTK resources explicitly with proper error handling"""
    nltk_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../nltk_data")
    os.makedirs(nltk_dir, exist_ok=True)
    nltk.data.path.append(nltk_dir)
    
    print(f"NLTK will download data to: {nltk_dir}")
    try:
        nltk.download('punkt', download_dir=nltk_dir, quiet=False)
        print("NLTK punkt downloaded successfully")
        return True
    except Exception as e:
        print(f"Failed to download NLTK resources: {str(e)}")
        print("Will use fallback sentence splitting")
        return False

# Initialize NLTK at import time
NLTK_AVAILABLE = initialize_nltk()

# Initialize the summarization model - using DistilBART which is smaller and faster than Pegasus
print("Loading summarization model...")
try:
    # Use DistilBART-CNN instead of Pegasus for better performance
    model_name = "sshleifer/distilbart-cnn-12-6"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    print(f"DistilBART summarization model loaded and running on {device}")
    SUMMARIZER_AVAILABLE = True
except Exception as e:
    print(f"Failed to load summarization model: {str(e)}")
    print("Will skip summarization step")
    SUMMARIZER_AVAILABLE = False

# Alternative sentence splitting function using regex in case NLTK fails
def regex_split_sentences(text):
    """Split text into sentences using regex as a fallback if NLTK fails"""
    if not text or pd.isna(text):
        return []
    
    # Clean the text
    text = re.sub(r'\s+', ' ', str(text).strip())
    
    # Simple regex-based sentence splitting
    # Split on periods, question marks, or exclamation points followed by a space and uppercase letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]

def split_sentences(text):
    """Split text into sentences for better merging"""
    if not text or pd.isna(text):
        return []
    
    # Clean the text
    text = re.sub(r'\s+', ' ', str(text).strip())
    
    try:
        # Try to use NLTK's sentence tokenizer
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except LookupError as e:
        # Fall back to regex-based splitting if NLTK data is not available
        print(f"NLTK tokenizer error: {str(e)}")
        print("Falling back to regex-based sentence splitting")
        return regex_split_sentences(text)

def merge_details(detail_a, detail_b):
    """Merge details from two questions by normalizing, splitting, and deduplicating"""
    # Handle NaN values
    detail_a = "" if pd.isna(detail_a) else str(detail_a)
    detail_b = "" if pd.isna(detail_b) else str(detail_b)
    
    # Split into sentences
    sents_a = split_sentences(detail_a)
    sents_b = split_sentences(detail_b)

    # Apply fuzzy deduplication to the combined sentences
    merged_sents = fuzzy_union(sents_a + sents_b)
    
    # Join back into a single text
    merged_text = ' '.join(merged_sents)
    return merged_text

def fuzzy_union(sents):
    """Deduplicate sentences using fuzzy matching to remove near-duplicates"""
    unique = []
    for s in sents:
        # Check if this sentence is too similar to any already included sentence
        if not any(fuzz.ratio(s.lower(), u.lower()) > 90 for u in unique):
            unique.append(s)
    return unique

def summarize_text(text, max_length=512, min_length=100):
    """Use DistilBART model to make the merged text more coherent and human-like"""
    if not SUMMARIZER_AVAILABLE:
        return text  # Return original text if summarizer isn't available
        
    if not text or len(text) < min_length:
        return text  # Don't summarize short texts
    
    try:
        # Tokenize and generate summary
        inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(device)
        summary_ids = model.generate(
            inputs["input_ids"], 
            num_beams=4,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            early_stopping=True
        )
        
        # Decode the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print(f"Summarization error: {str(e)}")
        return text  # Return original text if summarization fails

def save_merged_question_to_db(merged_data, original_ids=None):
    """
    Save a merged question to the merged questions database
    
    Args:
        merged_data: Dictionary with merged_question, merged_details, merged_answer
        original_ids: List of original question IDs that were merged
        
    Returns:
        new_id: The ID of the newly created merged question
    """
    # Generate a unique ID for the merged question
    new_id = str(uuid.uuid4())
    
    # Current timestamp
    timestamp = datetime.now().isoformat()
    
    # Create a record for the merged question
    merged_record = {
        'cqid': new_id,
        'question': merged_data['merged_question'],
        'details': merged_data['merged_details'],
        'answer': merged_data['merged_answer'],
        'category': merged_data.get('category', ''),
        'product_id': merged_data.get('product_id', ''),
        'created_at': timestamp,
        'updated_at': timestamp,
        'deleted_at': None,
        'is_merged': True,
        'original_ids': ','.join(original_ids) if original_ids else '',
        'merge_date': timestamp
    }
    
    # Load existing database or create new one
    if os.path.exists(MERGED_DB_PATH):
        merged_db = pd.read_csv(MERGED_DB_PATH)
        # Append the new record
        merged_db = pd.concat([merged_db, pd.DataFrame([merged_record])], ignore_index=True)
    else:
        # Create a new database with this record
        merged_db = pd.DataFrame([merged_record])
    
    # Save the updated database
    merged_db.to_csv(MERGED_DB_PATH, index=False)
    
    # Also save all merged questions to a single JSON file for reference
    json_path = os.path.join(MERGED_DATA_DIR, f"merged_question_{new_id}.json")
    with open(json_path, 'w') as f:
        json.dump(merged_record, f, indent=2)
    
    return new_id

def get_merged_questions():
    """
    Get all merged questions from the database
    
    Returns:
        DataFrame containing all merged questions
    """
    if os.path.exists(MERGED_DB_PATH):
        return pd.read_csv(MERGED_DB_PATH)
    else:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            'cqid', 'question', 'details', 'answer', 'category', 'product_id',
            'created_at', 'updated_at', 'deleted_at', 'is_merged', 'original_ids', 'merge_date'
        ])

def archive_original_questions(df, original_ids):
    """
    Mark original questions as archived/deleted after merging
    
    Args:
        df: DataFrame containing the original questions
        original_ids: List of IDs of questions to archive
        
    Returns:
        Updated DataFrame with archived questions
    """
    updated_df = df.copy()
    
    # Mark questions as deleted/archived
    for qid in original_ids:
        if qid in updated_df['cqid'].values:
            # Find the index of the question to archive
            idx = updated_df[updated_df['cqid'] == qid].index
            if not idx.empty:
                # Set deleted_at to current timestamp
                updated_df.loc[idx, 'deleted_at'] = datetime.now().isoformat()
                # Add a note that it was merged
                if 'notes' in updated_df.columns:
                    updated_df.loc[idx, 'notes'] = "Archived due to merge with similar question"
    
    return updated_df

def merge_questions(question_a, question_b, details_a=None, details_b=None, answer_a=None, answer_b=None, 
                    category_a=None, category_b=None, product_id=None, cqid_a=None, cqid_b=None):
    """
    Merge two similar questions along with their details and answers.
    Returns the merged question, details, answer, and logs the merge operation.
    """
    # For questions, use the longer one as it's likely more descriptive
    merged_question = question_a if len(question_a) >= len(question_b) else question_b
    
    # Merge details using fuzzy deduplication
    merged_details = merge_details(details_a, details_b)
    
    # For answers, prefer non-empty ones and use the longer if both exist
    if pd.isna(answer_a) or not answer_a:
        merged_answer = answer_b
    elif pd.isna(answer_b) or not answer_b:
        merged_answer = answer_a
    else:
        # Both answers exist, use the longer one
        merged_answer = answer_a if len(str(answer_a)) >= len(str(answer_b)) else answer_b
    
    # For category, use the first one that exists
    merged_category = category_a if category_a else category_b
    
    # For longer merged details, use summarization to make it more coherent
    if len(merged_details) > 300:  # Only summarize longer texts
        improved_details = summarize_text(merged_details)
        # Fall back to original if summarization failed or made it too short
        if len(improved_details) < len(merged_details) * 0.3:
            improved_details = merged_details
    else:
        improved_details = merged_details
    
    # Log the merge operation
    log_merge(question_a, question_b, merged_question, details_a, details_b, improved_details)
    
    # Create result object
    result = {
        "merged_question": merged_question,
        "merged_details": improved_details,
        "merged_answer": merged_answer,
        "category": merged_category,
        "product_id": product_id
    }
    
    # Save to database if IDs are provided
    if cqid_a and cqid_b:
        result["new_id"] = save_merged_question_to_db(result, original_ids=[cqid_a, cqid_b])
    
    return result

def log_merge(q1, q2, merged_q, d1, d2, merged_d):
    """Log merge operations for audit purposes"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question_1": q1,
        "question_2": q2,
        "merged_question": merged_q,
        "details_1": str(d1),
        "details_2": str(d2),
        "merged_details": merged_d
    }
    
    # Create a log file for this day
    log_file = os.path.join(AUDIT_DIR, f"merge_log_{datetime.now().strftime('%Y%m%d')}.jsonl")
    
    # Append to the log file
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    return log_file

def batch_save_merged_questions(merged_df, original_df=None):
    """
    Save all merged questions from a batch process and optionally archive originals
    
    Args:
        merged_df: DataFrame with merged questions from batch_process_similar_questions
        original_df: Original DataFrame containing all questions (to archive originals)
        
    Returns:
        Tuple of (merged_ids, updated_original_df)
    """
    merged_ids = []
    
    # Process each merged question
    for _, row in merged_df.iterrows():
        # Extract data
        merged_data = {
            'merged_question': row['merged_question'],
            'merged_details': row['merged_details'],
            'merged_answer': row['merged_answer'],
            'category': row.get('category', ''),
            'product_id': row.get('product_id', '')
        }
        
        # Get original IDs
        original_ids = [
            row['original_id_1'] if pd.notna(row['original_id_1']) else None,
            row['original_id_2'] if pd.notna(row['original_id_2']) else None
        ]
        original_ids = [oid for oid in original_ids if oid]
        
        # Save to database
        if original_ids:
            new_id = save_merged_question_to_db(merged_data, original_ids)
            merged_ids.append(new_id)
    
    # Archive original questions if original_df is provided
    if original_df is not None and merged_ids:
        # Collect all original IDs
        all_original_ids = []
        for _, row in merged_df.iterrows():
            if pd.notna(row['original_id_1']):
                all_original_ids.append(row['original_id_1'])
            if pd.notna(row['original_id_2']):
                all_original_ids.append(row['original_id_2'])
        
        # Archive originals
        updated_df = archive_original_questions(original_df, all_original_ids)
        
        # Save updated original dataset
        original_path = os.path.join(OUTPUT_DIR, "updated_data.csv")
        updated_df.to_csv(original_path, index=False)
        
        return merged_ids, updated_df
    
    return merged_ids, original_df

def load_merged_with_original_data(original_df):
    """
    Load both original and merged data, with merged data replacing archived originals
    
    Args:
        original_df: DataFrame with original data
        
    Returns:
        Combined DataFrame with merged questions included and originals archived
    """
    # Get merged questions
    merged_df = get_merged_questions()
    
    if merged_df.empty:
        return original_df
    
    # Create a copy of the original data
    combined_df = original_df.copy()
    
    # Add merged questions to the dataset
    combined_df = pd.concat([combined_df, merged_df], ignore_index=True)
    
    return combined_df

def batch_process_similar_questions(similar_pairs_df, threshold=1.0, original_df=None):
    """
    Process a dataframe of similar question pairs and merge those above the threshold
    
    Args:
        similar_pairs_df: DataFrame with similar question pairs
        threshold: Similarity threshold (1.0 for exact matches)
        original_df: Original dataframe to extract category and product_id information
        
    Returns:
        DataFrame with merged questions
    """
    results = []
    
    # Filter by threshold
    high_similarity_pairs = similar_pairs_df[similar_pairs_df['similarity'] >= threshold]
    
    for _, row in high_similarity_pairs.iterrows():
        # Extract category and product_id if available from original_df
        category_1 = row.get('category_1', None)
        category_2 = row.get('category_2', None)
        product_id = row.get('product_id', None)
        
        merged = merge_questions(
            row['question_1'],
            row['question_2'],
            row.get('details_1', None),
            row.get('details_2', None),
            row.get('answer_1', None),
            row.get('answer_2', None),
            category_1,
            category_2,
            product_id,
            row.get('cqid_1', None),
            row.get('cqid_2', None)
        )
        
        # Add the similarity and IDs to track the merge
        merged['similarity'] = row['similarity']
        merged['original_id_1'] = row.get('cqid_1', None)
        merged['original_id_2'] = row.get('cqid_2', None)
        
        results.append(merged)
    
    # Convert to DataFrame
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame(columns=['merged_question', 'merged_details', 'merged_answer', 
                                    'similarity', 'original_id_1', 'original_id_2',
                                    'category', 'product_id', 'new_id'])

if __name__ == "__main__":
    # Test the functionality
    q1 = "What is your information security policy?"
    q2 = "Does your company have an information security policy?"
    d1 = "We need to understand your security policies. This includes documentation and implementation."
    d2 = "Please provide details about the security policy documentation and how it is implemented across the organization."
    
    result = merge_questions(q1, q2, d1, d2, "Answer A", "Answer B", "Security", "Security", "123", "ID1", "ID2")
    print("Merged Question:", result["merged_question"])
    print("Merged Details:", result["merged_details"])
    print("New ID:", result.get("new_id", "Not saved"))

