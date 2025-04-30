import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("merging_pipeline")

# Load models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = hf_pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')

def compute_cosine_sim(str1, str2):
    """Compute cosine similarity between two strings using sentence embeddings."""
    emb1 = sentence_model.encode([str1])[0]
    emb2 = sentence_model.encode([str2])[0]
    sim = cosine_similarity([emb1], [emb2])[0][0]
    return sim

def merge_texts(texts, field_name):
    """Merge a list of texts using distilbart-cnn-12-6 summarization."""
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]
    # Concatenate and summarize
    combined = "\n".join([t for t in texts if t])
    if not combined.strip():
        return ""
    logger.info(f"Summarizing {field_name} for merge...")
    summary = summarizer(combined, max_length=128, min_length=10, do_sample=False)[0]['summary_text']
    return summary

def merge_questions(df, cq_ids, cqid_col='cqid'):
    merged_rows = []
    # Find all rows with the given cq_ids
    mask = df[cqid_col].isin(cq_ids)
    group = df[mask]
    if group.empty:
        logger.warning(f"No rows found for provided cq_ids: {cq_ids}")
        return df
    questions = group['question'].tolist()
    answers = group['answer'].tolist() if 'answer' in group.columns else [None]*len(questions)
    details = group['details'].tolist() if 'details' in group.columns else [None]*len(questions)
    q_sim = all(compute_cosine_sim(q1, q2) == 1.0 for i, q1 in enumerate(questions) for j, q2 in enumerate(questions) if i < j)
    a_sim = all(compute_cosine_sim(a1, a2) == 1.0 for i, a1 in enumerate(answers) for j, a2 in enumerate(answers) if i < j)
    d_sim = all(compute_cosine_sim(d1, d2) == 1.0 for i, d1 in enumerate(details) for j, d2 in enumerate(details) if i < j)
    if q_sim and a_sim and d_sim:
        merged_row = group.iloc[0].copy()
    else:
        merged_row = group.iloc[0].copy()
        merged_row['question'] = merge_texts(questions, 'question')
        merged_row['answer'] = merge_texts(answers, 'answer')
        merged_row['details'] = merge_texts(details, 'details')
    # Remove all original rows and add the merged row
    df = df[~mask]
    df = pd.concat([df, pd.DataFrame([merged_row])], ignore_index=True)
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge redundant questions in a dataset by cq_id.")
    parser.add_argument('--csv', default='../cleaned_dataset/all_complete_dataset.csv', help='Path to the main CSV file')
    parser.add_argument('--cq-ids', required=True, help='Comma-separated list of cq_ids to merge or path to a file with one cq_id per line')
    args = parser.parse_args()

    # Parse cq_ids
    if os.path.exists(args.cq_ids):
        with open(args.cq_ids) as f:
            cq_ids = [line.strip() for line in f if line.strip()]
    else:
        cq_ids = [cid.strip() for cid in args.cq_ids.split(",") if cid.strip()]

    df = pd.read_csv(args.csv)
    updated_df = merge_questions(df, cq_ids)
    updated_df.to_csv(args.csv, index=False)
    logger.info(f"Updated CSV saved to {args.csv}")
