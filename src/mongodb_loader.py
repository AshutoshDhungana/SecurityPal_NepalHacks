import os
import pandas as pd
import numpy as np
import json
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from pymongo import MongoClient
import uuid
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("mongodb_loader.log"), logging.StreamHandler()]
)
logger = logging.getLogger("mongodb_loader")

def get_absolute_path(path):
    """Helper function to get absolute path from relative path"""
    if os.path.isabs(path):
        return path
    # Get the directory where the script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go one level up to the project root
    base_dir = os.path.dirname(current_dir)
    # Join with the relative path
    return os.path.join(base_dir, path)

def compute_cluster_averages(embeddings, labels):
    """Compute average embeddings for each cluster."""
    cluster_averages = []
    num_clusters = np.max(labels) + 1  # assumes labels are 0-indexed
    for cluster_id in range(num_clusters):
        cluster_vectors = embeddings[labels == cluster_id]
        average_vector = np.mean(cluster_vectors, axis=0)
        cluster_averages.append(average_vector)
    return np.array(cluster_averages)

class MongoDBLoader:
    def __init__(self, 
                 dataset_dir="cleaned_dataset",
                 mongodb_uri="mongodb://localhost:27017/",
                 db_name="nepalhacks",
                 collection_name="clusters",
                 batch_size=100):
        """
        Initialize the MongoDB loader
        
        Args:
            dataset_dir: Directory containing cleaned datasets
            mongodb_uri: URI for MongoDB connection
            db_name: MongoDB database name
            collection_name: MongoDB collection name
            batch_size: Number of documents to insert in each batch
        """
        # Convert relative paths to absolute paths
        self.dataset_dir = get_absolute_path(dataset_dir)
        self.mongodb_uri = mongodb_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.batch_size = batch_size
        
        # MongoDB client
        self.client = None
        self.db = None
        self.collection = None
        
        logger.info(f"Initialized MongoDB loader")
        logger.info(f"Dataset directory: {self.dataset_dir}")
        logger.info(f"MongoDB URI: {self.mongodb_uri}")
        logger.info(f"Database name: {self.db_name}")
        logger.info(f"Collection name: {self.collection_name}")
    
    def connect_to_mongodb(self):
        """Connect to MongoDB database"""
        try:
            logger.info("Connecting to MongoDB...")
            self.client = MongoClient(self.mongodb_uri)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            # Check connection by running a simple command
            self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
            return True
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            return False
    
    def load_data(self, filename):
        """
        Load data from CSV/parquet file
        
        Args:
            filename: Name of the file to load
            
        Returns:
            DataFrame with loaded data
        """
        file_path = os.path.join(self.dataset_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading data from: {file_path}")
        
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def convert_to_mongo_format(self, df):
        """
        Convert dataframe to MongoDB document format
        
        Args:
            df: DataFrame with data to convert
            
        Returns:
            List of documents in MongoDB format
        """
        logger.info("Converting data to MongoDB format")
        
        # Group by cluster_id
        cluster_docs = []
        
        # Filter out noise points (cluster_id = -1) if present
        if 'cluster_id' in df.columns:
            df = df[df['cluster_id'] != -1].copy()
            # Convert cluster_id to string format for consistency
            df['cluster_id'] = df['cluster_id'].astype(str)
        
        # Make sure all necessary columns exist
        required_cols = ['cluster_id', 'category', 'product', 'id', 'question', 'answer', 'details', 'created_at', 'deleted_at']
        for col in required_cols:
            if col not in df.columns:
                if col in ['details', 'answer']:
                    # These are optional, set to empty if not present
                    df[col] = ""
                else:
                    logger.warning(f"Column {col} not found in DataFrame, using placeholder values")
                    if col == 'id':
                        df[col] = [str(uuid.uuid4()) for _ in range(len(df))]
                    elif col == 'product':
                        df[col] = "unknown_product"
                    elif col == 'category':
                        df[col] = "unknown_category"
                    elif col == 'cluster_id':
                        df[col] = ["cluster_" + str(i) for i in range(len(df))]
                    elif col in ['created_at', 'deleted_at']:
                        df[col] = None
        
        # Prepare for average embedding calculation using the provided logic
        # Collect all embeddings and labels for this group
        all_embeddings = []
        all_labels = []
        for _, row in df.iterrows():
            emb = row['embedding']
            label = row['cluster_id']
            if pd.notna(emb):
                if isinstance(emb, str):
                    try:
                        emb = json.loads(emb)
                    except Exception:
                        continue
                try:
                    emb = np.array([float(x) for x in emb], dtype=np.float32)
                    all_embeddings.append(emb)
                    all_labels.append(int(label))
                except Exception:
                    continue
        if all_embeddings and all_labels:
            all_embeddings = np.stack(all_embeddings)
            all_labels = np.array(all_labels)
            cluster_averages = compute_cluster_averages(all_embeddings, all_labels)
            # Map cluster_id to average embedding
            cluster_avg_map = {i: cluster_averages[i].tolist() for i in range(cluster_averages.shape[0])}
        else:
            cluster_avg_map = {}
        
        # Group by cluster_id
        grouped = df.groupby('cluster_id')
        
        # Calculate average embedding for each cluster if embedding data is available
        has_embeddings = 'embedding' in df.columns and not df['embedding'].isna().all()
        
        for cluster_id, group in tqdm(grouped, desc="Processing clusters"):
            # Initialize cluster document
            canonical_question = group.loc[group['is_canonical'] == True].iloc[0] if 'is_canonical' in group.columns else group.iloc[0]
            
            entries = []
            for _, row in group.iterrows():
                # Create entry for each question in the cluster
                entry = {
                    "productid": row['product'],
                    "canonicalid": row['id'],
                    "question": row['question'],
                    "answer": row['answer'],
                    "details": row['details'] if pd.notna(row['details']) else "",
                    "created_at": row['created_at'] if pd.notna(row['created_at']) else datetime.now().isoformat(),
                    "deleted_at": row['deleted_at'] if pd.notna(row['deleted_at']) else None
                }
                
                # Add embedding value if available
                if has_embeddings and pd.notna(row['embedding']):
                    try:
                        emb_value = row['embedding']
                        # Store the actual embedding as a list
                        if isinstance(emb_value, str):
                            try:
                                emb_value = json.loads(emb_value)
                            except Exception:
                                pass  # fallback: keep as string if not JSON
                        entry["embedding"] = emb_value
                        # Also keep the hash for backward compatibility
                        if isinstance(emb_value, list):
                            entry["embed_value"] = str(hash(str(emb_value[:min(5, len(emb_value))])))[:6]
                        else:
                            entry["embed_value"] = str(hash(str(emb_value)))[:6]
                    except Exception as e:
                        entry["embed_value"] = str(hash(str(row['id'])))[:6]
                        logger.debug(f"Error processing embedding for entry: {str(e)}")
                
                entries.append(entry)
            
            # Remove average embedding calculation logic
            # Determine if this cluster contains outdated questions
            has_outdated = any(pd.notna(row['deleted_at']) for _, row in group.iterrows())
            outdated_score = 0.7 if has_outdated else 0.0
            
            # Create timestamp
            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Create auto merge suggestion based on cluster characteristics
            cluster_size = len(group)
            merge_confidence = 0.9 if cluster_size > 1 else 0.0
            
            # Create the complete cluster document (no avg_embedding or avg_embed_value)
            cluster_doc = {
                "cluster_id": f"cluster_{cluster_id}",
                "category": canonical_question['category'],
                "entries": entries,
                "auto_merge_suggestion": {
                    "merge_confidence": merge_confidence
                },
                "outdated_score": outdated_score,
                "still_valid": not has_outdated,
                "sandbox": {
                    "status": "pending",
                    "review_notes": ""
                },
                "timestamp": timestamp
            }
            if cluster_id.isdigit():
                avg_embedding = cluster_avg_map.get(int(cluster_id))
                if avg_embedding is not None:
                    cluster_doc["avg_embedding"] = avg_embedding
            cluster_docs.append(cluster_doc)
        
        logger.info(f"Converted {len(cluster_docs)} clusters to MongoDB format")
        return cluster_docs
    
    def insert_documents(self, documents):
        """
        Insert documents into MongoDB collection
        
        Args:
            documents: List of documents to insert
            
        Returns:
            Number of documents inserted
        """
        if self.collection is None:
            if not self.connect_to_mongodb():
                logger.error("Failed to connect to MongoDB")
                return 0
        
        total_inserted = 0
        
        logger.info(f"Inserting {len(documents)} documents into MongoDB")
        
        # Insert in batches
        for i in tqdm(range(0, len(documents), self.batch_size), desc="Inserting batches"):
            batch = documents[i:i+self.batch_size]
            try:
                result = self.collection.insert_many(batch)
                total_inserted += len(result.inserted_ids)
            except Exception as e:
                logger.error(f"Error inserting batch {i//self.batch_size}: {str(e)}")
        
        logger.info(f"Successfully inserted {total_inserted} documents")
        return total_inserted
    
    def run_pipeline(self, filename, drop_existing=False):
        """
        Run the complete MongoDB loading pipeline
        
        Args:
            filename: Name of the file to load data from
            drop_existing: Whether to drop existing collection before inserting
            
        Returns:
            Number of documents inserted
        """
        logger.info(f"Starting MongoDB loading pipeline")
        
        # Connect to MongoDB
        if not self.connect_to_mongodb():
            logger.error("Failed to connect to MongoDB. Exiting.")
            return 0
        
        # Drop existing collection if requested
        if drop_existing:
            logger.info(f"Dropping existing collection: {self.collection_name}")
            self.collection.drop()
            self.collection = self.db[self.collection_name]
        
        # Load data
        df = self.load_data(filename)
        
        # Convert to MongoDB format
        documents = self.convert_to_mongo_format(df)
        
        # Insert documents
        inserted_count = self.insert_documents(documents)
        
        logger.info(f"MongoDB loading pipeline completed. Inserted {inserted_count} documents.")
        
        return inserted_count


def main():
    """
    Main function to run the MongoDB loading pipeline
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MongoDB loading pipeline")
    parser.add_argument("--dataset-dir", default="cleaned_dataset", help="Directory containing cleaned datasets")
    parser.add_argument("--mongodb-uri", default="mongodb://localhost:27017/", help="URI for MongoDB connection")
    parser.add_argument("--db-name", default="nepalhacks", help="MongoDB database name")
    parser.add_argument("--collection-name", default="clusters", help="MongoDB collection name")
    parser.add_argument("--filename", default="active_canonical.csv", help="Name of the file to load data from")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of documents to insert in each batch")
    parser.add_argument("--drop-existing", action="store_true", help="Drop existing collection before inserting")
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = MongoDBLoader(
            dataset_dir=args.dataset_dir,
            mongodb_uri=args.mongodb_uri,
            db_name=args.db_name,
            collection_name=args.collection_name,
            batch_size=args.batch_size
        )
        
        # Run pipeline
        pipeline.run_pipeline(args.filename, args.drop_existing)
        
    except Exception as e:
        logger.error(f"Error running MongoDB loading pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 