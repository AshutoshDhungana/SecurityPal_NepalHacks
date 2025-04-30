import os
import pandas as pd
import numpy as np
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import time
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("cleaned_ds_create.log"), logging.StreamHandler()]
)
logger = logging.getLogger("cleaned_ds_create")

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

class CleanedDatasetCreator:
    def __init__(self, 
                 processed_dir="processed_clusters",
                 models_dir="models",
                 output_dir="cleaned_dataset",
                 embeddings_included=True):
        """
        Initialize the cleaned dataset creator
        
        Args:
            processed_dir: Directory containing processed cluster results
            models_dir: Directory containing embedding data
            output_dir: Directory to save clean datasets
            embeddings_included: Whether to include embeddings (can make files very large)
        """
        # Convert relative paths to absolute paths
        self.processed_dir = get_absolute_path(processed_dir)
        self.models_dir = get_absolute_path(models_dir)
        self.output_dir = get_absolute_path(output_dir)
        self.embeddings_included = embeddings_included
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load embeddings metadata
        self.metadata_path = os.path.join(self.models_dir, "embeddings_metadata.json")
        if os.path.exists(self.metadata_path) and self.embeddings_included:
            try:
                logger.info(f"Loading embeddings metadata from {self.metadata_path}")
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded embeddings metadata")
            except Exception as e:
                logger.warning(f"Failed to load embeddings metadata: {str(e)}")
                self.metadata = None
                self.embeddings_included = False
        else:
            logger.warning(f"Embeddings metadata not found or embeddings not included")
            self.metadata = None
            if self.embeddings_included:
                logger.warning("Embeddings will not be included due to missing metadata")
                self.embeddings_included = False
        
        logger.info(f"Initialized cleaned dataset creator")
        logger.info(f"Processed directory: {self.processed_dir}")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Embeddings included: {self.embeddings_included}")
    
    def load_processed_data(self, product=None):
        """
        Load processed data from CSV files
        
        Args:
            product: If specified, only load this product's data
            
        Returns:
            DataFrame with processed data
        """
        try:
            if product:
                # Load specific product data
                file_path = os.path.join(self.processed_dir, f"{product}_processed_clusters.csv")
                if not os.path.exists(file_path):
                    logger.error(f"Processed file not found: {file_path}")
                    raise FileNotFoundError(f"Processed file not found: {file_path}")
                
                logger.info(f"Loading processed data for product: {product} from {file_path}")
                
                try:
                    # Try to load the file
                    df = pd.read_csv(file_path)
                    logger.info(f"Loaded {len(df)} records")
                    return df
                except Exception as e:
                    logger.error(f"Error loading file: {str(e)}")
                    raise
            else:
                # Load all processed data
                file_path = os.path.join(self.processed_dir, "all_processed_clusters.csv")
                if not os.path.exists(file_path):
                    # Try loading individual product files
                    logger.info("All processed file not found, attempting to load individual files")
                    
                    processed_files = [f for f in os.listdir(self.processed_dir) if f.endswith("_processed_clusters.csv")]
                    if not processed_files:
                        logger.error(f"No processed files found in: {self.processed_dir}")
                        raise FileNotFoundError(f"No processed files found in: {self.processed_dir}")
                    
                    all_dfs = []
                    for file in processed_files:
                        file_path = os.path.join(self.processed_dir, file)
                        logger.info(f"Loading processed data from: {file}")
                        try:
                            df = pd.read_csv(file_path)
                            all_dfs.append(df)
                            logger.info(f"Loaded {len(df)} records from {file}")
                        except Exception as e:
                            logger.error(f"Error loading file {file}: {str(e)}")
                            # Continue with other files
                    
                    if not all_dfs:
                        logger.error("No files could be loaded")
                        raise ValueError("No files could be loaded")
                    
                    merged_df = pd.concat(all_dfs, ignore_index=True)
                    logger.info(f"Loaded {len(merged_df)} total records from {len(all_dfs)} files")
                    return merged_df
                else:
                    logger.info(f"Loading all processed data from {file_path}")
                    try:
                        df = pd.read_csv(file_path)
                        logger.info(f"Loaded {len(df)} records")
                        return df
                    except Exception as e:
                        logger.error(f"Error loading file: {str(e)}")
                        raise
        except Exception as e:
            logger.error(f"Error in load_processed_data: {str(e)}")
            raise
    
    def load_embeddings(self, product, category, question_ids):
        """
        Load embeddings for a specific product/category/question_ids
        
        Args:
            product: Product name
            category: Category name
            question_ids: List of question IDs to load embeddings for
            
        Returns:
            Dictionary mapping question_id to embedding
        """
        if not self.embeddings_included or self.metadata is None:
            return {}
        
        if product not in self.metadata["products"]:
            logger.debug(f"Product {product} not found in metadata")
            return {}
        
        if category not in self.metadata["products"][product]["categories"]:
            logger.debug(f"Category {category} not found in metadata for product {product}")
            return {}
        
        category_data = self.metadata["products"][product]["categories"][category]
        embeddings_file = category_data["embeddings_file"]
        embeddings_path = os.path.join(self.models_dir, product, embeddings_file)
        
        if not os.path.exists(embeddings_path):
            logger.debug(f"Embeddings file not found: {embeddings_path}")
            return {}
        
        try:
            # Load embeddings
            embeddings = np.load(embeddings_path)
            metadata_question_ids = category_data["question_ids"]
            
            # Create a mapping from question_id to embedding
            embedding_map = {}
            for q_id, emb in zip(metadata_question_ids, embeddings):
                if q_id in question_ids:
                    embedding_map[q_id] = emb
            
            logger.debug(f"Loaded {len(embedding_map)} embeddings for {product}/{category}")
            return embedding_map
        
        except Exception as e:
            logger.error(f"Error loading embeddings for {product}/{category}: {str(e)}")
            return {}
    
    def add_embeddings_to_dataframe(self, df):
        """
        Add embeddings to dataframe as additional columns
        
        Args:
            df: DataFrame with processed data
            
        Returns:
            DataFrame with added embedding columns
        """
        if not self.embeddings_included or self.metadata is None:
            logger.info("Skipping embedding addition")
            return df
        
        logger.info(f"Adding embeddings to {len(df)} records")
        
        # Make a copy of the DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Create a new 'embedding' column
        result_df['embedding'] = None
        
        # Group by product and category to load embeddings efficiently
        if 'product' in result_df.columns and 'category' in result_df.columns:
            # Process groups
            for (product, category), group in tqdm(result_df.groupby(['product', 'category']), 
                                                desc="Adding embeddings by product/category"):
                # Get question IDs for this product/category
                id_column = 'id' if 'id' in group.columns else 'question_id'
                if id_column not in group.columns:
                    logger.warning(f"No question ID column found for {product}/{category}, skipping")
                    continue
                
                question_ids = group[id_column].tolist()
                
                # Load embeddings for these questions
                embeddings = self.load_embeddings(product, category, question_ids)
                
                # Add embeddings to DataFrame
                for idx, row in group.iterrows():
                    q_id = row[id_column]
                    if q_id in embeddings:
                        result_df.at[idx, 'embedding'] = embeddings[q_id].tolist()
        else:
            logger.warning("DataFrame does not have product or category columns, skipping embedding addition")
        
        # Count how many embeddings were added
        embedding_count = result_df['embedding'].count()
        logger.info(f"Added {embedding_count} embeddings out of {len(result_df)} records")
        
        return result_df
    
    def create_complete_dataset(self, df):
        """
        Create a complete dataset with all information
        
        Args:
            df: DataFrame with processed data
            
        Returns:
            DataFrame with the complete dataset
        """
        # Starting with the processed DataFrame
        result_df = df.copy()
        
        # Additional processing can be done here if needed
        # For example, converting date columns to proper datetime format
        if 'created_at' in result_df.columns:
            try:
                result_df['created_at'] = pd.to_datetime(result_df['created_at'])
            except:
                logger.warning("Could not convert created_at to datetime")
        
        if 'deleted_at' in result_df.columns:
            try:
                result_df['deleted_at'] = pd.to_datetime(result_df['deleted_at'], errors='coerce')
            except:
                logger.warning("Could not convert deleted_at to datetime")
        
        # Add embeddings if requested
        if self.embeddings_included:
            result_df = self.add_embeddings_to_dataframe(result_df)
        
        return result_df
    
    def save_dataset(self, df, product=None, format='csv'):
        """
        Save the dataset to file
        
        Args:
            df: DataFrame with complete dataset
            product: If specified, save with product name prefix
            format: Format to save ('csv' or 'parquet')
            
        Returns:
            Path to saved file
        """
        if product:
            base_filename = f"{product}_complete_dataset"
        else:
            base_filename = "all_complete_dataset"
        
        if format.lower() == 'csv':
            output_file = os.path.join(self.output_dir, f"{base_filename}.csv")
            df.to_csv(output_file, index=False)
        elif format.lower() == 'parquet':
            output_file = os.path.join(self.output_dir, f"{base_filename}.parquet")
            df.to_parquet(output_file, index=False)
        else:
            logger.warning(f"Unknown format {format}, defaulting to CSV")
            output_file = os.path.join(self.output_dir, f"{base_filename}.csv")
            df.to_csv(output_file, index=False)
            
        logger.info(f"Complete dataset saved to {output_file}")
        
        # Also save a version without embeddings if they were included
        # This makes the file more manageable for viewing/analysis
        if self.embeddings_included and 'embedding' in df.columns:
            df_no_embed = df.drop(columns=['embedding'])
            
            if format.lower() == 'csv':
                output_file_no_embed = os.path.join(self.output_dir, f"{base_filename}_no_embeddings.csv")
                df_no_embed.to_csv(output_file_no_embed, index=False)
            elif format.lower() == 'parquet':
                output_file_no_embed = os.path.join(self.output_dir, f"{base_filename}_no_embeddings.parquet")
                df_no_embed.to_parquet(output_file_no_embed, index=False)
            else:
                output_file_no_embed = os.path.join(self.output_dir, f"{base_filename}_no_embeddings.csv")
                df_no_embed.to_csv(output_file_no_embed, index=False)
                
            logger.info(f"Dataset without embeddings saved to {output_file_no_embed}")
        
        return output_file
    
    def create_datasets_by_status(self, df, product=None, format='csv'):
        """
        Create and save separate datasets by status
        
        Args:
            df: DataFrame with complete dataset
            product: If specified, save with product name prefix
            format: Format to save ('csv' or 'parquet')
        """
        # Create prefix for filenames
        prefix = f"{product}_" if product else ""
        
        # 1. Active canonical questions - these are the primary questions to keep
        if 'is_canonical' in df.columns and 'is_active' in df.columns:
            active_canonical = df[(df['is_canonical'] == True) & (df['is_active'] == True)]
            if len(active_canonical) > 0:
                if format.lower() == 'csv':
                    output_file = os.path.join(self.output_dir, f"{prefix}active_canonical.csv")
                    active_canonical.to_csv(output_file, index=False)
                elif format.lower() == 'parquet':
                    output_file = os.path.join(self.output_dir, f"{prefix}active_canonical.parquet")
                    active_canonical.to_parquet(output_file, index=False)
                logger.info(f"Active canonical dataset ({len(active_canonical)} records) saved to {output_file}")
        
        # 2. Archived canonical questions - these were once canonical but now archived
        if 'is_canonical' in df.columns and 'is_archived' in df.columns:
            archived_canonical = df[(df['is_canonical'] == True) & (df['is_archived'] == True)]
            if len(archived_canonical) > 0:
                if format.lower() == 'csv':
                    output_file = os.path.join(self.output_dir, f"{prefix}archived_canonical.csv")
                    archived_canonical.to_csv(output_file, index=False)
                elif format.lower() == 'parquet':
                    output_file = os.path.join(self.output_dir, f"{prefix}archived_canonical.parquet")
                    archived_canonical.to_parquet(output_file, index=False)
                logger.info(f"Archived canonical dataset ({len(archived_canonical)} records) saved to {output_file}")
        
        # 3. Active redundant questions - these are active but duplicates of canonical questions
        if 'is_redundant' in df.columns and 'is_active' in df.columns:
            active_redundant = df[(df['is_redundant'] == True) & (df['is_active'] == True)]
            if len(active_redundant) > 0:
                if format.lower() == 'csv':
                    output_file = os.path.join(self.output_dir, f"{prefix}active_redundant.csv")
                    active_redundant.to_csv(output_file, index=False)
                elif format.lower() == 'parquet':
                    output_file = os.path.join(self.output_dir, f"{prefix}active_redundant.parquet")
                    active_redundant.to_parquet(output_file, index=False)
                logger.info(f"Active redundant dataset ({len(active_redundant)} records) saved to {output_file}")
        
        # 4. Archived redundant questions
        if 'is_redundant' in df.columns and 'is_archived' in df.columns:
            archived_redundant = df[(df['is_redundant'] == True) & (df['is_archived'] == True)]
            if len(archived_redundant) > 0:
                if format.lower() == 'csv':
                    output_file = os.path.join(self.output_dir, f"{prefix}archived_redundant.csv")
                    archived_redundant.to_csv(output_file, index=False)
                elif format.lower() == 'parquet':
                    output_file = os.path.join(self.output_dir, f"{prefix}archived_redundant.parquet")
                    archived_redundant.to_parquet(output_file, index=False)
                logger.info(f"Archived redundant dataset ({len(archived_redundant)} records) saved to {output_file}")
        
        # 5. Noise points (not in any cluster)
        if 'cluster_id' in df.columns:
            noise_points = df[df['cluster_id'] == -1]
            if len(noise_points) > 0:
                if format.lower() == 'csv':
                    output_file = os.path.join(self.output_dir, f"{prefix}noise_points.csv")
                    noise_points.to_csv(output_file, index=False)
                elif format.lower() == 'parquet':
                    output_file = os.path.join(self.output_dir, f"{prefix}noise_points.parquet")
                    noise_points.to_parquet(output_file, index=False)
                logger.info(f"Noise points dataset ({len(noise_points)} records) saved to {output_file}")
    
    def run_pipeline(self, product=None, format='csv'):
        """
        Run the complete dataset creation pipeline
        
        Args:
            product: If specified, only process this product
            format: Format to save datasets ('csv' or 'parquet')
            
        Returns:
            DataFrame with the complete dataset
        """
        logger.info(f"Starting cleaned dataset creation pipeline")
        
        # Load processed data
        df = self.load_processed_data(product)
        
        # Create complete dataset
        complete_df = self.create_complete_dataset(df)
        
        # Save the complete dataset
        self.save_dataset(complete_df, product, format)
        
        # Create and save datasets by status
        self.create_datasets_by_status(complete_df, product, format)
        
        logger.info("Cleaned dataset creation pipeline completed")
        
        return complete_df


def main():
    """
    Main function to run the cleaned dataset creation pipeline
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cleaned dataset creation pipeline")
    parser.add_argument("--processed-dir", default="processed_clusters", help="Directory containing processed cluster results")
    parser.add_argument("--models-dir", default="models", help="Directory containing embedding data")
    parser.add_argument("--output-dir", default="cleaned_dataset", help="Directory to save clean datasets")
    parser.add_argument("--product", default=None, help="Process only specific product")
    parser.add_argument("--format", default="csv", choices=["csv", "parquet"], help="Format to save datasets")
    parser.add_argument("--no-embeddings", action="store_true", help="Exclude embeddings from the dataset")
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = CleanedDatasetCreator(
            processed_dir=args.processed_dir,
            models_dir=args.models_dir,
            output_dir=args.output_dir,
            embeddings_included=not args.no_embeddings
        )
        
        # Run pipeline
        pipeline.run_pipeline(args.product, args.format)
        
    except Exception as e:
        logger.error(f"Error running cleaned dataset creation pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
