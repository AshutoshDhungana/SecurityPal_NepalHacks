import os
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("suggestion_engine.log"), logging.StreamHandler()]
)
logger = logging.getLogger("suggestion_engine")

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

class SuggestionRule:
    """Base class for suggestion rules"""
    def __init__(self, name, description, priority=1):
        self.name = name
        self.description = description
        self.priority = priority  # Higher priority rules are evaluated first
    
    def apply(self, data):
        """Apply rule to data and generate suggestions"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def __str__(self):
        return f"{self.name} (Priority: {self.priority})"

class MergeSimilarEntriesRule(SuggestionRule):
    """Rule for merging highly similar entries in the same cluster"""
    def __init__(self, similarity_threshold=0.85, priority=10):
        super().__init__(
            name="Merge Similar Entries",
            description="Suggests merging entries that are highly similar in content",
            priority=priority
        )
        self.similarity_threshold = similarity_threshold
    
    def apply(self, data):
        """Find entries with high similarity that should be merged"""
        suggestions = []
        
        # Group by cluster_id
        for cluster_id, group in data.groupby('cluster_id'):
            if len(group) < 2:
                continue  # Skip clusters with only one entry
            
            # Extract questions and compute pairwise similarity
            questions = group['question'].tolist()
            similarity_matrix = squareform(pdist(
                [q.lower() for q in questions], 
                lambda x, y: 1 - (len(set(x.split()) & set(y.split())) / len(set(x.split()) | set(y.split())))
            ))
            
            # Find pairs above threshold
            for i in range(len(questions)):
                for j in range(i+1, len(questions)):
                    if similarity_matrix[i][j] >= self.similarity_threshold:
                        entry1 = group.iloc[i]
                        entry2 = group.iloc[j]
                        
                        suggestions.append({
                            'rule': self.name,
                            'type': 'merge',
                            'cluster_id': cluster_id,
                            'entries': [entry1['id'], entry2['id']],
                            'question_texts': [entry1['question'], entry2['question']],
                            'answer_texts': [entry1.get('answer', ''), entry2.get('answer', '')],
                            'similarity_score': similarity_matrix[i][j],
                            'justification': f"Entries have {similarity_matrix[i][j]:.2f} similarity (threshold: {self.similarity_threshold})",
                            'created_at': datetime.now().isoformat()
                        })
        
        logger.info(f"Found {len(suggestions)} merge suggestions")
        return suggestions

class UpdateOutdatedContentRule(SuggestionRule):
    """Rule for identifying outdated content that needs updating"""
    def __init__(self, age_threshold_days=180, priority=8):
        super().__init__(
            name="Update Outdated Content",
            description="Identifies content that hasn't been updated in a long time",
            priority=priority
        )
        self.age_threshold_days = age_threshold_days
    
    def apply(self, data):
        """Find entries that are outdated and need updating"""
        suggestions = []
        
        # Convert date columns to datetime if they exist
        date_cols = ['last_updated', 'updated_at', 'created_at', 'last_modified']
        date_col = None
        
        for col in date_cols:
            if col in data.columns:
                date_col = col
                try:
                    data[col] = pd.to_datetime(data[col])
                except:
                    logger.warning(f"Could not convert {col} to datetime")
                    date_col = None
        
        if date_col is None:
            logger.warning("No valid date column found, cannot apply outdated content rule")
            return suggestions
        
        # Calculate current date and threshold
        current_date = datetime.now()
        threshold_date = current_date - timedelta(days=self.age_threshold_days)
        
        # Find outdated entries
        outdated_mask = data[date_col] < threshold_date
        outdated_entries = data[outdated_mask]
        
        for _, entry in outdated_entries.iterrows():
            days_old = (current_date - entry[date_col]).days
            
            suggestions.append({
                'rule': self.name,
                'type': 'update',
                'entry_id': entry['id'],
                'question_text': entry['question'],
                'answer_text': entry.get('answer', ''),
                'days_since_update': days_old,
                'last_updated': entry[date_col].isoformat(),
                'justification': f"Entry hasn't been updated in {days_old} days (threshold: {self.age_threshold_days} days)",
                'created_at': current_date.isoformat()
            })
        
        logger.info(f"Found {len(suggestions)} outdated content suggestions")
        return suggestions

class ConsolidateDuplicatesRule(SuggestionRule):
    """Rule for identifying and consolidating duplicate entries across clusters"""
    def __init__(self, similarity_threshold=0.92, priority=9):
        super().__init__(
            name="Consolidate Duplicates",
            description="Identifies and suggests consolidating duplicate entries across different clusters",
            priority=priority
        )
        self.similarity_threshold = similarity_threshold
    
    def apply(self, data):
        """Find duplicate entries across different clusters"""
        suggestions = []
        
        # Generate a list of entries with their cluster IDs
        entries = []
        for _, row in data.iterrows():
            entries.append({
                'id': row['id'],
                'cluster_id': row.get('cluster_id', -1),
                'question': row['question'],
                'answer': row.get('answer', '')
            })
        
        # Compare entries across different clusters
        for i in range(len(entries)):
            for j in range(i+1, len(entries)):
                # Skip entries in the same cluster
                if entries[i]['cluster_id'] == entries[j]['cluster_id']:
                    continue
                
                # Compute similarity between questions
                q1 = entries[i]['question'].lower()
                q2 = entries[j]['question'].lower()
                # Simple token overlap similarity
                tokens1 = set(q1.split())
                tokens2 = set(q2.split())
                similarity = len(tokens1 & tokens2) / len(tokens1 | tokens2)
                
                if similarity >= self.similarity_threshold:
                    suggestions.append({
                        'rule': self.name,
                        'type': 'consolidate',
                        'entries': [entries[i]['id'], entries[j]['id']],
                        'cluster_ids': [entries[i]['cluster_id'], entries[j]['cluster_id']],
                        'question_texts': [entries[i]['question'], entries[j]['question']],
                        'answer_texts': [entries[i]['answer'], entries[j]['answer']],
                        'similarity_score': similarity,
                        'justification': f"Cross-cluster duplicate entries with {similarity:.2f} similarity (threshold: {self.similarity_threshold})",
                        'created_at': datetime.now().isoformat()
                    })
        
        logger.info(f"Found {len(suggestions)} consolidation suggestions")
        return suggestions

class SuggestionEngine:
    """Main engine for generating suggestions based on rules"""
    def __init__(self, 
                 processed_clusters_dir="processed_clusters",
                 output_dir="suggestions",
                 rules=None):
        """
        Initialize the suggestion engine
        
        Args:
            processed_clusters_dir: Directory containing processed cluster data
            output_dir: Directory to save suggestion results
            rules: List of SuggestionRule instances (if None, use default rules)
        """
        self.processed_clusters_dir = get_absolute_path(processed_clusters_dir)
        self.output_dir = get_absolute_path(output_dir)
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup default rules if none provided
        if rules is None:
            self.rules = [
                MergeSimilarEntriesRule(similarity_threshold=0.85, priority=10),
                ConsolidateDuplicatesRule(similarity_threshold=0.92, priority=9),
                UpdateOutdatedContentRule(age_threshold_days=180, priority=8)
            ]
        else:
            self.rules = rules
            
        # Sort rules by priority (highest first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        
        logger.info(f"Initialized suggestion engine with {len(self.rules)} rules")
        logger.info(f"Processed clusters directory: {self.processed_clusters_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_processed_clusters(self, product=None):
        """
        Load processed cluster data
        
        Args:
            product: If specified, only load this product's data
            
        Returns:
            DataFrame with processed cluster data
        """
        try:
            if product:
                file_path = os.path.join(self.processed_clusters_dir, f"{product}_processed_clusters.csv")
                if not os.path.exists(file_path):
                    logger.error(f"Processed clusters file not found: {file_path}")
                    raise FileNotFoundError(f"Processed clusters file not found: {file_path}")
                
                logger.info(f"Loading processed clusters for product: {product}")
                df = pd.read_csv(file_path)
                logger.info(f"Loaded {len(df)} entries")
                return df
            else:
                # Try loading all_processed_clusters.csv first
                file_path = os.path.join(self.processed_clusters_dir, "all_processed_clusters.csv")
                if os.path.exists(file_path):
                    logger.info("Loading all processed clusters")
                    df = pd.read_csv(file_path)
                    logger.info(f"Loaded {len(df)} entries")
                    return df
                
                # If all_processed_clusters.csv doesn't exist, try loading individual product files
                logger.info("All processed clusters file not found, attempting to load individual product files")
                cluster_files = [f for f in os.listdir(self.processed_clusters_dir) 
                                if f.endswith("_processed_clusters.csv")]
                
                if not cluster_files:
                    logger.error(f"No processed cluster files found in: {self.processed_clusters_dir}")
                    raise FileNotFoundError(f"No processed cluster files found in: {self.processed_clusters_dir}")
                
                all_dfs = []
                for file in cluster_files:
                    file_path = os.path.join(self.processed_clusters_dir, file)
                    logger.info(f"Loading: {file}")
                    df = pd.read_csv(file_path)
                    all_dfs.append(df)
                    logger.info(f"Loaded {len(df)} entries from {file}")
                
                merged_df = pd.concat(all_dfs, ignore_index=True)
                logger.info(f"Loaded {len(merged_df)} total entries from {len(all_dfs)} files")
                return merged_df
        except Exception as e:
            logger.error(f"Error loading processed clusters: {str(e)}")
            raise
    
    def generate_suggestions(self, data):
        """
        Apply all rules to the data and generate suggestions
        
        Args:
            data: DataFrame with processed cluster data
            
        Returns:
            List of suggestion dictionaries
        """
        all_suggestions = []
        
        logger.info(f"Generating suggestions using {len(self.rules)} rules")
        
        for rule in self.rules:
            logger.info(f"Applying rule: {rule}")
            try:
                rule_suggestions = rule.apply(data)
                all_suggestions.extend(rule_suggestions)
                logger.info(f"Rule {rule.name} generated {len(rule_suggestions)} suggestions")
            except Exception as e:
                logger.error(f"Error applying rule {rule.name}: {str(e)}")
        
        logger.info(f"Generated a total of {len(all_suggestions)} suggestions")
        return all_suggestions
    
    def save_suggestions(self, suggestions, product=None):
        """
        Save suggestions to JSON file
        
        Args:
            suggestions: List of suggestion dictionaries
            product: If specified, use this in the filename
            
        Returns:
            Path to saved file
        """
        if product:
            filename = f"{product}_suggestions.json"
        else:
            filename = "all_suggestions.json"
        
        file_path = os.path.join(self.output_dir, filename)
        
        with open(file_path, 'w') as f:
            json.dump({
                'suggestions': suggestions,
                'generated_at': datetime.now().isoformat(),
                'count': len(suggestions)
            }, f, indent=2)
        
        logger.info(f"Saved {len(suggestions)} suggestions to {file_path}")
        return file_path
    
    def run_pipeline(self, product=None):
        """
        Run the complete suggestion pipeline
        
        Args:
            product: If specified, only process this product
            
        Returns:
            Path to saved suggestions file
        """
        logger.info("Starting suggestion pipeline")
        
        # Load processed cluster data
        data = self.load_processed_clusters(product)
        
        # Generate suggestions
        suggestions = self.generate_suggestions(data)
        
        # Save suggestions
        output_path = self.save_suggestions(suggestions, product)
        
        logger.info(f"Suggestion pipeline completed successfully")
        return output_path

def main():
    parser = argparse.ArgumentParser(description="Generate suggestions based on cluster data")
    parser.add_argument("--product", help="Process only a specific product")
    parser.add_argument("--processed-clusters-dir", default="processed_clusters", 
                        help="Directory containing processed cluster data")
    parser.add_argument("--output-dir", default="suggestions", 
                        help="Directory to save suggestion results")
    parser.add_argument("--merge-threshold", type=float, default=0.85,
                        help="Similarity threshold for merge suggestions (0-1)")
    parser.add_argument("--consolidate-threshold", type=float, default=0.92,
                        help="Similarity threshold for consolidation suggestions (0-1)")
    parser.add_argument("--age-threshold", type=int, default=180,
                        help="Age threshold in days for outdated content suggestions")
    args = parser.parse_args()
    
    # Create rules with custom thresholds
    rules = [
        MergeSimilarEntriesRule(similarity_threshold=args.merge_threshold, priority=10),
        ConsolidateDuplicatesRule(similarity_threshold=args.consolidate_threshold, priority=9),
        UpdateOutdatedContentRule(age_threshold_days=args.age_threshold, priority=8)
    ]
    
    # Create and run the suggestion engine
    engine = SuggestionEngine(
        processed_clusters_dir=args.processed_clusters_dir,
        output_dir=args.output_dir,
        rules=rules
    )
    
    output_path = engine.run_pipeline(args.product)
    logger.info(f"Suggestions saved to: {output_path}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 