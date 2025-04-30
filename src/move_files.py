import os
import shutil
import sys
from pathlib import Path

def create_directories():
    """Create the necessary directories if they don't exist."""
    # Get the root directory (parent of src)
    root_dir = Path(__file__).parent.parent.absolute()
    
    dirs = [
        os.path.join(root_dir, "data"),
        os.path.join(root_dir, "models"),
        os.path.join(root_dir, "output"),
        os.path.join(root_dir, "static")
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Created or verified directory: {directory}")

def move_files():
    """Move files to their appropriate directories."""
    # Get the root directory (parent of src)
    root_dir = Path(__file__).parent.parent.absolute()
    
    # Define file mappings (source -> destination)
    file_mappings = {
        # Original data files to data directory
        os.path.join(root_dir, "(Answer Library Entry) Danfe Corp KL.xlsx - AnswerLibraryEntry (1).csv"): 
            os.path.join(root_dir, "data", "AnswerLibraryEntry.csv"),
            
        os.path.join(root_dir, "(Canonical Questions Products) Danfe Corp KL.xlsx - CanonicalQuestionProducts.csv"): 
            os.path.join(root_dir, "data", "CanonicalQuestionProducts.csv"),
            
        os.path.join(root_dir, "(Product) Danfe Corp KL.xlsx - Product.csv"): 
            os.path.join(root_dir, "data", "Product.csv"),
        
        # Embeddings and models to models directory
        os.path.join(root_dir, "qna_embeddings.npy"): 
            os.path.join(root_dir, "models", "qna_embeddings.npy"),
            
        os.path.join(root_dir, "qna_embeddings_reduced.npy"): 
            os.path.join(root_dir, "models", "qna_embeddings_reduced.npy"),
        
        # Generated files to output directory
        os.path.join(root_dir, "merged_data.csv"): 
            os.path.join(root_dir, "output", "merged_data.csv"),
            
        os.path.join(root_dir, "clustered_data.csv"): 
            os.path.join(root_dir, "output", "clustered_data.csv"),
            
        os.path.join(root_dir, "deleted_data.csv"): 
            os.path.join(root_dir, "output", "deleted_data.csv"),
            
        os.path.join(root_dir, "not_deleted_data.csv"): 
            os.path.join(root_dir, "output", "not_deleted_data.csv"),
            
        os.path.join(root_dir, "question_analysis_results.csv"): 
            os.path.join(root_dir, "output", "question_analysis_results.csv")
    }
    
    # Move files
    for source, destination in file_mappings.items():
        if os.path.exists(source):
            try:
                # Create destination directory if it doesn't exist
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                
                print(f"Copying: {source} -> {destination}")
                # Copy the file (using shutil to handle cross-device links)
                shutil.copy2(source, destination)
                print(f"Successfully copied: {os.path.basename(source)}")
            except Exception as e:
                print(f"Error copying {source}: {e}")
        else:
            print(f"Warning: Source file not found: {source}")

def main():
    print("Starting file organization...")
    create_directories()
    move_files()
    print("File organization complete!")

if __name__ == "__main__":
    main() 