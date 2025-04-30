# KL Enhancer Pipeline

This pipeline processes security questionnaire data, generates embeddings for questions, identifies similar questions, and clusters them based on semantic similarity.

## Features

- Data loading from GitHub repositories
- Preprocessing and merging of datasets
- Sentence embedding generation using transformer models
- Similarity search using FAISS
- Finding similar question pairs with configurable similarity thresholds
- Dimensionality reduction with PCA
- Clustering using HDBSCAN
- Separation of deleted and non-deleted data
- Example search functionality to find similar questions to a query
- Interactive web interface using Streamlit

## Project Structure

```
.
├── data/                # Input data files
├── models/              # Saved models and embeddings
├── output/              # Generated output files
├── src/                 # Source code
│   ├── kl_enhancer.py   # Core pipeline functionality
│   └── app.py           # Streamlit web application
├── static/              # Static assets for the web interface
├── requirements.txt     # Dependencies
└── README.md            # Documentation
```

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment (optional but recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Command Line Pipeline

Run the main pipeline from the src directory:

```
cd src
python kl_enhancer.py
```

This will:

1. Load data from specified URLs
2. Preprocess and merge the data
3. Generate embeddings for questions
4. Create a similarity index
5. Find similar question pairs
6. Perform dimensionality reduction and clustering
7. Separate deleted and non-deleted data
8. Save output files

### Streamlit Web Interface

To run the interactive web interface:

```
cd src
streamlit run app.py
```

The Streamlit app provides:

1. **Similar Questions Search**: Find semantically similar questions to your query
2. **Data Explorer**: Browse, filter, and search the dataset
3. **Clustering Analysis**: Generate and analyze question clusters
4. **Statistics**: View dataset statistics and visualizations

## Output Files

The pipeline generates several output files in the `output/` directory:

- `merged_data.csv`: Merged dataset
- `similar_qna_pairs.csv`: Similar question pairs
- `clustered_data.csv`: Dataset with cluster labels
- `deleted_data.csv`: Data with deleted_at field populated
- `not_deleted_data.csv`: Data with deleted_at field empty

Models and embeddings are stored in the `models/` directory:

- `qna_embeddings.npy`: Question embeddings
- `qna_embeddings_reduced.npy`: Reduced dimensionality embeddings

## Advanced Usage

The script is structured as a collection of functions that can be imported and used individually:

```python
from kl_enhancer import load_data, generate_embeddings, search_similar_questions

# Load data
ans_df, can_df = load_data()

# Process as needed
...

# Search for similar questions
results = search_similar_questions("What is the security policy?", model, index, embeddings, merged_df)
```

## Customization

You can customize various parameters:

- Embedding model: Change the model_name parameter in generate_embeddings()
- Similarity threshold: Adjust the similarity_threshold in find_similar_pairs()
- Clustering parameters: Modify min_cluster_size and min_samples in perform_clustering()
