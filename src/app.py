import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
from pathlib import Path

# Add the src directory to the path so we can import kl_enhancer
src_path = Path(__file__).parent.absolute()
sys.path.append(str(src_path))

# Set pandas options to allow for larger styled dataframes
pd.set_option("styler.render.max_elements", 400000)  # Increased from default 262144

# Import functions from kl_enhancer.py
from kl_enhancer import (
    load_data, 
    preprocess_data, 
    generate_embeddings_for_product, 
    create_similarity_index,
    search_similar_questions,
    find_similar_pairs,
    perform_clustering,
    separate_deleted_data,
    find_potential_duplicates,
    generate_duplicate_report,
    find_similar_to_new_question,
    filter_by_product,
    get_product_list,
    generate_embeddings
)

# Import merge functionality
try:
    from merge import (
        merge_questions, 
        batch_process_similar_questions,
        batch_save_merged_questions,
        load_merged_with_original_data,
        get_merged_questions,
        NLTK_AVAILABLE,
        SUMMARIZER_AVAILABLE
    )
    MERGE_AVAILABLE = True
    
    # Show appropriate warnings about missing features
    if not NLTK_AVAILABLE:
        print("WARNING: NLTK punkt not available. Using regex fallback for sentence splitting.")
    if not SUMMARIZER_AVAILABLE:
        print("WARNING: DistilBART summarization model not available. Merging will skip summarization step.")
        
except ImportError as e:
    print(f"Merge functionality not available: {str(e)}")
    print("To enable merging, please install: pip install transformers==4.21.0 rapidfuzz nltk torch")
    MERGE_AVAILABLE = False
    NLTK_AVAILABLE = False
    SUMMARIZER_AVAILABLE = False

# Define paths
DATA_DIR = "../data"
OUTPUT_DIR = "../output"
MODELS_DIR = "../models"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="KL Enhancer - Security Questionnaire Analysis",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4A90E2;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .product-selector {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        border-left: 5px solid #4A90E2;
    }
    .result-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid #4A90E2;
    }
    .archived-card {
        background-color: #f1f1f1;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid #FF9800;
        opacity: 0.85;
    }
    .similarity-score {
        font-size: 0.9rem;
        color: #555;
    }
    .answer-text {
        background-color: #e6f7ff;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .details-text {
        font-style: italic;
        color: #666;
        margin-top: 10px;
    }
    .category-badge {
        background-color: #4A90E2;
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
        margin-top: 5px;
    }
    .archived-badge {
        background-color: #FF9800;
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
        margin-top: 5px;
        margin-left: 5px;
    }
    .divider {
        border-top: 1px solid #ddd;
        margin: 20px 0;
    }
    .context-info {
        background-color: #e6f7ff;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
        border-left: 4px solid #4A90E2;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("<h1 class='main-header'>KL Enhancer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Security Questionnaire Analysis & Similarity Search</p>", unsafe_allow_html=True)

# STEP 1: Product Selection - First thing the user sees and does
# Load products data from kl_enhancer
products_df = get_product_list()

with st.container():
    st.markdown("<div class='product-selector'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #4A90E2; margin-top: 0;'>Step 1: Select Product Context</h3>", unsafe_allow_html=True)
    st.markdown("<p>Choose which product's knowledge base you want to analyze. All operations will be limited to this product.</p>", unsafe_allow_html=True)
    
    if products_df is not None:
        # Create a dictionary of product names and IDs for display
        product_options = [("All Products", "all")]
        for _, row in products_df.iterrows():
            product_options.append((row['product_name'], row['product_id']))
        
        # Convert to dictionaries for selectbox
        product_names = [item[0] for item in product_options]
        product_ids = [item[1] for item in product_options]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create a selectbox for product selection
            selected_product_name = st.selectbox(
                "Select a product:", 
                product_names,
                index=0,
                help="Filter all analysis to only questions for the selected product"
            )
            
            # Get the corresponding product ID
            selected_product_id = product_ids[product_names.index(selected_product_name)]
        
        with col2:
            if selected_product_id != "all":
                st.success(f"Active product context: {selected_product_name}")
            else:
                st.info("Analyzing all products")
    else:
        st.error("Product data not found. Using all data.")
        selected_product_id = "all"
        selected_product_name = "All Products"
    
    st.markdown("</div>", unsafe_allow_html=True)

# Function to load saved data if it exists
def load_existing_data(product_id=None):
    """Load existing processed data, prioritizing product-specific files"""
    # First check for product-specific files
    if product_id and product_id != "all":
        product_merged_path = os.path.join(OUTPUT_DIR, f"merged_data_{product_id}.csv")
        product_embeddings_path = os.path.join(MODELS_DIR, f"qna_embeddings_{product_id}.npy")
        
        if os.path.exists(product_merged_path) and os.path.exists(product_embeddings_path):
            st.success(f"Found product-specific data for {product_id}")
            merged_df = pd.read_csv(product_merged_path)
            embeddings = np.load(product_embeddings_path)
            return merged_df, embeddings
    
    # Fall back to general data
    merged_path = os.path.join(OUTPUT_DIR, "merged_data.csv")
    embeddings_path = os.path.join(MODELS_DIR, "qna_embeddings.npy")
    
    if os.path.exists(merged_path):
        merged_df = pd.read_csv(merged_path)
        
        # Load merged questions and combine with original data
        if MERGE_AVAILABLE:
            try:
                merged_df = load_merged_with_original_data(merged_df)
            except Exception as e:
                print(f"Error loading merged questions: {str(e)}")
        
        # Filter by product if specified
        if product_id and product_id != "all":
            merged_df = filter_by_product(merged_df, product_id)
            
            # If filtering by product, we need to regenerate embeddings
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Generate embeddings for the filtered data
            questions = merged_df['question'].tolist()
            embeddings = model.encode(questions)
            
            return merged_df, embeddings
        
        # For all products, load the saved embeddings
        if os.path.exists(embeddings_path):
            embeddings = np.load(embeddings_path)
            return merged_df, embeddings
    
    return None, None

# Modified search function to include deleted_at field in results
def search_similar_questions_with_deleted_status(query_text, model, index, embeddings, df, top_k=5):
    """Search for similar questions and include deleted_at status"""
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
            "category": df.iloc[indices[0][i]]['category'],
            "deleted_at": df.iloc[indices[0][i]]['deleted_at'],
            "is_archived": pd.notna(df.iloc[indices[0][i]]['deleted_at'])
        })
    
    return results

# Create sidebar AFTER product selection
st.sidebar.title("Options")

# Show currently selected product in sidebar
if selected_product_id != "all":
    st.sidebar.success(f"Selected product: {selected_product_name}")
else:
    st.sidebar.info("Analyzing all products")

# Data processing section - STEP 2
st.sidebar.header("Step 2: Data Processing")

# Check if data is already processed for the selected product
merged_df, embeddings = load_existing_data(selected_product_id)

if merged_df is not None and embeddings is not None:
    st.sidebar.success("✅ Data ready for analysis")
    st.sidebar.info(f"Found {len(merged_df)} questions" + 
                   (f" for {selected_product_name}" if selected_product_id != "all" else ""))
    data_processed = True
else:
    data_processed = False
    st.sidebar.warning("❌ Data needs processing")

# Add button to process data
process_button_text = "Process Data" if not data_processed else "Reprocess Data"
if st.sidebar.button(process_button_text):
    with st.spinner(f'Processing data for {selected_product_name}...'):
        # Create a placeholder for progress messages
        progress_placeholder = st.empty()
        
        # Step 1: Load data
        progress_placeholder.info("Step 1/5: Loading data...")
        ans_df, can_df, _ = load_data()
        
        # Step 2: Preprocess data with product filtering as the first step
        progress_placeholder.info(f"Step 2/5: Preprocessing data for {selected_product_name}...")
        merged_df = preprocess_data(ans_df, can_df, selected_product_id)
        
        # Save merged data with product-specific name if appropriate
        output_path = os.path.join(OUTPUT_DIR, "merged_data.csv")
        if selected_product_id != "all":
            output_path = os.path.join(OUTPUT_DIR, f"merged_data_{selected_product_id}.csv")
        merged_df.to_csv(output_path, index=False)
        
        # Step 3: Generate embeddings
        progress_placeholder.info("Step 3/5: Generating embeddings...")
        embeddings, model = generate_embeddings(merged_df, selected_product_id)
        
        # Step 4: Create similarity index
        progress_placeholder.info("Step 4/5: Creating similarity index...")
        index, norm_embeddings = create_similarity_index(embeddings)
        
        # Step 5: Find similar pairs
        progress_placeholder.info("Step 5/5: Finding similar pairs...")
        similar_pairs_df = find_similar_pairs(index, norm_embeddings, merged_df)
        similar_pairs_path = os.path.join(OUTPUT_DIR, "similar_qna_pairs.csv")
        if selected_product_id != "all":
            similar_pairs_path = os.path.join(OUTPUT_DIR, f"similar_qna_pairs_{selected_product_id}.csv")
        similar_pairs_df.to_csv(similar_pairs_path, index=False)
        
        # Clear progress message and show success
        progress_placeholder.empty()
        product_info = f" for {selected_product_name}" if selected_product_id != "all" else ""
        st.success(f"✅ Analysis ready! Processed {len(merged_df)} questions{product_info}.")
        data_processed = True

# Create tabs for different functionalities
tabs = ["Similar Questions Search", "Data Explorer", "Clustering Analysis", "Statistics", "Duplicate Detection"]
if MERGE_AVAILABLE:
    tabs.append("Merge Questions")

tab_objects = st.tabs(tabs)
tab1, tab2, tab3, tab4, tab5 = tab_objects[0], tab_objects[1], tab_objects[2], tab_objects[3], tab_objects[4]
if MERGE_AVAILABLE:
    tab6 = tab_objects[5]

# Helper function to display product context info in each tab
def show_product_context():
    if selected_product_id != "all":
        st.markdown(f"""
        <div class="context-info">
            <strong>Product Context:</strong> {selected_product_name}
            <br><small>All results are filtered to this product only.</small>
        </div>
        """, unsafe_allow_html=True)

# Tab 1: Similar Questions Search
with tab1:
    st.header("Search for Similar Questions")
    
    # Show product context
    show_product_context()
    
    if not data_processed:
        st.warning("Please process the data first using the sidebar option.")
    else:
        # Create search interface
        query = st.text_input("Enter your security question:", placeholder="e.g., What is your information security policy?")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            top_k = st.slider("Number of results:", min_value=1, max_value=20, value=5)
        
        # Add option to include archived questions
        with col2:
            show_archived = st.checkbox("Include archived questions in results", value=True, 
                                        help="Archived questions are those that have been deleted (have a value in the deleted_at field)")
        
        if st.button("Search") and query:
            with st.spinner('Searching for similar questions...'):
                # Load the model
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                
                # Create index again for search
                index, norm_embeddings = create_similarity_index(embeddings)
                
                # Perform search with modified function
                results = search_similar_questions_with_deleted_status(query, model, index, norm_embeddings, merged_df, top_k=top_k)
                
                # Filter active vs archived questions
                active_results = [r for r in results if not r['is_archived']]
                archived_results = [r for r in results if r['is_archived']]
                
                # Display active results
                st.subheader(f"Active Questions ({len(active_results)})")
                
                if not active_results:
                    st.info("No active similar questions found.")
                else:
                    for result in active_results:
                        with st.container():
                            st.markdown(f"""
                            <div class="result-card">
                                <h4>{result['question']}</h4>
                                <div class="similarity-score">Similarity: {result['similarity']:.4f}</div>
                                <div class="category-badge">{result['category']}</div>
                                
                                {'<div class="answer-text"><strong>Answer:</strong> ' + str(result['answer']) + '</div>' if pd.notna(result['answer']) else '<div class="answer-text">No answer available</div>'}
                                
                                {'<div class="details-text"><strong>Details:</strong> ' + str(result['details']) + '</div>' if pd.notna(result['details']) else ''}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Display archived results if checkbox is checked
                if show_archived and archived_results:
                    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                    st.subheader(f"Archived Questions ({len(archived_results)})")
                    
                    for result in archived_results:
                        with st.container():
                            archived_date = pd.to_datetime(result['deleted_at']).strftime('%Y-%m-%d') if pd.notna(result['deleted_at']) else ""
                            st.markdown(f"""
                            <div class="archived-card">
                                <h4>{result['question']}</h4>
                                <div class="similarity-score">Similarity: {result['similarity']:.4f}</div>
                                <span class="category-badge">{result['category']}</span>
                                <span class="archived-badge">Archived on {archived_date}</span>
                                
                                {'<div class="answer-text"><strong>Answer:</strong> ' + str(result['answer']) + '</div>' if pd.notna(result['answer']) else '<div class="answer-text">No answer available</div>'}
                                
                                {'<div class="details-text"><strong>Details:</strong> ' + str(result['details']) + '</div>' if pd.notna(result['details']) else ''}
                            </div>
                            """, unsafe_allow_html=True)
                
                elif show_archived and not archived_results:
                    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                    st.info("No archived similar questions found.")

# Tab 2: Data Explorer
with tab2:
    st.header("Data Explorer")
    
    # Show product context
    show_product_context()
    
    if not data_processed:
        st.warning("Please process the data first using the sidebar option.")
    else:
        # Create a data explorer interface
        st.subheader("Explore the merged dataset")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            categories = ["All"] + sorted(merged_df['category'].unique().tolist())
            selected_category = st.selectbox("Filter by category:", categories)
        
        with col2:
            search_term = st.text_input("Search in questions:", placeholder="Enter keywords...")
            
        with col3:
            archive_status = st.selectbox("Archive status:", ["All", "Active Only", "Archived Only"])
        
        # Apply filters
        filtered_df = merged_df.copy()
        
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
            
        if search_term:
            filtered_df = filtered_df[filtered_df['question'].str.contains(search_term, case=False, na=False)]
        
        if archive_status == "Active Only":
            filtered_df = filtered_df[filtered_df['deleted_at'].isna()]
        elif archive_status == "Archived Only":
            filtered_df = filtered_df[filtered_df['deleted_at'].notna()]
        
        # Display filtered data
        st.write(f"Showing {len(filtered_df)} records")
        
        # Add visual indication of archived records
        def highlight_archived(row):
            if pd.notna(row['deleted_at']):
                return ['background-color: #fff3e0'] * len(row)
            return [''] * len(row)
        
        # Apply styling and display dataframe
        styled_df = filtered_df[['cqid', 'product_id', 'category', 'question', 'answer', 'details', 'deleted_at']].style.apply(highlight_archived, axis=1)
        
        # Only use styling for reasonably sized dataframes
        if len(filtered_df) > 5000:
            st.warning(f"Displaying without styling due to large data size ({len(filtered_df)} rows)")
            st.dataframe(filtered_df[['cqid', 'product_id', 'category', 'question', 'answer', 'details', 'deleted_at']], height=400)
        else:
            try:
                st.dataframe(styled_df, height=400)
            except Exception as e:
                st.warning(f"Error applying styling: {str(e)}. Displaying without styling.")
                st.dataframe(filtered_df[['cqid', 'product_id', 'category', 'question', 'answer', 'details', 'deleted_at']], height=400)
        
        # Download button
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download filtered data as CSV",
            data=csv_data,
            file_name="filtered_data.csv",
            mime="text/csv"
        )

# Tab 3: Clustering Analysis
with tab3:
    st.header("Clustering Analysis")
    
    # Show product context
    show_product_context()
    
    if not data_processed:
        st.warning("Please process the data first using the sidebar option.")
    else:
        # Create clustering interface
        st.subheader("Analyze question clusters")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            min_cluster_size = st.slider("Minimum cluster size:", min_value=2, max_value=50, value=10)
        with col2:
            min_samples = st.slider("Minimum samples:", min_value=1, max_value=10, value=1)
        with col3:
            include_archived = st.checkbox("Include archived questions", value=False)
        
        if st.button("Generate Clusters"):
            with st.spinner('Generating clusters...'):
                # Filter out archived questions if needed
                cluster_df = merged_df.copy()
                if not include_archived:
                    cluster_df = cluster_df[cluster_df['deleted_at'].isna()]
                    st.info(f"Analyzing {len(cluster_df)} active questions (excluding {len(merged_df) - len(cluster_df)} archived questions)")
                
                # Apply product filtering if needed (should already be done at merged_df level)
                if selected_product_id != "all":
                    cluster_df = cluster_df[cluster_df['product_id'] == selected_product_id]
                
                # Generate embeddings for this specific subset if necessary
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                cluster_embeddings = model.encode(cluster_df['question'].tolist())
                
                # Perform clustering on the filtered data
                labels, reduced_embeddings = perform_clustering(cluster_embeddings, min_cluster_size, min_samples)
                
                # Add cluster labels to dataframe
                clustered_df = merged_df.copy()
                clustered_df['cluster'] = labels
                
                # Filter if needed
                if not include_archived:
                    clustered_df = clustered_df[clustered_df['deleted_at'].isna()]
                
                # Save clustered data
                clustered_data_path = os.path.join(OUTPUT_DIR, "clustered_data.csv")
                clustered_df.to_csv(clustered_data_path, index=False)
                
                # Calculate cluster statistics
                cluster_counts = clustered_df['cluster'].value_counts().reset_index()
                cluster_counts.columns = ['Cluster', 'Count']
                
                # Remove noise points (cluster -1)
                valid_clusters = cluster_counts[cluster_counts['Cluster'] != -1]
                
                # Display cluster statistics
                st.subheader("Cluster Statistics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Clusters", len(valid_clusters))
                col2.metric("Largest Cluster Size", cluster_counts['Count'].max())
                col3.metric("Noise Points", cluster_counts[cluster_counts['Cluster'] == -1]['Count'].values[0] if -1 in cluster_counts['Cluster'].values else 0)
                
                # Bar chart of cluster sizes
                st.bar_chart(valid_clusters.set_index('Cluster'))
                
                # Display sample questions from each cluster
                st.subheader("Sample Questions from Clusters")
                
                # Get top clusters by size
                top_clusters = valid_clusters.sort_values('Count', ascending=False).head(5)['Cluster'].tolist()
                
                for cluster_id in top_clusters:
                    with st.expander(f"Cluster {cluster_id} ({cluster_counts[cluster_counts['Cluster'] == cluster_id]['Count'].values[0]} questions)"):
                        # Get sample questions from this cluster
                        sample_questions = clustered_df[clustered_df['cluster'] == cluster_id].sample(min(5, cluster_counts[cluster_counts['Cluster'] == cluster_id]['Count'].values[0]))
                        for i, (_, row) in enumerate(sample_questions.iterrows()):
                            archived_badge = '<span class="archived-badge">Archived</span>' if pd.notna(row['deleted_at']) else ''
                            st.markdown(f"**{i+1}. {row['question']}** ({row['category']}) {archived_badge}", unsafe_allow_html=True)

# Tab 4: Statistics
with tab4:
    st.header("Dataset Statistics")
    
    # Show product context
    show_product_context()
    
    if not data_processed:
        st.warning("Please process the data first using the sidebar option.")
    else:
        # Display key statistics about the dataset
        st.subheader("Key Metrics")
        
        # Calculate archived stats
        # Use merged_df which is already filtered by product if needed
        deleted_df, not_deleted_df = separate_deleted_data(merged_df)
        archived_count = len(deleted_df)
        active_count = len(not_deleted_df)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Questions", len(merged_df))
        col2.metric("Active Questions", active_count)
        col3.metric("Archived Questions", archived_count)
        col4.metric("Categories", merged_df['category'].nunique())
        
        # Category distribution
        st.subheader("Category Distribution")
        category_counts = merged_df['category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        
        # Display top 10 categories
        top_categories = category_counts.head(10)
        st.bar_chart(top_categories.set_index('Category'))
        
        # Display active vs. archived data
        st.subheader("Active vs. Archived Questions")
        
        status_data = {
            'Status': ['Active', 'Archived'],
            'Count': [active_count, archived_count]
        }
        status_data_df = pd.DataFrame(status_data)
        st.bar_chart(status_data_df.set_index('Status'))
        
        # Questions with answers vs. without answers
        st.subheader("Questions With/Without Answers")
        with_answer = merged_df['answer'].notna().sum()
        without_answer = merged_df['answer'].isna().sum()
        
        answer_data = {
            'Status': ['With Answer', 'Without Answer'],
            'Count': [with_answer, without_answer]
        }
        answer_data_df = pd.DataFrame(answer_data)
        st.bar_chart(answer_data_df.set_index('Status'))
        
        # Archive status by category
        st.subheader("Archive Status by Category")
        
        # Calculate percentage of archived questions per category
        category_archive_stats = []
        for category in merged_df['category'].unique():
            cat_df = merged_df[merged_df['category'] == category]
            archived = cat_df['deleted_at'].notna().sum()
            total = len(cat_df)
            category_archive_stats.append({
                'Category': category,
                'Total': total,
                'Archived': archived,
                'Active': total - archived,
                'Archived %': round(100 * archived / total, 1)
            })
        
        archive_stats_df = pd.DataFrame(category_archive_stats)
        archive_stats_df = archive_stats_df.sort_values('Archived %', ascending=False).head(10)
        
        st.dataframe(archive_stats_df)

# Tab 5: Duplicate Detection
with tab5:
    st.header("Duplicate Detection & Suggestions")
    
    # Show product context
    show_product_context()
    
    if not data_processed:
        st.warning("Please process the data first using the sidebar option.")
    else:
        st.subheader("1. Automatic Duplicate Detection")
        
        # Display explanation about duplicate detection
        st.info("""
        This tool automatically detects potential duplicate questions in the database based on semantic similarity.
        Questions with similarity scores above 0.85 are considered potential duplicates. This can help:
        - Clean up the knowledge base
        - Consolidate similar questions
        - Improve consistency in answers
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            similarity_threshold = st.slider(
                "Similarity threshold:", 
                min_value=0.70, 
                max_value=0.99, 
                value=0.85,
                help="Higher values mean more strict matching (fewer but more accurate duplicates)"
            )
        
        with col2:
            show_archived = st.checkbox(
                "Include archived questions", 
                value=False,
                help="Whether to include archived questions in duplicate detection"
            )
        
        if st.button("Find Potential Duplicates"):
            with st.spinner('Analyzing for potential duplicates...'):
                # Load the model if needed
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                
                print("Creating index...")
                # Create index again for search
                index, norm_embeddings = create_similarity_index(embeddings)

                
                # Filter out archived questions if needed
                if not show_archived:
                    filter_df = merged_df[merged_df['deleted_at'].isna()].reset_index(drop=True)
                    # If selected product is not "all", make sure we're only using questions from that product
                    if selected_product_id != "all":
                        filter_df = filter_df[filter_df['product_id'] == selected_product_id].reset_index(drop=True)
                    
                    filtered_embeddings = model.encode(filter_df['question'].tolist())
                    filtered_index, filtered_norm_embeddings = create_similarity_index(filtered_embeddings)
                    
                    # Perform duplicate detection
                    duplicate_df = find_potential_duplicates(
                        filtered_index, 
                        filtered_norm_embeddings, 
                        filter_df, 
                        similarity_threshold
                    )
                else:
                    # If selected product is not "all", filter merged_df
                    if selected_product_id != "all":
                        filter_df = merged_df[merged_df['product_id'] == selected_product_id].reset_index(drop=True)
                        filtered_embeddings = model.encode(filter_df['question'].tolist())
                        filtered_index, filtered_norm_embeddings = create_similarity_index(filtered_embeddings)
                        
                        # Perform duplicate detection with filtered data
                        duplicate_df = find_potential_duplicates(
                            filtered_index, 
                            filtered_norm_embeddings, 
                            filter_df, 
                            similarity_threshold
                        )
                    else:
                        # Perform duplicate detection with all questions
                        duplicate_df = find_potential_duplicates(
                            index, 
                            norm_embeddings, 
                            merged_df, 
                            similarity_threshold
                        )
                
                # Generate summary statistics
                summary_df = generate_duplicate_report(duplicate_df)
                
                if duplicate_df.empty:
                    st.info("No potential duplicates found with the current threshold. Try lowering the similarity threshold.")
                else:
                    # Display summary statistics
                    st.subheader("Summary")
                    for idx, row in summary_df.iterrows():
                        st.metric(row['Metric'], row['Value'])
                    
                    # Display duplicate pairs
                    st.subheader("Potential Duplicate Pairs")
                    st.write(f"Found {len(duplicate_df)} potential duplicate pairs.")
                    
                    # Format and display the duplicate pairs
                    for i, row in duplicate_df.iterrows():
                        with st.expander(f"Pair #{i+1}: Similarity {row['similarity']:.4f}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Question 1:**")
                                st.info(row['question_1'])
                                st.markdown(f"**Category:** {row['category_1']}")
                                if row['is_archived_1']:
                                    st.warning("⚠️ This question is archived")
                            
                            with col2:
                                st.markdown("**Question 2:**")
                                st.info(row['question_2'])
                                st.markdown(f"**Category:** {row['category_2']}")
                                if row['is_archived_2']:
                                    st.warning("⚠️ This question is archived")
                            
                            st.markdown("**Suggested Action:**")
                            if row['is_archived_1'] and not row['is_archived_2']:
                                st.success(f"✅ Keep Question 2, as Question 1 is already archived")
                            elif not row['is_archived_1'] and row['is_archived_2']:
                                st.success(f"✅ Keep Question 1, as Question 2 is already archived")
                            elif row['is_archived_1'] and row['is_archived_2']:
                                st.warning("Both questions are archived - no action needed")
                            else:
                                # Both are active, suggest keeping the newer one
                                st.success(f"✅ Suggested to keep the newer question: {row['newer_question']}")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.subheader("2. New Question Suggestion")
        st.info("""
        Enter a new question you're considering adding to the knowledge base. 
        The system will automatically suggest semantically similar existing questions 
        to avoid creating duplicates.
        """)
        
        new_question = st.text_area(
            "Enter a new question:", 
            placeholder="e.g., Does your company have an information security policy?",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            similarity_threshold = st.slider(
                "Minimum similarity to show:", 
                min_value=0.50, 
                max_value=0.95, 
                value=0.70,
                key="new_q_threshold"
            )
        
        with col2:
            show_archived_similar = st.checkbox(
                "Show archived similar questions", 
                value=True,
                key="show_archived_similar"
            )
        
        if st.button("Check for Similar Questions") and new_question:
            with st.spinner('Searching for similar existing questions...'):
                # Load the model
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                
                # Create index again for search or use the existing one
                index, norm_embeddings = create_similarity_index(embeddings)
                
                # Filter merged_df for specific product if needed
                search_df = merged_df
                if selected_product_id != "all":
                    search_df = merged_df[merged_df['product_id'] == selected_product_id].reset_index(drop=True)
                    # Regenerate embeddings for filtered data
                    search_embeddings = model.encode(search_df['question'].tolist())
                    index, norm_embeddings = create_similarity_index(search_embeddings)
                
                # Perform similarity search
                similar_questions = find_similar_to_new_question(
                    new_question, 
                    model, 
                    index, 
                    norm_embeddings, 
                    search_df, 
                    top_k=10,
                    similarity_threshold=similarity_threshold
                )
                
                # Filter based on archive status if needed
                if not show_archived_similar:
                    similar_questions = [q for q in similar_questions if not q['is_archived']]
                
                # Display results
                if not similar_questions:
                    st.success("✅ No similar questions found! This appears to be a unique question.")
                else:
                    st.warning(f"⚠️ Found {len(similar_questions)} similar existing questions!")
                    
                    # Determine if this might be a duplicate
                    high_similarity = any(q['similarity'] > 0.85 for q in similar_questions)
                    if high_similarity:
                        st.error("❌ This question is very similar to existing questions and may be a duplicate!")
                    
                    # Display similar questions
                    for i, question in enumerate(similar_questions):
                        archived_badge = '📦 ARCHIVED' if question['is_archived'] else ''
                        
                        # Choose styling based on similarity score
                        if question['similarity'] > 0.85:
                            container_style = "danger"
                            similarity_label = "Very High Similarity"
                        elif question['similarity'] > 0.8:
                            container_style = "warning" 
                            similarity_label = "High Similarity"
                        else:
                            container_style = "info"
                            similarity_label = "Moderate Similarity"
                        
                        with st.container():
                            st.markdown(f"""
                            <div style="border-left: 5px solid {
                                '#d9534f' if container_style == 'danger' else 
                                '#f0ad4e' if container_style == 'warning' else 
                                '#5bc0de'
                            }; padding-left: 10px; margin-bottom: 10px;">
                                <h4>{question['question']} {archived_badge}</h4>
                                <p><b>{similarity_label}</b>: {question['similarity']:.4f}</p>
                                <p><b>Category:</b> {question['category']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            with st.expander("Show answer and details"):
                                st.markdown(f"**Answer:** {question['answer'] if pd.notna(question['answer']) else 'No answer available'}")
                                if pd.notna(question['details']):
                                    st.markdown(f"**Details:** {question['details']}")

# Tab 6: Merge Questions
if MERGE_AVAILABLE and 'tab6' in locals():
    with tab6:
        st.header("Merge Similar Questions")
        
        # Show product context
        show_product_context()
        
        if not data_processed:
            st.warning("Please process the data first using the sidebar option.")
        else:
            st.info("""
            This tool helps you merge similar questions to reduce duplication in your knowledge base.
            You can manually select questions to merge or use automatic merging for questions above a similarity threshold.
            """)
            
            # Tabs for manual vs automatic merging
            merge_tab1, merge_tab2 = st.tabs(["Manual Merge", "Automatic Batch Merge"])
            
            # Tab for manual merging
            with merge_tab1:
                st.subheader("Manually Merge Two Questions")
                
                # Display warnings about missing dependencies
                if not NLTK_AVAILABLE:
                    st.warning("⚠️ NLTK sentence tokenizer not available. Using regex fallback for sentence splitting.")
                if not SUMMARIZER_AVAILABLE:
                    st.warning("⚠️ DistilBART summarization model not available. Merged text will not be summarized for better coherence.")
                
                # Select questions to merge
                col1, col2 = st.columns(2)
                
                # Create a safety wrapper for getting dataframe row
                def get_question_data(cqid, default_msg="Question data not available"):
                    try:
                        if cqid in merged_df['cqid'].values:
                            return merged_df[merged_df['cqid'] == cqid].iloc[0]
                        else:
                            st.error(f"Question ID {cqid} not found in dataset")
                            return None
                    except Exception as e:
                        st.error(f"Error retrieving question data: {str(e)}")
                        return None
                
                with col1:
                    q1_id = st.selectbox(
                        "Select first question:", 
                        options=merged_df['cqid'].tolist(),
                        format_func=lambda x: merged_df[merged_df['cqid'] == x]['question'].values[0] if x in merged_df['cqid'].values else str(x)
                    )
                    q1_data = get_question_data(q1_id)
                    
                    if q1_data is not None:
                        st.write("**Question 1 Details:**")
                        st.info(q1_data['question'])
                        if pd.notna(q1_data['details']):
                            st.write("**Details:**")
                            st.text_area("", value=q1_data['details'], height=100, key='details1', disabled=True)
                        if pd.notna(q1_data['answer']):
                            st.write("**Answer:**")
                            st.text_area("", value=q1_data['answer'], height=100, key='answer1', disabled=True)
                
                with col2:
                    # Ensure we have at least 2 options or select the same for demonstration
                    index_option = 1 if len(merged_df) > 1 else 0
                    
                    q2_id = st.selectbox(
                        "Select second question:", 
                        options=merged_df['cqid'].tolist(),
                        format_func=lambda x: merged_df[merged_df['cqid'] == x]['question'].values[0] if x in merged_df['cqid'].values else str(x),
                        index=index_option
                    )
                    q2_data = get_question_data(q2_id)
                    
                    if q2_data is not None:
                        st.write("**Question 2 Details:**")
                        st.info(q2_data['question'])
                        if pd.notna(q2_data['details']):
                            st.write("**Details:**")
                            st.text_area("", value=q2_data['details'], height=100, key='details2', disabled=True)
                        if pd.notna(q2_data['answer']):
                            st.write("**Answer:**")
                            st.text_area("", value=q2_data['answer'], height=100, key='answer2', disabled=True)
                
                st.markdown("---")
                st.subheader("Merge Result")
                
                # Button to perform merge
                if st.button("Merge These Questions") and q1_data is not None and q2_data is not None:
                    try:
                        # Save to database with original question IDs
                        new_id = merge_questions(
                            q1_data['question'],
                            q2_data['question'],
                            q1_data.get('details', None),
                            q2_data.get('details', None),
                            q1_data.get('answer', None),
                            q2_data.get('answer', None),
                            q1_data.get('category', None),
                            q2_data.get('category', None),
                            q1_data.get('product_id', None),
                            q1_data['cqid'],
                            q2_data['cqid']
                        )
                        
                        # Update the merged_df to include the new merged question
                        updated_df = load_merged_with_original_data(merged_df)
                        
                        # Store updated dataframe for future use
                        output_path = os.path.join(OUTPUT_DIR, "merged_data.csv")
                        updated_df.to_csv(output_path, index=False)
                        
                        st.success(f"""
                        ✅ Merged question saved! 
                        
                        The original questions have been archived and replaced with the merged question.
                        Reload the app to see the changes.
                        """)
                        
                        # Show a button to reload the app
                        if st.button("Reload App"):
                            st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error saving merged question: {str(e)}")
            
            # Tab for automatic batch merging
            with merge_tab2:
                st.subheader("Automatically Merge Similar Questions")
                
                # Display warnings about missing dependencies
                if not NLTK_AVAILABLE:
                    st.warning("⚠️ NLTK sentence tokenizer not available. Using regex fallback for sentence splitting.")
                if not SUMMARIZER_AVAILABLE:
                    st.warning("⚠️ DistilBART summarization model not available. Merged text will not be summarized for better coherence.")
                
                # Set similarity threshold
                similarity = st.slider(
                    "Similarity Threshold", 
                    min_value=0.80, 
                    max_value=1.0, 
                    value=0.95,
                    step=0.01,
                    help="Only merge questions with similarity above this threshold. Higher values mean stricter matching."
                )
                
                # Option to include archived questions
                include_archived = st.checkbox("Include archived questions in merging", value=False)
                
                # Find potential duplicates to merge
                if st.button("Find and Merge Similar Questions"):
                    with st.spinner('Finding similar questions to merge...'):
                        # Create index for search
                        index, norm_embeddings = create_similarity_index(embeddings)
                        
                        # Filter the dataframe if needed
                        filter_df = merged_df
                        if not include_archived:
                            filter_df = merged_df[merged_df['deleted_at'].isna()].reset_index(drop=True)
                        
                        # Find potential duplicate questions
                        duplicate_df = find_potential_duplicates(index, norm_embeddings, filter_df, similarity_threshold=similarity)
                        
                        if duplicate_df.empty:
                            st.info(f"No questions with similarity above {similarity} found. Try lowering the threshold.")
                        else:
                            # Add details and answers to the duplicate dataframe for merging
                            duplicate_df['details_1'] = duplicate_df.apply(
                                lambda row: filter_df[filter_df['cqid'] == row['cqid_1']]['details'].values[0] 
                                if row['cqid_1'] in filter_df['cqid'].values and len(filter_df[filter_df['cqid'] == row['cqid_1']]['details'].values) > 0 
                                else None, axis=1
                            )
                            duplicate_df['details_2'] = duplicate_df.apply(
                                lambda row: filter_df[filter_df['cqid'] == row['cqid_2']]['details'].values[0]
                                if row['cqid_2'] in filter_df['cqid'].values and len(filter_df[filter_df['cqid'] == row['cqid_2']]['details'].values) > 0
                                else None, axis=1
                            )
                            duplicate_df['answer_1'] = duplicate_df.apply(
                                lambda row: filter_df[filter_df['cqid'] == row['cqid_1']]['answer'].values[0]
                                if row['cqid_1'] in filter_df['cqid'].values and len(filter_df[filter_df['cqid'] == row['cqid_1']]['answer'].values) > 0
                                else None, axis=1
                            )
                            duplicate_df['answer_2'] = duplicate_df.apply(
                                lambda row: filter_df[filter_df['cqid'] == row['cqid_2']]['answer'].values[0]
                                if row['cqid_2'] in filter_df['cqid'].values and len(filter_df[filter_df['cqid'] == row['cqid_2']]['answer'].values) > 0
                                else None, axis=1
                            )
                            
                            # Process batch merging
                            merged_df_result = batch_process_similar_questions(duplicate_df, threshold=similarity)
                            
                            # Display merge results
                            st.success(f"✅ Successfully merged {len(merged_df_result)} pairs of similar questions!")
                            
                            # Show the merged questions
                            st.subheader("Merged Questions")
                            st.write(f"Found and merged {len(merged_df_result)} pairs of questions with similarity above {similarity}.")
                            
                            for i, row in merged_df_result.iterrows():
                                with st.expander(f"Merged Pair #{i+1}: {row['merged_question'][:80]}..."):
                                    st.write("**Original Questions:**")
                                    try:
                                        q1 = duplicate_df[duplicate_df['cqid_1'] == row['original_id_1']]['question_1'].values[0] if row['original_id_1'] in duplicate_df['cqid_1'].values else "Question data not available"
                                        q2 = duplicate_df[duplicate_df['cqid_2'] == row['original_id_2']]['question_2'].values[0] if row['original_id_2'] in duplicate_df['cqid_2'].values else "Question data not available"
                                        st.info(f"1. {q1}")
                                        st.info(f"2. {q2}")
                                    except (IndexError, KeyError) as e:
                                        st.warning(f"Could not retrieve original questions: {str(e)}")
                                        st.info("1. Original question data unavailable")
                                        st.info("2. Original question data unavailable")
                                    
                                    st.write("**Merged Question:**")
                                    st.success(row['merged_question'])
                                    
                                    st.write("**Merged Details:**")
                                    st.text_area("", value=row['merged_details'], height=100, key=f'details_{i}', disabled=True)
                                    
                                    st.write("**Merged Answer:**")
                                    st.text_area("", value=row['merged_answer'], height=100, key=f'answer_{i}', disabled=True)
                            
                            # Option to save all merged questions
                            if st.button("Save All Merged Questions"):
                                try:
                                    # Save merged questions and archive originals
                                    merged_ids, updated_df = batch_save_merged_questions(merged_df_result, merged_df)
                                    
                                    if merged_ids:
                                        # Save the updated dataframe
                                        output_path = os.path.join(OUTPUT_DIR, "merged_data.csv")
                                        updated_df.to_csv(output_path, index=False)
                                        
                                        st.success(f"""
                                        ✅ Successfully saved {len(merged_ids)} merged questions! 
                                        
                                        The original questions have been archived. Reload the app to see the changes.
                                        """)
                                        
                                        # Display merged question IDs
                                        with st.expander("Merged Question IDs"):
                                            for mid in merged_ids:
                                                st.code(mid)
                                        
                                        # Show a button to reload the app
                                        if st.button("Reload App"):
                                            st.experimental_rerun()
                                    else:
                                        st.warning("No questions were merged.")
                                except Exception as e:
                                    st.error(f"Error saving merged questions: {str(e)}")
                                    st.info("Please check the console for more information.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>KL Enhancer v1.0 | Security Questionnaire Analysis Tool</p>", unsafe_allow_html=True) 