import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_processing import process_data
from utils.clustering import (
    perform_clustering_with_params, 
    generate_cluster_characteristics,
    create_cluster_distribution_chart,
    create_time_cluster_analysis
)
import warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def show_data_processing_page():
    """Enhanced Data Processing Page."""
    
    # Page header
    st.markdown("""
    <div class="main-title-container">
        <h1 class="main-title">📊 Data Processing & HDBSCAN Analysis</h1>
        <p class="main-subtitle">Upload your dataset and explore optimal clustering parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    st.markdown("### 📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Choose your crime dataset (CSV format)",
        type=['csv'],
        help="Upload a CSV file containing crime data"
    )
    
    # Handle file upload
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Data overview
            st.markdown("### 📈 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #667eea; margin: 0;">📊 Records</h3>
                    <h2 style="margin: 0.5rem 0 0 0;">{:,}</h2>
                </div>
                """.format(len(df)), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #667eea; margin: 0;">📋 Columns</h3>
                    <h2 style="margin: 0.5rem 0 0 0;">{}</h2>
                </div>
                """.format(len(df.columns)), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #667eea; margin: 0;">❌ Missing</h3>
                    <h2 style="margin: 0.5rem 0 0 0;">{:,}</h2>
                </div>
                """.format(df.isnull().sum().sum()), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class="metric-card">
                    <h3 style="color: #667eea; margin: 0;">🔄 Duplicates</h3>
                    <h2 style="margin: 0.5rem 0 0 0;">{:,}</h2>
                </div>
                """.format(df.duplicated().sum()), unsafe_allow_html=True)
            
            # Sample data preview
            with st.expander("🔍 Preview Dataset", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Data Processing Section
            st.markdown("### 🚀 Data Processing & Feature Engineering")
            
            if st.button("🔄 Process Data", type="primary", use_container_width=True):
                processed_df, processing_stats = process_data(df.copy())
                
                # Store processed data
                st.session_state.processed_df = processed_df
                st.session_state.processing_stats = processing_stats
                st.session_state.original_df = df.copy()
                
                # Clear previous analysis results
                clear_analysis_session_state()
                
                st.success("✅ Data processing completed!")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.error("Please ensure your CSV file has the correct format.")
    
    # Show current status if data is processed
    if st.session_state.get('processed_df') is not None:
        display_processed_data_info()
        
        # HDBSCAN Tuning Section
        show_hdbscan_tuning_interface()
    
    elif uploaded_file is None:
        show_instructions()

def display_processed_data_info():
    """Display information about currently processed data with safe error handling."""
    st.success("✅ Data processing completed and ready for HDBSCAN analysis!")
    
    if st.session_state.get('processing_stats'):
        stats = st.session_state.processing_stats
        
        # FIX: Safe access dengan fallback values
        try:
            final_shape = stats.get('final_shape', (0, 0))
            records_count = final_shape[0] if isinstance(final_shape, (list, tuple)) else stats.get('records_after_cleaning', 0)
            features_count = final_shape[1] if isinstance(final_shape, (list, tuple)) else stats.get('features_count', 0)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Records", f"{records_count:,}")
            with col2:
                st.metric("Features", features_count)
            with col3:
                st.metric("Crime Categories", stats.get('crime_categories', 0))
            with col4:
                st.metric("Location Categories", stats.get('location_categories', 0))
                
            # Debug info jika ada error
            if 'error' in stats:
                st.warning(f"⚠️ Processing warning: {stats['error']}")
                
        except Exception as e:
            st.error(f"❌ Error displaying stats: {str(e)}")
            # Fallback: Show basic info from DataFrame directly
            if st.session_state.get('processed_df') is not None:
                df = st.session_state.processed_df
                st.info(f"📊 **Processed Data**: {len(df):,} records, {len(df.columns)} columns")

def show_hdbscan_tuning_interface():
    """Show HDBSCAN tuning interface - terpisah agar bisa dipanggil tanpa rerun."""
    
    # HDBSCAN Tuning Section
    st.markdown("---")
    st.markdown("### 🎯 HDBSCAN Parameter Tuning")
    
    # Parameter selection interface berdasarkan notebook
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("""
        **Explore different HDBSCAN parameters to find optimal clustering:**
        - **Min Cluster Size**: Minimum number of points to form a cluster  
        - **Min Samples**: Core point threshold for cluster formation
        
        📊 **Parameter combinations from notebook:** 4 min sizes × 4 min samples = 16 tests
        """)
    
    with col2:
        st.markdown("**Min Cluster Size:**")
        min_cluster_sizes = [300, 500, 1000, 2000]
        selected_min_sizes = st.multiselect(
            "Select sizes to test:",
            min_cluster_sizes,
            default=st.session_state.get('selected_min_sizes', min_cluster_sizes),
            key="min_sizes_select"
        )
        st.session_state.selected_min_sizes = selected_min_sizes
    
    with col3:
        st.markdown("**Min Samples:**")
        min_samples_options = [5, 10, 20, 50]
        selected_min_samples = st.multiselect(
            "Select samples to test:",
            min_samples_options,
            default=st.session_state.get('selected_min_samples', min_samples_options),
            key="min_samples_select"
        )
        st.session_state.selected_min_samples = selected_min_samples
    
    # Run tuning button
    if st.button("🔍 Run HDBSCAN Tuning", type="secondary", use_container_width=True):
        if not selected_min_sizes or not selected_min_samples:
            st.error("Please select at least one option for both parameters!")
        else:
            # Run tuning with selected parameters
            tuning_results = run_hdbscan_tuning_custom(
                st.session_state.processed_df, 
                selected_min_sizes, 
                selected_min_samples
            )
            
            if tuning_results is not None:
                st.session_state.tuning_results = tuning_results
                # FIX: Langsung tampilkan results tanpa rerun
    
    # Display tuning results table - langsung ditampilkan jika ada
    if st.session_state.get('tuning_results') is not None:
        show_tuning_results()
    
    # Final Clustering Section - langsung ditampilkan jika tuning sudah ada
    if st.session_state.get('tuning_results') is not None:
        show_final_clustering_section()
    
    # Show analysis results if completed - langsung ditampilkan jika sudah selesai
    if st.session_state.get('analysis_completed', False):
        show_analysis_results()

def show_tuning_results():
    """Display tuning results tanpa validation."""
    st.markdown("#### 📊 HDBSCAN Tuning Results")
    tuning_df = st.session_state.tuning_results
    
    # Display table
    display_df = tuning_df.copy()
    display_df['noise_percentage'] = display_df['noise_percentage'].round(1)
    display_df = display_df.sort_values(['min_cluster_size', 'min_samples'])
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Row selection for visualization
    st.markdown("**📋 Click on a row to select parameters for final clustering:**")
    
    selected_indices = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    # Store selected row
    if len(selected_indices.selection.rows) > 0:
        selected_idx = selected_indices.selection.rows[0]
        selected_row = display_df.iloc[selected_idx]
        
        st.session_state.viz_min_cluster_size = int(selected_row['min_cluster_size'])
        st.session_state.viz_min_samples = int(selected_row['min_samples'])
        
        st.success(f"""
        ✅ **Selected Configuration:**  
        • Min Cluster Size: {int(selected_row['min_cluster_size'])}  
        • Min Samples: {int(selected_row['min_samples'])}  
        • Expected Clusters: {selected_row['n_clusters']}  
        • Expected Noise: {selected_row['n_noise']} ({selected_row['noise_percentage']:.1f}%)
        """)
    
    # Best configuration suggestion
    best_config = tuning_df.loc[tuning_df['n_clusters'].idxmax()]
    st.info(f"""
    💡 **Suggested Configuration**: Min Size = {int(best_config['min_cluster_size'])}, 
    Min Samples = {int(best_config['min_samples'])} 
    (Results in {best_config['n_clusters']} clusters with {best_config['noise_percentage']:.1f}% noise)
    """)

def show_final_clustering_section():
    """Show final clustering section."""
    st.markdown("---")
    st.markdown("### 🎪 Final Clustering & Analysis")
    
    # Get selected parameters dengan validasi ketat
    viz_min_cluster_size = st.session_state.get('viz_min_cluster_size', 2000)
    viz_min_samples = st.session_state.get('viz_min_samples', 50)
    
    # CRITICAL: Pastikan integer conversion
    try:
        viz_min_cluster_size = int(viz_min_cluster_size)
        viz_min_samples = int(viz_min_samples)
    except (ValueError, TypeError):
        st.error("❌ Invalid parameter types!")
        return
    
    st.info(f"""
    **Using Selected Parameters:**  
    • Min Cluster Size: {viz_min_cluster_size}  
    • Min Samples: {viz_min_samples}
    """)
    
    # Show clustering status if already completed
    if st.session_state.get('analysis_completed'):
        st.success("✅ Final clustering already completed! Check results below or go to Visualizations page.")
    
    if st.button("🚀 Generate Final Clustering", type="primary", use_container_width=True):
        # CRITICAL: Pass integer parameters explicitly
        clustered_df, scaled_features, labels, cluster_stats = perform_clustering_with_params(
            st.session_state.processed_df, 
            viz_min_cluster_size,  # Sudah integer
            viz_min_samples       # Sudah integer
        )
        
        if clustered_df is not None:
            # Store results
            st.session_state.clustered_df = clustered_df
            st.session_state.cluster_stats = cluster_stats
            st.session_state.scaled_features = scaled_features
            st.session_state.analysis_completed = True
            
            st.success("✅ Final clustering completed! Go to Visualizations page to explore results.")
            st.rerun()  # Refresh untuk show results

def clear_analysis_session_state():
    """Clear analysis-related session state when new data is processed."""
    keys_to_clear = [
        'tuning_results', 'clustered_df', 'cluster_stats', 'scaled_features',
        'analysis_completed', 'viz_min_cluster_size', 'viz_min_samples',
        'embedding', 'scaler', 'features_for_clustering'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def run_hdbscan_tuning_custom(df, min_cluster_sizes, min_samples_list):
    """Run HDBSCAN tuning with improved error handling for deployment."""
    
    with st.spinner("🔍 Running HDBSCAN parameter tuning..."):
        import numpy as np
        import warnings
        from sklearn.preprocessing import StandardScaler
        
        # Suppress all warnings for deployment
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                import hdbscan
            except ImportError:
                st.error("❌ HDBSCAN not installed properly")
                return None
            
            try:
                import umap.umap_ as umap
            except ImportError:
                try:
                    import umap
                except ImportError:
                    st.error("❌ UMAP not installed properly")
                    return None
        
        # Features selection with validation
        required_features = ['primary_type_encoded', 'weekday', 'hour', 'crime_scene',
                           'location_category_encoded', 'latitude', 'longitude']
        
        # Validate features
        available_features = [col for col in required_features if col in df.columns]
        if len(available_features) < 7:
            st.error(f"❌ Missing required features: {set(required_features) - set(available_features)}")
            return None
        
        fitur = df[required_features].copy()
        st.write(f"📊 **Data for clustering**: {len(fitur):,} records")
        
        # Enhanced feature validation for deployment
        for col in required_features:
            if col in fitur.columns:
                # Convert to numeric if needed
                if fitur[col].dtype == 'object':
                    try:
                        fitur[col] = pd.to_numeric(fitur[col], errors='coerce')
                    except Exception as e:
                        st.error(f"❌ Cannot convert {col} to numeric: {str(e)}")
                        return None
                
                # Handle infinite values
                if np.isinf(fitur[col]).any():
                    fitur[col] = fitur[col].replace([np.inf, -np.inf], np.nan)
                
                # Fill remaining NaN values
                if fitur[col].isnull().any():
                    fitur[col] = fitur[col].fillna(fitur[col].median())
        
        # Standard scaling with error handling
        try:
            scaler = StandardScaler()
            scaled_fitur = scaler.fit_transform(fitur)
            st.write(f"📊 **Scaled features shape**: {scaled_fitur.shape}")
        except Exception as e:
            st.error(f"❌ Scaling error: {str(e)}")
            return None
        
        # UMAP dimensionality reduction with enhanced error handling
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Adjust n_neighbors based on data size
                n_neighbors = min(30, max(2, len(fitur) // 10))
                
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors, 
                    min_dist=0.1, 
                    n_components=2, 
                    random_state=42,
                    n_jobs=1,
                    verbose=False
                )
                embedding = reducer.fit_transform(scaled_fitur)
                st.write(f"📊 **UMAP embedding shape**: {embedding.shape}")
                
        except Exception as e:
            st.error(f"❌ UMAP error: {str(e)}")
            st.error("This might be due to insufficient data or memory constraints")
            return None
        
        # HDBSCAN tuning loop with better error handling
        tuning_results = []
        total_combinations = len(min_cluster_sizes) * len(min_samples_list)
        
        if total_combinations > 20:
            st.warning("⚠️ Large parameter space detected. This may take some time...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        current_step = 0
        
        st.write("🔍 **HDBSCAN Tuning Progress:**")
        results_placeholder = st.empty()
        
        for min_size in min_cluster_sizes:
            for min_samples in min_samples_list:
                current_step += 1
                status_text.text(f"Testing: min_size={min_size}, min_samples={min_samples} ({current_step}/{total_combinations})")
                
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=int(min_size),
                            min_samples=int(min_samples),
                            metric='euclidean',
                            prediction_data=True,
                            cluster_selection_epsilon=0.0
                        )
                        labels = clusterer.fit_predict(embedding)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    noise = np.sum(labels == -1)
                    noise_percentage = (noise / len(labels)) * 100
                    
                    result_text = f"min_size: {int(min_size)}, min_samples: {int(min_samples)}, Clusters: {n_clusters}, Noise: {noise}"
                    
                    tuning_results.append({
                        'min_cluster_size': int(min_size),
                        'min_samples': int(min_samples),
                        'n_clusters': n_clusters,
                        'n_noise': noise,
                        'noise_percentage': noise_percentage,
                        'result_text': result_text
                    })
                    
                    # Update display with recent results only (for performance)
                    recent_results = tuning_results[-5:] if len(tuning_results) > 5 else tuning_results
                    results_text = "\n".join([r['result_text'] for r in recent_results])
                    results_placeholder.text(f"Recent results:\n{results_text}")
                    
                except Exception as e:
                    st.warning(f"⚠️ Error with min_size={min_size}, min_samples={min_samples}: {str(e)}")
                    continue
                
                # Update progress
                progress_bar.progress(current_step / total_combinations)
        
        # Cleanup
        progress_bar.empty()
        status_text.empty()
        
        if not tuning_results:
            st.error("❌ No successful tuning results. Please check your data or try different parameters.")
            return None
        
        # Convert to DataFrame
        tuning_df = pd.DataFrame(tuning_results)
        
        # Store for later use
        st.session_state.embedding = embedding
        st.session_state.scaler = scaler
        st.session_state.features_for_clustering = fitur
        st.session_state.reducer = reducer
        st.session_state.scaled_fitur = scaled_fitur
        
        st.success(f"✅ Tuning completed! Successfully tested {len(tuning_results)}/{total_combinations} combinations.")
        
        return tuning_df

def show_analysis_results():
    """Display analysis results with charts from notebook."""
    st.markdown("### 🎊 Final Analysis Results")
    
    # Success message
    st.markdown("""
    <div class="success-box">
        <h3 style="margin: 0; color: white;">✅ Analysis Completed Successfully!</h3>
        <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9);">
            Your data has been processed and clustered using HDBSCAN.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Processing statistics
    st.markdown("#### 📊 Data Processing")
    stats = st.session_state.processing_stats
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p><strong>Records After Cleaning:</strong> {stats['final_shape'][0]:,}</p>
            <p><strong>Features After Processing:</strong> {stats['final_shape'][1]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p><strong>Missing Values Removed:</strong> {stats['missing_removed']:,}</p>
            <p><strong>Duplicates Removed:</strong> {stats['duplicates_removed']:,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Clustering results
    st.markdown("#### 🎯 Clustering Results")
    cluster_stats = st.session_state.cluster_stats
    noise_percentage = (cluster_stats['n_noise'] / cluster_stats['total_points']) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Clusters Found", cluster_stats['n_clusters'])
    with col2:
        st.metric("Noise Points", f"{cluster_stats['n_noise']:,}")
    with col3:
        st.metric("Noise Percentage", f"{noise_percentage:.1f}%")
    
    # Charts from notebook implementation
    st.markdown("#### 📈 Cluster Analysis Charts")
    
    # Cluster distribution chart
    fig_dist = create_cluster_distribution_chart(st.session_state.clustered_df)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Time analysis chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig_time = create_time_cluster_analysis(st.session_state.clustered_df)
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        # Location category distribution
        if 'location_category' in st.session_state.clustered_df.columns:
            loc_counts = st.session_state.clustered_df['location_category'].value_counts()
            st.markdown("**Location Categories:**")
            for cat, count in loc_counts.head(10).items():
                st.write(f"• {cat}: {count:,}")
    
    # Download section
    st.markdown("### 💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = st.session_state.clustered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Clustered Dataset",
            data=csv,
            file_name="crime_data_clustered.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Download tuning results
        if st.session_state.get('tuning_results') is not None:
            tuning_csv = st.session_state.tuning_results.to_csv(index=False)
            st.download_button(
                label="📊 Download Tuning Results",
                data=tuning_csv,
                file_name="hdbscan_tuning_results.csv",
                mime="text/csv",
                use_container_width=True
            )

def show_instructions():
    """Show instructions for data upload with notebook context."""
    st.markdown("""
    ### 📋 Instructions
    
    **Required CSV Format (sesuai Chicago Crime dataset):**
    
    Your dataset should contain the following columns:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Essential Columns:**
        - `date` - Crime occurrence date/time
        - `primary_type` - Type of crime
        - `location_description` - Where crime occurred
        - `latitude` - Geographic latitude
        - `longitude` - Geographic longitude
        """)
    
    with col2:
        st.markdown("""
        **Optional Columns:**
        - `case_number` - Unique case ID
        - `description` - Crime description
        - `arrest` - Whether arrest was made
        - `domestic` - Domestic violence indicator
        """)
    
    st.markdown("### 📊 Sample Data Format")
    
    # Sample data based on Chicago Crime dataset
    sample_data = {
        'unique_key': [12830127, 13297608],
        'case_number': ['JF399700', 'JG528761'],
        'date': ['2022-09-15 01:30:00 UTC', '2023-02-24 23:50:00 UTC'],
        'primary_type': ['CRIMINAL SEXUAL ASSAULT', 'CRIMINAL SEXUAL ASSAULT'],
        'description': ['AGGRAVATED - OTHER', 'NON-AGGRAVATED'],
        'location_description': ['DRUG STORE', 'BAR OR TAVERN'],
        'latitude': [41.886815464, 41.885908101],
        'longitude': [-87.628361716, -87.626289429]
    }
    st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
    
    st.markdown("""
    ### 🎯 HDBSCAN Processing Pipeline (sesuai notebook):
    
    1. **Data Cleaning**: Remove missing values and duplicates
    2. **Date Processing**: Extract day, month, year, hour, weekday
    3. **Feature Engineering**: Create crime_scene, day_time, location_category
    4. **Label Encoding**: Encode categorical variables
    5. **Feature Scaling**: Standardize numerical features  
    6. **UMAP Reduction**: Reduce to 2D for clustering
    7. **Parameter Testing**: Test multiple HDBSCAN configurations
    8. **Final Clustering**: Use optimal parameters
    
    **📁 Upload your CSV file above to get started!**
    """)