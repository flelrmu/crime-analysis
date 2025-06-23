import streamlit as st

def show_sidebar():
    """Modern sidebar with gradient styling and page navigation"""
    
    with st.sidebar:
        # Logo and title
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: #667eea; margin: 0;">🚔 Crime Analysis</h1>
            <p style="color: rgba(255,255,255,0.7); margin: 0;">HDBSCAN Clustering</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get current page
        current_page = st.session_state.get('current_page', 'data_processing')
        
        # Navigation Section
        st.markdown("""
        <div style="
            background: rgba(30, 30, 46, 0.9);
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.1);
        ">
            <h3 style="color: #667eea; margin-top: 0;">🎯 Navigation</h3>
        """, unsafe_allow_html=True)
        
        # Data Processing Page
        if st.button("📊 Data Processing", 
                    key="sidebar_data", 
                    use_container_width=True,
                    type="primary" if current_page == 'data_processing' else "secondary"):
            st.session_state.current_page = 'data_processing'
            st.rerun()
        
        # Visualizations Page
        if st.button("📈 Visualizations", 
                    key="sidebar_viz", 
                    use_container_width=True,
                    type="primary" if current_page == 'visualizations' else "secondary"):
            st.session_state.current_page = 'visualizations'
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Data Status Section
        st.markdown("""
        <div style="
            background: rgba(30, 30, 46, 0.9);
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.1);
        ">
            <h3 style="color: #667eea; margin-top: 0;">📊 Data Status</h3>
        """, unsafe_allow_html=True)
        
        # Show data processing status
        if st.session_state.get('processed_df') is not None:
            st.success("✅ Data Processed")
            if st.session_state.get('processing_stats'):
                stats = st.session_state.processing_stats
                st.write(f"📋 Records: {stats['final_shape'][0]:,}")
                st.write(f"🏷️ Features: {stats['final_shape'][1]}")
        else:
            st.warning("⏳ No Data Processed")
        
        # Show clustering status
        if st.session_state.get('analysis_completed'):
            st.success("✅ Clustering Complete")
            if st.session_state.get('cluster_stats'):
                clusters = st.session_state.cluster_stats['n_clusters']
                st.write(f"🎪 Clusters: {clusters}")
        else:
            st.warning("⏳ Clustering Pending")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Actions Section
        if st.session_state.get('processed_df') is not None:
            st.markdown("""
            <div style="
                background: rgba(30, 30, 46, 0.9);
                padding: 1.5rem;
                border-radius: 15px;
                margin-bottom: 1.5rem;
                border: 1px solid rgba(255,255,255,0.1);
            ">
                <h3 style="color: #667eea; margin-top: 0;">⚡ Quick Actions</h3>
            """, unsafe_allow_html=True)
            
            if st.button("🔄 Reset All Data", use_container_width=True, type="secondary"):
                # Clear all session state
                keys_to_clear = [
                    'processed_df', 'processing_stats', 'original_df',
                    'tuning_results', 'clustered_df', 'cluster_stats', 'scaled_features',
                    'analysis_completed', 'viz_min_cluster_size', 'viz_min_samples',
                    'embedding', 'scaler', 'features_for_clustering', 'selected_min_sizes',
                    'selected_min_samples'
                ]
                
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.success("✅ All data cleared!")
                st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Info Section
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
        ">
            <h3 style="margin-top: 0;">📊 Features</h3>
            <ul style="margin: 0; padding-left: 1rem;">
                <li>HDBSCAN Clustering</li>
                <li>UMAP Dimensionality Reduction</li>
                <li>Interactive Maps & Charts</li>
                <li>Crime Pattern Analysis</li>
                <li>Parameter Tuning Interface</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: rgba(255,255,255,0.6);">
            <small>🔬 Advanced Crime Analytics<br>
            Powered by HDBSCAN & UMAP</small>
        </div>
        """, unsafe_allow_html=True)