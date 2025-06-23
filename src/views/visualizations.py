import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

@st.cache_data
def create_cluster_distribution_chart_optimized(df):
    """Create cluster distribution chart with caching for performance."""
    try:
        # Filter out noise for cleaner visualization
        df_clean = df[df['cluster'] != -1].copy()
        
        # EXACT seperti notebook: cluster_profile = df_clustered['cluster'].value_counts().reset_index()
        cluster_counts = df_clean['cluster'].value_counts().reset_index()
        cluster_counts.columns = ['cluster', 'count']
        
        # EXACT seperti notebook: pastikan tipe cluster numerik dan urutkan
        cluster_counts['cluster'] = cluster_counts['cluster'].astype(int)
        cluster_counts = cluster_counts.sort_values('cluster')
        
        # Create optimized bar chart
        fig = px.bar(
            cluster_counts,
            x='cluster',
            y='count',
            title='📊 Distribusi Jumlah Data per Cluster',
            labels={'cluster': 'Cluster', 'count': 'Jumlah Data'},
            color='count',
            color_continuous_scale='viridis',
            text='count'
        )
        
        # Optimize layout for performance
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=False,
            height=500
        )
        
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        
        return fig
    except Exception as e:
        st.error(f"Error creating cluster distribution chart: {str(e)}")
        return None

@st.cache_data
def create_crime_type_heatmap_optimized(df):
    """Create optimized crime type heatmap."""
    try:
        # Sample data if too large for performance
        if len(df) > 50000:
            df_sample = df.sample(n=50000, random_state=42)
            st.info("📊 Using sample of 50k records for performance")
        else:
            df_sample = df.copy()
        
        df_clean = df_sample[df_sample['cluster'] != -1].copy()
        
        # EXACT seperti notebook: pastikan cluster adalah integer dan urut
        df_clean['cluster'] = df_clean['cluster'].astype(int)
        
        # Create crosstab EXACT seperti notebook
        crime_type_counts = pd.crosstab(df_clean['cluster'], df_clean['primary_type'])
        
        # Urutkan index cluster seperti notebook
        crime_type_counts = crime_type_counts.sort_index()
        
        # Create optimized heatmap
        fig = px.imshow(
            crime_type_counts.T,  # Transpose seperti notebook
            title='🎯 Jenis Kejahatan per Cluster (Crime Type Heatmap)',
            labels=dict(x="Cluster", y="Primary Type", color="Jumlah Kejadian"),
            aspect="auto",
            color_continuous_scale="YlGnBu",
            text_auto=True
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=600
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating crime type heatmap: {str(e)}")
        return None

@st.cache_data
def create_geographic_scatter_optimized(df):
    """Create optimized geographic scatter plot."""
    try:
        # Sample for performance if dataset is large
        if len(df) > 20000:
            df_sample = df.sample(n=20000, random_state=42)
            st.info("📊 Showing sample of 20k points for performance")
        else:
            df_sample = df.copy()
        
        # Convert cluster to string untuk discrete colors seperti notebook
        df_plot = df_sample.copy()
        df_plot['cluster'] = df_plot['cluster'].astype(str)
        
        # Create optimized mapbox scatter plot
        fig = px.scatter_mapbox(
            df_plot,
            lat='latitude',
            lon='longitude',
            color='cluster',
            size_max=5,  # Smaller size for performance
            opacity=0.6,
            zoom=10,
            height=600,
            mapbox_style='carto-positron',
            hover_name='primary_type',
            hover_data=['day_time', 'name_days'] if 'day_time' in df_plot.columns else ['primary_type'],
            title='🗺️ Distribusi Geografis Data per Cluster'
        )
        
        fig.update_layout(
            legend_title_text='Cluster',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating geographic scatter: {str(e)}")
        return None

@st.cache_data
def create_hourly_distribution_boxplot(df):
    """Create optimized hourly distribution boxplot."""
    try:
        # Filter out noise for cleaner visualization
        df_clean = df[df['cluster'] != -1].copy()
        
        if 'hour' not in df_clean.columns:
            st.warning("⚠️ 'hour' column not found in data")
            return None
        
        # EXACT seperti notebook: pastikan cluster adalah integer dan urut
        df_clean['cluster'] = df_clean['cluster'].astype(int)
        
        # Sample data if too large for performance
        if len(df_clean) > 30000:
            df_clean = df_clean.sample(n=30000, random_state=42)
            st.info("📊 Using sample of 30k records for performance")
        
        # Create boxplot EXACT seperti notebook
        fig = px.box(
            df_clean,
            x='cluster',
            y='hour',
            title='⏰ Distribusi Jam Kejadian per Cluster',
            labels={'cluster': 'Cluster', 'hour': 'Jam Kejadian'},
            color='cluster',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        # Sort x-axis secara numerik seperti notebook
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=False,
            height=500,
            xaxis={'type': 'category', 'categoryorder': 'category ascending'}
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating hourly distribution boxplot: {str(e)}")
        return None

@st.cache_data
def create_location_category_cluster_heatmap(df):
    """Create heatmap untuk location category per cluster seperti notebook."""
    try:
        # Sample data if too large
        if len(df) > 50000:
            df_sample = df.sample(n=50000, random_state=42)
            st.info("📊 Using sample of 50k records for performance")
        else:
            df_sample = df.copy()
        
        df_clean = df_sample[df_sample['cluster'] != -1].copy()
        
        if 'location_category' not in df_clean.columns:
            st.warning("⚠️ 'location_category' column not found")
            return None
        
        # EXACT seperti notebook
        df_clean['cluster'] = df_clean['cluster'].astype(int)
        
        # Create crosstab untuk location category per cluster
        location_counts = pd.crosstab(df_clean['cluster'], df_clean['location_category'])
        location_counts = location_counts.sort_index()
        
        # Create heatmap dengan Plotly
        fig = px.imshow(
            location_counts.T,
            title='🏢 Kategori Lokasi per Cluster',
            labels=dict(x="Cluster", y="Location Category", color="Jumlah"),
            aspect="auto",
            color_continuous_scale="Blues",
            text_auto=True
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=600
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating location category heatmap: {str(e)}")
        return None

@st.cache_data
def create_day_cluster_heatmap(df):
    """Create heatmap hari per cluster seperti notebook."""
    try:
        df_clean = df[df['cluster'] != -1].copy()
        
        if 'name_days' not in df_clean.columns:
            st.warning("⚠️ 'name_days' column not found")
            return None
        
        # EXACT seperti notebook
        df_clean['cluster'] = df_clean['cluster'].astype(int)
        
        # Create pivot table seperti notebook
        pivot = pd.crosstab(df_clean['name_days'], df_clean['cluster'])
        
        # Urutkan hari seperti notebook
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = pivot.reindex(day_order)
        
        # Urutkan cluster
        pivot = pivot[sorted(pivot.columns)]
        
        # Create heatmap dengan Plotly
        fig = px.imshow(
            pivot,
            title='📅 Heatmap Jumlah Kejadian per Hari per Cluster',
            labels=dict(x="Cluster", y="Hari", color="Jumlah Kejadian"),
            aspect="auto",
            color_continuous_scale="YlGnBu",
            text_auto=True
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating day-cluster heatmap: {str(e)}")
        return None


@st.cache_data
def create_clustering_evaluation_metrics(df, embedding, labels):
    """Create clustering evaluation metrics seperti notebook cell ke-26."""
    try:
        if embedding is None or labels is None:
            st.warning("⚠️ Embedding or labels not available")
            return None
        
        # EXACT seperti notebook
        def interpret_metric(metric_name, value):
            if metric_name == 'Silhouette Score':
                if value >= 0.7:
                    return 'Sangat Bagus'
                elif value >= 0.5:
                    return 'Bagus'
                elif value >= 0.3:
                    return 'Cukup'
                else:
                    return 'Kurang'
            elif metric_name == 'Davies-Bouldin Index':
                if value < 0.3:
                    return 'Sangat Bagus'
                elif value < 0.6:
                    return 'Bagus'
                elif value < 1.0:
                    return 'Cukup'
                else:
                    return 'Kurang'
            elif metric_name == 'Calinski-Harabasz Index':
                if value > 50000:
                    return 'Sangat Bagus'
                elif value > 10000:
                    return 'Bagus'
                elif value > 2000:
                    return 'Cukup'
                else:
                    return 'Kurang'
            else:
                return '-'
        
        # EXACT function seperti notebook
        def evaluate_clustering(X, labels):
            mask = labels != -1
            X_filtered = X[mask]
            labels_filtered = labels[mask]
            
            metrics = {
                'Silhouette Score': np.nan,
                'Davies-Bouldin Index': np.nan,
                'Calinski-Harabasz Index': np.nan
            }
            
            if len(np.unique(labels_filtered)) > 1:
                metrics['Silhouette Score'] = silhouette_score(X_filtered, labels_filtered)
                metrics['Davies-Bouldin Index'] = davies_bouldin_score(X_filtered, labels_filtered)
                metrics['Calinski-Harabasz Index'] = calinski_harabasz_score(X_filtered, labels_filtered)
            
            # Create DataFrame
            df_eval = pd.DataFrame.from_dict(metrics, orient='index', columns=['Nilai'])
            df_eval['Keterangan'] = df_eval.apply(lambda row: interpret_metric(row.name, row['Nilai']), axis=1)
            return df_eval
        
        hasil_evaluasi = evaluate_clustering(embedding, labels)
        return hasil_evaluasi
        
    except Exception as e:
        st.error(f"Error creating evaluation metrics: {str(e)}")
        return None

@st.cache_data
def create_silhouette_analysis_chart(embedding, labels):
    """Create silhouette analysis chart seperti notebook cell ke-27."""
    try:
        if embedding is None or labels is None:
            st.warning("⚠️ Embedding or labels not available")
            return None
        
        # EXACT seperti notebook
        sample_silhouette_values = silhouette_samples(embedding, labels)
        
        # Create DataFrame
        df_silhouette = pd.DataFrame({
            'cluster': labels,
            'silhouette': sample_silhouette_values
        })
        
        # Hapus noise (-1)
        df_silhouette = df_silhouette[df_silhouette['cluster'] != -1]
        
        # Hitung rata-rata silhouette score per cluster
        cluster_silhouette_scores = df_silhouette.groupby('cluster')['silhouette'].mean().reset_index()
        cluster_silhouette_scores = cluster_silhouette_scores.sort_values(by='cluster')
        
        # Create line chart dengan Plotly
        fig = px.line(
            cluster_silhouette_scores,
            x='cluster',
            y='silhouette',
            title='📊 Silhouette Score per Cluster',
            labels={'cluster': 'Cluster', 'silhouette': 'Rata-rata Silhouette Score'},
            markers=True,
            line_shape='linear'
        )
        
        # Add text annotations
        fig.update_traces(
            mode='lines+markers+text',
            text=[f"{val:.2f}" for val in cluster_silhouette_scores['silhouette']],
            textposition='top center'
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=500,
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating silhouette analysis: {str(e)}")
        return None

def show_visualization_page():
    """Comprehensive Visualizations Page dengan semua chart dari notebook."""
    st.markdown("""
    <div class="main-title-container">
        <h1 class="main-title">📊 Crime Data Visualizations</h1>
        <p class="main-subtitle">Explore clustering results and patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if clustering results are available
    if not st.session_state.get('clustered_df') is not None:
        st.warning("⚠️ No clustering results found. Please run clustering analysis first.")
        return
    
    df = st.session_state.clustered_df
    embedding = st.session_state.get('embedding')
    labels = st.session_state.get('final_labels')
    
    # Performance info
    st.info(f"📊 **Dataset**: {len(df):,} records | **Clusters**: {df['cluster'].nunique()-1} | **Features**: {len(df.columns)}")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_clusters = len(df[df['cluster'] != -1]['cluster'].unique())
        st.metric("Valid Clusters", n_clusters)
    with col2:
        n_noise = len(df[df['cluster'] == -1])
        st.metric("Noise Points", n_noise)
    with col3:
        noise_pct = (n_noise / len(df)) * 100
        st.metric("Noise %", f"{noise_pct:.1f}%")
    with col4:
        largest_cluster = df[df['cluster'] != -1]['cluster'].value_counts().iloc[0] if n_clusters > 0 else 0
        st.metric("Largest Cluster", largest_cluster)
    
    # Comprehensive tabs dengan semua visualisasi dari notebook
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distribution", 
        "🎯 Crime Analysis", 
        "⏰ Temporal", 
        "🗺️ Geographic"
    ])
    
    with tab1:
        st.markdown("### 📊 Cluster Distribution")
        with st.spinner("Creating distribution chart..."):
            fig_dist = create_cluster_distribution_chart_optimized(df)
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)
        
        # NEW: Location category distribution
        st.markdown("### 🏢 Location Category Distribution")
        with st.spinner("Creating location category heatmap..."):
            fig_location = create_location_category_cluster_heatmap(df)
            if fig_location:
                st.plotly_chart(fig_location, use_container_width=True)
    
    with tab2:
        st.markdown("### 🎯 Crime Type Analysis")
        with st.spinner("Creating crime type heatmap..."):
            fig_crime = create_crime_type_heatmap_optimized(df)
            if fig_crime:
                st.plotly_chart(fig_crime, use_container_width=True)
    
    with tab3:
        st.markdown("### ⏰ Temporal Analysis")
        
        # Hourly distribution
        if 'hour' in df.columns:
            with st.spinner("Creating hourly distribution..."):
                fig_hour = create_hourly_distribution_boxplot(df)
                if fig_hour:
                    st.plotly_chart(fig_hour, use_container_width=True)
        
        # NEW: Day-cluster heatmap
        st.markdown("### 📅 Daily Patterns")
        if 'name_days' in df.columns:
            with st.spinner("Creating day-cluster heatmap..."):
                fig_day = create_day_cluster_heatmap(df)
                if fig_day:
                    st.plotly_chart(fig_day, use_container_width=True)
    
    with tab4:
        st.markdown("### 🗺️ Geographic Distribution")
        if 'latitude' in df.columns and 'longitude' in df.columns:
            with st.spinner("Creating geographic map..."):
                fig_geo = create_geographic_scatter_optimized(df)
                if fig_geo:
                    st.plotly_chart(fig_geo, use_container_width=True)
        else:
            st.warning("⚠️ Geographic data not available for mapping")
