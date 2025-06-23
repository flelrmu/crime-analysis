import streamlit as st
import pandas as pd
import numpy as np
import hdbscan
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import calendar
import warnings

# Import UMAP dengan handling warning
try:
    import umap.umap_ as umap
    warnings.filterwarnings('ignore', category=UserWarning, module='umap')
except ImportError:
    try:
        import umap
        warnings.filterwarnings('ignore', category=UserWarning, module='umap')
    except ImportError:
        st.error("❌ UMAP package not installed. Please run: pip install umap-learn")
        st.stop()

def perform_clustering(df, min_cluster_size=200, min_samples=15):
    """Perform HDBSCAN clustering based on notebook implementation."""
    
    with st.spinner("🔍 Performing HDBSCAN clustering..."):
        # Select features for clustering berdasarkan notebook dan dataset Chicago Crime
        required_features = ['primary_type_encoded', 'weekday', 'hour', 'crime_scene',
                           'location_category_encoded', 'latitude', 'longitude']
        available_features = [col for col in required_features if col in df.columns]
        
        if len(available_features) < 6:
            st.error(f"Missing required features: {set(required_features) - set(available_features)}")
            return None, None, None, None
        
        # Prepare features
        features = df[available_features].copy()
        
        # Handle any remaining NaN values
        features = features.dropna()
        
        if len(features) < min_cluster_size:
            st.error(f"Not enough data points ({len(features)}) for clustering with min_cluster_size={min_cluster_size}")
            return None, None, None, None
        
        # Scale features dengan force_all_finite=True untuk menghindari deprecation warning
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # UMAP dimensionality reduction dengan parameter yang dioptimalkan
        try:
            # Suppress warnings untuk UMAP
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reducer = umap.UMAP(
                    n_neighbors=min(30, len(features) - 1),  # Adjust based on data size
                    min_dist=0.1, 
                    n_components=2, 
                    random_state=42,
                    n_jobs=1  # Fix untuk warning n_jobs
                )
                embedding = reducer.fit_transform(scaled_features)
        except Exception as e:
            st.error(f"❌ UMAP error: {str(e)}")
            return None, None, None, None
        
        # HDBSCAN clustering on embedding
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                prediction_data=True
            )
            labels = clusterer.fit_predict(embedding)
        except Exception as e:
            st.error(f"❌ HDBSCAN error: {str(e)}")
            return None, None, None, None
        
        # Filter original dataframe to match processed features
        df_filtered = df.loc[features.index].copy()
        
        # Add results to filtered DataFrame
        df_clustered = df_filtered.copy()
        df_clustered['cluster'] = labels
        df_clustered['umap_x'] = embedding[:, 0]
        df_clustered['umap_y'] = embedding[:, 1]
        
        # Calculate cluster statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = sum(labels == -1)
        
        # Handle cluster persistence safely
        try:
            persistence = clusterer.cluster_persistence_
        except AttributeError:
            persistence = []
        
        cluster_stats = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'persistence': persistence,
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'total_points': len(labels)
        }
        
        st.success(f"✅ Clustering completed: {n_clusters} clusters found, {n_noise} noise points")
        
        return df_clustered, scaled_features, labels, cluster_stats

def perform_clustering_with_params(df, min_cluster_size, min_samples):
    """Perform final clustering EXACT seperti notebook cell ke-11."""
    
    # Convert parameters to integers
    try:
        min_cluster_size = int(min_cluster_size)
        min_samples = int(min_samples)
    except (ValueError, TypeError):
        st.error("❌ Min cluster size and min samples must be integers!")
        return None, None, None, None
    
    # Validasi parameter
    if min_cluster_size <= 0 or min_samples <= 0:
        st.error("❌ Parameters must be positive integers!")
        return None, None, None, None
    
    if st.session_state.get('embedding') is None:
        st.error("❌ Please run tuning first!")
        return None, None, None, None
    
    with st.spinner(f"🔍 Clustering with min_size={min_cluster_size}, min_samples={min_samples}..."):
        # Use stored embedding dari tuning
        embedding = st.session_state.embedding
        features = st.session_state.features_for_clustering
        
        try:
            # EXACT parameters seperti notebook cell ke-11
            cluster = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',  # EXACT seperti notebook
                prediction_data=True,
                cluster_selection_epsilon=0.0  # Ensure default
            )
            labels = cluster.fit_predict(embedding)
            
            # EXACT seperti notebook: persistence = cluster.cluster_persistence_
            try:
                persistence = cluster.cluster_persistence_
            except AttributeError:
                persistence = []
            
            # EXACT calculation seperti notebook
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            
            # Debug output untuk target combination (2000, 50)
            if min_cluster_size == 2000 and min_samples == 50:
                st.write(f"🎯 **TARGET RESULT** - min_size={min_cluster_size}, min_samples={min_samples}")
                st.write(f"   Clusters: {n_clusters}, Noise: {n_noise}")
                st.write(f"   Expected from notebook: 17 clusters, 67 noise points")
                
                if n_clusters == 17 and n_noise == 67:
                    st.success("✅ PERFECT MATCH dengan notebook!")
                else:
                    st.warning(f"❌ Result mismatch: got {n_clusters} clusters, {n_noise} noise vs expected 17 clusters, 67 noise")
            
            # Print labels dan noise seperti notebook cell ke-11
            st.write(f"**Labels summary:** Total points: {len(labels)}, Unique labels: {len(set(labels))}")
            st.write(f"**Noise:** {n_noise} points")
            
            # Show persistence scores seperti notebook
            if len(persistence) > 0:
                st.write("**Cluster Persistence Scores:**")
                for i, score in enumerate(persistence):
                    st.write(f"Cluster {i}: Persistence = {score:.2f}")
            
            # EXACT seperti notebook cell ke-12: df_clustered = df.copy()
            df_filtered = df.loc[features.index].copy()
            df_clustered = df_filtered.copy()
            df_clustered['cluster'] = labels
            df_clustered['umap_x'] = embedding[:, 0]
            df_clustered['umap_y'] = embedding[:, 1]
            
            # EXACT seperti notebook cell ke-13: Add day_time and name_days
            import calendar
            
            # Add day_time seperti notebook
            if 'hour' in df_clustered.columns:
                df_clustered['day_time'] = df_clustered['hour'].apply(get_day_time_label)
            
            # Add name_days seperti notebook
            if 'weekday' in df_clustered.columns:
                df_clustered['name_days'] = df_clustered['weekday'].map(dict(enumerate(calendar.day_name)))
            
            cluster_stats = {
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'persistence': persistence,
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'total_points': len(labels)
            }
            
            st.success(f"✅ Clustering completed: {n_clusters} clusters found, {n_noise} noise points")
            
            # Return scaled features menggunakan stored scaler
            return df_clustered, st.session_state.scaler.transform(features), labels, cluster_stats
            
        except Exception as e:
            st.error(f"❌ Clustering error: {str(e)}")
            return None, None, None, None

def get_day_time_label(hour):
    """Get day time label EXACT seperti notebook cell ke-13."""
    if pd.isna(hour):
        return 'night'
    if 1 <= hour < 5:
        return 'early morning'
    elif 5 <= hour < 11:
        return 'morning'
    elif 11 <= hour < 15:
        return 'noon'
    elif 15 <= hour < 18:
        return 'afternoon'
    else:
        return 'night'

def generate_cluster_characteristics(clustered_df):
    """Generate cluster characteristics summary based on notebook."""
    # Remove noise points
    df_profile = clustered_df[clustered_df['cluster'] != -1].copy()
    
    # Calculate cluster counts
    cluster_counts = df_profile['cluster'].value_counts().rename_axis('cluster').reset_index(name='count')
    
    # Calculate averages for numeric columns
    numeric_summary = df_profile.groupby('cluster')[['hour', 'latitude', 'longitude']].mean().reset_index()
    
    # Calculate mode for categorical columns
    def mode(series):
        return series.mode().iloc[0] if not series.mode().empty else None
    
    categorical_columns = ['primary_type', 'day_time', 'name_days', 'location_category']
    available_cat_cols = [col for col in categorical_columns if col in df_profile.columns]
    
    if available_cat_cols:
        categorical_summary = df_profile.groupby('cluster')[available_cat_cols].agg(mode).reset_index()
        
        # Merge all summaries
        cluster_profile = cluster_counts.merge(numeric_summary, on='cluster')
        cluster_profile = cluster_profile.merge(categorical_summary, on='cluster')
    else:
        cluster_profile = cluster_counts.merge(numeric_summary, on='cluster')
    
    return cluster_profile

def create_cluster_visualizations(df):
    """Create geographic cluster visualization."""
    # Filter out noise points
    df_clean = df[df['cluster'] != -1]
    
    # Geographic scatter plot
    hover_data = ['crime_scene']
    if 'day_time' in df.columns:
        hover_data.append('day_time')
    if 'location_category' in df.columns:
        hover_data.append('location_category')
    
    fig_geo = px.scatter_mapbox(
        df_clean,
        lat='latitude',
        lon='longitude',
        color='cluster',
        hover_data=hover_data,
        mapbox_style='carto-positron',
        title='Crime Clusters by Geographic Location',
        height=600,
        color_continuous_scale='viridis'
    )
    
    # Center map on data
    fig_geo.update_layout(
        mapbox=dict(
            center=dict(
                lat=df['latitude'].mean(),
                lon=df['longitude'].mean()
            ),
            zoom=10
        )
    )
    
    return fig_geo

def create_cluster_distribution_chart(clustered_df):
    """Create cluster distribution bar chart based on notebook."""
    cluster_counts = clustered_df['cluster'].value_counts().sort_index()
    
    fig = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        title="Crime Clusters Distribution",
        labels={'x': 'Cluster ID', 'y': 'Number of Crime Incidents'},
        color=cluster_counts.values,
        color_continuous_scale="viridis"
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        showlegend=False
    )
    
    return fig

def create_time_cluster_analysis(clustered_df):
    """Create time-based cluster analysis."""
    # Filter out noise
    df_clean = clustered_df[clustered_df['cluster'] != -1]
    
    # Hour vs cluster scatter plot
    cluster_profile = generate_cluster_characteristics(clustered_df)
    
    fig_time = px.scatter(
        cluster_profile,
        x='cluster',
        y='hour',
        size='count',
        color='count',
        title='⏰ Average Crime Time per Cluster',
        labels={'hour': 'Average Hour', 'cluster': 'Cluster'},
        color_continuous_scale='viridis'
    )
    
    fig_time.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    
    return fig_time

def create_location_heatmap(clustered_df):
    """Create location category heatmap."""
    df_clean = clustered_df[clustered_df['cluster'] != -1]
    
    if 'location_category' not in df_clean.columns:
        return None
    
    # Create crosstab
    location_counts = pd.crosstab(df_clean['cluster'], df_clean['location_category'])
    
    fig_heatmap = px.imshow(
        location_counts.T,
        title='🎯 Location Categories per Cluster',
        labels=dict(x="Cluster", y="Location Category", color="Count"),
        aspect="auto",
        color_continuous_scale="YlGnBu"
    )
    
    fig_heatmap.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    
    return fig_heatmap