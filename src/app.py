import streamlit as st
import sys
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='umap')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from views.data_processing import show_data_processing_page
from views.visualizations import show_visualization_page
from styles.themes import apply_modern_theme
from components.sidebar import show_sidebar

def main():
    st.set_page_config(
        page_title="Crime Analysis Dashboard",
        page_icon="🚔",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply modern theme
    apply_modern_theme()
    
    # Show sidebar (dengan navigation di dalamnya)
    show_sidebar()
    
    # Get current page from session state
    current_page = st.session_state.get('current_page', 'data_processing')
    
    # Page routing berdasarkan selection di sidebar (tanpa predictions)
    if current_page == 'data_processing':
        show_data_processing_page()
    elif current_page == 'visualizations':
        show_visualization_page()

if __name__ == "__main__":
    main()