import streamlit as st
import sys
import os

# Early warning suppression for deployment
import warnings
warnings.filterwarnings('ignore')

# Configure for deployment
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config.deployment import configure_for_deployment
    configure_for_deployment()
except ImportError:
    # Fallback configuration
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

# Suppress specific HDBSCAN warnings
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import hdbscan

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from views.data_processing import show_data_processing_page
from views.visualizations import show_visualization_page
from styles.themes import apply_modern_theme
from components.sidebar import show_sidebar

def main():
    """Main application function with improved error handling."""
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="Crime Analysis Dashboard",
        page_icon="🚔",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply modern theme
    try:
        apply_modern_theme()
    except Exception as e:
        st.warning(f"Theme loading issue: {str(e)}")
    
    # Show sidebar with error handling
    try:
        show_sidebar()
    except Exception as e:
        st.error(f"Sidebar error: {str(e)}")
        st.stop()
    
    # Get current page from session state
    current_page = st.session_state.get('current_page', 'data_processing')
    
    # Page routing with error handling
    try:
        if current_page == 'data_processing':
            show_data_processing_page()
        elif current_page == 'visualizations':
            show_visualization_page()
        else:
            st.error("❌ Invalid page selection")
            st.session_state.current_page = 'data_processing'
            st.rerun()
    except Exception as e:
        st.error(f"Page loading error: {str(e)}")
        st.write("**Debug Info:**")
        st.write(f"Current page: {current_page}")
        st.write(f"Session state keys: {list(st.session_state.keys())}")

if __name__ == "__main__":
    main()