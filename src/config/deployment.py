import warnings
import os

def configure_for_deployment():
    """Configure application for deployment environment."""
    
    # Suppress all warnings for clean deployment
    warnings.filterwarnings('ignore')
    
    # Set environment variables for better performance
    os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT for faster startup
    os.environ['UMAP_DISABLE_PROGRESS_BAR'] = '1'  # Disable UMAP progress bars
    
    # Memory optimization
    import gc
    gc.set_threshold(700, 10, 10)  # More aggressive garbage collection
    
    return True