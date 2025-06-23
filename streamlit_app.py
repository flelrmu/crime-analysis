"""
Main entry point for Streamlit deployment.
This file should be in the root directory for proper deployment.
"""

import sys
import os

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Import and run the main application
from src.app import main

if __name__ == "__main__":
    main()