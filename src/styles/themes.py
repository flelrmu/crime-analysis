import streamlit as st

# filepath: crime-analysis-app/src/styles/themes.py
class Theme:
    def __init__(self, name, primary_color, secondary_color, background_color, text_color):
        self.name = name
        self.primary_color = primary_color
        self.secondary_color = secondary_color
        self.background_color = background_color
        self.text_color = text_color

# Define a modern theme
modern_theme = Theme(
    name="Modern",
    primary_color="#1f77b4",  # Blue
    secondary_color="#ff7f0e",  # Orange
    background_color="#f0f2f6",  # Light gray
    text_color="#333333"  # Dark gray
)

# Define a dark theme
dark_theme = Theme(
    name="Dark",
    primary_color="#ffffff",  # White
    secondary_color="#ff7f0e",  # Orange
    background_color="#1e1e1e",  # Dark gray
    text_color="#ffffff"  # White
)

# Define a light theme
light_theme = Theme(
    name="Light",
    primary_color="#007bff",  # Bootstrap Blue
    secondary_color="#6c757d",  # Bootstrap Gray
    background_color="#ffffff",  # White
    text_color="#212529"  # Dark gray
)

# List of available themes
themes = {
    "modern": modern_theme,
    "dark": dark_theme,
    "light": light_theme
}

def apply_modern_theme():
    """Apply modern dark theme with gradients and animations"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Root variables */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --dark-bg: #0e1117;
        --card-bg: #1e1e2e;
        --text-primary: #ffffff;
        --text-secondary: #a0a0a0;
        --accent-color: #ff6b6b;
        --border-radius: 15px;
        --shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        --glow: 0 0 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Global styles */
    .stApp {
        background: var(--dark-bg);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Main title container */
    .main-title-container {
        text-align: center;
        padding: 2rem 0;
        background: var(--primary-gradient);
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        box-shadow: var(--shadow);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        animation: float 3s ease-in-out infinite;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    /* Navigation container */
    .nav-container {
        margin-bottom: 2rem;
    }
    
    /* Custom buttons */
    .stButton > button {
        background: var(--primary-gradient);
        color: white;
        border: none;
        border-radius: var(--border-radius);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease;
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Cards */
    .metric-card {
        background: var(--card-bg);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--glow);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--primary-gradient);
    }
    
    /* Success box */
    .success-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
        color: white;
        animation: slideInFromLeft 0.5s ease;
    }
    
    /* Prediction box */
    .prediction-box {
        background: var(--success-gradient);
        border-radius: var(--border-radius);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
        color: white;
        animation: pulse 2s infinite;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--card-bg);
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: var(--card-bg);
        border-radius: var(--border-radius);
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: var(--shadow);
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: var(--glow);
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: var(--card-bg);
        border-radius: var(--border-radius);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #667eea;
        box-shadow: var(--glow);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: var(--card-bg);
        border-radius: var(--border-radius);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: var(--primary-gradient);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--card-bg);
        border-radius: var(--border-radius);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Info box */
    .stInfo {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: var(--border-radius);
        border: none;
        color: white;
    }
    
    /* Warning box */
    .stWarning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: var(--border-radius);
        border: none;
        color: white;
    }
    
    /* Error box */
    .stError {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        border-radius: var(--border-radius);
        border: none;
        color: #8B0000;
    }
    
    /* Animations */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    @keyframes glow {
        from { box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3); }
        to { box-shadow: 0 8px 32px rgba(118, 75, 162, 0.3); }
    }
    
    @keyframes slideInFromLeft {
        0% { transform: translateX(-100%); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Plotly charts */
    .js-plotly-plot {
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: var(--shadow);
    }
    
    /* DataFrame */
    .stDataFrame {
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: var(--shadow);
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--dark-bg);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    </style>
    """, unsafe_allow_html=True)