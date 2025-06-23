# filepath: crime-analysis-app/config/settings.py
class Config:
    """Configuration settings for the Crime Analysis App."""
    
    # General settings
    APP_NAME = "Crime Analysis Dashboard"
    APP_VERSION = "1.0.0"
    
    # Streamlit settings
    STREAMLIT_THEME = {
        "primaryColor": "#1f77b4",
        "backgroundColor": "#ffffff",
        "secondaryBackgroundColor": "#f0f2f6",
        "textColor": "#333333",
        "font": "sans serif"
    }
    
    # Database settings (if applicable)
    DATABASE_URI = "sqlite:///crime_data.db"  # Example for SQLite, change as needed
    
    # API keys (if applicable)
    GOOGLE_MAPS_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"
    
    # File upload settings
    MAX_UPLOAD_SIZE_MB = 10  # Maximum file upload size in MB
    
    # Logging settings
    LOGGING_LEVEL = "INFO"
    
    # Other settings can be added as needed
    @staticmethod
    def init_app(app):
        """Initialize the app with the configuration settings."""
        pass  # Add any initialization logic if necessary