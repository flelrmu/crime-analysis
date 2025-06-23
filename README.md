# Crime Analysis App

## Overview
The Crime Analysis App is a web application built using Streamlit that allows users to analyze crime data through data processing, visualizations, and predictions. The app provides an interactive interface for users to upload datasets, perform clustering analysis, and predict crime categories based on various parameters.

## Features
- **Data Processing**: Upload and clean crime data, handle missing values, and perform clustering analysis.
- **Visualizations**: Generate various charts and maps to visualize crime data and clustering results.
- **Predictions**: Input parameters to predict the most likely crime category using a trained machine learning model.

## Project Structure
```
crime-analysis-app
├── src
│   ├── app.py                  # Main entry point of the application
│   ├── components               # Contains reusable components
│   │   ├── sidebar.py           # Sidebar navigation and user input
│   │   ├── metrics.py           # Functions to display metrics
│   │   └── charts.py            # Functions to create charts
│   ├── pages                    # Different pages of the app
│   │   ├── data_processing.py    # Data processing page
│   │   ├── visualizations.py      # Visualizations page
│   │   └── predictions.py         # Predictions page
│   ├── utils                    # Utility functions
│   │   ├── data_processing.py     # Data processing utilities
│   │   ├── clustering.py          # Clustering functions
│   │   ├── prediction.py          # Prediction functions
│   │   └── mapping.py             # Mapping functions
│   └── styles                   # CSS styles
│       ├── main.css              # Main styles
│       ├── components.css         # Component-specific styles
│       └── themes.py             # Theme definitions
├── assets                       # Static assets
│   ├── icons                    # Icon files
│   │   ├── crime.svg            # Crime icon
│   │   ├── cluster.svg          # Clustering icon
│   │   └── prediction.svg       # Prediction icon
│   └── images                   # Image files
│       └── logo.png             # Application logo
├── config                       # Configuration files
│   ├── settings.py              # Application settings
├── requirements.txt             # Project dependencies
├── .streamlit                   # Streamlit configuration
│   └── config.toml              # Streamlit settings
└── README.md                    # Project documentation
```

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/crime-analysis-app.git
   cd crime-analysis-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run src/app.py
   ```

## Usage
- Navigate to the **Data Processing** page to upload your crime dataset and perform data cleaning and clustering.
- Go to the **Visualizations** page to view various charts and maps based on the processed data.
- Use the **Predictions** page to input parameters and predict the most likely crime category.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.