from streamlit import metrics, markdown

def display_metrics(total_records, total_columns, missing_values, duplicates_removed):
    """Display various metrics related to crime data."""
    markdown("<h2 style='color: #1f77b4;'>📊 Data Metrics</h2>", unsafe_allow_html=True)

    col1, col2, col3, col4 = metrics.columns(4)

    with col1:
        metrics.metric("Total Records", f"{total_records:,}", delta=None)

    with col2:
        metrics.metric("Total Columns", total_columns, delta=None)

    with col3:
        metrics.metric("Missing Values", f"{missing_values:,}", delta=None)

    with col4:
        metrics.metric("Duplicates Removed", f"{duplicates_removed:,}", delta=None)

def display_model_metrics(accuracy, features_used, total_categories):
    """Display metrics related to the prediction model."""
    markdown("<h2 style='color: #1f77b4;'>🔮 Model Performance Metrics</h2>", unsafe_allow_html=True)

    col1, col2, col3 = metrics.columns(3)

    with col1:
        metrics.metric("Model Accuracy", f"{accuracy:.2%}", delta=None)

    with col2:
        metrics.metric("Features Used", features_used, delta=None)

    with col3:
        metrics.metric("Total Crime Categories", total_categories, delta=None)