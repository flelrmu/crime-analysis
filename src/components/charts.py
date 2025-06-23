from plotly import express as px
import pandas as pd
import streamlit as st

def create_crime_scene_chart(df):
    """Create a bar chart for crime scene distribution."""
    crime_scene_counts = df['crime_scene'].value_counts().reset_index()
    crime_scene_counts.columns = ['Crime Scene', 'Count']
    
    fig = px.bar(
        crime_scene_counts,
        x='Crime Scene',
        y='Count',
        title='Distribution of Crime Scenes',
        color='Count',
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={'Count': 'Number of Incidents'},
        template='plotly_dark'
    )
    
    return fig

def create_crime_category_chart(df):
    """Create a pie chart for crime category distribution."""
    crime_category_counts = df['crime_category'].value_counts().reset_index()
    crime_category_counts.columns = ['Crime Category', 'Count']
    
    fig = px.pie(
        crime_category_counts,
        values='Count',
        names='Crime Category',
        title='Distribution of Crime Categories',
        color='Crime Category',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    return fig

def create_location_category_chart(df):
    """Create a sunburst chart for location category distribution."""
    location_category_counts = df['location_category'].value_counts().reset_index()
    location_category_counts.columns = ['Location Category', 'Count']
    
    fig = px.sunburst(
        location_category_counts,
        path=['Location Category'],
        values='Count',
        title='Location Category Distribution',
        color='Count',
        color_continuous_scale=px.colors.sequential.Inferno
    )
    
    return fig

def display_charts(df):
    """Display all charts on the Streamlit app."""
    st.markdown("### 📊 Crime Scene Distribution")
    crime_scene_fig = create_crime_scene_chart(df)
    st.plotly_chart(crime_scene_fig, use_container_width=True)

    st.markdown("### 📊 Crime Category Distribution")
    crime_category_fig = create_crime_category_chart(df)
    st.plotly_chart(crime_category_fig, use_container_width=True)

    st.markdown("### 📊 Location Category Distribution")
    location_category_fig = create_location_category_chart(df)
    st.plotly_chart(location_category_fig, use_container_width=True)