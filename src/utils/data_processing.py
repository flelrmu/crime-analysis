import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def handle_missing_values(df):
    """Handle missing values sesuai requirement."""
    missing_before = df.isnull().sum().sum()
    df = df.dropna(axis=0)
    missing_after = df.isnull().sum().sum()
    return df, missing_before, missing_after

def handle_duplicates(df):
    """Count duplicates tapi TIDAK remove."""
    duplicates_before = df.duplicated().sum()
    duplicates_after = df.duplicated().sum()
    return df, duplicates_before, duplicates_after

def process_date_column(df):
    """Process date column untuk feature engineering."""
    if 'date' in df.columns:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['hour'] = df['date'].dt.hour
        df['weekday'] = df['date'].dt.dayofweek
        
        df = df.drop(['date'], axis=1)
    
    return df

def crime_scene_mapping(hour):
    """Crime scene mapping berdasarkan jam."""
    if pd.isna(hour):
        return 4
    if 1 <= hour < 5:
        return 0  # early morning
    elif 5 <= hour < 11:
        return 1  # morning
    elif 11 <= hour < 15:
        return 2  # noon
    elif 15 <= hour < 18:
        return 3  # afternoon
    else:
        return 4  # night

def create_location_map_with_data(df):
    """Create location map dengan kategori lokasi."""
    location_map = {
        "Residential": [
            "APARTMENT", "RESIDENCE", "HOUSE", "RESIDENCE-GARAGE",
            "RESIDENTIAL YARD (FRONT/BACK)",
            "RESIDENCE PORCH/HALLWAY", "RESIDENCE - GARAGE",
            "RESIDENCE - PORCH / HALLWAY", "RESIDENCE - YARD (FRONT / BACK)",
            "PORCH", "YARD", "VESTIBULE"
        ],
        "Commercial": [
            "DRUG STORE", "BAR OR TAVERN", "GROCERY FOOD STORE", "BANK",
            "CONVENIENCE STORE", "SMALL RETAIL STORE", "RESTAURANT",
            "DEPARTMENT STORE", "HOTEL / MOTEL", "MOTEL",
            "TAVERN/LIQUOR STORE", "TAVERN / LIQUOR STORE", "GAS STATION",
            "GAS STATION DRIVE/PROP.", "CAR WASH", "APPLIANCE STORE",
            "AUTO / BOAT / RV DEALERSHIP", "PAWN SHOP", "BARBERSHOP",
            "BARBER SHOP/BEAUTY SALON", "NEWSSTAND", "CLEANING STORE",
            "CURRENCY EXCHANGE", "COMMERCIAL / BUSINESS OFFICE", "HOTEL/MOTEL"
        ],
        "Transportation": [
            "STREET", "SIDEWALK", "PARKING LOT/GARAGE(NON.RESID.)",
            "PARKING LOT / GARAGE (NON RESIDENTIAL)", "CTA PLATFORM",
            "CTA BUS", "CTA TRAIN", "CTA BUS STOP", "CTA STATION",
            "CTA GARAGE / OTHER PROPERTY",
            "CTA PARKING LOT / GARAGE / OTHER PROPERTY",
            "CTA TRACKS - RIGHT OF WAY", "CTA PROPERTY",
            "VEHICLE NON-COMMERCIAL",
            "VEHICLE - OTHER RIDE SHARE SERVICE (E.G., UBER, LYFT)",
            "VEHICLE - OTHER RIDE SHARE SERVICE (LYFT, UBER, ETC.)",
            "VEHICLE - OTHER RIDE SERVICE", "VEHICLE-COMMERCIAL",
            "VEHICLE - COMMERCIAL", "TAXICAB", "DELIVERY TRUCK",
            "VEHICLE - DELIVERY TRUCK", "VEHICLE - COMMERCIAL: TROLLEY BUS",
            "DRIVEWAY - RESIDENTIAL", "OTHER COMMERCIAL TRANSPORTATION",
            "ALLEY", "PARKING LOT", "BRIDGE",
            "HIGHWAY/EXPRESSWAY", "HIGHWAY / EXPRESSWAY",
            'CTA "L" TRAIN', "GARAGE"
        ],
        "Educational": [
            "SCHOOL, PRIVATE, BUILDING", "SCHOOL, PRIVATE, GROUNDS",
            "SCHOOL - PRIVATE GROUNDS", "SCHOOL - PRIVATE BUILDING",
            "SCHOOL, PUBLIC, BUILDING", "SCHOOL, PUBLIC, GROUNDS",
            "SCHOOL - PUBLIC GROUNDS", "SCHOOL - PUBLIC BUILDING",
            "COLLEGE/UNIVERSITY GROUNDS", "COLLEGE / UNIVERSITY - GROUNDS",
            "COLLEGE/UNIVERSITY RESIDENCE HALL", "DAY CARE CENTER"
        ],
        "Healthcare": [
            "HOSPITAL BUILDING/GROUNDS", "HOSPITAL BUILDING / GROUNDS",
            "MEDICAL/DENTAL OFFICE", "MEDICAL / DENTAL OFFICE",
            "NURSING HOME/RETIREMENT HOME", "NURSING / RETIREMENT HOME",
            "ANIMAL HOSPITAL"
        ],
        "Recreational": [
            "ATHLETIC CLUB", "PARK PROPERTY",
            "LAKEFRONT / WATERFRONT / RIVERBANK",
            "LAKEFRONT/WATERFRONT/RIVERBANK", "MOVIE HOUSE / THEATER",
            "MOVIE HOUSE/THEATER", "BOWLING ALLEY",
            "SPORTS ARENA/STADIUM", "SPORTS ARENA / STADIUM",
            "POOL ROOM", "FOREST PRESERVE", "WOODED AREA",
            "FARM", "KENNEL", "CEMETARY"
        ],
        "Religious": [
            "CHURCH/SYNAGOGUE/PLACE OF WORSHIP",
            "CHURCH / SYNAGOGUE / PLACE OF WORSHIP"
        ],
        "Government": [
            "GOVERNMENT BUILDING/PROPERTY",
            "GOVERNMENT BUILDING / PROPERTY",
            "FEDERAL BUILDING", "FIRE STATION",
            "POLICE FACILITY / VEHICLE PARKING LOT",
            "POLICE FACILITY/VEH PARKING LOT",
            "JAIL / LOCK-UP FACILITY"
        ],
        "Industrial": [
            "FACTORY/MANUFACTURING BUILDING",
            "FACTORY / MANUFACTURING BUILDING",
            "WAREHOUSE", "GARAGE/AUTO REPAIR"
        ],
        "Vacant": [
            "VACANT LOT / LAND", "VACANT LOT/LAND",
            "VACANT LOT", "ABANDONED BUILDING"
        ],
        "Airport": [
            loc for loc in df['location_description'].unique()
            if isinstance(loc, str) and (loc.startswith("AIRPORT") or loc == "AIRCRAFT")
        ],
        "Vehicle": ["AUTO", "TRAILER"],
        "Public Housing (CHA)": [
            "CHA APARTMENT", "CHA GROUNDS",
            "CHA HALLWAY / STAIRWELL / ELEVATOR",
            "CHA HALLWAY/STAIRWELL/ELEVATOR",
            "CHA PARKING LOT/GROUNDS", "CHA PARKING LOT / GROUNDS"
        ],
        "Transit-Related": [
            "OTHER RAILROAD PROP / TRAIN DEPOT",
            "OTHER RAILROAD PROPERTY / TRAIN DEPOT",
            'CTA "L" TRAIN'   
        ],
        "Indoor Public Space": ["HALLWAY", "STAIRWELL"],
        "Vending / Machines": ["COIN OPERATED MACHINE"],
        "Library": ["LIBRARY"],
        "Financial": [
            "ATM (AUTOMATIC TELLER MACHINE)", "CREDIT UNION", "SAVINGS AND LOAN"
        ],
        "Construction": ["CONSTRUCTION SITE"],
        "Marine": ["BOAT/WATERCRAFT", "BOAT / WATERCRAFT"],
        "Other/Unknown": ["OTHER", "OTHER (SPECIFY)"]
    }
    return location_map

def map_location_category(desc, location_map):
    """Map location description ke kategori."""
    if pd.isna(desc):
        return "Other/Unknown"
    
    for category, desc_list in location_map.items():
        if desc in desc_list:
            return category
    
    return "Other/Unknown"

def process_data(df):
    """Process data untuk clustering analysis."""
    try:
        with st.spinner("🔄 Processing data..."):
            original_df = df.copy()
            
            # Step 1: Handle missing values
            df_processed = df.dropna(axis=0).copy()
            missing_before = df.isnull().sum().sum()
            missing_after = df_processed.isnull().sum().sum()
            
            # Step 2: Count duplicates (tidak di-remove)
            duplicates_before = df.duplicated().sum()
            duplicates_after = df_processed.duplicated().sum()
            
            st.write(f"data kosong = {missing_before}")
            st.write(f"data duplikat = {duplicates_before}")
            st.write(f"data kosong setelah cleaning: {missing_after}")
            st.write(f"data duplikat setelah cleaning: {duplicates_after}")
            
            # Step 3: Process date column
            if 'date' in df_processed.columns:
                df_processed['date'] = pd.to_datetime(df_processed['date'])
                df_processed['day'] = df_processed['date'].dt.day
                df_processed['month'] = df_processed['date'].dt.month
                df_processed['year'] = df_processed['date'].dt.year
                df_processed['hour'] = df_processed['date'].dt.hour
                df_processed['weekday'] = df_processed['date'].dt.dayofweek
                df_processed = df_processed.drop(['date'], axis=1)
            
            # Step 4: Crime scene mapping
            if 'hour' in df_processed.columns:
                def crime_scene(hour):
                    if 1 <= hour < 5:
                        return 0
                    elif 5 <= hour < 11:
                        return 1
                    elif 11 <= hour < 15:
                        return 2
                    elif 15 <= hour < 18:
                        return 3
                    else:
                        return 4
                
                df_processed['crime_scene'] = df_processed['hour'].apply(crime_scene)
            
            # Step 5: Location category mapping
            if 'location_description' in df_processed.columns:
                location_map = create_location_map_with_data(df_processed)
                df_processed['location_category'] = df_processed['location_description'].apply(
                    lambda x: map_location_category(x, location_map)
                )
            
            # Step 6: Label encoding
            if 'location_category' in df_processed.columns and 'primary_type' in df_processed.columns:
                from sklearn.preprocessing import LabelEncoder
                
                le_location = LabelEncoder()
                
                # Encode location_category
                df_processed['location_category_encoded'] = le_location.fit_transform(df_processed['location_category'])
                
                # Encode primary_type with same encoder
                df_processed['primary_type_encoded'] = le_location.fit_transform(df_processed['primary_type'])
                
                # Store encoder
                st.session_state.location_encoder = le_location
            
            # Processing statistics
            processing_stats = {
                'missing_removed': missing_before - missing_after,
                'duplicates_removed': 0,
                'final_shape': df_processed.shape,
                'crime_categories': df_processed['primary_type'].nunique() if 'primary_type' in df_processed.columns else 0,
                'location_categories': df_processed['location_category'].nunique() if 'location_category' in df_processed.columns else 0,
                'original_shape': original_df.shape,
                'records_after_cleaning': df_processed.shape[0],
                'features_count': df_processed.shape[1]
            }
            
            st.write(f"📊 **Final processed data**: {df_processed.shape[0]:,} records, {df_processed.shape[1]} columns")
            
            return df_processed, processing_stats
            
    except Exception as e:
        st.error(f"❌ Error in data processing: {str(e)}")
        fallback_stats = {
            'missing_removed': 0,
            'duplicates_removed': 0,
            'final_shape': df.shape,
            'crime_categories': 0,
            'location_categories': 0,
            'original_shape': df.shape,
            'records_after_cleaning': len(df),
            'features_count': len(df.columns),
            'error': str(e)
        }
        return df.copy(), fallback_stats