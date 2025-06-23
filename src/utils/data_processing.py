import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def handle_missing_values(df):
    """Handle missing values EXACT sesuai notebook."""
    missing_before = df.isnull().sum().sum()
    df = df.dropna(axis=0)  # EXACT seperti notebook
    missing_after = df.isnull().sum().sum()
    return df, missing_before, missing_after

def handle_duplicates(df):
    """Count duplicates tapi TIDAK remove (sesuai notebook)."""
    duplicates_before = df.duplicated().sum()
    # CRITICAL: Notebook TIDAK melakukan drop_duplicates!
    duplicates_after = df.duplicated().sum()
    return df, duplicates_before, duplicates_after

def process_date_column(df):
    """Process date column EXACT sesuai notebook."""
    if 'date' in df.columns:
        # FIX: Use .copy() to avoid SettingWithCopyWarning
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # EXACT seperti notebook
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['hour'] = df['date'].dt.hour
        df['weekday'] = df['date'].dt.dayofweek
        
        df = df.drop(['date'], axis=1)
    
    return df

def crime_scene_mapping(hour):
    """Crime scene mapping EXACT sesuai notebook."""
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
    """Create location map dengan Airport dinamis EXACT seperti notebook."""
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
    """Map location EXACT sesuai notebook."""
    if pd.isna(desc):
        return "Other/Unknown"
    
    for category, desc_list in location_map.items():
        if desc in desc_list:
            return category
    
    return "Other/Unknown"

def process_data(df):
    """Process data EXACT sesuai notebook dengan bug yang sama."""
    try:
        with st.spinner("🔄 Processing data..."):
            original_df = df.copy()
            
            # Step 1: Handle missing values EXACT seperti notebook
            df_processed = df.dropna(axis=0).copy()  # FIX: Add .copy()
            missing_before = df.isnull().sum().sum()
            missing_after = df_processed.isnull().sum().sum()
            
            # Step 2: Count duplicates (tidak di-remove seperti notebook)
            duplicates_before = df.duplicated().sum()
            duplicates_after = df_processed.duplicated().sum()
            
            st.write(f"data kosong = {missing_before}")
            st.write(f"data duplikat = {duplicates_before}")
            st.write(f"data kosong setelah cleaning: {missing_after}")
            st.write(f"data duplikat setelah cleaning: {duplicates_after}")
            
            # Step 3: Process date column EXACT seperti notebook
            if 'date' in df_processed.columns:
                df_processed['date'] = pd.to_datetime(df_processed['date'])
                df_processed['day'] = df_processed['date'].dt.day
                df_processed['month'] = df_processed['date'].dt.month
                df_processed['year'] = df_processed['date'].dt.year
                df_processed['hour'] = df_processed['date'].dt.hour
                df_processed['weekday'] = df_processed['date'].dt.dayofweek
                df_processed = df_processed.drop(['date'], axis=1)
            
            # Step 4: Crime scene mapping EXACT seperti notebook
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
            
            # Step 5: Location category mapping EXACT seperti notebook
            if 'location_description' in df_processed.columns:
                # Use exact location_map from notebook
                location_map = create_location_map_with_data(df_processed)
                df_processed['location_category'] = df_processed['location_description'].apply(
                    lambda x: map_location_category(x, location_map)
                )
            
            # Step 6: CRITICAL - REPLICATE EXACT BUG dari notebook
            if 'location_category' in df_processed.columns and 'primary_type' in df_processed.columns:
                from sklearn.preprocessing import LabelEncoder
                
                # EXACT BUG replication: gunakan encoder yang sama untuk keduanya
                le_location = LabelEncoder()
                
                # Encode location_category dulu
                df_processed['location_category_encoded'] = le_location.fit_transform(df_processed['location_category'])
                
                # BUG: Gunakan encoder yang sama untuk primary_type (seperti di notebook)
                # Ini menyebabkan primary_type di-encode berdasarkan location categories!
                df_processed['primary_type_encoded'] = le_location.fit_transform(df_processed['primary_type'])
                
                # Store untuk debugging
                st.session_state.location_encoder = le_location
            
            # FIX: Pastikan semua field ada dalam processing_stats
            processing_stats = {
                'missing_removed': missing_before - missing_after,
                'duplicates_removed': 0,  # Tidak ada yang diremove
                'final_shape': df_processed.shape,  # FIX: Pastikan key ini ada
                'crime_categories': df_processed['primary_type'].nunique() if 'primary_type' in df_processed.columns else 0,
                'location_categories': df_processed['location_category'].nunique() if 'location_category' in df_processed.columns else 0,
                'original_shape': original_df.shape,
                'records_after_cleaning': df_processed.shape[0],  # FIX: Tambahan fallback
                'features_count': df_processed.shape[1]  # FIX: Tambahan fallback
            }
            
            st.write(f"📊 **Final processed data**: {df_processed.shape[0]:,} records, {df_processed.shape[1]} columns")
            
            return df_processed, processing_stats
            
    except Exception as e:
        st.error(f"❌ Error in data processing: {str(e)}")
        # FIX: Return safe fallback stats
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