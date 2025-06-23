def crime_scene_mapping(hour):
    """Map hour to crime scene time period"""
    if 1 <= hour < 5:
        return 'early morning'
    elif 5 <= hour < 11:
        return 'morning'
    elif 11 <= hour < 15:
        return 'noon'
    elif 15 <= hour < 18:
        return 'afternoon'
    else:
        return 'night'

def map_crime_description(desc):
    """Map crime description to category"""
    mapping_dict = {
        "ASSAULT": [
            "AGGRAVATED", "BATTERY", "ASSAULT", "STRONGARM", "RECKLESS CONDUCT",
            "HANDS", "FISTS", "FEET", "INTIMIDATION"
        ],
        "THEFT": [
            "THEFT", "PURSE-SNATCHING", "POCKET-PICKING", "RETAIL", "STOLEN", 
            "BURGLARY", "ATTEMPT THEFT", "FROM BUILDING", "LOOTING"
        ],
        "DRUG": [
            "POSS", "POSSESS", "CANNABIS", "HEROIN", "COCAINE", "DELIVER", "MANUFACTURE",
            "DRUG", "SYNTHETIC", "NARCOTICS", "LOOK-ALIKE", "PCP", "AMPHETAMINES", "METH"
        ],
        "SEX OFFENSE": [
            "SEXUAL", "SEX", "INDECENT", "PORNOGRAPHY", "SOLICITATION", "FAMILY MEMBER",
            "PEEPING", "CHILD", "MOLEST", "PREDATORY", "EXPLOITATION"
        ],
        "WEAPON": [
            "HANDGUN", "KNIFE", "CUTTING INSTR", "FIREARM", "WEAPON", "GUN", "ARMED", 
            "AMMUNITION", "BOMB", "EXPLOSIVE", "INCENDIARY"
        ],
        "FRAUD": [
            "FRAUD", "FORGERY", "COUNTERFEIT", "IMPERSONATION", "IDENTITY", "CONFIDENCE",
            "FINANCIAL", "CREDIT CARD", "EMBEZZLEMENT"
        ],
        "HOMICIDE": [
            "HOMICIDE", "MURDER", "KILLING"
        ],
        "PROPERTY DAMAGE": [
            "ARSON", "VANDALISM", "CRIMINAL DEFACEMENT", "TO PROPERTY", "TO VEHICLE",
            "TO LAND"
        ],
        "TRAFFIC": [
            "VEHICLE", "AUTOMOBILE", "TRUCK", "DRIVING", "LICENSE", "REGISTRATION"
        ],
        "OTHER": [
            "OTHER", "VIOLATION", "CONSPIRACY", "ESCAPE", "OBSTRUCT", "RESIST",
            "UNLAWFUL", "PROHIBITED", "FAIL TO REGISTER", "ORDER OF PROTECTION"
        ]
    }
    
    for category, keywords in mapping_dict.items():
        for keyword in keywords:
            if keyword in desc.upper():
                return category
    return "OTHER"

def map_location_category(desc):
    """Map location description to category"""
    location_map = {
        "Residential": [
            "APARTMENT", "RESIDENCE", "HOUSE", "RESIDENCE-GARAGE", "RESIDENTIAL YARD (FRONT/BACK)",
            "RESIDENCE PORCH/HALLWAY", "RESIDENCE - GARAGE", "RESIDENCE - PORCH / HALLWAY",
            "RESIDENCE - YARD (FRONT / BACK)", "PORCH", "YARD", "VESTIBULE"
        ],
        "Commercial": [
            "DRUG STORE", "BAR OR TAVERN", "GROCERY FOOD STORE", "BANK", "CONVENIENCE STORE", 
            "SMALL RETAIL STORE", "RESTAURANT", "DEPARTMENT STORE", "HOTEL / MOTEL", "MOTEL", 
            "TAVERN/LIQUOR STORE", "TAVERN / LIQUOR STORE", "GAS STATION", "GAS STATION DRIVE/PROP.",
            "CAR WASH", "APPLIANCE STORE", "AUTO / BOAT / RV DEALERSHIP", "PAWN SHOP", "BARBERSHOP",
            "BARBER SHOP/BEAUTY SALON", "NEWSSTAND", "CLEANING STORE", "CURRENCY EXCHANGE"
        ],
        "Transportation": [
            "STREET", "SIDEWALK", "PARKING LOT/GARAGE(NON.RESID.)", "PARKING LOT / GARAGE (NON RESIDENTIAL)",
            "CTA PLATFORM", "CTA BUS", "CTA TRAIN", "CTA BUS STOP", "CTA STATION", "CTA GARAGE / OTHER PROPERTY",
            "CTA PARKING LOT / GARAGE / OTHER PROPERTY", "CTA TRACKS - RIGHT OF WAY", "CTA PROPERTY",
            "VEHICLE NON-COMMERCIAL", "VEHICLE - OTHER RIDE SHARE SERVICE (E.G., UBER, LYFT)",
            "VEHICLE - OTHER RIDE SHARE SERVICE (LYFT, UBER, ETC.)", "VEHICLE - OTHER RIDE SERVICE",
            "VEHICLE-COMMERCIAL", "VEHICLE - COMMERCIAL", "TAXICAB", "DELIVERY TRUCK", "VEHICLE - DELIVERY TRUCK",
            "VEHICLE - COMMERCIAL: TROLLEY BUS", "DRIVEWAY - RESIDENTIAL", "OTHER COMMERCIAL TRANSPORTATION"
        ],
        "Educational": [
            "SCHOOL, PRIVATE, BUILDING", "SCHOOL, PRIVATE, GROUNDS", "SCHOOL - PRIVATE GROUNDS",
            "SCHOOL - PRIVATE BUILDING", "SCHOOL, PUBLIC, BUILDING", "SCHOOL, PUBLIC, GROUNDS",
            "SCHOOL - PUBLIC GROUNDS", "SCHOOL - PUBLIC BUILDING", "COLLEGE/UNIVERSITY GROUNDS",
            "COLLEGE / UNIVERSITY - GROUNDS", "COLLEGE/UNIVERSITY RESIDENCE HALL"
        ],
        "Healthcare": [
            "HOSPITAL BUILDING/GROUNDS", "HOSPITAL BUILDING / GROUNDS", "MEDICAL/DENTAL OFFICE",
            "MEDICAL / DENTAL OFFICE", "NURSING HOME/RETIREMENT HOME", "NURSING / RETIREMENT HOME",
            "ANIMAL HOSPITAL"
        ],
        "Recreational": [
            "ATHLETIC CLUB", "PARK PROPERTY", "LAKEFRONT / WATERFRONT / RIVERBANK", "LAKEFRONT/WATERFRONT/RIVERBANK",
            "MOVIE HOUSE / THEATER", "MOVIE HOUSE/THEATER", "BOWLING ALLEY", "SPORTS ARENA/STADIUM",
            "SPORTS ARENA / STADIUM", "POOL ROOM", "FOREST PRESERVE", "WOODED AREA", "FARM", "KENNEL", "CEMETARY"
        ],
        "Religious": [
            "CHURCH/SYNAGOGUE/PLACE OF WORSHIP", "CHURCH / SYNAGOGUE / PLACE OF WORSHIP"
        ],
        "Government": [
            "GOVERNMENT BUILDING/PROPERTY", "GOVERNMENT BUILDING / PROPERTY", "FEDERAL BUILDING",
            "FIRE STATION", "POLICE FACILITY / VEHICLE PARKING LOT", "POLICE FACILITY/VEH PARKING LOT", "JAIL / LOCK-UP FACILITY"
        ],
        "Industrial": [
            "FACTORY/MANUFACTURING BUILDING", "FACTORY / MANUFACTURING BUILDING", "WAREHOUSE", "GARAGE/AUTO REPAIR"
        ],
        "Vacant": [
            "VACANT LOT / LAND", "VACANT LOT/LAND", "VACANT LOT", "ABANDONED BUILDING"
        ],
        "Airport": [
            "AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA", "AIRPORT BUILDING NON-TERMINAL - SECURE AREA",
            "AIRPORT EXTERIOR - NON-SECURE AREA", "AIRPORT EXTERIOR - SECURE AREA", 
            "AIRPORT PARKING LOT", "AIRPORT TERMINAL LOWER LEVEL - NON-SECURE AREA",
            "AIRPORT TERMINAL LOWER LEVEL - SECURE AREA", "AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA",
            "AIRPORT TERMINAL UPPER LEVEL - SECURE AREA", "AIRPORT TRANSPORTATION SYSTEM (ATS)",
            "AIRCRAFT"
        ],
        "Vehicle": [
            "AUTO", "TRAILER"
        ],
        "Other/Unknown": [
            "OTHER", "OTHER (SPECIFY)", "OTHER RAILROAD PROP / TRAIN DEPOT", "OTHER RAILROAD PROPERTY / TRAIN DEPOT",
            "COIN OPERATED MACHINE", "STAIRWELL", "HALLWAY", "LIBRARY", "CHA APARTMENT", "CHA GROUNDS",
            "CHA HALLWAY / STAIRWELL / ELEVATOR", "CHA HALLWAY/STAIRWELL/ELEVATOR", "CHA PARKING LOT/GROUNDS",
            "CHA PARKING LOT / GROUNDS"
        ]
    }
    
    for category, desc_list in location_map.items():
        if desc in desc_list:
            return category
    return "Other/Unknown"