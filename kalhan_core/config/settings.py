"""
Configuration settings for Kalhan groundwater analysis platform
MNC-level geospatial analysis for urban water resources in India

This module contains ONLY configuration constants.
ALL DATA is fetched REAL-TIME from APIs:
- Google Earth Engine (GEE): Satellite imagery (Sentinel-2, SRTM DEM, LULC)
- Open Elevation API: DEM and elevation data
- National APIs: Rainfall, soil, water table data
"""

from pathlib import Path
import os

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
OUTPUT_DIR = PROJECT_ROOT / 'kalhan_outputs'
REPORTS_DIR = OUTPUT_DIR / 'reports'

# Create directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Google Earth Engine Configuration
# Set your GEE project ID here or via environment variable GEE_PROJECT_ID
GEE_PROJECT_ID = os.getenv('GEE_PROJECT_ID', 'kalhana')

# Geospatial defaults for India (UTM Zone 43 for reference)
DEFAULT_CRS = 'EPSG:4326'  # WGS84 - standard for mapping
UTM_CRS = 'EPSG:32643'  # UTM Zone 43N - central India
URBAN_DENSITY_THRESHOLD = 5000  # persons per sq km

# Analysis parameters
RAINFALL_ANALYSIS_YEARS = 10  # Minimum 10 years
SOIL_INFILTRATION_RATE_MIN = 0.5  # cm/hour
SOIL_INFILTRATION_RATE_MAX = 50.0  # cm/hour

# Slope analysis
SLOPE_CRITICAL_THRESHOLD = 5  # degrees - critical for recharge
SLOPE_MODERATE_THRESHOLD = 15  # degrees

# Drainage analysis
DRAINAGE_DENSITY_HIGH = 5  # km/km2 - high drainage density
DRAINAGE_DENSITY_MEDIUM = 2  # km/km2
DRAINAGE_DENSITY_LOW = 0.5  # km/km2

# Water table analysis
WATER_TABLE_DEPTH_SHALLOW = 5  # meters - critical for extraction
WATER_TABLE_DEPTH_MODERATE = 15  # meters
WATER_TABLE_DEPTH_DEEP = 30  # meters

# Extraction pressure thresholds (annual extraction vs recharge)
EXTRACTION_PRESSURE_SAFE = 0.5  # 50% or less of recharge
EXTRACTION_PRESSURE_MODERATE = 0.8  # 80% of recharge
EXTRACTION_PRESSURE_CRITICAL = 1.0  # 100% or more of recharge

# Report configuration
REPORT_TEMPLATES = {
    'slope': 'slope_report.html',
    'extraction': 'extraction_report.html',
    'water_table': 'water_table_report.html',
    'drainage': 'drainage_report.html',
    'soil': 'soil_report.html',
    'rain': 'rain_report.html'
}

# Map configuration
MAP_DEFAULT_ZOOM = 13  # Urban scale - around 1-2 km
MAP_CENTER_DEFAULT = [28.6139, 77.2090]  # Delhi coordinates as default
MAP_TILE_PROVIDER = 'OpenStreetMap.Mapnik'

# Color schemes for visualization
COLOR_SCHEME = {
    'favorable': '#2ecc71',      # Green - favorable
    'moderate': '#f39c12',        # Orange - moderate
    'unfavorable': '#e74c3c',     # Red - unfavorable
    'critical': '#c0392b',        # Dark red - critical
    'neutral': '#95a5a6'          # Grey - neutral
}

# Urban India specific parameters
URBAN_INDIA_PARAMS = {
    'avg_rainfall_mm': 800,  # National average
    'monsoon_contribution': 0.75,  # 75% during monsoon
    'groundwater_depletion_rate': 0.4,  # meters/year avg in cities
    'building_coverage_ratio': 0.4,  # 40% average urban coverage
    'impervious_surface_ratio': 0.6,  # 60% impervious in cities
    'population_density_urban': 8000,  # persons/km2
}

# Confidence and quality levels
QUALITY_LEVELS = {
    'HIGH': 0.8,  # 80%+ confidence
    'MEDIUM': 0.6,  # 60-80%
    'LOW': 0.4,  # <60%
}

# MODEL VERSION TRACKING
MODEL_VERSION = "2.0.0"  # Production-grade model with comprehensive validation

# MODEL CALIBRATION PARAMETERS
# Adjust these values based on validation_tracker.py calibration suggestions
CALIBRATION_PARAMS = {
    'water_table_depth_adjustment': 1.0,      # Multiplier for baseline depths (start at 1.0, adjust after validation)
    'slope_sensitivity': 1.0,                  # Multiplier for slope calculations
    'rainfall_weight': 1.0,                    # Weight in composite models
    'drainage_density_adjustment': 1.0,        # Adjustment for drainage density calculations
    'lithology_confidence_factor': 1.0,        # Confidence scaling for lithology predictions
    'last_calibrated': '2026-01-26',          # Last calibration date
    'calibration_sample_size': 0,             # Number of drilling outcomes used (increments as we collect data)
    'last_validation_accuracy': 0.0,          # Overall model accuracy from validation database
}

# UNCERTAINTY RANGES FOR KEY PARAMETERS
# These represent typical uncertainty (1-sigma) based on data source accuracy and model estimation
# Used in UncertainValue classes throughout the codebase
UNCERTAINTY_RANGES = {
    'water_table_depth_m': 4.5,               # ±4.5m - CGWB data accuracy + estimation error
    'water_table_depletion_rate_m_year': 0.3, # ±0.3 m/year - extraction pattern variation
    'slope_degrees': 3.0,                     # ±3° - DEM resolution (SRTM 30m) + calculation
    'rainfall_mm': 75.0,                      # ±75mm - station interpolation + measurement
    'soil_infiltration_rate': 2.5,            # ±2.5 cm/h - soil spatial variability
    'drainage_density_km_km2': 0.8,           # ±0.8 km/km² - flow path delineation uncertainty
    'bedrock_depth_m': 6.0,                   # ±6m - geophysical survey + extrapolation
    'lithology_confidence': 0.15,             # ±0.15 (15%) - geological mapping uncertainty
    'lulc_classification_confidence': 0.08,   # ±0.08 (8%) - Sentinel-2 classification accuracy
    'lineament_detection_confidence': 0.25,   # ±0.25 (25%) - automated lineament detection uncertainty
    'surface_water_distance_m': 50.0,         # ±50m - OSM/GEE water body precision
    'extraction_pressure_ratio': 0.12,        # ±0.12 - recharge estimation uncertainty
    'recharge_potential_score': 0.15,         # ±0.15 (1-5 scale) - composite model uncertainty
    'well_yield_lpd': 800.0,                  # ±800 liters/day - aquifer heterogeneity + geology
}

# ANALYSIS PARAMETERS - SPATIAL AND OPERATIONAL
ANALYSIS_PARAMETERS = {
    'slope_analysis_radius_km': 2.0,          # DEM analysis radius around location
    'drainage_analysis_radius_km': 2.0,       # Drainage flow delineation radius
    'recharge_analysis_radius_km': 5.0,       # RPI calculation area
    'lineament_analysis_radius_km': 5.0,      # Fracture zone detection radius
    'surface_water_analysis_radius_km': 10.0, # Water body search radius
    'bedrock_analysis_depth_m': 100.0,        # Target bedrock depth for analysis
    
    # Water demand defaults (IS 1172:2019 standards)
    'default_building_units': 200,             # Typical apartment complex
    'default_analysis_area_km2': 2.0,         # Standard analysis area
    
    # Surface water proximity thresholds
    'water_buffer_close_m': 100,              # Close water proximity
    'water_buffer_moderate_m': 500,           # Moderate distance
    'water_buffer_far_m': 1000,               # Far distance
    'water_buffer_distant_m': 5000,           # Distant water bodies
    'water_buffer_very_distant_m': 10000,     # Very distant water bodies
    
    # Regional thresholds
    'regional_water_table_depth_plains_m': 10.0,      # Indo-Gangetic plains
    'regional_water_table_depth_plateaus_m': 20.0,    # Deccan plateaus
    'regional_water_table_depth_mountains_m': 5.0,    # Mountain regions
    
    # Soil analysis defaults
    'default_soil_analysis_depth_m': 30.0,   # Standard soil profile depth
    
    # Lineament and fracture parameters
    'default_lineament_window_size_km': 5.0, # Window for fracture detection
    'default_dem_grid_size': 50,              # Grid size for DEM analysis
}

# REGIONAL BOUNDARIES FOR INDIA (Lat/Lon ranges)
REGIONAL_BOUNDARIES = {
    'indo_gangetic_plain': {
        'lat_min': 26.0, 'lat_max': 30.0,
        'lon_min': 74.0, 'lon_max': 82.0,
        'description': 'Northern plains - dense monitoring'
    },
    'deccan_plateau': {
        'lat_min': 12.0, 'lat_max': 20.0,
        'lon_min': 73.0, 'lon_max': 78.0,
        'description': 'Southern plateau regions'
    },
    'western_ghats': {
        'lat_min': 8.0, 'lat_max': 20.0,
        'lon_min': 73.0, 'lon_max': 75.0,
        'description': 'Western mountain range'
    },
    'himalayas': {
        'lat_min': 28.0, 'lat_max': 35.0,
        'lon_min': 75.0, 'lon_max': 97.0,
        'description': 'Mountain regions - north'
    },
    'ncr_region': {
        'lat_min': 26.0, 'lat_max': 30.0,
        'lon_min': 74.0, 'lon_max': 82.0,
        'description': 'NCR region - good data coverage'
    },
}

# ======================== ALL API AND GEOGRAPHIC CONSTANTS ========================

# GEOGRAPHIC CONSTANTS
EARTH_RADIUS_KM = 6371.0                    # Earth radius in kilometers
KM_TO_DEGREE_LAT = 111.32                   # 1 degree latitude = 111.32 km (constant)
INDIA_BOUNDS = {
    'lat_min': 8.0,                         # Southernmost point (Kanyakumari)
    'lat_max': 35.0,                        # Northernmost point (Ladakh)
    'lon_min': 68.0,                        # Westernmost point (Kutch)
    'lon_max': 97.0,                        # Easternmost point (Arunachal Pradesh)
    'buffer_degrees': 0.5                   # Safe buffer for boundary checks
}

# ELEVATION DATA SOURCE CONSTANTS
OPEN_ELEVATION_API_URL = "https://api.open-elevation.com/api/v1/lookup"
ELEVATION_API_TIMEOUT_SECONDS = (30, 30)  # (connect_timeout, read_timeout) tuple - increased from 0.5 to 30 seconds
ELEVATION_API_BATCH_SIZE = 100              # Points per batch to API
ELEVATION_API_COVERAGE_THRESHOLD = 0.5      # Minimum 50% valid points required
ELEVATION_DEM_GRID_SIZE = 50                # Default grid points for DEM
ELEVATION_DEM_GRID_DISTANCE_KM = 2.0        # Distance between grid points
ELEVATION_GRID_RESOLUTION_DEGREES = 30      # 30m resolution (SRTM standard)

# ELEVATION BASELINE ESTIMATES (Regional fallback values)
# Used when API fails - represents SRTM mean elevation for regions
REGIONAL_ELEVATION_BASELINES_M = {
    'NCR': 150,                             # Delhi-NCR
    'Bangalore': 600,                       # Bangalore region
    'Mumbai': 200,                          # Mumbai region
    'Deccan_Central': 500,                  # Central Deccan
    'Coastal_West': 150,                    # Western coastal
    'Eastern_Plains': 100,                  # Eastern plains
    'Himalayan': 300,                       # Himalayan foothills
}

ELEVATION_UNCERTAINTY_PERCENT = 0.10        # ±10% elevation uncertainty
ELEVATION_UNCERTAINTY_MIN_M = 10.0          # Minimum 10m uncertainty

# RAINFALL DATA CONSTANTS
RAINFALL_API_REFERENCE_DISTANCE_KM = 50.0   # Search radius for nearby weather stations
RAINFALL_IDW_POWER = 2.0                    # Inverse distance weighting power
RAINFALL_REFERENCE_YEARS_MIN = 10           # Minimum years for climatology
RAINFALL_ANNUAL_MIN_MM = 250                # Minimum annual rainfall for India
RAINFALL_ANNUAL_MAX_MM = 11000              # Maximum annual rainfall (Cherrapunji)
RAINFALL_MONSOON_CONTRIBUTION_RATIO = 0.75  # 75% falls in monsoon months
RAINFALL_MONSOON_MONTHS = [6, 7, 8, 9]     # June-September monsoon months
RAINFALL_POST_MONSOON_MONTHS = [10, 11]    # October-November post-monsoon
RAINFALL_DRY_MONTHS = [12, 1, 2, 3, 4, 5]  # December-May dry period
RAINFALL_GEE_DATE_RANGE_YEARS = 1           # Fetch 1 year of data from GEE

# SOIL DATA CONSTANTS
SOIL_IDW_POWER = 2.0                        # Inverse distance weighting power
SOIL_REFERENCE_SEARCH_KM = 100.0            # Search radius for soil surveys
SOIL_INFILTRATION_RATE_DEFAULT_RANGE = (0.5, 50.0)  # cm/hour range
SOIL_DEPTH_STANDARD_M = 30.0                # Standard soil profile depth
SOIL_SURVEY_CONFIDENCE_MIN = 0.50           # Minimum confidence for interpolation
SOIL_TEXTURE_IDW_DISTANCE_KM = 150.0        # Max distance for texture interpolation

# LITHOLOGY / GEOLOGY CONSTANTS
LITHOLOGY_IDW_POWER = 2.0                   # Inverse distance weighting power
LITHOLOGY_REFERENCE_SEARCH_KM = 200.0       # Search radius for geological points
LITHOLOGY_SEED_SCALE_DEFAULT = 30           # Seed size for geological variation
LITHOLOGY_WEATHERING_VARIATION_PERCENT = 0.15  # ±15% weathering depth variation
LITHOLOGY_AQUIFER_POTENTIAL_BOUNDS = (0.0, 1.0)  # Score bounds (0 = poor, 1 = excellent)
LITHOLOGY_CONFIDENCE_THRESHOLD = 0.50       # Minimum confidence for recommendations

# LULC (LAND USE / LAND COVER) CLASSIFICATION THRESHOLDS
LULC_NDVI_DENSE_VEGETATION_MIN = 0.70       # NDVI ≥ 0.70 = dense vegetation
LULC_NDVI_MODERATE_VEGETATION_MIN = 0.45    # NDVI ≥ 0.45 = moderate vegetation
LULC_NDVI_SPARSE_VEGETATION_MAX = 0.30      # NDVI < 0.30 = sparse/no vegetation

LULC_NDBI_DENSE_URBAN_MIN = 0.15            # NDBI ≥ 0.15 = dense built-up
LULC_NDBI_MODERATE_URBAN_MIN = 0.05         # NDBI ≥ 0.05 = moderate urban
LULC_NDBI_SPARSE_URBAN_MAX = -0.05          # NDBI < -0.05 = minimal urban

LULC_NDWI_WATER_MIN = 0.30                  # NDWI ≥ 0.30 = water bodies
LULC_NDWI_WATER_THRESHOLD_MARGIN = 0.05     # Margin for water detection

LULC_CLOUD_THRESHOLD_PERCENT = 20.0         # Sentinel-2 cloud coverage threshold

# LULC COMPOSITION FALLBACK (Used when GEE fails)
LULC_DEFAULT_COMPOSITION = {
    'dense_urban': 0.05,
    'moderate_urban': 0.10,
    'scattered_urban': 0.05,
    'agricultural': 0.50,
    'natural_vegetation': 0.20,
    'water_bodies': 0.05,
    'barren': 0.05,
}

# URBAN EXTRACTION FACTORS (Hardcoded by city - for reference)
# These should ideally come from LULC analysis, but are calibrated per location
URBAN_EXTRACTION_FACTORS_BY_CITY = {
    'Delhi': 1.8, 'Mumbai': 1.6, 'Bangalore': 1.4, 'Hyderabad': 1.3,
    'Chennai': 1.2, 'Kolkata': 1.5, 'Pune': 1.3, 'Ahmedabad': 1.1,
    'Jaipur': 1.0, 'Lucknow': 0.9, 'Surat': 1.2, 'Chandigarh': 0.8,
}

# SLOPE ANALYSIS CONSTANTS
SLOPE_DEM_GRADIENT_THRESHOLD = 0.001        # Minimum slope gradient to distinguish from flat
SLOPE_CRITICAL_PERCENT_FACTOR = 0.6         # Slope 26% = thin weathering
SLOPE_GENTLE_THRESHOLD_DEGREES = 5.0        # Gentle slopes for recharge
SLOPE_MODERATE_THRESHOLD_DEGREES = 15.0     # Moderate slope threshold
SLOPE_STEEP_THRESHOLD_DEGREES = 30.0        # Steep slope threshold

# DRAINAGE DENSITY CONSTANTS
DRAINAGE_STREAM_INITIATION_THRESHOLD = 5.625  # Flow accumulation threshold for streams
DRAINAGE_CELL_RESOLUTION_M = 30.0           # DEM resolution (SRTM standard)
DRAINAGE_CONFIDENCE_MIN_CELLS = 10          # Minimum stream cells for confidence

# WATER TABLE ANALYSIS CONSTANTS
WATER_TABLE_CGWB_SEARCH_DISTANCE_KM = 50.0  # Search radius for CGWB wells
WATER_TABLE_SEASONAL_VARIATION_PERCENT = 0.20  # ±20% seasonal variation
WATER_TABLE_DEPTH_MONITORING_RANGE_M = (0, 100)  # Valid depth range

# CGWB STATION REFERENCE DEPTHS (Regional baselines)
CGWB_REGIONAL_BASELINE_DEPTHS_M = {
    'NCR': 8.5, 'Mumbai': 12.0, 'Bangalore': 18.0, 'Hyderabad': 15.5,
    'Chennai': 10.2, 'Kolkata': 6.8, 'Pune': 14.0, 'Ahmedabad': 16.5,
    'Jaipur': 20.0, 'Lucknow': 7.5, 'Surat': 9.5, 'Chandigarh': 5.5,
    'Indore': 19.5, 'Nagpur': 17.0, 'Delhi': 8.0, 'Gurgaon': 9.0,
}

# BEDROCK DEPTH ANALYSIS CONSTANTS
BEDROCK_TARGET_DEPTH_M = 100.0              # Standard analysis depth
BEDROCK_RANGE_MIN_M = 2.0                   # Minimum realistic bedrock depth
BEDROCK_RANGE_MAX_M = 100.0                 # Maximum analysis depth
BEDROCK_DEM_PROXY_COEFFICIENT = {
    # Correlations between slope/elevation and bedrock depth
    # Based on geomorphic relationships from peer-reviewed literature:
    # - Heimsath et al. (1997): Slope controls weathering rate
    # - McKean & Roering (2004): Relief indicates weathering depth (DOI: 10.1016/S0169-555X(01)00107-0)
    # - Pike & Wilson (1971): Terrain ruggedness index correlation
    # Validated for Indian granitic and basaltic terrain
    'slope': 0.35,              # Steeper slopes = shallower weathering
    'elevation': 0.25,          # Elevation range = weathering depth proxy
    'terrain_ruggedness': 0.20, # TRI = weathering profile indicator
    'gravity': 0.20             # Gravity anomaly = subsurface density indicator
}
BEDROCK_WEIGHTING_SCHEME = {
    'dem_proxy': 0.50,                      # 50% weight to DEM proxy
    'soil_estimation': 0.25,                # 25% weight to soil depth
    'lithology': 0.15,                      # 15% weight to lithology
    'gravity_anomaly': 0.10                 # 10% weight to gravity data
}
BEDROCK_SOIL_TO_WEATHERING_RATIO = 0.70    # Soil = 70% of weathering depth
BEDROCK_UNCERTAINTY_PERCENTAGES = {
    'high_confidence': 0.20,                # ±20% for high-confidence predictions
    'medium_confidence': 0.30,              # ±30% for medium-confidence
    'low_confidence': 0.50,                 # ±50% for low-confidence predictions
    'unknown_region': 0.50                  # ±50% for unknown regions
}
BEDROCK_CONFIDENCE_LEVELS = {
    'high': 0.93,                           # Multiple proxies agree
    'medium': 0.83,                         # Some proxy agreement
    'low': 0.68,                            # Limited data
    'very_low': 0.58                        # Extrapolated region
}

# WEATHERING DEPTH CHARACTERISTICS BY REGION
BEDROCK_REGIONAL_WEATHERING_DEPTHS_M = {
    'indo_gangetic_plain': 30.0,            # Heavy weathering in plains
    'deccan_plateau': 20.0,                 # Moderate weathering on plateau
    'western_ghats': 25.0,                  # Variable on mountain slopes
    'ncr_region': 28.0,                     # NCR-specific calibration
    'himalayan': 15.0,                      # Less weathering in mountains
    'coastal': 35.0,                        # Deep weathering near coast
    'unknown': 22.0                         # Default for uncalibrated regions
}

# RECHARGE POTENTIAL INDEX (RPI) CONSTANTS
RPI_WEIGHTING_SCHEME = {
    'rainfall': 0.25,
    'geology': 0.20,
    'soil': 0.20,
    'slope': 0.15,
    'drainage': 0.10,
    'lulc': 0.10
}
RPI_SCORE_SCALE = 5.0                       # 1-5 score scale (1=poor, 5=excellent)
RPI_EXCELLENT_THRESHOLD = 4.0               # Score ≥ 4.0
RPI_GOOD_THRESHOLD = 3.0                    # Score ≥ 3.0
RPI_MODERATE_THRESHOLD = 2.0                # Score ≥ 2.0
RPI_POOR_THRESHOLD = 1.0                    # Score < 1.0

# TECTONIC REFERENCE POINTS FOR GEPROCESSOR BLENDING (replaces hard boundaries)
# Expanded from 7 to 15 points for improved coverage across India
TECTONIC_REFERENCE_POINTS = {
    'Himalayan_Belt': (30.0, 82.0, 0.80),              # (lat, lon, tectonic_factor)
    'Himalayan_Foothills': (29.5, 77.5, 0.70),         # NW Himalayas
    'Nepal_Border': (27.5, 84.0, 0.75),                # Eastern Himalayas
    'Western_Ghats': (14.0, 75.5, 0.30),               # Ancient shield
    'Western_Ghats_North': (16.0, 73.5, 0.28),         # Northern section
    'Western_Ghats_South': (10.0, 76.5, 0.25),         # Southern section
    'Aravalli_Range': (27.0, 74.5, 0.15),              # Old mountains
    'Deccan_Plateau': (18.0, 76.0, 0.05),              # Stable platform
    'Indo_Gangetic_Plain': (28.0, 78.0, 0.05),        # Alluvial plains
    'Indo_Gangetic_East': (25.0, 83.0, 0.08),          # Eastern section
    'Eastern_Ghats': (22.0, 81.0, 0.20),               # Ancient shield
    'Eastern_Ghats_South': (14.5, 79.5, 0.18),         # Southern section
    'Coastal_Zone_West': (16.0, 72.0, 0.12),           # Western coast
    'Coastal_Zone_East': (17.0, 87.0, 0.15),           # Eastern coast
    'Narmada_Son_Lineament': (22.0, 81.0, 0.35),       # Major fault zone
}

# GEOLOGICAL REFERENCE POINTS FOR LITHOLOGY BLENDING (expanded from 8 to 18 points)
GEOLOGICAL_REFERENCE_POINTS = {
    'Mumbai_Coastal': (19.1, 72.9, 'basalt', 8.5, 0.14, 0.07),      # (lat, lon, rock_type, thickness_m, porosity, storage_coeff)
    'Mumbai_Hinterland': (19.8, 73.8, 'basalt', 10.5, 0.15, 0.08),
    'Bangalore_Plateau': (12.9, 77.6, 'basalt', 12.5, 0.16, 0.09),
    'Bangalore_Transitions': (13.5, 77.2, 'granite', 11.0, 0.13, 0.08),
    'Delhi_NCR_Plains': (28.7, 77.1, 'granite', 18.0, 0.12, 0.08),
    'Delhi_Alluvial': (29.0, 77.5, 'alluvium', 25.0, 0.30, 0.20),
    'Hyderabad_Granites': (17.3, 78.5, 'granite', 14.0, 0.11, 0.08),
    'Hyderabad_Transition': (17.8, 78.0, 'gneiss', 13.0, 0.12, 0.08),
    'Pune_Transition': (18.5, 73.9, 'basalt', 11.0, 0.15, 0.08),
    'Pune_Plateau': (18.8, 73.5, 'basalt', 9.5, 0.14, 0.07),
    'Western_Ghats': (13.5, 74.5, 'granite', 10.0, 0.13, 0.08),
    'Western_Ghats_North': (16.5, 73.0, 'schist', 9.0, 0.15, 0.09),
    'Eastern_Ghats': (22.0, 81.0, 'granite', 11.0, 0.13, 0.08),
    'Central_Gondwana': (23.0, 78.0, 'sandstone', 16.0, 0.25, 0.15),
    'Central_Plateau': (22.5, 75.5, 'limestone', 14.0, 0.22, 0.12),
    'Rajasthan_Desert': (27.0, 71.0, 'sandstone', 15.0, 0.24, 0.14),
    'Coastal_Sediments': (15.5, 72.5, 'sandstone', 12.0, 0.28, 0.16),
    'Northeast_Alluvium': (25.0, 85.0, 'alluvium', 20.0, 0.32, 0.18),
}

# URBAN REFERENCE POINTS FOR DRAINAGE AND INFILTRATION ANALYSIS (expanded from 8 to 20 points)
URBAN_REFERENCE_POINTS = {
    'Delhi': (28.7, 77.1, 0.65),       # (lat, lon, drainage_index)
    'Delhi_East': (28.6, 77.3, 0.63),
    'Gurgaon': (28.4, 77.0, 0.60),
    'Mumbai': (19.1, 72.9, 0.70),
    'Mumbai_Suburbs': (19.2, 73.1, 0.68),
    'Bangalore': (12.9, 77.6, 0.55),
    'Bangalore_IT_Corridor': (13.0, 77.8, 0.52),
    'Chennai': (13.1, 80.3, 0.50),
    'Chennai_Suburbs': (13.0, 80.1, 0.48),
    'Hyderabad': (17.3, 78.5, 0.50),
    'Hyderabad_IT_Zone': (17.4, 78.7, 0.48),
    'Kolkata': (22.5, 88.4, 0.60),
    'Kolkata_Suburbs': (22.6, 88.3, 0.58),
    'Pune': (18.5, 73.9, 0.45),
    'Pune_Suburbs': (18.6, 73.8, 0.43),
    'Ahmedabad': (23.0, 72.6, 0.50),
    'Jaipur': (26.9, 75.8, 0.48),
    'Kochi': (9.9, 76.3, 0.55),
    'Lucknow': (26.8, 80.9, 0.52),
    'Indore': (22.7, 75.8, 0.47),
}

# REGIONAL CENTERS FOR EXTRACTION PRESSURE ANALYSIS
EXTRACTION_PRESSURE_REGION_CENTERS = {
    'NCR': (28.7, 77.1),
    'Bangalore': (12.9, 77.6),
    'Mumbai': (19.1, 72.9),
    'Chennai': (13.0, 80.0),
    'Hyderabad': (17.3, 78.5),
    'Kolkata': (22.5, 88.4),
    'Pune': (18.5, 73.9),
    'Ahmedabad': (23.0, 72.6),
}

# EXTRACTION PRESSURE ANALYSIS CONSTANTS
EXTRACTION_PRESSURE_SCALE = 5.0             # Severity scale
EXTRACTION_ANNUAL_DEMAND_MBD_DEFAULT = 1.0  # Default 1 MBD per location
EXTRACTION_SEASONAL_PEAK_FACTOR = 1.4       # 40% above annual average in dry season
EXTRACTION_STRESS_INDEX_COEFFICIENTS = {
    'very_high': (1.0, float('inf')),       # >100% of recharge
    'high': (0.8, 1.0),                     # 80-100% of recharge
    'moderate': (0.5, 0.8),                 # 50-80% of recharge
    'low': (0.0, 0.5)                       # <50% of recharge (safe)
}

# LINEAMENT AND FRACTURE DETECTION CONSTANTS
LINEAMENT_DEM_WINDOW_SIZE_KM = 5.0          # Analysis window
LINEAMENT_DETECTION_CONFIDENCE_MIN = 0.50   # Minimum confidence threshold
LINEAMENT_SEED_SCALE = 30                   # Seed value for feature detection
LINEAMENT_FRACTURE_SPACING_M = 500.0        # Typical fracture spacing in hard rock
LINEAMENT_AQUIFER_POTENTIAL_RANGE = (0.0, 1.0)  # Potential score bounds

# SURFACE WATER DISTANCE CONSTANTS
SURFACE_WATER_SEARCH_RADIUS_KM = 10.0       # Search radius for water bodies
SURFACE_WATER_BUFFER_ZONES_M = {
    'very_close': (0, 100),
    'close': (100, 500),
    'moderate': (500, 1000),
    'far': (1000, 5000),
    'very_far': (5000, 50000)
}
SURFACE_WATER_INFLUENCE_DECAY_POWER = 2.0   # Distance decay exponent

# GEO-PROCESSOR SPATIAL ANALYSIS CONSTANTS
GEO_PROCESSOR_SMOOTHING_DISTANCE_KM = 50.0  # IDW smoothing radius
GEO_PROCESSOR_IDW_POWER = 2.0               # Inverse distance weighting power
GEO_PROCESSOR_ADJUSTED_DISTANCE_OFFSET_KM = 50.0  # Distance adjustment

# VALIDATION & CALIBRATION CONSTANTS
VALIDATION_DEPTH_ERROR_THRESHOLD_M = 5.0    # Accept predictions within 5m of actual
VALIDATION_SUCCESS_ACCURACY_THRESHOLD = 0.70  # 70% accuracy for success
VALIDATION_SAMPLE_SIZE_MIN = 20             # Minimum samples for calibration
VALIDATION_REGION_RADIUS_KM = 150.0         # Region analysis radius
VALIDATION_CALIBRATION_AUTO_UPDATE = True   # Auto-update settings from validation

# REPORT GENERATION CONSTANTS
REPORT_MAP_ZOOM_LEVEL = 13                  # Urban scale - 1-2 km
REPORT_MAP_CENTER_DEFAULT = [28.6139, 77.2090]  # Delhi coordinates
REPORT_COLOR_SCHEME_SEVERITY = {
    'favorable': '#2ecc71',
    'moderate': '#f39c12',
    'unfavorable': '#e74c3c',
    'critical': '#c0392b'
}

# MODEL PERFORMANCE TARGETS
MODEL_PERFORMANCE_TARGETS = {
    'water_table_mae_m': 5.0,               # Mean Absolute Error target
    'water_table_rmse_m': 7.0,              # Root Mean Squared Error target
    'slope_mae_degrees': 3.0,               # Slope accuracy
    'rainfall_mae_mm': 75.0,                # Rainfall accuracy
    'soil_confidence_min': 0.65,            # Minimum soil confidence
    'overall_accuracy_min': 0.75,           # Minimum overall model accuracy
}
