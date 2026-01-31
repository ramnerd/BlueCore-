"""
Core utility functions for Kalhan geospatial analysis platform
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Standard result container for all analyses"""
    analysis_type: str
    location: Dict[str, float]  # {'latitude': float, 'longitude': float}
    timestamp: str
    confidence_score: float  # 0-1
    severity_level: str  # 'favorable', 'moderate', 'unfavorable', 'critical'
    key_findings: Dict[str, Any]
    recommendations: List[str]
    methodology: str
    data_sources: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate result before serialization"""
        errors = []
        
        # Required field checks
        if not self.analysis_type:
            errors.append("Missing analysis_type")
        
        if not self.location or 'latitude' not in self.location or 'longitude' not in self.location:
            errors.append("Missing or invalid location coordinates")
        
        # Confidence score range
        if not (0 <= self.confidence_score <= 1):
            errors.append(f"Invalid confidence: {self.confidence_score} (must be 0-1)")
        
        # Severity level enum
        valid_severities = ['favorable', 'moderate', 'unfavorable', 'critical']
        if self.severity_level not in valid_severities:
            errors.append(f"Invalid severity: {self.severity_level} (must be one of {valid_severities})")
        
        # Key findings not empty
        if not self.key_findings:
            errors.append("Empty key_findings")
        
        # Timestamp format
        try:
            datetime.fromisoformat(self.timestamp)
        except ValueError:
            errors.append(f"Invalid timestamp format: {self.timestamp}")
        
        return (len(errors) == 0, errors)


class DataValidator:
    """Validates and sanitizes input data to prevent injection attacks"""
    
    @staticmethod
    def validate_coordinates(latitude: float, longitude: float, 
                           strict: bool = True) -> Tuple[bool, str]:
        """Validate and sanitize geographic coordinates"""
        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except (TypeError, ValueError) as e:
            return False, f"Invalid coordinate type: {e}"
        
        # Global bounds
        if not (-90 <= latitude <= 90):
            return False, f"Latitude {latitude}° outside global range [-90°, 90°]"
        
        if not (-180 <= longitude <= 180):
            return False, f"Longitude {longitude}° outside global range [-180°, 180°]"
        
        # India-specific bounds (if strict)
        if strict:
            INDIA_LAT_MIN, INDIA_LAT_MAX = 6.0, 37.0
            INDIA_LON_MIN, INDIA_LON_MAX = 67.0, 98.0
            
            if not (INDIA_LAT_MIN <= latitude <= INDIA_LAT_MAX):
                return False, f"Latitude {latitude}° outside India bounds [{INDIA_LAT_MIN}°, {INDIA_LAT_MAX}°]"
            if not (INDIA_LON_MIN <= longitude <= INDIA_LON_MAX):
                return False, f"Longitude {longitude}° outside India bounds [{INDIA_LON_MIN}°, {INDIA_LON_MAX}°]"
        
        return True, "Valid coordinates"
    
    @staticmethod
    def sanitize_location_id(location_id: str, max_length: int = 100) -> str:
        """Sanitize location ID to prevent SQL injection and other attacks"""
        # Remove all non-alphanumeric except hyphens and underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', str(location_id))
        # Limit length
        return sanitized[:max_length]
    
    @staticmethod
    def validate_depth(depth_m: float) -> Tuple[bool, str]:
        """Validate depth parameter"""
        try:
            depth_m = float(depth_m)
        except (TypeError, ValueError):
            return False, f"Invalid depth type: {type(depth_m).__name__}"
        
        # Reasonable depth range for groundwater
        if not (0.0 <= depth_m <= 500.0):
            return False, f"Depth {depth_m}m outside valid range [0-500m]"
        
        return True, "Valid depth"
    
    @staticmethod
    def validate_area(area_km2: float) -> Tuple[bool, str]:
        """Validate analysis area parameter"""
        try:
            area_km2 = float(area_km2)
        except (TypeError, ValueError):
            return False, f"Invalid area type: {type(area_km2).__name__}"
        
        # Reasonable urban analysis area
        if not (0.01 <= area_km2 <= 1000.0):
            return False, f"Area {area_km2}km² outside valid range [0.01-1000km²]"
        
        return True, "Valid area"
    
    @staticmethod
    def validate_building_count(count: int) -> Tuple[bool, str]:
        """Validate building unit count"""
        try:
            count = int(count)
        except (TypeError, ValueError):
            return False, f"Invalid count type: {type(count).__name__}"
        
        if not (1 <= count <= 100000):
            return False, f"Building count {count} outside valid range [1-100000]"
        
        return True, "Valid building count"


class GeoProcessor:
    """Handle geospatial processing tasks"""
    
    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance in kilometers using Haversine formula
        """
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    @staticmethod
    def create_buffer_bbox(latitude: float, longitude: float, 
                          radius_km: float) -> Dict[str, float]:
        """
        Create bounding box around a point (in decimal degrees)
        Approximate: 1 degree ≈ 111 km
        """
        lat_offset = radius_km / 111
        lon_offset = (radius_km / 111) / np.cos(np.radians(latitude))
        
        return {
            'min_lat': latitude - lat_offset,
            'max_lat': latitude + lat_offset,
            'min_lon': longitude - lon_offset,
            'max_lon': longitude + lon_offset
        }
    
    @staticmethod
    def generate_grid_points(bbox: Dict[str, float], grid_size: int) -> List[Tuple[float, float]]:
        """
        Generate grid points within bounding box for analysis
        """
        lats = np.linspace(bbox['min_lat'], bbox['max_lat'], grid_size)
        lons = np.linspace(bbox['min_lon'], bbox['max_lon'], grid_size)
        
        points = [(lat, lon) for lat in lats for lon in lons]
        return points


class ConfidenceCalculator:
    """Calculate confidence scores for analyses"""
    
    @staticmethod
    def calculate_data_confidence(data_points: int, 
                                 ideal_points: int = 1000) -> float:
        """Calculate confidence based on data density"""
        confidence = min(data_points / ideal_points, 1.0)
        return round(confidence, 2)
    
    @staticmethod
    def combine_scores(scores: List[float], weights: Optional[List[float]] = None) -> float:
        """Combine multiple confidence scores with optional weights"""
        scores = np.array(scores)
        
        if weights is None:
            weights = np.ones(len(scores)) / len(scores)
        else:
            weights = np.array(weights) / np.sum(weights)
        
        combined = np.average(scores, weights=weights)
        return round(float(combined), 2)


class ReportExporter:
    """Export analysis results in multiple formats"""
    
    @staticmethod
    def to_json(result: AnalysisResult, output_path: Path) -> None:
        """Export result to JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(result.to_json())
        
        logger.info(f"JSON report exported to {output_path}")
    
    @staticmethod
    def to_csv(data: pd.DataFrame, output_path: Path) -> None:
        """Export DataFrame to CSV"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_csv(output_path, index=False)
        logger.info(f"CSV report exported to {output_path}")
    
    @staticmethod
    def to_geojson(features: List[Dict], output_path: Path) -> None:
        """Export features to GeoJSON format"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2, default=str)
        
        logger.info(f"GeoJSON exported to {output_path}")


class SeverityClassifier:
    """Classify analysis results into severity levels"""
    
    THRESHOLDS = {
        'slope': {'unfavorable': 0.3, 'moderate': 0.6, 'favorable': 1.0},
        'extraction': {'favorable': 0.5, 'moderate': 0.8, 'critical': 1.5},
        'water_table': {'favorable': 15, 'moderate': 5, 'critical': 0},
        'drainage': {'high': 5, 'medium': 2, 'low': 0},
        'soil_infiltration': {'low': 2, 'medium': 15, 'high': 50}
    }
    
    @staticmethod
    def classify(metric_type: str, value: float) -> str:
        """Classify a metric value into severity level"""
        thresholds = SeverityClassifier.THRESHOLDS.get(metric_type, {})
        
        if metric_type in ['slope', 'water_table', 'drainage', 'soil_infiltration']:
            for threshold_name, threshold_val in sorted(thresholds.items(), 
                                                       key=lambda x: x[1]):
                if metric_type == 'water_table':
                    if value >= threshold_val:
                        return threshold_name
                elif metric_type == 'extraction':
                    if value <= threshold_val:
                        return threshold_name
                else:
                    if value <= threshold_val:
                        return threshold_name
        
        return 'moderate'


def get_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()
