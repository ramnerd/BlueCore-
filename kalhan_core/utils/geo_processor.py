"""
Geographic Processing Utilities
Provides smooth spatial interpolation and distance-weighted blending
Replaces hard geographic boundaries with continuous functions
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class GeoProcessor:
    """Geographic calculations and spatial blending"""
    
    EARTH_RADIUS_KM = 6371
    
    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate great-circle distance between two points using Haversine formula
        Returns distance in kilometers
        
        Args:
            lat1, lon1, lat2, lon2: Geographic coordinates in degrees
            
        Raises:
            ValueError: If coordinates are outside valid ranges
        """
        # Validate inputs
        if not (-90 <= lat1 <= 90):
            raise ValueError(f"Invalid latitude: lat1={lat1} (must be -90 to 90)")
        if not (-90 <= lat2 <= 90):
            raise ValueError(f"Invalid latitude: lat2={lat2} (must be -90 to 90)")
        if not (-180 <= lon1 <= 180):
            raise ValueError(f"Invalid longitude: lon1={lon1} (must be -180 to 180)")
        if not (-180 <= lon2 <= 180):
            raise ValueError(f"Invalid longitude: lon2={lon2} (must be -180 to 180)")
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return GeoProcessor.EARTH_RADIUS_KM * c
    
    @staticmethod
    def inverse_distance_weight(location: Tuple[float, float],
                               reference_points: Dict[str, Tuple[float, float, float]],
                               power: float = 2.0,
                               smoothing_distance_km: float = 50.0) -> Dict[str, float]:
        """
        Calculate inverse distance weights for reference points
        
        Args:
            location: (latitude, longitude) of query point
            reference_points: {name: (lat, lon, value), ...}
            power: Distance power (2.0 = quadratic decay)
            smoothing_distance_km: Distance to add before computing weights (avoids zero division)
        
        Returns:
            {name: weight, ...} normalized to sum to 1.0
            
        Raises:
            ValueError: If location is outside valid bounds
            
        Example:
            reference_points = {
                'NCR': (28.7, 77.1, 0.32),
                'Mumbai': (19.1, 72.9, 0.22),
                'Bangalore': (12.9, 77.6, 0.28)
            }
            weights = GeoProcessor.inverse_distance_weight(
                location=(25.0, 75.0),
                reference_points=reference_points,
                power=2.0
            )
            # Returns: {'NCR': 0.6, 'Mumbai': 0.25, 'Bangalore': 0.15}
        """
        # Validate input location
        lat_query, lon_query = location
        GeoProcessor.validate_india_coordinates(lat_query, lon_query)
        
        weights = {}
        total_weight = 0
        
        for name, (ref_lat, ref_lon, ref_value) in reference_points.items():
            distance_km = GeoProcessor.calculate_distance(
                lat_query, lon_query, ref_lat, ref_lon
            )
            
            # Add smoothing distance to avoid division by zero
            adjusted_distance = distance_km + smoothing_distance_km
            
            # Inverse distance weight
            weight = 1.0 / (adjusted_distance ** power)
            weights[name] = weight
            total_weight += weight
        
        # Normalize to probabilities
        normalized_weights = {
            name: weight / total_weight 
            for name, weight in weights.items()
        }
        
        return normalized_weights
    
    @staticmethod
    def blend_regional_values(location: Tuple[float, float],
                             regional_values: Dict[str, Tuple[float, float, float]],
                             power: float = 2.0) -> float:
        """
        Blend regional values using inverse distance weighting
        Creates smooth transitions instead of hard boundaries
        
        Args:
            location: (latitude, longitude) of query point
            regional_values: {region_name: (ref_lat, ref_lon, value), ...}
            power: Distance power for IDW
            
        Returns:
            Blended value (smooth interpolation)
        """
        weights = GeoProcessor.inverse_distance_weight(
            location, regional_values, power=power
        )
        
        blended_value = 0.0
        for region, weight in weights.items():
            if region in regional_values:
                _, _, value = regional_values[region]
                blended_value += weight * value
        
        return blended_value
    
    @staticmethod
    def blend_regional_dict(location: Tuple[float, float],
                           regional_dicts: Dict[str, Tuple[float, float, Dict]],
                           keys_to_blend: List[str],
                           power: float = 2.0) -> Dict:
        """
        Blend dictionaries from multiple regions
        Useful for complex regional properties
        
        Args:
            location: (latitude, longitude)
            regional_dicts: {region: (lat, lon, {property_dict}), ...}
            keys_to_blend: Which keys to interpolate
            power: IDW power
            
        Returns:
            Blended dictionary with smooth values
            
        Example:
            regional_data = {
                'NCR': (28.7, 77.1, {'infiltration': 0.32, 'conductivity': 0.15}),
                'Mumbai': (19.1, 72.9, {'infiltration': 0.22, 'conductivity': 0.08})
            }
            result = GeoProcessor.blend_regional_dict(
                location=(25.0, 75.0),
                regional_dicts=regional_data,
                keys_to_blend=['infiltration', 'conductivity']
            )
        """
        weights = {}
        total_weight = 0
        
        # Calculate weights
        for region, (ref_lat, ref_lon, _) in regional_dicts.items():
            distance_km = GeoProcessor.calculate_distance(
                location[0], location[1], ref_lat, ref_lon
            )
            adjusted_distance = distance_km + 50.0
            weight = 1.0 / (adjusted_distance ** power)
            weights[region] = weight
            total_weight += weight
        
        # Normalize weights
        normalized_weights = {
            region: w / total_weight 
            for region, w in weights.items()
        }
        
        # Blend each property
        blended = {}
        for key in keys_to_blend:
            blended_value = 0.0
            for region, weight in normalized_weights.items():
                if region in regional_dicts:
                    _, _, properties = regional_dicts[region]
                    if key in properties:
                        blended_value += weight * properties[key]
            blended[key] = blended_value
        
        return blended
    
    @staticmethod
    def validate_india_coordinates(latitude: float, longitude: float) -> None:
        """
        Validate that coordinates are within India bounds with buffer
        
        Args:
            latitude, longitude: Geographic coordinates in degrees
            
        Raises:
            ValueError: If coordinates are outside India bounds
        """
        INDIA_BOUNDS = {
            'lat_min': 8.0,
            'lat_max': 35.0,
            'lon_min': 68.0,
            'lon_max': 97.0,
            'buffer_degrees': 0.5
        }
        
        buffer = INDIA_BOUNDS['buffer_degrees']
        lat_min = INDIA_BOUNDS['lat_min'] - buffer
        lat_max = INDIA_BOUNDS['lat_max'] + buffer
        lon_min = INDIA_BOUNDS['lon_min'] - buffer
        lon_max = INDIA_BOUNDS['lon_max'] + buffer
        
        if not (lat_min <= latitude <= lat_max):
            raise ValueError(f"Latitude {latitude:.4f} outside India bounds ({lat_min} to {lat_max})")
        if not (lon_min <= longitude <= lon_max):
            raise ValueError(f"Longitude {longitude:.4f} outside India bounds ({lon_min} to {lon_max})")
    
    @staticmethod
    def get_distance_to_reference(location: Tuple[float, float],
                                 reference_points: Dict[str, Tuple[float, float]],
                                 closest_n: int = 1) -> Dict[str, float]:
        """
        Get distances to reference points
        Useful for uncertainty estimation (far from data = higher uncertainty)
        
        Args:
            location: (latitude, longitude)
            reference_points: {name: (lat, lon), ...}
            closest_n: Return distances to N closest points
            
        Returns:
            {name: distance_km, ...} sorted by distance
        """
        distances = {}
        lat_q, lon_q = location
        
        for name, (ref_lat, ref_lon) in reference_points.items():
            dist = GeoProcessor.calculate_distance(lat_q, lon_q, ref_lat, ref_lon)
            distances[name] = dist
        
        # Sort and return closest N
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])[:closest_n]
        return {name: dist for name, dist in sorted_distances}
    
    @staticmethod
    def is_location_in_india(latitude: float, longitude: float, 
                           buffer_degrees: float = 0.5) -> bool:
        """
        Check if location is within India (with buffer)
        Prevents extrapolation outside India
        """
        # Approximate India bounds
        lat_min, lat_max = 8.0, 35.0
        lon_min, lon_max = 68.0, 97.0
        
        # Add buffer to be more conservative
        lat_min -= buffer_degrees
        lat_max += buffer_degrees
        lon_min -= buffer_degrees
        lon_max += buffer_degrees
        
        in_bounds = (lat_min <= latitude <= lat_max) and (lon_min <= longitude <= lon_max)
        
        if not in_bounds:
            logger.warning(
                f"Location ({latitude}, {longitude}) outside India bounds. "
                f"Results may be unreliable."
            )
        
        return in_bounds


class UncertainValue:
    """
    Represents a value with uncertainty
    Replaces single-point estimates with ranges
    """
    
    def __init__(self, value: float, uncertainty: float = 0.0, 
                 confidence_level: float = 0.68, unit: str = ""):
        """
        Args:
            value: Central estimate
            uncertainty: Standard deviation or range
            confidence_level: 0.68 = 1-sigma, 0.95 = 2-sigma, etc.
            unit: Unit of measurement (optional)
        """
        self.value = value
        self.uncertainty = uncertainty
        self.confidence_level = confidence_level
        self.unit = unit
    
    @property
    def lower_bound(self) -> float:
        """Lower confidence bound"""
        return self.value - self.uncertainty
    
    @property
    def upper_bound(self) -> float:
        """Upper confidence bound"""
        return self.value + self.uncertainty
    
    def __repr__(self) -> str:
        if self.uncertainty > 0:
            if self.unit:
                return f"{self.value:.2f} ± {self.uncertainty:.2f} {self.unit}"
            else:
                return f"{self.value:.2f} ± {self.uncertainty:.2f}"
        else:
            if self.unit:
                return f"{self.value:.2f} {self.unit}"
            else:
                return f"{self.value:.2f}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'value': round(self.value, 4),
            'uncertainty': round(self.uncertainty, 4),
            'lower_bound': round(self.lower_bound, 4),
            'upper_bound': round(self.upper_bound, 4),
            'confidence_level': self.confidence_level,
            'unit': self.unit
        }


def estimate_distance_uncertainty(location: Tuple[float, float],
                                 reference_points: Dict[str, Tuple[float, float]],
                                 base_uncertainty: float = 0.05) -> float:
    """
    Estimate uncertainty based on distance to nearest reference point
    Locations far from data should have higher uncertainty
    
    Args:
        location: (latitude, longitude)
        reference_points: {name: (lat, lon, ...), ...}  # Can have extra values, only lat/lon used
        base_uncertainty: Uncertainty at reference point (0-1 scale)
        
    Returns:
        Uncertainty factor (0-1), typically between base_uncertainty and 0.30
    """
    # Find nearest reference point - extract only lat/lon from tuples
    min_distance = float('inf')
    for ref_tuple in reference_points.values():
        # Handle different tuple sizes: (lat, lon), (lat, lon, val), etc.
        ref_lat, ref_lon = ref_tuple[0], ref_tuple[1]
        dist = GeoProcessor.calculate_distance(
            location[0], location[1], ref_lat, ref_lon
        )
        min_distance = min(min_distance, dist)
    
    # Uncertainty increases with distance
    # At 0 km: base_uncertainty
    # At 100 km: ~1.5x base
    # At 500 km: ~2.5x base
    distance_penalty = min(min_distance / 200, 0.25)  # Max 25% additional uncertainty
    
    total_uncertainty = base_uncertainty + distance_penalty
    return min(total_uncertainty, 0.30)  # Cap at 30%
