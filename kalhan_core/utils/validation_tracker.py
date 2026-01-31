"""
MODEL VALIDATION & OUTCOME TRACKING
Stores drilling outcomes and calculates model accuracy
Enables continuous model calibration based on field data
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DrillingOutcome:
    """Record of actual drilling result"""
    location_id: str
    latitude: float
    longitude: float
    prediction_date: datetime
    outcome_date: datetime
    
    # Model predictions (before drilling)
    predicted_depth_m: float
    predicted_yield_lpd: Optional[float]
    predicted_success: bool
    prediction_confidence: float
    
    # Actual drilling results
    drilling_depth_m: float
    water_found: bool
    actual_yield_lpd: Optional[float]
    notes: str = ""
    contractor_name: str = ""
    
    # Outcomes
    depth_error_m: float = 0.0
    depth_error_percent: float = 0.0
    success_prediction_correct: bool = False
    yield_error_percent: float = 0.0
    
    def calculate_errors(self):
        """Calculate prediction errors"""
        self.depth_error_m = abs(self.drilling_depth_m - self.predicted_depth_m)
        if self.predicted_depth_m > 0:
            self.depth_error_percent = (self.depth_error_m / self.predicted_depth_m) * 100
        self.success_prediction_correct = (self.water_found == self.predicted_success)
        if self.predicted_yield_lpd and self.actual_yield_lpd:
            self.yield_error_percent = abs(
                (self.actual_yield_lpd - self.predicted_yield_lpd) / self.predicted_yield_lpd
            ) * 100


class ValidationDatabase:
    """SQLite database for tracking predictions and outcomes"""
    
    def __init__(self, db_path: str = "model_validation.db"):
        """Initialize validation database"""
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Create tables if not exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                location_id TEXT UNIQUE,
                latitude REAL,
                longitude REAL,
                prediction_date TIMESTAMP,
                predicted_depth_m REAL,
                predicted_yield_lpd REAL,
                predicted_success INTEGER,
                prediction_confidence REAL,
                model_version TEXT,
                analysis_json TEXT
            )
        ''')
        
        # Outcomes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY,
                prediction_id INTEGER,
                outcome_date TIMESTAMP,
                drilling_depth_m REAL,
                water_found INTEGER,
                actual_yield_lpd REAL,
                notes TEXT,
                contractor_name TEXT,
                FOREIGN KEY(prediction_id) REFERENCES predictions(id)
            )
        ''')
        
        # Error analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_analysis (
                id INTEGER PRIMARY KEY,
                prediction_id INTEGER,
                depth_error_m REAL,
                depth_error_percent REAL,
                success_correct INTEGER,
                yield_error_percent REAL,
                calculated_date TIMESTAMP,
                FOREIGN KEY(prediction_id) REFERENCES predictions(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Initialized validation database: {self.db_path}")
    
    def record_prediction(self, location_id: str, latitude: float, longitude: float,
                         predicted_depth_m: float, predicted_yield_lpd: Optional[float],
                         predicted_success: bool, prediction_confidence: float,
                         model_version: str, analysis_json: str) -> int:
        """Record model prediction before drilling
        
        Args:
            location_id: Unique location identifier (non-empty string)
            latitude, longitude: Geographic coordinates (must be in India bounds)
            predicted_depth_m: Predicted depth in meters (0-1000m range)
            predicted_yield_lpd: Predicted yield in liters per day (optional, 0-500000 lpd)
            predicted_success: Whether water found is predicted
            prediction_confidence: Confidence score 0.0-1.0
            model_version: Model version string
            analysis_json: JSON string with analysis details
            
        Returns:
            Inserted prediction ID
            
        Raises:
            ValueError: If any inputs fail validation
        """
        # INPUT VALIDATION
        from kalhan_core.utils.geo_processor import GeoProcessor
        
        if not isinstance(location_id, str) or not location_id.strip():
            raise ValueError("Invalid location_id: must be non-empty string")
        
        try:
            GeoProcessor.validate_india_coordinates(latitude, longitude)
        except ValueError as e:
            raise ValueError(f"Invalid coordinates: {e}")
        
        if not isinstance(predicted_depth_m, (int, float)):
            raise ValueError(f"Depth must be numeric: {predicted_depth_m}")
        if predicted_depth_m < 0:
            raise ValueError(f"Depth cannot be negative: {predicted_depth_m}")
        if predicted_depth_m > 1000:
            raise ValueError(f"Depth unrealistic (>1000m): {predicted_depth_m}")
        
        if not (0 <= prediction_confidence <= 1.0):
            raise ValueError(f"Confidence must be 0-1: {prediction_confidence}")
        
        if predicted_yield_lpd is not None:
            if not isinstance(predicted_yield_lpd, (int, float)):
                raise ValueError(f"Yield must be numeric: {predicted_yield_lpd}")
            if predicted_yield_lpd < 0:
                raise ValueError(f"Yield cannot be negative: {predicted_yield_lpd}")
            if predicted_yield_lpd > 500000:
                raise ValueError(f"Yield unrealistic (>500000 lpd): {predicted_yield_lpd}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO predictions 
                (location_id, latitude, longitude, prediction_date, predicted_depth_m,
                 predicted_yield_lpd, predicted_success, prediction_confidence,
                 model_version, analysis_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                location_id, latitude, longitude, datetime.now(),
                predicted_depth_m, predicted_yield_lpd, int(predicted_success),
                prediction_confidence, model_version, analysis_json
            ))
            
            prediction_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Recorded prediction {prediction_id} for {location_id}")
            return prediction_id
            
        except Exception as e:
            logger.error(f"Failed to record prediction: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def record_outcome(self, prediction_id: int, drilling_depth_m: float,
                      water_found: bool, actual_yield_lpd: Optional[float] = None,
                      notes: str = "", contractor_name: str = "") -> bool:
        """Record actual drilling outcome"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO outcomes
                (prediction_id, outcome_date, drilling_depth_m, water_found, 
                 actual_yield_lpd, notes, contractor_name)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_id, datetime.now(), drilling_depth_m,
                int(water_found), actual_yield_lpd, notes, contractor_name
            ))
            
            # Calculate and store errors
            prediction = cursor.execute(
                'SELECT predicted_depth_m, predicted_yield_lpd, predicted_success FROM predictions WHERE id = ?',
                (prediction_id,)
            ).fetchone()
            
            if prediction:
                pred_depth, pred_yield, pred_success = prediction
                depth_error = abs(drilling_depth_m - pred_depth)
                depth_error_pct = (depth_error / pred_depth * 100) if pred_depth > 0 else 0
                success_correct = int((water_found == bool(pred_success)))
                
                yield_error_pct = 0
                if pred_yield and actual_yield_lpd:
                    yield_error_pct = abs((actual_yield_lpd - pred_yield) / pred_yield) * 100
                
                cursor.execute('''
                    INSERT INTO error_analysis
                    (prediction_id, depth_error_m, depth_error_percent, success_correct,
                     yield_error_percent, calculated_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    prediction_id, depth_error, depth_error_pct, success_correct,
                    yield_error_pct, datetime.now()
                ))
            
            conn.commit()
            logger.info(f"Recorded outcome for prediction {prediction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record outcome: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_accuracy_metrics(self) -> Dict:
        """Calculate overall model accuracy metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get all error records
            errors = cursor.execute('''
                SELECT depth_error_m, depth_error_percent, success_correct, yield_error_percent
                FROM error_analysis
            ''').fetchall()
            
            if not errors:
                return {'status': 'insufficient_data', 'samples': 0}
            
            depth_errors = [e[0] for e in errors]
            depth_error_pcts = [e[1] for e in errors]
            success_correct = [e[2] for e in errors]
            yield_errors = [e[3] for e in errors if e[3] > 0]
            
            metrics = {
                'total_outcomes': len(errors),
                'depth_prediction': {
                    'mae_m': round(np.mean(depth_errors), 2),
                    'rmse_m': round(np.sqrt(np.mean(np.array(depth_errors)**2)), 2),
                    'median_error_m': round(np.median(depth_errors), 2),
                    'mean_error_percent': round(np.mean(depth_error_pcts), 1),
                    'std_dev_m': round(np.std(depth_errors), 2)
                },
                'success_prediction': {
                    'accuracy_percent': round((sum(success_correct) / len(success_correct)) * 100, 1),
                    'correct_predictions': sum(success_correct),
                    'total_predictions': len(success_correct)
                }
            }
            
            if yield_errors:
                metrics['yield_prediction'] = {
                    'mae_percent': round(np.mean(yield_errors), 1),
                    'median_error_percent': round(np.median(yield_errors), 1),
                    'samples': len(yield_errors)
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate accuracy: {e}")
            return {'error': str(e)}
        finally:
            conn.close()
    
    def get_calibration_suggestions(self) -> Dict:
        """Suggest model parameter adjustments based on outcomes"""
        metrics = self.get_accuracy_metrics()
        
        if 'error' in metrics or metrics.get('status') == 'insufficient_data':
            return {'status': 'insufficient_data', 'reason': 'Need at least 5-10 outcomes for calibration'}
        
        suggestions = {
            'total_outcomes_analyzed': metrics['total_outcomes'],
            'adjustments': []
        }
        
        # Depth calibration
        depth_mae = metrics['depth_prediction']['mae_m']
        if depth_mae > 5:
            suggestions['adjustments'].append({
                'parameter': 'water_table_depth_baseline',
                'direction': 'increase' if depth_mae > 0 else 'decrease',
                'magnitude_m': round(depth_mae / 2, 1),
                'reason': f'Consistent depth error of {depth_mae:.1f}m'
            })
        
        # Success accuracy
        success_accuracy = metrics['success_prediction']['accuracy_percent']
        if success_accuracy < 70:
            suggestions['adjustments'].append({
                'parameter': 'success_prediction_thresholds',
                'direction': 'review',
                'current_accuracy_percent': success_accuracy,
                'reason': 'Success prediction below 70% accuracy'
            })
        
        return suggestions
    
    def get_performance_by_region(self) -> Dict[str, Dict]:
        """Analyze model performance by geographic region using GeoProcessor distance blending"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            results = cursor.execute('''
                SELECT p.latitude, p.longitude, e.depth_error_m, e.success_correct
                FROM predictions p
                JOIN error_analysis e ON p.id = e.prediction_id
            ''').fetchall()
            
            if not results:
                return {}
            
            # Use major cities as region centers (REPLACES hard lat/lon boxes)
            from kalhan_core.utils.geo_processor import GeoProcessor
            
            region_centers = {
                'NCR': (28.7, 77.1),
                'Bangalore': (12.9, 77.6),
                'Mumbai': (19.1, 72.9),
                'Chennai': (13.1, 80.3),
                'Hyderabad': (17.3, 78.5),
                'Kolkata': (22.5, 88.4),
                'Pune': (18.5, 73.9),
                'Ahmedabad': (23.0, 72.6)
            }
            
            # Group results by nearest region center (within 150km)
            regions = {region: {'samples': [], 'errors': [], 'successes': []} 
                      for region in region_centers.keys()}
            
            # Assign each result to nearest region center
            for lat, lon, error, success in results:
                min_dist = float('inf')
                nearest_region = None
                
                for region, (center_lat, center_lon) in region_centers.items():
                    dist = GeoProcessor.calculate_distance(lat, lon, center_lat, center_lon)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_region = region
                
                # Only assign if within 150km of center (otherwise unclassified)
                if min_dist < 150 and nearest_region:
                    regions[nearest_region]['samples'].append((lat, lon))
                    regions[nearest_region]['errors'].append(error)
                    regions[nearest_region]['successes'].append(success)
            
            # Calculate statistics for each region
            performance = {}
            for region, data in regions.items():
                if len(data['samples']) > 0:
                    performance[region] = {
                        'samples': len(data['samples']),
                        'mae_m': round(np.mean(data['errors']), 2),
                        'success_rate_percent': round((sum(data['successes']) / len(data['successes'])) * 100, 1)
                    }
            
            return performance
            
        except Exception as e:
            logger.error(f"Failed to calculate regional performance: {e}")
            return {}
        finally:
            conn.close()
    
    def export_report(self, output_file: str = "model_validation_report.json"):
        """Export validation report"""
        report = {
            'generated': datetime.now().isoformat(),
            'accuracy_metrics': self.get_accuracy_metrics(),
            'calibration_suggestions': self.get_calibration_suggestions(),
            'regional_performance': self.get_performance_by_region()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report exported to {output_file}")
        return report
