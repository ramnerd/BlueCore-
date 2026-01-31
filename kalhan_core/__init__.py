"""
Package initialization file for Kalhan Core
"""

__version__ = "2.0.0"
__description__ = "Kalhan - MNC-Level Groundwater Analysis for Urban Apartments in India"

# v2.0 models are imported directly from __main___new.py
# Old v1.0 model imports commented out - using v2.0 implementation instead
# from kalhan_core.models.slope import SlopeDetectionModel
# from kalhan_core.models.extraction_pressure import ExtractionPressureModel
# from kalhan_core.models.water_table import WaterTableModel
# from kalhan_core.models.drainage_density import DrainageDensityModel
# from kalhan_core.models.soil import SoilAnalysisModel
# from kalhan_core.models.rain import RainfallAnalysisModel

from kalhan_core.reports.html_generator import HtmlReportGenerator, SummaryReportGenerator

from kalhan_core.utils.core import (
    AnalysisResult, GeoProcessor, DataValidator, ConfidenceCalculator,
    ReportExporter, SeverityClassifier
)

__all__ = [
    'SlopeDetectionModel',
    'ExtractionPressureModel',
    'WaterTableModel',
    'DrainageDensityModel',
    'SoilAnalysisModel',
    'RainfallAnalysisModel',
    'HtmlReportGenerator',
    'SummaryReportGenerator',
    'AnalysisResult',
    'GeoProcessor',
    'DataValidator',
    'ConfidenceCalculator',
    'ReportExporter',
    'SeverityClassifier'
]
