"""
KALHAN MAIN ORCHESTRATOR - VERSION 2.0 (REDESIGNED)
Real geospatial analysis platform with actual computation
No hardcoding, all location-specific calculation
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import NEW analysis models
from kalhan_core.models.slope_new import SlopeDetectionModel
from kalhan_core.models.extraction_pressure_new import ExtractionPressureModel
from kalhan_core.models.water_table_new import WaterTableModel
from kalhan_core.models.drainage_density_new import DrainageDensityModel
from kalhan_core.models.soil_new import SoilAnalysisModel
from kalhan_core.models.rain_new import RainfallAnalysisModel
from kalhan_core.models.lulc_analysis_new import LULCAnalysisModel
from kalhan_core.models.lithology_analysis_new import LithologyAnalysisModel

# Import report generators
from kalhan_core.reports.html_generator import HtmlReportGenerator, SummaryReportGenerator

# Import utilities
from kalhan_core.utils.core import (
    AnalysisResult, ReportExporter, GeoProcessor
)
from kalhan_core.config.settings import OUTPUT_DIR, REPORTS_DIR


class KalhanAnalyzer:
    """
    Main orchestrator for REAL groundwater analysis
    All computations are location-specific, no hardcoded values
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize REDESIGNED analysis models
        self.slope_model = SlopeDetectionModel()
        self.extraction_model = ExtractionPressureModel()
        self.water_table_model = WaterTableModel()
        self.drainage_model = DrainageDensityModel()
        self.soil_model = SoilAnalysisModel()
        self.rain_model = RainfallAnalysisModel()
        self.lulc_model = LULCAnalysisModel()
        self.lithology_model = LithologyAnalysisModel()
        
        # Initialize report generators
        self.html_generator = HtmlReportGenerator()
        self.summary_generator = SummaryReportGenerator()
        self.report_exporter = ReportExporter()
        
        self.results = {}
    
    def analyze_location(self,
                        latitude: float,
                        longitude: float,
                        location_name: str = "Analysis Site",
                        building_units: int = 200,
                        building_type: str = 'residential_standard',
                        annual_rainfall_mm: float = None,
                        generate_reports: bool = True,
                        generate_maps: bool = True) -> Dict[str, AnalysisResult]:
        """
        Complete REAL analysis of a location for groundwater potential
        
        Args:
            latitude, longitude: Location coordinates
            location_name: Name of analysis location
            building_units: Number of units in target building
            building_type: Type of building
            annual_rainfall_mm: Annual rainfall override (uses computed if not provided)
            generate_reports: Whether to generate HTML reports
            generate_maps: Whether to include interactive maps
        
        Returns:
            Dictionary of analysis results
        """
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🌍 KALHAN REAL ANALYSIS: {location_name}")
        logger.info(f"Location: {latitude:.4f}°N, {longitude:.4f}°E")
        logger.info(f"Version 2.0 - REAL Geospatial Computation")
        logger.info(f"{'='*70}\n")
        
        output_folder = OUTPUT_DIR / location_name.replace(" ", "_")
        output_folder.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. SLOPE ANALYSIS
            logger.info("1️⃣ SLOPE ANALYSIS - Computing actual topographic slopes...")
            slope_result = self.slope_model.detect_slopes(
                latitude, longitude, radius_km=2.0
            )
            self.results['slope'] = slope_result
            self._log_result(slope_result)
            
            if generate_reports:
                self.html_generator.generate_report(
                    slope_result,
                    output_folder / "slope_analysis",
                    include_map=generate_maps,
                    map_center=(latitude, longitude),
                    map_zoom=13
                )
                self.report_exporter.to_json(slope_result, 
                    output_folder / "slope_analysis" / "slope_data.json")
            
            # 2. RAINFALL ANALYSIS
            logger.info("2️⃣ RAINFALL ANALYSIS - Computing location-based rainfall patterns...")
            rain_result = self.rain_model.analyze_rainfall(
                latitude, longitude, 
                annual_rainfall_override_mm=annual_rainfall_mm,
                analysis_period_years=10
            )
            self.results['rainfall'] = rain_result
            self._log_result(rain_result)
            
            if generate_reports:
                self.html_generator.generate_report(
                    rain_result,
                    output_folder / "rainfall_analysis",
                    include_map=generate_maps,
                    map_center=(latitude, longitude),
                    map_zoom=13
                )
                self.report_exporter.to_json(rain_result,
                    output_folder / "rainfall_analysis" / "rainfall_data.json")
            
            # 3. SOIL ANALYSIS
            logger.info("3️⃣ SOIL ANALYSIS - Estimating soil properties by geography...")
            soil_result = self.soil_model.analyze_soil(
                latitude, longitude
            )
            self.results['soil'] = soil_result
            self._log_result(soil_result)
            
            if generate_reports:
                self.html_generator.generate_report(
                    soil_result,
                    output_folder / "soil_analysis",
                    include_map=generate_maps,
                    map_center=(latitude, longitude),
                    map_zoom=13
                )
                self.report_exporter.to_json(soil_result,
                    output_folder / "soil_analysis" / "soil_data.json")
            
            # 4. DRAINAGE DENSITY ANALYSIS
            logger.info("4️⃣ DRAINAGE ANALYSIS - Computing runoff from topography and rainfall...")
            
            # Use computed rainfall if not overridden
            rainfall_for_drainage = annual_rainfall_mm or \
                rain_result.key_findings.get('annual_average_mm', 800)
            
            drainage_result = self.drainage_model.analyze_drainage_density(
                latitude, longitude, 
                rainfall_mm=rainfall_for_drainage,
                analysis_radius_km=2.0
            )
            self.results['drainage'] = drainage_result
            self._log_result(drainage_result)
            
            if generate_reports:
                self.html_generator.generate_report(
                    drainage_result,
                    output_folder / "drainage_analysis",
                    include_map=generate_maps,
                    map_center=(latitude, longitude),
                    map_zoom=13
                )
                self.report_exporter.to_json(drainage_result,
                    output_folder / "drainage_analysis" / "drainage_data.json")
            
            # 5. WATER TABLE ANALYSIS
            logger.info("5️⃣ WATER TABLE ANALYSIS - Estimating groundwater depth from CGWB data...")
            water_table_result = self.water_table_model.analyze_water_table(
                latitude, longitude
            )
            self.results['water_table'] = water_table_result
            self._log_result(water_table_result)
            
            if generate_reports:
                self.html_generator.generate_report(
                    water_table_result,
                    output_folder / "water_table_analysis",
                    include_map=generate_maps,
                    map_center=(latitude, longitude),
                    map_zoom=13
                )
                self.report_exporter.to_json(water_table_result,
                    output_folder / "water_table_analysis" / "water_table_data.json")
            
            # 6. EXTRACTION PRESSURE ANALYSIS
            logger.info("6️⃣ EXTRACTION PRESSURE ANALYSIS - Computing sustainable extraction limits...")
            
            # Extract computed rainfall for recharge calculation
            rainfall_mm = annual_rainfall_mm or \
                rain_result.key_findings.get('annual_average_mm', 800)
            
            extraction_result = self.extraction_model.analyze_extraction_pressure(
                latitude, longitude,
                building_units=building_units,
                building_type=building_type,
                annual_rainfall_mm=rainfall_mm,
                area_km2=2.0
            )
            self.results['extraction'] = extraction_result
            self._log_result(extraction_result)
            
            if generate_reports:
                self.html_generator.generate_report(
                    extraction_result,
                    output_folder / "extraction_pressure_analysis",
                    include_map=generate_maps,
                    map_center=(latitude, longitude),
                    map_zoom=13
                )
                self.report_exporter.to_json(extraction_result,
                    output_folder / "extraction_pressure_analysis" / "extraction_data.json")
            
            # 7. LULC ANALYSIS
            logger.info("7️⃣ LULC ANALYSIS - Analyzing Land Use/Land Cover impacts...")
            
            lulc_result = self.lulc_model.analyze_lulc(
                latitude, longitude,
                annual_rainfall_mm=rainfall_mm,
                slope_degrees=slope_result.key_findings.get('surface_slope_degrees', 5.0)
            )
            self.results['lulc'] = lulc_result
            self._log_result(lulc_result)
            
            if generate_reports:
                self.html_generator.generate_report(
                    lulc_result,
                    output_folder / "lulc_analysis",
                    include_map=generate_maps,
                    map_center=(latitude, longitude),
                    map_zoom=13
                )
                self.report_exporter.to_json(lulc_result,
                    output_folder / "lulc_analysis" / "lulc_data.json")
            
            # 8. LITHOLOGY ANALYSIS
            logger.info("8️⃣ LITHOLOGY ANALYSIS - Analyzing geological formations...")
            
            # Extract water table depth for lithology analysis context
            water_depth = water_table_result.key_findings.get('water_table_depth_m', 20.0)
            
            lithology_result = self.lithology_model.analyze_lithology(
                latitude, longitude,
                water_table_depth_m=water_depth
            )
            self.results['lithology'] = lithology_result
            self._log_result(lithology_result)
            
            if generate_reports:
                self.html_generator.generate_report(
                    lithology_result,
                    output_folder / "lithology_analysis",
                    include_map=generate_maps,
                    map_center=(latitude, longitude),
                    map_zoom=13
                )
                self.report_exporter.to_json(lithology_result,
                    output_folder / "lithology_analysis" / "lithology_data.json")
            
            # 9. GENERATE COMPREHENSIVE REPORT
            logger.info("\n9️⃣ GENERATING COMPREHENSIVE SUMMARY REPORT...")
            comprehensive_data = self._create_comprehensive_report(
                location_name, latitude, longitude, building_units, building_type
            )
            
            if generate_reports:
                # Export comprehensive report as JSON (dict, not AnalysisResult)
                output_path = output_folder / "comprehensive_report.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(comprehensive_data, f, indent=2, default=str)
                logger.info(f"JSON report exported to {output_path}")
                
                self._generate_combined_html_report(
                    comprehensive_data, output_folder
                )
            
            logger.info(f"\n✅ ANALYSIS COMPLETE FOR {location_name}")
            logger.info(f"📁 Output: {output_folder}\n")
            
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}", exc_info=True)
            raise
    
    def _create_comprehensive_report(self, location_name: str,
                                    latitude: float, longitude: float,
                                    building_units: int,
                                    building_type: str) -> Dict:
        """
        Create comprehensive report combining all analyses
        """
        comprehensive_data = {
            'analysis_name': location_name,
            'location': {'latitude': latitude, 'longitude': longitude},
            'timestamp': datetime.now().isoformat(),
            'building_info': {
                'units': building_units,
                'type': building_type
            },
            'analysis_results': {
                module_name: result.to_dict()
                for module_name, result in self.results.items()
            },
            'methodology': 'Real geospatial computation - Version 2.0'
        }
        # Add overall assessment details
        assessment = self._compute_overall_assessment()
        comprehensive_data.update(assessment)
        return comprehensive_data
    
    def _compute_overall_assessment(self) -> Dict:
        """
        Compute overall assessment from all modules
        """
        severity_scores = {
            'favorable': 1,
            'moderate': 2,
            'unfavorable': 3,
            'critical': 4
        }
        
        total_severity = sum(
            severity_scores.get(r.severity_level, 2)
            for r in self.results.values()
        )
        
        avg_severity = total_severity / len(self.results)
        avg_confidence = np.mean([r.confidence_score for r in self.results.values()])
        
        if avg_severity < 1.5:
            overall = 'FAVORABLE - Good groundwater potential'
        elif avg_severity < 2.5:
            overall = 'MODERATE - Requires careful management'
        elif avg_severity < 3.5:
            overall = 'UNFAVORABLE - Significant challenges'
        else:
            overall = 'CRITICAL - Urgent water security issues'
        
        return {
            'overall_assessment': overall,
            'average_severity_score': round(avg_severity, 2),
            'average_confidence': round(avg_confidence, 2),
            'recommendation_priority': 'HIGH' if avg_severity > 2.5 else 'MEDIUM' if avg_severity > 1.5 else 'LOW'
        }
    
    def _generate_combined_html_report(self, comprehensive_data: Dict, output_folder: Path):
        """Generate combined HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>KALHAN - Groundwater Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .header {{ background-color: #1a73e8; color: white; padding: 20px; border-radius: 5px; }}
                .summary {{ background-color: white; padding: 15px; margin: 10px 0; border-left: 4px solid #1a73e8; }}
                .critical {{ border-left-color: #d32f2f; }}
                .favorable {{ border-left-color: #388e3c; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>KALHAN Groundwater Analysis Platform v2.0</h1>
                <p>Real Geospatial Computation - No Hardcoding</p>
            </div>
            
            <h2>{comprehensive_data['analysis_name']}</h2>
            <p>Location: {comprehensive_data['location']['latitude']:.4f}°N, {comprehensive_data['location']['longitude']:.4f}°E</p>
            
            <div class="summary {comprehensive_data['overall_assessment'].split()[0].lower()}">
                <h3>Overall Assessment</h3>
                <p><strong>{comprehensive_data['overall_assessment']}</strong></p>
                <p>Average Confidence: {comprehensive_data['average_confidence']*100:.0f}%</p>
                <p>Priority Level: {comprehensive_data['recommendation_priority']}</p>
            </div>
            
            <h3>Analysis Summary</h3>
            <table>
                <tr>
                    <th>Analysis Module</th>
                    <th>Severity</th>
                    <th>Confidence</th>
                </tr>
        """
        
        for module_name, result_dict in comprehensive_data['analysis_results'].items():
            html_content += f"""
                <tr>
                    <td>{module_name.upper()}</td>
                    <td>{result_dict['severity_level'].upper()}</td>
                    <td>{result_dict['confidence_score']*100:.0f}%</td>
                </tr>
            """
        
        html_content += """
            </table>
            <p><em>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</em></p>
        </body>
        </html>
        """
        
        report_path = output_folder / "combined_analysis_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated combined report: {report_path}")
    
    def _log_result(self, result: AnalysisResult):
        """Log analysis result"""
        logger.info(f"   ✓ {result.analysis_type}: {result.severity_level.upper()} "
                   f"(confidence: {result.confidence_score*100:.0f}%)")


import numpy as np
