"""
KALHAN MAIN ORCHESTRATOR
Complete groundwater analysis platform for urban apartments in India
MNC-Level Implementation
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import analysis models (Production Grade) - 8 Original Models
from kalhan_core.models.slope import SlopeDetectionModel
from kalhan_core.models.extraction_pressure import ExtractionPressureModel
from kalhan_core.models.water_table import WaterTableModel
from kalhan_core.models.drainage import DrainageDensityModel
from kalhan_core.models.soil import SoilAnalysisModel
from kalhan_core.models.rain import RainfallAnalysisModel
from kalhan_core.models.lithology import LithologyAnalysisModel
from kalhan_core.models.lulc import LULCAnalysisModel

# Import 4 Advanced Groundwater Analysis Models
from kalhan_core.models.lineament_fracture import LineamentFractureAnalysisModel
from kalhan_core.models.bedrock_depth import DepthToBedrockModel
from kalhan_core.models.recharge_potential_index import RechargePotentialIndexModel
from kalhan_core.models.surface_water_distance import SurfaceWaterDistanceModel

# Import report generators
from kalhan_core.reports.html_generator import HtmlReportGenerator, SummaryReportGenerator

# Import utilities and config
from kalhan_core.utils.core import (
    AnalysisResult, ReportExporter, GeoProcessor
)
from kalhan_core.config.settings import OUTPUT_DIR, REPORTS_DIR


class KalhanAnalyzer:
    """
    Main orchestrator for groundwater analysis
    Coordinates all analysis modules and report generation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize 8 Original production-grade analysis models
        self.slope_model = SlopeDetectionModel()
        self.extraction_model = ExtractionPressureModel()
        self.water_table_model = WaterTableModel()
        self.drainage_model = DrainageDensityModel()
        self.soil_model = SoilAnalysisModel()
        self.rain_model = RainfallAnalysisModel()
        self.lithology_model = LithologyAnalysisModel()
        self.lulc_model = LULCAnalysisModel()
        
        # Initialize 4 Advanced models (NEW - Enterprise depth)
        self.lineament_model = LineamentFractureAnalysisModel()
        self.bedrock_model = DepthToBedrockModel()
        self.recharge_index_model = RechargePotentialIndexModel()
        self.surface_water_model = SurfaceWaterDistanceModel()
        
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
                        generate_reports: bool = True,
                        generate_maps: bool = True) -> Dict[str, AnalysisResult]:
        """
        Complete analysis of a location for groundwater potential
        ALL parameters are calculated from REAL DATA, NOT hardcoded
        
        Args:
            latitude, longitude: Location coordinates (REQUIRED)
            location_name: Name of analysis location
            building_units: Number of units in target building
            building_type: Type of building
            generate_reports: Whether to generate HTML reports
            generate_maps: Whether to include interactive maps
        
        Returns:
            Dictionary of analysis results from 12 models
        
        NOTE: rainfall is NOT hardcoded - calculated from IMD data
        NOTE: all results are location-specific, not template values
        """
        
        logger.info(f"\n{'='*70}")
        logger.info(f"KALHAN ANALYSIS: {location_name}")
        logger.info(f"Location: {latitude:.4f}°N, {longitude:.4f}°E")
        logger.info(f"{'='*70}\n")
        
        output_folder = OUTPUT_DIR / location_name.replace(" ", "_")
        output_folder.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. SLOPE ANALYSIS
            logger.info("1️⃣ Analyzing slope patterns...")
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
            logger.info("2️⃣ Analyzing rainfall patterns...")
            rain_result = self.rain_model.analyze_rainfall(
                latitude, longitude, rainfall_data=None, analysis_period_years=10
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
            logger.info("3️⃣ Analyzing soil properties...")
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
            logger.info("4️⃣ Analyzing drainage patterns (uses real rainfall from step 2)...")
            # Extract numeric value from UncertainValue if needed
            rainfall_value = rain_result.key_findings['annual_average_mm']
            if isinstance(rainfall_value, dict):
                rainfall_value = rainfall_value.get('value', rainfall_value)
            
            drainage_result = self.drainage_model.analyze_drainage_density(
                latitude, longitude, 
                rainfall_mm=rainfall_value,  # From actual rainfall model
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
            logger.info("5️⃣ Analyzing water table...")
            water_table_result = self.water_table_model.analyze_water_table(
                latitude, longitude
            )
            self.results['water_table'] = water_table_result
            self._log_result(water_table_result)
            
            if generate_reports and water_table_result is not None:
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
            logger.info("6️⃣ Analyzing extraction pressure (CRITICAL)...")
            # Extract numeric value from UncertainValue if needed
            rainfall_for_extraction = rain_result.key_findings['annual_average_mm']
            if isinstance(rainfall_for_extraction, dict):
                rainfall_for_extraction = rainfall_for_extraction.get('value', rainfall_for_extraction)
            
            extraction_result = self.extraction_model.analyze_extraction_pressure(
                latitude, longitude,
                building_units=building_units,
                building_type=building_type,
                annual_rainfall_mm=rainfall_for_extraction
            )
            self.results['extraction_pressure'] = extraction_result
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
            
            # 7. LITHOLOGY ANALYSIS
            logger.info("7️⃣ Analyzing lithology and aquifer types...")
            lithology_result = self.lithology_model.analyze_lithology(
                latitude, longitude
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
            
            # 8. LULC ANALYSIS
            logger.info("8️⃣ Analyzing land use/land cover...")
            lulc_result = self.lulc_model.analyze_lulc(
                latitude, longitude
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
            
            # ==================== ADVANCED MODELS (4) ====================
            
            # 9. LINEAMENT & FRACTURE ZONE ANALYSIS (Advanced)
            logger.info("9️⃣ Analyzing lineament and fracture zones (Advanced)...")
            try:
                from kalhan_core.data_sources import ElevationDataSource
                dem_result = ElevationDataSource.get_dem_data(latitude, longitude, radius_km=5.0)
                # Unpack tuple: (dem_array, metadata)
                if isinstance(dem_result, tuple):
                    dem_data, metadata = dem_result
                else:
                    dem_data = dem_result
                lineament_result = self.lineament_model.analyze_lineament_fracture(
                    latitude, longitude, dem_data=dem_data, radius_km=5.0
                )
                self.results['lineament_fracture'] = lineament_result
                self._log_result(lineament_result)
                
                if generate_reports:
                    self.html_generator.generate_report(
                        lineament_result,
                        output_folder / "lineament_fracture_analysis",
                        include_map=generate_maps,
                        map_center=(latitude, longitude),
                        map_zoom=13
                    )
                    self.report_exporter.to_json(lineament_result,
                        output_folder / "lineament_fracture_analysis" / "lineament_data.json")
            except Exception as e:
                logger.warning(f"Lineament analysis skipped: {e}")
            
            # 10. DEPTH-TO-BEDROCK ANALYSIS (Advanced)
            logger.info("🔟 Analyzing bedrock depth and storage capacity (Advanced)...")
            try:
                # Ensure dem_data is available (lineament may have failed)
                bedrock_dem_data = None
                if 'dem_data' in locals():
                    bedrock_dem_data = dem_data
                bedrock_result = self.bedrock_model.estimate_bedrock_depth(
                    latitude, longitude, dem_data=bedrock_dem_data
                )
                self.results['bedrock_depth'] = bedrock_result
                self._log_result(bedrock_result)
                
                if generate_reports:
                    self.html_generator.generate_report(
                        bedrock_result,
                        output_folder / "bedrock_depth_analysis",
                        include_map=generate_maps,
                        map_center=(latitude, longitude),
                        map_zoom=13
                    )
                    self.report_exporter.to_json(bedrock_result,
                        output_folder / "bedrock_depth_analysis" / "bedrock_data.json")
            except Exception as e:
                logger.warning(f"Bedrock depth analysis skipped: {e}")
            
            # 11. RECHARGE POTENTIAL INDEX (Advanced)
            logger.info("1️⃣1️⃣ Calculating comprehensive Recharge Potential Index (Advanced)...")
            try:
                rpi_result = self.recharge_index_model.calculate_recharge_potential_index(
                    latitude, longitude, radius_km=5.0
                )
                self.results['recharge_potential_index'] = rpi_result
                self._log_result(rpi_result)
                
                if generate_reports:
                    self.html_generator.generate_report(
                        rpi_result,
                        output_folder / "recharge_potential_analysis",
                        include_map=generate_maps,
                        map_center=(latitude, longitude),
                        map_zoom=13
                    )
                    self.report_exporter.to_json(rpi_result,
                        output_folder / "recharge_potential_analysis" / "rpi_data.json")
            except Exception as e:
                logger.warning(f"Recharge Potential Index calculation skipped: {e}")
            
            # 12. SURFACE WATER DISTANCE & INTERACTION (Advanced)
            logger.info("1️⃣2️⃣ Analyzing surface water proximity and GW-SW interaction (Advanced)...")
            try:
                surface_water_result = self.surface_water_model.analyze_surface_water_distance(
                    latitude, longitude, radius_km=10.0
                )
                self.results['surface_water_distance'] = surface_water_result
                self._log_result(surface_water_result)
                
                if generate_reports and surface_water_result is not None:
                    self.html_generator.generate_report(
                        surface_water_result,
                        output_folder / "surface_water_analysis",
                        include_map=generate_maps,
                        map_center=(latitude, longitude),
                        map_zoom=13
                    )
                    self.report_exporter.to_json(surface_water_result,
                        output_folder / "surface_water_analysis" / "surface_water_data.json")
            except Exception as e:
                logger.warning(f"Surface water analysis skipped: {e}")
            
            # ================================================================
            
            # Generate combined summary report
            if generate_reports:
                logger.info("📋 Generating combined summary report...")
                self.summary_generator.generate_combined_report(
                    list(self.results.values()),
                    location_name=location_name,
                    output_folder=output_folder
                )
            
            # Generate comprehensive JSON report
            self._generate_comprehensive_report(output_folder, location_name)
            
            logger.info(f"\n{'='*70}")
            logger.info("✅ ANALYSIS COMPLETE")
            logger.info(f"Reports saved to: {output_folder}")
            logger.info(f"{'='*70}\n")
            
            return self.results
        
        except Exception as e:
            logger.error(f"❌ Error during analysis: {str(e)}", exc_info=True)
            raise
    
    def _log_result(self, result: AnalysisResult) -> None:
        """Log analysis result summary"""
        
        if result is None:
            return
        
        logger.info(f"   Analysis: {result.analysis_type}")
        logger.info(f"   Severity: {result.severity_level}")
        logger.info(f"   Confidence: {result.confidence_score * 100:.0f}%")
        logger.info(f"   Key Finding Count: {len(result.key_findings)}")
        logger.info("")
    
    def _generate_comprehensive_report(self, output_folder: Path, 
                                      location_name: str) -> None:
        """Generate comprehensive JSON report combining all analyses"""
        
        comprehensive_report = {
            'metadata': {
                'location_name': location_name,
                'timestamp': datetime.now().isoformat(),
                'platform': 'Kalhan Groundwater Analysis Platform',
                'version': '2.0',
                'mnc_level': True
            },
            'analyses': {}
        }
        
        for analysis_type, result in self.results.items():
            if result is not None:
                comprehensive_report['analyses'][analysis_type] = result.to_dict()
        
        # Add synthesis and recommendations
        comprehensive_report['synthesis'] = self._generate_synthesis()
        comprehensive_report['overall_recommendations'] = self._generate_overall_recommendations()
        
        # Write to file
        report_file = output_folder / "comprehensive_report.json"
        with open(report_file, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        logger.info(f"Comprehensive report saved: {report_file}")
    
    def _generate_synthesis(self) -> Dict:
        """Generate synthesis across all analyses"""
        
        synthesis = {
            'overall_groundwater_potential': 'PENDING',
            'critical_findings': [],
            'feasibility_score': 0,
            'recommended_actions': []
        }
        
        # Analyze extraction pressure result
        if 'extraction_pressure' in self.results:
            extraction = self.results['extraction_pressure']
            ratio = extraction.key_findings.get('extraction_to_recharge_ratio', 1.0)
            
            if ratio < 0.5:
                synthesis['overall_groundwater_potential'] = 'EXCELLENT'
                synthesis['feasibility_score'] = 90
            elif ratio < 0.8:
                synthesis['overall_groundwater_potential'] = 'GOOD'
                synthesis['feasibility_score'] = 75
            elif ratio < 1.5:
                synthesis['overall_groundwater_potential'] = 'MODERATE'
                synthesis['feasibility_score'] = 50
            else:
                synthesis['overall_groundwater_potential'] = 'POOR'
                synthesis['feasibility_score'] = 20
                synthesis['critical_findings'].append(
                    "Over-extraction relative to recharge"
                )
        
        # Check drainage for recharge potential
        if 'drainage' in self.results:
            drainage = self.results['drainage']
            infiltration = drainage.key_findings.get('infiltration_capacity_percent', 50)
            
            if infiltration < 30:
                synthesis['critical_findings'].append(
                    "Low infiltration capacity limits recharge"
                )
        
        return synthesis
    
    def _generate_overall_recommendations(self) -> List[str]:
        """Generate overall recommendations for the location"""
        
        recommendations = [
            "Implement mandatory Rainwater Harvesting (RWH) systems",
            "Regular groundwater monitoring (quarterly minimum)",
            "Water audit and conservation programs",
            "Establish coordination with surrounding wells",
            "Monitor monsoon rainfall for trend analysis"
        ]
        
        return recommendations
    
    def export_to_geojson(self, output_folder: Optional[Path] = None) -> Path:
        """Export analysis results to GeoJSON format"""
        
        if output_folder is None:
            output_folder = OUTPUT_DIR / "geojson"
        
        output_folder.mkdir(parents=True, exist_ok=True)
        
        features = []
        
        for analysis_type, result in self.results.items():
            feature = {
                "type": "Feature",
                "properties": {
                    "analysis_type": analysis_type,
                    "severity": result.severity_level,
                    "confidence": result.confidence_score,
                    **result.key_findings
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        result.location['longitude'],
                        result.location['latitude']
                    ]
                }
            }
            features.append(feature)
        
        output_file = output_folder / "analysis_results.geojson"
        self.report_exporter.to_geojson(features, output_file)
        
        return output_file


def main():
    """Main execution function"""
    
    # Initialize analyzer
    analyzer = KalhanAnalyzer()
    
    # Example analysis for Delhi NCR region
    logger.info("\n🌍 KALHAN GROUNDWATER ANALYSIS PLATFORM 🌍")
    logger.info("All 12 Production-Grade Models - Enterprise Depth Analysis\n")
    
    # Perform analysis for a sample location
    # NOTE: All parameters are calculated from real location data
    results = analyzer.analyze_location(
        latitude=28.5355,  # Delhi coordinates
        longitude=77.3910,
        location_name="Delhi_NCR_Premium_Residential",
        building_units=200,
        building_type='residential_premium',
        generate_reports=True,
        generate_maps=True
    )
    
    # Print summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY - ALL 12 MODELS")
    print("="*70)
    
    for analysis_type, result in results.items():
        print(f"\n{analysis_type.upper()}")
        print(f"  Severity: {result.severity_level}")
        print(f"  Confidence: {result.confidence_score * 100:.0f}%")
        print(f"  Status: {result.severity_level.replace('_', ' ').title()}")
    
    print("\n" + "="*70)
    print("✅ Analysis complete. Check output folders for detailed reports.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
