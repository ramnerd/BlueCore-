"""
HTML Report Generator
Creates interactive, professional HTML reports with embedded maps
MNC-level visualization and presentation
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from kalhan_core.utils.core import AnalysisResult
from kalhan_core.config.settings import (
    COLOR_SCHEME, MAP_DEFAULT_ZOOM, MAP_CENTER_DEFAULT,
    MAP_TILE_PROVIDER, OUTPUT_DIR
)

logger = logging.getLogger(__name__)


class HtmlReportGenerator:
    """Generate professional HTML reports with interactive maps"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.color_scheme = COLOR_SCHEME
    
    def generate_report(self,
                       analysis_result: AnalysisResult,
                       output_folder: Optional[Path] = None,
                       include_map: bool = True,
                       map_center: Optional[tuple] = None,
                       map_zoom: int = 13) -> Path:
        """
        Generate a comprehensive HTML report
        
        Args:
            analysis_result: AnalysisResult object from analysis module
            output_folder: Where to save the report
            include_map: Whether to include interactive map
            map_center: (latitude, longitude) for map center
            map_zoom: Map zoom level
        
        Returns:
            Path to generated HTML file
        """
        
        if output_folder is None:
            output_folder = OUTPUT_DIR / analysis_result.analysis_type
        
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Set map center
        if map_center is None:
            map_center = (
                analysis_result.location['latitude'],
                analysis_result.location['longitude']
            )
        
        # Generate HTML content
        html_content = self._generate_html(
            analysis_result, include_map, map_center, map_zoom
        )
        
        # Write to file
        output_file = output_folder / f"{analysis_result.analysis_type}_report.html"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Report generated: {output_file}")
        
        return output_file
    
    def _generate_html(self,
                      result: AnalysisResult,
                      include_map: bool,
                      map_center: tuple,
                      map_zoom: int) -> str:
        """Generate complete HTML content with input validation"""
        
        # Input validation for safety
        if not isinstance(map_center, (tuple, list)) or len(map_center) != 2:
            self.logger.warning(f"Invalid map_center {map_center}, using default")
            map_center = (MAP_CENTER_DEFAULT[0], MAP_CENTER_DEFAULT[1])
        
        # Validate and sanitize latitude/longitude
        try:
            map_lat = float(map_center[0])
            map_lon = float(map_center[1])
            map_lat = max(-90, min(90, map_lat))  # Clamp to valid range
            map_lon = max(-180, min(180, map_lon))  # Clamp to valid range
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid map coordinates, using default")
            map_lat, map_lon = MAP_CENTER_DEFAULT[0], MAP_CENTER_DEFAULT[1]
        
        # Validate zoom level
        map_zoom = max(1, min(19, int(map_zoom)))
        
        severity_color = self.color_scheme.get(
            result.severity_level, self.color_scheme['neutral']
        )
        
        # Determine severity icon
        severity_icons = {
            'favorable': '✓',
            'moderate': '⚠️',
            'unfavorable': '⚠️⚠️',
            'critical': '🔴'
        }
        severity_icon = severity_icons.get(result.severity_level, '')
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kalhan Report - {result.analysis_type.replace('_', ' ').title()}</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-transform: capitalize;
        }}
        
        .location-info {{
            font-size: 1.1em;
            opacity: 0.9;
            margin: 10px 0;
        }}
        
        .severity-badge {{
            display: inline-block;
            background: {severity_color};
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: bold;
            margin-top: 15px;
            font-size: 1.1em;
        }}
        
        .main-content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 40px;
        }}
        
        @media (max-width: 900px) {{
            .main-content {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .section {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            border-left: 5px solid {severity_color};
        }}
        
        .section h2 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.5em;
            border-bottom: 2px solid {severity_color};
            padding-bottom: 10px;
        }}
        
        .findings {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 15px;
        }}
        
        .findings-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }}
        
        .finding-item {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid {severity_color};
        }}
        
        .finding-label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }}
        
        .finding-value {{
            font-size: 1.4em;
            font-weight: bold;
            color: #333;
        }}
        
        .map-container {{
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            height: 400px;
            width: 100%;
        }}
        
        #map {{
            height: 100%;
            width: 100%;
        }}
        
        .recommendations {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        
        .recommendations h3 {{
            color: #667eea;
            margin-bottom: 15px;
        }}
        
        .recommendation-item {{
            padding: 12px;
            margin-bottom: 10px;
            background: #f0f7ff;
            border-left: 4px solid #667eea;
            border-radius: 4px;
        }}
        
        .recommendation-item:before {{
            content: "→ ";
            color: #667eea;
            font-weight: bold;
        }}
        
        .metadata {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            font-size: 0.9em;
            color: #555;
        }}
        
        .metadata-item {{
            margin-bottom: 10px;
            padding: 8px;
            background: white;
            border-radius: 4px;
        }}
        
        .metadata-label {{
            font-weight: bold;
            color: #667eea;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #ddd;
            font-size: 0.9em;
        }}
        
        .confidence-score {{
            display: inline-block;
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background: conic-gradient(to right, {severity_color} 0deg {result.confidence_score * 360}deg, #ddd {result.confidence_score * 360}deg);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.2em;
            color: white;
            margin: 20px auto;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }}
        
        .confidence-label {{
            text-align: center;
            font-size: 0.9em;
            color: #666;
            margin-top: 10px;
        }}
        
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        
        .data-table th {{
            background: {severity_color};
            color: white;
            padding: 12px;
            text-align: left;
        }}
        
        .data-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }}
        
        .data-table tr:hover {{
            background: #f5f5f5;
        }}
        
        .severity-low {{ color: #27ae60; }}
        .severity-moderate {{ color: #f39c12; }}
        .severity-high {{ color: #e74c3c; }}
        .severity-critical {{ color: #c0392b; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{severity_icon} {result.analysis_type.replace('_', ' ').title()}</h1>
            <div class="location-info">
                📍 {result.location['latitude']:.4f}° N, {result.location['longitude']:.4f}° E
            </div>
            <div class="severity-badge">
                Severity: {result.severity_level.replace('_', ' ').upper()}
            </div>
        </div>
        
        <div class="main-content">
            <!-- Left Column: Findings -->
            <div>
                <div class="section">
                    <h2>📊 Key Findings</h2>
                    <div class="confidence-score" style="background: conic-gradient(to right, {severity_color} 0deg, {severity_color} {result.confidence_score * 360}deg, #ddd {result.confidence_score * 360}deg);"></div>
                    <div class="confidence-label">Confidence: {result.confidence_score * 100:.0f}%</div>
                    <div class="findings-grid">
{self._generate_findings_grid(result.key_findings)}
                    </div>
                </div>
            </div>
            
            <!-- Right Column: Map -->
            <div>
                <div class="map-container">
                    <div id="map"></div>
                </div>
            </div>
        </div>
        
        <!-- Recommendations Section -->
        <div style="padding: 0 40px;">
            <div class="section">
                <h2>💡 Recommendations</h2>
                <div class="recommendations">
{self._generate_recommendations_html(result.recommendations)}
                </div>
            </div>
        </div>
        
        <!-- Metadata Section -->
        <div style="padding: 40px;">
            <div class="section">
                <h2>📋 Methodology & Data Sources</h2>
                <div class="metadata">
                    <div class="metadata-item">
                        <span class="metadata-label">Analysis Type:</span> {result.analysis_type}
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">Methodology:</span> {result.methodology}
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">Data Sources:</span> {', '.join(result.data_sources)}
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">Report Generated:</span> {result.timestamp}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>© 2024 Kalhan - Groundwater Resource Analysis Platform</p>
            <p>MNC-Level Geospatial Analysis for Urban Water Resources</p>
            <p>For urban apartments and real estate development in India</p>
        </div>
    </div>
    
    <script>
        // Initialize map with safe data injection
        var mapConfig = {json.dumps({"lat": map_lat, "lon": map_lon, "zoom": map_zoom})};
        var map = L.map('map').setView([mapConfig.lat, mapConfig.lon], mapConfig.zoom);
        
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors',
            maxZoom: 19
        }}).addTo(map);
        
        // Add marker with safe color injection
        var markerConfig = {json.dumps({"color": severity_color, "analysisType": result.analysis_type.title()})};
        L.circleMarker([mapConfig.lat, mapConfig.lon], {{
            radius: 12,
            fillColor: markerConfig.color,
            color: '#333',
            weight: 3,
            opacity: 1,
            fillOpacity: 0.8
        }}).addTo(map).bindPopup('<strong>' + markerConfig.analysisType + '</strong><br>Analysis Location');
        
        // Make map responsive
        window.addEventListener('resize', function() {{
            map.invalidateSize();
        }});
    </script>
</body>
</html>
"""
        
        return html
    
    def _generate_findings_grid(self, findings: Dict[str, Any]) -> str:
        """Generate HTML for findings grid"""
        
        html = ""
        for key, value in list(findings.items())[:6]:  # Limit to 6 items
            # Format the label
            label = key.replace('_', ' ').title()
            
            # Format the value
            if isinstance(value, float):
                value_str = f"{value:.2f}"
            elif isinstance(value, list):
                value_str = f"{len(value)} items"
            else:
                value_str = str(value)[:40]  # Truncate long strings
            
            html += f"""
                    <div class="finding-item">
                        <div class="finding-label">{label}</div>
                        <div class="finding-value">{value_str}</div>
                    </div>
"""
        
        return html
    
    def _generate_recommendations_html(self, recommendations: List[str]) -> str:
        """Generate HTML for recommendations"""
        
        html = ""
        for rec in recommendations:
            html += f'                    <div class="recommendation-item">{rec}</div>\n'
        
        return html


class SummaryReportGenerator:
    """Generate summary reports combining multiple analyses"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_combined_report(self,
                                analysis_results: List[AnalysisResult],
                                location_name: str = "Analysis Site",
                                output_folder: Optional[Path] = None) -> Path:
        """
        Generate a combined summary report from multiple analyses
        
        Args:
            analysis_results: List of AnalysisResult objects
            location_name: Name of the location
            output_folder: Output directory
        
        Returns:
            Path to generated HTML file
        """
        
        if output_folder is None:
            output_folder = OUTPUT_DIR / "combined_reports"
        
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Generate HTML
        html_content = self._generate_html(analysis_results, location_name)
        
        # Write to file
        output_file = output_folder / f"combined_analysis_{location_name}.html"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Combined report generated: {output_file}")
        
        return output_file
    
    def _generate_html(self, results: List[AnalysisResult], location: str) -> str:
        """Generate combined HTML report"""
        
        # Build summary table
        summary_rows = ""
        for result in results:
            if result is None:
                continue
            summary_rows += f"""
            <tr>
                <td><strong>{result.analysis_type.replace('_', ' ').title()}</strong></td>
                <td>{result.severity_level.upper()}</td>
                <td>{result.confidence_score * 100:.0f}%</td>
            </tr>
"""
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Kalhan Combined Analysis - {location}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Kalhan Groundwater Analysis Summary</h1>
        <p><strong>Location:</strong> {location}</p>
        <p><strong>Analyses Performed:</strong> {len(results)}</p>
        
        <h2>Analysis Results Summary</h2>
        <table>
            <tr>
                <th>Analysis Type</th>
                <th>Severity Level</th>
                <th>Confidence</th>
            </tr>
            {summary_rows}
        </table>
        
        <div class="footer">
            <p>© 2024 Kalhan Platform | MNC-Level Geospatial Analysis</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html
