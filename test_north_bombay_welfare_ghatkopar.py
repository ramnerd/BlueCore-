"""
TEST FILE: NORTH BOMBAY WELFARE SOCIETY SCHOOL - GHATKOPAR
Groundwater analysis for institutional facility in Ghatkopar, Mumbai
Location: Ghatkopar, Maharashtra (19.0896°N, 72.9250°E)

This location is significant for:
- Educational institution water management in Western India
- Groundwater conditions in high-density metropolitan area (Mumbai)
- Multi-use water demand (school facilities, residential areas)
- Urban groundwater sustainability and extraction analysis
- Aquifer characteristics in sedimentary and basaltic terrain
"""

import logging
from kalhan_core.__main__ import KalhanAnalyzer

# Configure logging for test
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

def test_north_bombay_welfare_ghatkopar():
    """
    Comprehensive groundwater analysis for North Bombay Welfare Society School, Ghatkopar
    Coordinates: 19.0896°N, 72.9250°E (Ghatkopar, Mumbai, Maharashtra)
    """
    
    # Initialize analyzer
    analyzer = KalhanAnalyzer()
    
    # Define location parameters
    latitude = 19.0896
    longitude = 72.9250
    location_name = "North_Bombay_Welfare_Society_School_Ghatkopar_Mumbai"
    
    print("\n" + "="*80)
    print(f"KALHAN ANALYSIS: North Bombay Welfare Society School, Ghatkopar, Mumbai")
    print(f"Location: {latitude}°N, {longitude}°E")
    print(f"Region: Ghatkopar, Mumbai - Urban institutional facility")
    print("="*80 + "\n")
    
    try:
        # Perform comprehensive analysis
        results = analyzer.analyze_location(
            latitude=latitude,
            longitude=longitude,
            location_name=location_name,
            building_units=1,  # Single institutional facility
            building_type='school',  # Educational institution
            generate_reports=True,
            generate_maps=True
        )
        
        # Display results summary
        print("\n" + "="*80)
        print("ANALYSIS RESULTS SUMMARY")
        print("="*80)
        
        for analysis_type, result in results.items():
            if result is None:
                print(f"\n[SKIP] {analysis_type}: No data available (API unavailable)")
            else:
                print(f"\n[OK] {analysis_type}: SUCCESS")
                print(f"  Confidence: {result.confidence_score if hasattr(result, 'confidence_score') else 'N/A'}")
                print(f"  Severity: {result.severity_level if hasattr(result, 'severity_level') else 'N/A'}")
                
                if hasattr(result, 'key_findings') and result.key_findings:
                    print(f"  Key Findings: {list(result.key_findings.keys())[:3]}")
        
        print("\n" + "="*80)
        print("Analysis completed successfully!")
        print("Reports generated in: kalhan_outputs/reports/North_Bombay_Welfare_Society_School_Ghatkopar_Mumbai/")
        print("="*80 + "\n")
        
        return results
        
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("\n" + "="*80)
    print("KALHAN GROUNDWATER ANALYSIS - NORTH BOMBAY WELFARE GHATKOPAR TEST".center(80))
    print("="*80 + "\n")
    
    results = test_north_bombay_welfare_ghatkopar()
    
    if results:
        print("\n[PASS] Test completed successfully")
        print("Results stored in: kalhan_outputs/reports/")
    else:
        print("\n[FAIL] Test failed")
        exit(1)
