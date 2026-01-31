#!/usr/bin/env python3
"""
Quick setup script to configure Google Earth Engine Project ID
Run this script to interactively set up your GEE project ID
"""

import os
import sys
from pathlib import Path

def main():
    print("\n" + "="*80)
    print("KALHAN - Google Earth Engine Project ID Setup".center(80))
    print("="*80 + "\n")
    
    # Get the project ID from user
    project_id = input("Enter your Google Earth Engine Project ID: ").strip()
    
    if not project_id or project_id == '':
        print("❌ Project ID cannot be empty!")
        sys.exit(1)
    
    if project_id == 'your-gee-project-id':
        print("❌ Please enter your actual GEE project ID, not the placeholder!")
        sys.exit(1)
    
    # Option 1: Set environment variable
    print("\n" + "-"*80)
    print("OPTION 1: Set environment variable (for this session)")
    print("-"*80)
    print(f"PowerShell: $env:GEE_PROJECT_ID = '{project_id}'")
    print(f"Command Prompt: set GEE_PROJECT_ID={project_id}")
    
    set_env = input("\nSet environment variable now? (y/n): ").lower().strip()
    if set_env == 'y':
        os.environ['GEE_PROJECT_ID'] = project_id
        print(f"✓ Environment variable set: GEE_PROJECT_ID = {project_id}")
    
    # Option 2: Edit settings file
    print("\n" + "-"*80)
    print("OPTION 2: Update settings.py (permanent)")
    print("-"*80)
    
    settings_file = Path(__file__).parent / 'kalhan_core' / 'config' / 'settings.py'
    
    if settings_file.exists():
        update_settings = input(f"Update {settings_file.name}? (y/n): ").lower().strip()
        
        if update_settings == 'y':
            try:
                with open(settings_file, 'r') as f:
                    content = f.read()
                
                # Replace the placeholder
                old_line = "GEE_PROJECT_ID = os.getenv('GEE_PROJECT_ID', 'your-gee-project-id')"
                new_line = f"GEE_PROJECT_ID = os.getenv('GEE_PROJECT_ID', '{project_id}')"
                
                if old_line in content:
                    content = content.replace(old_line, new_line)
                    
                    with open(settings_file, 'w') as f:
                        f.write(content)
                    
                    print(f"✓ Settings file updated with project ID: {project_id}")
                else:
                    print("⚠️  Could not find the placeholder line in settings.py")
                    print("   Please manually edit the file and replace 'your-gee-project-id'")
            
            except Exception as e:
                print(f"❌ Error updating settings file: {e}")
                print("   Please manually edit kalhan_core/config/settings.py")
    
    # Summary
    print("\n" + "="*80)
    print("SETUP COMPLETE!".center(80))
    print("="*80)
    print(f"\nYour GEE Project ID: {project_id}")
    print("\nNext steps:")
    print("1. Run your test: python test_kamakoti_peetha.py")
    print("2. If authentication fails, run: python -c \"import ee; ee.Authenticate()\"")
    print("3. Check the output for 'Google Earth Engine initialized' message\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
