"""
Setup configuration for Kalhan package
Enables installation and proper module importing
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kalhan",
    version="2.0.0",
    author="Kalhan Development Team",
    description="MNC-Level Groundwater Analysis Platform for Urban India",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Hydrology",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "folium>=0.14.0",
        "geopandas>=0.13.0",
        "shapely>=2.0.0",
        "rasterio>=1.3.0",
        "pyproj>=3.6.0",
        "requests>=2.31.0",
        "python-dateutil>=2.8.0",
        "openpyxl>=3.1.0",
    ],
    entry_points={
        "console_scripts": [
            "kalhan=kalhan_core.__main__:main",
        ],
    },
)
