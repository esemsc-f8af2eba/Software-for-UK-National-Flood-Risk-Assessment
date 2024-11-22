#!/usr/bin/env python

from setuptools import setup

# Modify, update, and improve this file as necessary.

setup(
    name="Flood Tool",
    version="0.1",
    description="Flood Risk Analysis Tool",
    author="ACDS project Team EXE",  # update this
    packages=["flood_tool"],  # DON'T install the scoring package
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "folium",
        "scikit-learn",
        "seaborn",
        "plotly",
        "xgboost",
        "imblearn",
        "ipywidgets",
        "geojson",
        "lightgbm",
        "scipy"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "pytest-timeout>=2.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
