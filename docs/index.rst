################################
The Team X Flood Prediction Tool
################################

This package implements a flood risk prediction and visualization tool.

Installation Instructions
-------------------------
**Prerequisite**

**Ensure Python3 is installed on your machine**

```bash
python3 --version
```

This project uses Conda as a package manager for managing software packages, dependencies and environments. You should have conda configured on your local machine before installing the project.
You can check if Conda is installed by running:

```bash
conda -V
```

We will now use Miniconda to create the `deluge` environment. To start, we first use `git` to clone the repository containing the materials of flooding risk project.

```bash
git clone https://github.com/ese-ada-lovelace-2024/ads-deluge-exe.git
```

Navigate into the repository (e.g., cd ads-deluge-exe). Then configure the conda environment:

```bash
conda env create -f environment.yml
conda avtivate deluge
```

To deactivate the environment you can run:

```bash
conda deactivate
```

**VS Code**

You can run the code in this project interactively in VSCode or use Jupyter Notebook for example. To get started:
1. Make sure you are setting up environment correctly and in the correct file path in the terminal.
2. In VS Code, run the following command.

```bash
code .
```

Quick Usage guide
-----------------

**Risk Tool**

The tool.py file combines the main functionality of the flooding risk tool. It provides methods for working with flood data associated with UK postcode and geographic coordinates. This enables predictions for:

- Risk labels
- Median house prices
- Identification of local authority
- historic flooding

---

#### **To get started:**

The first step is to import the tool from the flood_tool module:

```python
import sys
sys.path.append('..')
import flood_tool as ft
tool = ft.Tool()
```

Now, the class is initialized, we can simply call functions from tool directly. See below for description and instruction of each function involved in this tool.py.

1. Flood risk prediction: Predict flood risk levels (with riskLabel being the target variable) for postcodes or geographic coordinates based on trained models.

#### Flood risk for postcodes: 
```python
tool.predict_flood_class_from_postcodes(postcodes=['RH16 2QE'], method = 'predicting_all_risks')
```

#### Flood risk from easting/northing: 
```python
tool.predict_flood_class_from_OSGB36_location(eastings=[535295.0], northings=[123643.0], method='predicting_risk_from_easting_northing')
```

#### Flood risk from longitude/latitude: 
```python
tool.predict_flood_class_from_WGS84_locations(longitudes=[0], latitudes=[50], method = 'predicting_risk_from_latitude_longitude')
```

2. Median house price prediction: Predicting median house prices for a collection of postcodes using Random Forest regression.
```python
tool.predict_median_house_price(postcodes=['RH16 2QE'], method = 'house_price_rf')
```

3. Local authority prediction: Predicting local authorities for a sequence of easting and northing (OSGB36 locations).
```python
tool.predict_local_authority(eastings=[535295.0], northings=[123643.0], method='local_authority')
```

4. Historic flooding prediction: predicting whether a collection of postcodes has experienced historic flooding.
```python
tool.predict_historic_flooding(postcodes=['RH16 2QE'], method='historic_flooding')
```

### Visualization

Once the packages are imported into the Data Visualization.ipynb, we then can use it to explaore/analysis the data and generating interactive plot (e.g., maps).

Follow the instructions below to understand the functionality provided.

1. Import functions from analysis.py to be able to perform EDA.
2. Import functions from mapping.py to plot predicted variables (local authority, median house prices, historic flooding and risk labels). 


Further Documentation
---------------------

.. toctree::
   :maxdepth: 2

   data_formats
   models
   coordinates
   visualization


Function APIs
-------------

.. automodule:: flood_tool
  :members:
  :imported-members:

