***************************
Visualising Risk & Rainfall
***************************

This document provides an overview of the visualization of flood risk and rainfall data using interactive heatmaps. The visualization is created using the `folium` library in Python, which allows for the generation of interactive maps with various layers and plugins.

Flood risk visualization
========================

This visualization generates an interactive heatmap of high-risk areas using the `folium` library and the `HeatMap` plugin. It highlights areas with significant rainfall or water levels exceeding predefined thresholds.

**Data Source and Content**

The data for high-risk areas is sourced from a filtered dataset (`high_risk`) containing geographic and risk-related metrics. The dataset includes the following:

1. **Criteria for High-Risk Data Points**:
   - **Rainfall**: Greater than a specified threshold.
   - **Water Level**: Exceeds a defined threshold.

2. **Columns Used for the Heatmap**:
   - **`latitude`**: Geographic latitude of the data point.
   - **`longitude`**: Geographic longitude of the data point.
   - **`value`**: Represents the intensity of the risk, determined by rainfall or water level, and used as the heat intensity in the visualization.

**Visualization Interpretation**

1. **Geographic Distribution**:
   - The heatmap illustrates the geographic locations of high-risk areas, with color intensity indicating the severity of the risk.

2. **Risk Intensity**:
   - The `value` column determines the "heat" or intensity for each point on the map. Higher values result in deeper colors, signifying greater risk.

3. **Interactivity**:
   - The generated HTML map is interactive and can be viewed in a browser. Users can zoom in, zoom out, and explore the risk distribution in different regions.

**Code Workflow**

1. **Initialize the Map**:
   - A `folium.Map` object is created, centered at `[51.5, -0.1]` (latitude and longitude) with an initial zoom level of 6.

2. **Prepare Heatmap Data**:
   - Extracts data from the `high_risk` DataFrame, specifically the `latitude`, `longitude`, and `value` columns.
   - Filters out rows with missing values and converts the data into a list of lists.

3. **Add Heatmap Layer**:
   - Uses `HeatMap` from `folium.plugins` to create a heatmap layer based on the processed data.
   - Adds the heatmap layer to the base map.

4. **Save the Map**:
   - Saves the interactive map as an HTML file named `risk_map.html`.

5. **Display the Map**:
   - The `risk_map` object can be returned or viewed directly in Jupyter notebooks or other interactive environments.

**Parameters**

- **`high_risk`**: A Pandas DataFrame containing:
  - `latitude` (float): Latitude of the location.
  - `longitude` (float): Longitude of the location.
  - `value` (float): Represents the risk intensity (e.g., rainfall or water level).

**Outputs**

- **Interactive HTML File**:
  - The generated heatmap is saved as `risk_map.html`.
  - The map can be opened in a browser to interactively explore the high-risk areas.

**Example Usage**

.. code-block:: python

   import folium
   from folium.plugins import HeatMap

   # Create the map
   risk_map = folium.Map(location=[51.5, -0.1], zoom_start=6)

   # Prepare heatmap data
   heat_data = high_risk[['latitude', 'longitude', 'value']].dropna().values.tolist()
   HeatMap(heat_data).add_to(risk_map)

   # Save and display the map
   risk_map.save('risk_map.html')
   risk_map

Expected Output:
- An interactive map showing the heat intensity of flood risks.

**Limitations**

- The accuracy of the heatmap depends on the quality and completeness of the `high_risk` dataset.
- Missing or incorrect `latitude` and `longitude` values may result in errors or distortions in the visualization.
- Large datasets may affect the performance of the rendered heatmap.

**References**

- `folium` Documentation: https://python-visualization.github.io/folium/
- `HeatMap` Plugin: https://python-visualization.github.io/folium/plugins.html

---