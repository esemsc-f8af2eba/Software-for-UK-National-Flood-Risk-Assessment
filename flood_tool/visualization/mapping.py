import folium
import numpy as np
import pandas as pd
import plotly.express as px

from flood_tool.geo import get_gps_lat_long_from_easting_northing

all = ['plot_circle']


def plot_circle(lat, lon, radius, map=None, **kwargs):
    '''
    Plot a circle on a map (creating a new folium map instance if necessary).

    Parameters
    ----------

    lat: float
        latitude of circle to plot (degrees)
    lon: float
        longitude of circle to plot (degrees)
    radius: float
        radius of circle to plot (m)
    map: folium.Map
        existing map object

    Returns
    -------

    Folium map object

    Examples
    --------

    >>> import folium
    >>> plot_circle(52.79, -2.95, 1e3, map=None) # doctest: +SKIP
    '''

    if not map:
        map = folium.Map(location=[lat, lon], control_scale=True)

    folium.Circle(
        location=[lat, lon],
        radius=radius,
        fill=True,
        fillOpacity=0.6,
        **kwargs,
    ).add_to(map)

    return map


def plot_historically_flooded_areas(df, tool):
    """
    Creates a scatter mapbox visualization of historically flooded areas.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing columns 'postcode', 'easting', 'northing',
        and other relevant data.
    tool : object
        An object with a `predict_historic_flooding` method
        to predict flooding for postcodes.
    flooding_value : float, optional
        The value used to filter historically flooded areas (default is 1.0).
    method : str, optional
        The method to use for historic flooding prediction
        (default is 'historic_flooding').

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The scatter mapbox figure.
    """

    lat, long = get_gps_lat_long_from_easting_northing(df['easting'],
                                                       df['northing'])
    latlon_data = pd.DataFrame(np.column_stack((lat, long)),
                               columns=['Latitude', 'Longitude'])

    df_with_latlon = pd.concat([df, latlon_data], axis=1)
    df_with_latlon.set_index('postcode', inplace=True)

    predicted_flooding = pd.DataFrame(
        tool.predict_historic_flooding(df['postcode'],
                                       method='historic_flooding'),
        columns=['historicallyFlooded']
    )
    combined_data = pd.concat([df_with_latlon, predicted_flooding], axis=1)

    fig = px.scatter_mapbox(
        combined_data,
        lat="Latitude",
        lon="Longitude",
        color='historicallyFlooded',
        color_continuous_scale="Viridis",
        mapbox_style="carto-positron",
        title="Historically Flooded Areas",
        zoom=5,
        hover_name=combined_data.index,
    )

    return fig


def plot_predict_median_house_price(df, tool):
    """
    Creates a scatter mapbox visualization of historically flooded areas.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing columns 'postcode', 'easting', 'northing',
        and other relevant data.
    tool : object
        An object with a `predict_historic_flooding` method
        to predict flooding for postcodes.
    flooding_value : float, optional
        The value used to filter historically flooded areas (default is 1.0).
    method : str, optional
        The method to use for historic flooding prediction
        (default is 'historic_flooding').

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The scatter mapbox figure.
    """

    lat, long = get_gps_lat_long_from_easting_northing(df['easting'],
                                                       df['northing'])
    latlon_data = pd.DataFrame(np.column_stack((lat, long)),
                               columns=['Latitude', 'Longitude'])

    df_with_latlon = pd.concat([df, latlon_data], axis=1)
    df_with_latlon.set_index('postcode', inplace=True)

    predicted_median = pd.DataFrame(
        tool.predict_median_house_price(df['postcode'],
                                        method='house_price_rf'),
        columns=['medianPrice']
    )
    combined_data = pd.concat([df_with_latlon, predicted_median], axis=1)

    fig = px.scatter_mapbox(
        combined_data,
        lat="Latitude",
        lon="Longitude",
        color='medianPrice',
        color_continuous_scale="Viridis",
        mapbox_style="carto-positron",
        title="Median House Price",
        zoom=5,
        hover_name=combined_data.index,
    )

    return fig


def plot_predict_flood_class(df, tool):
    """
    Creates a scatter mapbox visualization of historically flooded areas.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing columns 'postcode', 'easting', 'northing',
        and other relevant data.
    tool : object
        An object with a `predict_historic_flooding` method
        to predict flooding for postcodes.
    flooding_value : float, optional
        The value used to filter historically flooded areas (default is 1.0).
    method : str, optional
        The method to use for historic flooding prediction
        (default is 'historic_flooding').

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The scatter mapbox figure.
    """

    lat, long = get_gps_lat_long_from_easting_northing(df['easting'],
                                                       df['northing'])
    latlon_data = pd.DataFrame(np.column_stack((lat, long)),
                               columns=['Latitude', 'Longitude'])

    df_with_latlon = pd.concat([df, latlon_data], axis=1)
    df_with_latlon.set_index('postcode', inplace=True)

    predicted_class = pd.DataFrame(
        tool.predict_flood_class_from_postcode(df['postcode'],
                                               method='predicting_all_risks'),
        columns=['Predicted Flood Class']
    )
    combined_data = pd.concat([df_with_latlon, predicted_class], axis=1)

    fig = px.scatter_mapbox(
        combined_data,
        lat="Latitude",
        lon="Longitude",
        color='Predicted Flood Class',
        color_continuous_scale="Viridis",
        mapbox_style="carto-positron",
        title="Flood Class",
        zoom=5,
        hover_name=combined_data.index,
    )

    return fig
