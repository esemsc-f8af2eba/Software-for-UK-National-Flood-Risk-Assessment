'''Interactions with rainfall and river data.'''

from os import path
import urllib
import json
import re

from typing import List, Tuple

import numpy as np
import pandas as pd

from ..tool import _data_dir, _example_dir

__all__ = [
    'get_station_data_from_csv',
    'get_stations',
    'get_latest_rainfall_readings',
    'get_live_station_data',
    'get_live_weather_data'
]

# Paths to data files
_wet_day_file = path.join(_example_dir, 'wet_day.csv')
_typical_day_file = path.join(_example_dir, 'typical_day.csv')
_station_file = path.join(_data_dir, 'stations.csv')


def get_station_data_from_csv(filename: str,
                              station_reference: str = None) -> pd.Series:
    '''Return readings for a specified recording station from .csv file.

    Parameters
    ----------

    filename: str
        filename to read
    station_reference : str, optional
        station_reference to return.

    Returns
    -------
    pandas.Series
        Series of data values

    Examples
    --------
    >>> data = get_station_data_from_csv(_wet_day_file, '0184TH')
    '''
    frame = pd.read_csv(filename)
    frame['dateTime'] = pd.to_datetime(frame['dateTime'])
    frame['value'] = pd.to_numeric(frame['value'], errors='coerce')

    if station_reference is not None:
        frame = frame.loc[frame.stationReference == station_reference]
        frame.drop('stationReference', axis=1, inplace=True)
        frame.set_index('dateTime', inplace=True)

    else:
        frame.set_index(['stationReference', 'dateTime'], inplace=True)

    return frame.sort_index()


def get_stations(filename: str = _station_file) -> pd.DataFrame:
    '''Return a DataFrame of the measurement stations.

    Parameters
    ----------

    filename: str, optional
        filename to read

    Returns
    -------
    pandas.DataFrame
        DataFrame of the measurement stations.

    Examples
    --------
    >>> stations = get_stations()
    >>> stations.stationReference.head(5) # doctest: +NORMALIZE_WHITESPACE
    0      000008
    1      000028
    2    000075TP
    3    000076TP
    4    000180TP
    Name: stationReference, dtype: object
    '''

    return pd.read_csv(filename)


# See https://environment.data.gov.uk/flood-monitoring/doc/reference
# and https://environment.data.gov.uk/flood-monitoring/doc/rainfall
# for full API documentation.

API_URL = 'http://environment.data.gov.uk/flood-monitoring/'


rainfall_station = re.compile(r'.*/(.*)-rainfall-(.*)-t-15_min-(.*)/.*')


def split_rainfall_api_id(input: str) -> tuple[str, str, str]:
    '''Split rainfall station API id into component parts
    using a regular expression.
    '''

    match = rainfall_station.match(input)

    try:
        return match.group(1), match.group(2), match.group(3)
    except AttributeError:
        return np.nan


def get_latest_rainfall_readings() -> pd.DataFrame:
    '''Return last readings for all rainfall stations via live API.

    >>> data = get_latest_rainfall_readings()
    '''

    url = API_URL + 'data/readings?parameter=rainfall&latest'
    data = urllib.request.urlopen(url)
    data = json.load(data)

    dframe = pd.DataFrame(data['items'])

    # split id into parts
    id_data = dframe['@id'].apply(split_rainfall_api_id)
    id_data = id_data.dropna()
    dframe['stationReference'] = id_data.apply(lambda x: x[0])
    dframe['qualifier'] = id_data.apply(lambda x:
                                        x[1].replace('_', ' '). title())
    dframe['unitName'] = id_data.apply(lambda x: x[2])
    dframe.drop(['@id', 'measure'], axis=1, inplace=True)

    dframe['dateTime'] = dframe['dateTime'].apply(pd.to_datetime)

    dframe.set_index(['stationReference', 'dateTime'], inplace=True)

    dframe['parameter'] = 'rainfail'

    dframe['value'] = pd.to_numeric(dframe['value'], errors='coerce')

    return dframe.dropna().sort_index()


def get_live_station_data(station_reference: str) -> pd.DataFrame:
    '''Return recent readings for a specified recording station from live API.

    Parameters
    ----------

    station_reference
        station_reference to return.

    Examples
    --------

    >>> data = get_live_station_data('0184TH')
    '''

    url = API_URL + f'id/stations/{station_reference}/readings?_sorted'
    data = urllib.request.urlopen(url)
    data = json.load(data)

    dframe = pd.DataFrame(data['items'])

    return dframe


def get_live_weather_data(lat: float, long: float,
                          current: List[str] = ['temperature_2m', 'rain'],
                          hourly: List[str] = ['temperature_2m',
                                               'relative_humidity_2m', 'rain']
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''Return current and forecast weather data for a specified location
    from live weather forecasting API.

    Uses the Open-Meteo API.

    See https://open-meteo.com/en/docs for full API documentation.

    Parameters
    ----------

    lat
        Latitude of location
    long
        Longitude of location

    Examples
    --------

    >>> live_data, forecast = get_live_weather_data(51.5, -0.1)
    '''

    base_url = 'https://api.open-meteo.com/v1/forecast?'
    position = f'latitude={lat:.4f}&longitude={long:.4f}'
    current_enc = f'current={','.join(current)}'
    hourly_enc = f'hourly={','.join(hourly)}'

    data = urllib.request.urlopen(base_url + position
                                  + '&' + current_enc + '&' + hourly_enc)
    data = json.load(data)

    live = pd.DataFrame(columns=current,
                        index=[pd.to_datetime(data['current']['time'])])

    forecast = pd.DataFrame(columns=hourly,
                            index=pd.to_datetime(data['hourly']['time']))

    for key in current:
        live[key] = float(data['current'][key])

    for key in hourly:
        forecast[key] = np.array(data['hourly'][key], float)

    return live, forecast
