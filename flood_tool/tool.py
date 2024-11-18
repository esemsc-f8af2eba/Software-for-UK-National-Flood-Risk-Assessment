'''Example module in template package.'''

import os

from collections.abc import Sequence
from typing import List

import numpy as np
import pandas as pd

from .geo import *  # noqa: F401, F403

__all__ = [
    'Tool',
    '_data_dir',
    '_example_dir',
    'flood_class_from_postcode_methods',
    'flood_class_from_location_methods',
    'house_price_methods',
    'local_authority_methods',
    'historic_flooding_methods',
]

_data_dir = os.path.join(os.path.dirname(__file__), 'resources')
_example_dir = os.path.join(os.path.dirname(__file__), 'example_data')


# dictionaries with keys of short name and values of long name of
# classification/regression methods

# You should add your own methods here
flood_class_from_postcode_methods = {
    'all_zero_risk': 'All zero risk',
}
flood_class_from_location_methods = {
    'all_zero_risk': 'All zero risk',
}
historic_flooding_methods = {
    'all_false': 'All False',
}
house_price_methods = {
    'all_england_median': 'All England median',
}
local_authority_methods = {
    'all_nan': 'All NaN',
}

IMPUTATION_CONSTANTS = {
    'soilType': 'Unsurveyed/Urban',
    'elevation': 60.0,
    'nearestWatercourse': '',
    'distanceToWatercourse': 80,
    'localAuthority': np.nan
}


class Tool(object):
    '''Class to interact with a postcode database file.'''

    def __init__(self, labelled_unit_data: str = '',
                 unlabelled_unit_data: str = '',
                 sector_data: str = '', district_data: str = '',
                 additional_data: dict = {}):

        '''
        Parameters
        ----------

        unlabelled_unit_data : str, optional
            Filename of a .csv file containing geographic location
            data for postcodes.

        labelled_unit_data: str, optional
            Filename of a .csv containing class labels for specific
            postcodes.

        sector_data : str, optional
            Filename of a .csv file containing information on households
            by postcode sector.

        district_data : str, optional
            Filename of a .csv file containing information on households
            by postcode district.

        additional_data: dict, optional
            Dictionary containing additional .csv files containing addtional
            information on households.
        '''

        # Set defaults if no inputs provided
        if labelled_unit_data == '':
            labelled_unit_data = os.path.join(_data_dir,
                                              'postcodes_labelled.csv')

        if unlabelled_unit_data == '':
            unlabelled_unit_data = os.path.join(_example_dir,
                                                'postcodes_unlabelled.csv')

        if sector_data == '':
            sector_data = os.path.join(_data_dir,
                                       'sector_data.csv')
        if district_data == '':
            district_data = os.path.join(_data_dir,
                                         'district_data.csv')

        self._postcodedb = pd.read_csv(labelled_unit_data)

        # continue your work here

    def fit(self, models: List = [], update_labels: str = '',
            update_hyperparameters: bool = False, **kwargs):
        '''Fit/train models using a labelled set of samples.

        Parameters
        ----------

        models : sequence of model keys
            Models to fit/train
        update_labels : str, optional
            Filename of a .csv file containing an updated
            labelled set of samples
            in the same format as the original labelled set.

            If not provided, the data set provided at
            initialisation is used.
        update_hyperparameters : bool, optional
            If True, models may tune their hyperparameters, where
            possible. If False, models will use their default hyperparameters.
        Examples
        --------
        >>> tool = Tool()
        >>> fcp_methods = list(flood_class_from_postcode_methods.keys())
        >>> tool.fit(fcp_methods[0])  # doctest: +SKIP
        >>> classes = tool.predict_flood_class_from_postcode(
        ...    ['M34 7QL'], fcp_methods[0])  # doctest: +SKIP
        '''

        if update_labels:
            print('updating labelled sample file')
            # update your labelled postcodes data set here

        for model in models:
            if update_hyperparameters:
                print(f'tuning {model} hyperparameters')
                # Do your hyperparameter tuning for the specified model
            else:
                print(f'training {model}')
                # Do your regular fitting/training for the specified model

    def lookup_easting_northing(self, postcodes: Sequence,
                                dtype: np.dtype = np.float64) -> pd.DataFrame:
        '''Get a dataframe of OS eastings and northings from a sequence of
        input postcodes in the labelled or unlabelled datasets.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        dtype: numpy.dtype, optional
            Data type of the easting and northing columns.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing columns of 'easthing' and 'northing',
            indexed by the input postcodes. Invalid postcodes (i.e. those
            not in the available postcodes file) return as NaN.

        Examples
        --------

        >>> tool = Tool()
        >>> results = tool.lookup_easting_northing(['RH16 2QE'])
        >>> results  # doctest: +NORMALIZE_WHITESPACE
                   easting  northing
        RH16 2QE  535295.0  123643.0
        >>> results = tool.lookup_easting_northing(['RH16 2QE', 'AB1 2PQ'])
        >>> results  # doctest: +NORMALIZE_WHITESPACE
                   easting  northing
        RH16 2QE  535295.0  123643.0
        AB1 2PQ        NaN       NaN
        '''

        postcodes = pd.Index(postcodes)

        frame = self._postcodedb.copy()
        frame = frame.set_index('postcode')
        frame = frame.reindex(postcodes)

        return frame.loc[postcodes, ['easting', 'northing']].astype(dtype)

    def lookup_lat_long(self, postcodes: Sequence,
                        dtype: np.dtype = np.float64) -> pd.DataFrame:
        '''Get a Pandas dataframe containing GPS latitude and longitude
        information for a sequence of postcodes in the labelled or
        unlabelled datasets.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        dtype: numpy.dtype, optional
            Data type of the latitude and longitude columns.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only WGS84 latitude and longitude pairs for
            the input postcodes. Missing/Invalid postcodes (i.e. those not in
            the input unlabelled postcodes file) return as NaNs in the latitude
            and longitude columns.

        Examples
        --------
        >>> tool = Tool()
        >>> tool.lookup_lat_long(['M34 7QL']) # doctest: +SKIP
                latitude  longitude
        postcode
        M34 7QL  53.4461    -2.0997
        '''
        # INCOMPLETE: continue your work here
        return pd.DataFrame(columns=['longitude', 'latitude'],
                            index=postcodes, dtype=dtype)

    def impute_missing_values(self, dataframe: pd.DataFrame,
                              method: str = 'mean',
                              constant_values: dict = IMPUTATION_CONSTANTS
                              ) -> pd.DataFrame:
        '''Impute missing values in a dataframe.

        Parameters
        ----------

        dataframe : pandas.DataFrame
            DataFrame (in the format of the unlabelled postcode data)
            potentially containing missing values as NaNs, or with missing
            columns.

        method : str, optional
            Method to use for imputation. Options include:
            - 'mean', to use the mean for the labelled dataset
            - 'constant', to use a constant value for imputation
            - 'knn' to use k-nearest neighbours imputation from the
              labelled dataset

        constant_values : dict, optional
            Dictionary containing constant values to
            use for imputation in the format {column_name: value}.
            Only used if method is 'constant'.

        Returns
        -------

        pandas.DataFrame
            DataFrame with missing values imputed.

        Examples
        --------

        >>> tool = Tool()
        >>> missing = os.path.join(_example_dir, 'postcodes_missing_data.csv')
        >>> data = pd.read_csv(missing)
        >>> data = tool.impute_missing_values(data)  # doctest: +SKIP
        '''
        # INCOMPLETE: continue your work here. Feel free to add more
        # methods for imputation. If you do, remember to update the docstring
        return dataframe

    def predict_flood_class_from_postcode(self, postcodes: Sequence[str],
                                          method: str = 'all_zero_risk'
                                          ) -> pd.Series:
        '''
        Generate series predicting flood probability classification
        for a collection of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        method : str (optional)
            optionally specify (via a key in the
            `get_flood_class_from_postcode_methods` dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by postcodes.
            Returns NaN for postcode units not in the available postcode files.
        '''

        if method == 'all_zero_risk':
            return pd.Series(
                data=np.ones(len(postcodes), int),
                index=np.asarray(postcodes),
                name='riskLabel',
            )
        else:
            raise NotImplementedError(f'method {method} not implemented')

    def predict_flood_class_from_OSGB36_location(
            self, eastings: Sequence[float], northings: Sequence[float],
            method: str = 'all_zero_risk') -> pd.Series:
        '''
        Generate series predicting flood probability classification
        for a collection of locations given as eastings and northings
        on the Ordnance Survey National Grid (OSGB36) datum.

        Parameters
        ----------

        eastings : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : str (optional)
            optionally specify (via a key in the
            get_flood_class_from_location_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by locations
            as an (easting, northing) tuple.
        '''

        if method == 'all_zero_risk':
            return pd.Series(
                data=np.ones(len(eastings), int),
                index=((est, nth) for est, nth in zip(eastings, northings)),
                name='riskLabel',
            )
        else:
            raise NotImplementedError(f'method {method} not implemented')

    def predict_flood_class_from_WGS84_locations(
            self, longitudes: Sequence[float], latitudes: Sequence[float],
            method: str = 'all_zero_risk') -> pd.Series:
        '''
        Generate series predicting flood probability classification
        for a collection of WGS84 datum locations.

        Parameters
        ----------

        longitudes : sequence of floats
            Sequence of WGS84 longitudes.
        latitudes : sequence of floats
            Sequence of WGS84 latitudes.
        method : str (optional)
            optionally specify (via a key in
            get_flood_class_from_location_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels multi-indexed by
            location as a (longitude, latitude) pair.
        '''

        if method == 'all_zero_risk':
            idx = pd.MultiIndex.from_tuples([(lng, lat) for lng, lat in
                                             zip(longitudes, latitudes)])
            return pd.Series(
                data=np.ones(len(longitudes), int),
                index=idx,
                name='riskLabel',
            )
        else:
            raise NotImplementedError(f'method {method} not implemented')

    def predict_median_house_price(
            self, postcodes: Sequence[str],
            method: str = 'all_england_median'
            ) -> pd.Series:
        '''
        Generate series predicting median house price for a collection
        of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        method : str (optional)
            optionally specify (via a key in the
            get_house_price_methods dict) the regression
            method to be used.

        Returns
        -------

        pandas.Series
            Series of median house price estimates indexed by postcodes.
        '''

        if method == 'all_england_median':
            return pd.Series(
                data=np.full(len(postcodes), 245000.0),
                index=np.asarray(postcodes),
                name='medianPrice',
            )
        else:
            raise NotImplementedError(f'method {method} not implemented')

    def predict_local_authority(
        self, eastings: Sequence[float], northings: Sequence[float],
        method: str = 'do_nothing'
    ) -> pd.Series:
        '''
        Generate series predicting local authorities in m for a sequence
        of OSGB36 locations.

        Parameters
        ----------

        eastingss : sequence of floats
            Sequence of OSGB36 eastings.
        northings : sequence of floats
            Sequence of OSGB36 northings.
        method : str (optional)
            optionally specify (via a key in the
            local_authority_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of predicted local authorities for the input
            locations, and multiindexed by the location as a
            (easting, northing) tuple.
        '''

        if method == 'all_nan':
            idx = pd.MultiIndex.from_tuples([(est, nth) for est, nth in
                                             zip(eastings, northings)])
            return pd.Series(
                data=np.full(len(eastings), np.nan),
                index=idx,
                name='localAuthority',
            )
        else:
            raise NotImplementedError(f'method {method} not implemented')

    def predict_historic_flooding(
            self, postcodes: Sequence[str],
            method: str = 'all_false'
            ) -> pd.Series:
        '''
        Generate series predicting whether a collection of postcodes
        has experienced historic flooding.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        method : str (optional)
            optionally specify (via a key in the
            historic_flooding_methods dict) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series indicating whether a postcode experienced historic
            flooding, indexed by the postcodes.
        '''

        if method == 'all_false':
            return pd.Series(
                data=np.full(len(postcodes), False),
                index=np.asarray(postcodes),
                name='historicallyFlooded',
            )
        else:
            raise NotImplementedError(f'method {method} not implemented')

    def estimate_total_value(self, postal_data: Sequence[str]) -> pd.Series:
        '''
        Return a series of estimates of the total property values
        of a sequence of postcode units or postcode sectors.

        The estimate is based on the median house price for the area and an
        estimate of the number of properties it contains.

        Parameters
        ----------

        postal_data : sequence of strs
            Sequence of postcode units or postcode sectors (either
            may be used).


        Returns
        -------

        pandas.Series
            Series of total property value estimates indexed by locations.
        '''

        raise NotImplementedError

    def estimate_annual_human_flood_risk(self, postcodes: Sequence[str],
                                         risk_labels: [pd.Series | None] = None
                                         ) -> pd.Series:
        '''
        Return a series of estimates of the risk to human life for a
        collection of postcodes.

        Risk is defined here as an impact coefficient multiplied by the
        estimated number of people under threat multiplied by the probability
        of an event.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        risk_labels: pandas.Series (optional)
            Series containing flood risk classifiers, as
            predicted by get_flood_class_from_postcodes.

        Returns
        -------

        pandas.Series
            Series of total annual human flood risk estimates
            indexed by postcode.
        '''

        risk_labels = (risk_labels or
                       self.get_flood_class_from_postcodes(postcodes))

        raise NotImplementedError

    def estimate_annual_flood_economic_risk(
            self, postcodes: Sequence[str],
            risk_labels: [pd.Series | None] = None
            ) -> pd.Series:
        '''
        Return a series of estimates of the total economic property risk
        for a collection of postcodes.

        Risk is defined here as a damage coefficient multiplied by the
        value under threat multiplied by the probability of an event.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcode units.
        risk_labels: pandas.Series (optional)
            optionally provide a Pandas Series containing flood risk
            classifiers, as predicted by get_flood_class_from_postcodes.

        Returns
        -------

        pandas.Series
            Series of total annual economic flood risk estimates indexed
            by postcode.
        '''

        risk_labels = (risk_labels or
                       self.get_flood_class_from_postcodes(postcodes))

        raise NotImplementedError
