'''Example module in template package.'''

import os

from collections.abc import Sequence
from typing import List

import numpy as np
import pandas as pd

from .geo import *  # noqa: F401, F403
from .geo import get_gps_lat_long_from_easting_northing

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans


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
    'predicting_all_risks': 'all risks'
}
flood_class_from_location_methods = {
    'all_zero_risk': 'All zero risk',
    'predicting_risk_from_easting_northing': 'easting/northing based risk',
    'predicting_risk_from_latitude_longitude': 'latitude/longitude based risk'
}
historic_flooding_methods = {
    'all_false': 'All False',
    'historic_flooding': 'KNN model for historic flooding',
}
house_price_methods = {
    'all_england_median': 'All England median',
    'house_price_rf': (
        'Predicting Median House Price using Random Forest Regressor'
    ),
}
local_authority_methods = {
    'all_nan': 'All NaN',
    'local_authority': 'Random Forest Model for Local Authority Prediction',
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
        self._sector_data = pd.read_csv(sector_data)
        self._unlabelled_unit_data = pd.read_csv(unlabelled_unit_data)
        self._district_data = pd.read_csv(district_data)

        # # continue your work here
        # self.fit(models=['local_authority'])
        # self.fit(models=['house_price_rf'])
        # self.fit(models=['historic_flooding'])
        # self.fit(models=['predicting_all_risks'])

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

            self._postcodedb = pd.read_csv(update_labels)

        for model in models:

            if model == 'local_authority':
                print('Training model for Local Authority Prediction...')
                la_data_cols = ['easting', 'northing', 'localAuthority']
                la_data = self._postcodedb[la_data_cols]
                X = la_data.drop('localAuthority', axis=1)
                y = la_data['localAuthority']

                rf_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    random_state=42,
                    class_weight='balanced',
                )

                rf_model.fit(X, y)

                self.rf_model = rf_model

                print("Model trained successfully.")

            if model == 'house_price_rf':
                print('Training model for Median House Price Prediction...')

                data = pd.concat([self._postcodedb[:-1],
                                  self._unlabelled_unit_data])
                data['postcodeSector'] = \
                    data['postcode'].str.split(' ').str[0] + \
                    ' ' + data['postcode'].str.split(' ').str[1].str[0]
                data['postcodeDistrict'] = \
                    data['postcode'].str.split(' ').str[0]

                data = data.merge(self._district_data,
                                  on='postcodeDistrict',
                                  how='left')
                data = data.merge(self._sector_data,
                                  on='postcodeSector',
                                  how='left')
                postcode_col = data['postcode']

                data.drop(columns=['nearestWatercourse',
                                   'catsPerHousehold',
                                   'headcount',
                                   'historicallyFlooded',
                                   'riskLabel',
                                   'postcode',
                                   'postcodeDistrict',
                                   'postcodeSector'], inplace=True)

                data = data.dropna(subset=['medianPrice'])

                X = data.drop('medianPrice', axis=1).copy()
                y = data['medianPrice'].copy()

                def log_function(x):
                    x_min = np.min(x, axis=0)
                    # Calculate the minimum of each column
                    x_shifted = x - x_min
                    # Shift by the minimum
                    return np.log1p(x_shifted)
                    # Apply log(1 + x_shifted)

                robust_scaler_features = ['distanceToWatercourse']

                onehot_encoding_features = ['soilType']
                ordinal_encoder_features = ['localAuthority']
                standard_scaler_features = ['easting',
                                            'northing',
                                            'elevation',
                                            'dogsPerHousehold',
                                            'households',
                                            'numberOfPostcodeUnits']
                robust_scaler_features = ['distanceToWatercourse']

                onehot_encoding_features = ['soilType']
                ordinal_encoder_features = ['localAuthority']
                standard_scaler_features = ['easting',
                                            'northing',
                                            'elevation',
                                            'dogsPerHousehold',
                                            'households',
                                            'numberOfPostcodeUnits']

                log_transformer = \
                    FunctionTransformer(log_function,
                                        feature_names_out='one-to-one')

                log_transform_pipeline = Pipeline(steps=[
                    ('log_transform', log_transformer)
                ])

                distanceToWatercourse_pipeline = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('log', log_transform_pipeline),
                    ('scaler', RobustScaler())
                ])

                standard_scaler_features_pipeline = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('log', log_transform_pipeline),
                    ('scaler', StandardScaler())
                ])

                onehot_encoder_pipeline = Pipeline(steps=[
                    ('onehot',
                     OneHotEncoder(handle_unknown='ignore',
                                   sparse_output=False))
                ])

                ordinal_encoder_pipeline = Pipeline(steps=[
                    ('ordinal',
                     OrdinalEncoder(handle_unknown='use_encoded_value',
                                    unknown_value=-1))
                ])

                preprocessor = ColumnTransformer(transformers=[
                    ('robust_scaler',
                     distanceToWatercourse_pipeline,
                     robust_scaler_features),
                    ('num',
                     standard_scaler_features_pipeline,
                     standard_scaler_features),
                    ('onehot_encoder',
                     onehot_encoder_pipeline,
                     onehot_encoding_features),
                    ('ordinal_encoder',
                     ordinal_encoder_pipeline,
                     ordinal_encoder_features)],
                    remainder='passthrough')

                prep_pipe = Pipeline(steps=[
                    ('preprocessor', preprocessor)
                ])

                X = data.drop(columns=['medianPrice']).copy()
                y = data['medianPrice'].copy()

                X = prep_pipe.fit_transform(X)
                self.hp_data = pd.concat([postcode_col,
                                          pd.DataFrame(X)], axis=1)

                house_model = RandomForestRegressor(n_estimators=50,
                                                    min_samples_split=2,
                                                    min_samples_leaf=4,
                                                    max_depth=20,
                                                    random_state=42)

                house_model.fit(X, y)

                self.house_model = house_model

                print("House Price Model trained successfully.")

            if model == 'historic_flooding':
                print('Training KNN for historic flood')
                pc_data_col = ["postcode",
                               "medianPrice",
                               "easting",
                               "northing",
                               "elevation",
                               "distanceToWatercourse",
                               "riskLabel",
                               "historicallyFlooded",
                               'soilType']
                pc_data = self._postcodedb[pc_data_col]
                X = pc_data.drop(columns=["historicallyFlooded",
                                          "riskLabel",
                                          "medianPrice"])
                y = pc_data['historicallyFlooded']

                X[["outward", "inward"]] = \
                    X["postcode"].str.split(" ", expand=True)

                X = X.drop(columns=["postcode"])

                log_transform_cols = ['elevation', 'distanceToWatercourse']

                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ])

                encoded_transformer = Pipeline(steps=[
                    ('encoder',
                     OrdinalEncoder(handle_unknown="use_encoded_value",
                                    unknown_value=-1))
                ])

                # Log transformation for skewed features
                def log_positive_negative(x):
                    return np.where(x > 0, np.log1p(x), -np.log1p(-x))

                log_custom = \
                    FunctionTransformer(log_positive_negative,
                                        feature_names_out="one-to-one")
                log_transformer = Pipeline([('imputer',
                                             SimpleImputer(strategy='median')),
                                            ('log_transform', log_custom)])

                # Column transformer combining all transformations
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('log', log_transformer, log_transform_cols),
                        ('cat', categorical_transformer, ["soilType"]),
                        ("encode_cat",
                         encoded_transformer, ["outward", "inward"])
                    ], remainder="passthrough"
                )

                # Applying transformations
                X_transformed = preprocessor.fit_transform(X)

                # Training KNN model
                knn_model = KNeighborsClassifier(weights='distance',
                                                 n_neighbors=5)
                knn_model.fit(X_transformed, y)

                # Save the model and preprocessor for future use
                self.knn_model = knn_model
                self.preprocessor = preprocessor

                print(f'{model} training complete!')

            if model == 'predicting_all_risks':
                print('Training model for Postcode-based Prediction...')
                # Extract relevant columns for postcodes
                pc_data_cols = ['postcode',
                                'elevation',
                                'distanceToWatercourse',
                                'soilType',
                                'localAuthority',
                                'historicallyFlooded',
                                'riskLabel',
                                'easting',
                                'northing',
                                'medianPrice']
                pc_data = self._postcodedb[pc_data_cols]
                X = pc_data.drop(columns=['riskLabel', 'postcode'], axis=1)
                X[['outward_code', 'inward_code']] = \
                    pc_data['postcode'].str.split(' ', expand=True)

                y = pc_data['riskLabel']

                # Apply preprocessing

                # Preprocessing Pipeline

                def log_function(x):
                    return np.log1p(x)

                log_custom = \
                    FunctionTransformer(log_function,
                                        feature_names_out='one-to-one')

                log_transform_cols = ['elevation', 'distanceToWatercourse']
                robust_cols = ['medianPrice']
                cat_cols_ohe = ['soilType',
                                'localAuthority',
                                'historicallyFlooded']
                standard_cols = ['easting', 'northing']
                cat_cols_ordinal = ['outward_code', 'inward_code']

                # Create pipelines for transformations
                log_transformer = Pipeline([('log_transform', log_custom)])
                robust_transformer = Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('robust_scaler', RobustScaler())])
                categorical_transformer_ohe = \
                    Pipeline([('onehot',
                               OneHotEncoder(handle_unknown='ignore',
                                             sparse_output=False))])
                categorical_transformer_ordinal = \
                    Pipeline([('ordinal',
                               OrdinalEncoder(
                                   handle_unknown='use_encoded_value',
                                   unknown_value=-1))])
                standard_transformer = \
                    Pipeline([('scaler', StandardScaler()),
                              ('kmeans',
                               KMeans(n_clusters=10, random_state=42))])

                # Combine all transformers into a single ColumnTransformer
                prep_pipe = ColumnTransformer(
                    transformers=[
                        ('log', log_transformer, log_transform_cols),
                        ('num_robust', robust_transformer, robust_cols),
                        ('onehot', categorical_transformer_ohe, cat_cols_ohe),
                        ('standard', standard_transformer, standard_cols),
                        ('ordinal',
                         categorical_transformer_ordinal,
                         cat_cols_ordinal),
                    ],
                    remainder='passthrough'
                )
                print('Preprocessing data...')
                X_transformed = prep_pipe.fit_transform(X)

                self.prep_pipe = prep_pipe

                # Initiate the model
                rf_model = RandomForestClassifier(
                    n_estimators=300,
                    min_samples_split=5,
                    max_depth=None,
                    random_state=42,
                    class_weight='balanced',
                )

                rf_model.fit(X_transformed, y)

                self.rf_model = rf_model

                print("Model trained successfully.")

            if model == 'predicting_risk_from_easting_northing':
                print('Training model for '
                      'easting/northing-based prediction...')
                # Extract relevant columns for postcodes
                EN_data_cols = ['easting', 'northing', 'riskLabel']
                EN_data = self._postcodedb[EN_data_cols]
                # print(EN_data)
                X = EN_data.drop('riskLabel', axis=1)
                y = EN_data['riskLabel']

                # preprocessing pipeline
                # Step 1: Specify columns for each transformation
                standard_cols = ['easting', 'northing']

                # Step 2: Create pipelines for transformations
                # Standard scaling pipeline for lat/long
                standard_transformer = Pipeline([
                    ('scaler', StandardScaler()),  # Scale numeric features
                    ('kmeans',
                     KMeans(n_clusters=10,
                            random_state=42))])

                # Combine all transformers into a single ColumnTransformer
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('standard',
                         standard_transformer,
                         standard_cols)],
                    remainder='passthrough'
                    # Pass through any remaining columns
                )

                # Final pipeline
                prep_pipe_en = preprocessor

                # Fitting the preprocessor
                X_transformed = prep_pipe_en.fit_transform(X)

                self.prep_pipe_en = prep_pipe_en

                # RandomForest Classifier
                rf_model_en = RandomForestClassifier(
                    n_estimators=400,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='log2',
                    max_depth=20,
                    random_state=42,
                    class_weight='balanced',
                )

                rf_model_en.fit(X_transformed, y)

                self.rf_model_en = rf_model_en

                print("Model trained successfully.")

            if model == 'predicting_risk_from_latitude_longitude':
                print('Training model for '
                      'latitude/longitude-based prediction...')
                # Extract relevant columns for postcodes
                LL_data_cols = ['easting', 'northing', 'riskLabel']
                LL_data = self._postcodedb[LL_data_cols]

                easting = LL_data['easting']
                northing = LL_data['northing']

                LL_result = get_gps_lat_long_from_easting_northing(
                    easting, northing)

                # Combine latitude and longitude into a 2D array
                latitudes, longitudes = LL_result
                combined_result = np.column_stack((latitudes, longitudes))

                # Include Lat/Long to the DataFrame
                combined_result_df = pd.DataFrame(
                    combined_result,
                    columns=['latitude', 'longitude'])

                LL_data = LL_data.assign(
                    latitude=combined_result_df['latitude'])
                LL_data = LL_data.assign(
                    longitude=combined_result_df['longitude'])
                # LL_data['latitude'] = combined_result_df['latitude']
                # LL_data['longitude'] = combined_result_df['longitude']
                # EN_data_cols = ['easting', 'northing', 'riskLabel']
                # EN_data = self._postcodedb[EN_data_cols]

                X = LL_data.drop(['riskLabel', 'easting', 'northing'], axis=1)
                y = LL_data['riskLabel']

                # Apply preprocessing pipeline
                # Step 1: Specify columns for each transformation
                standard_cols = ['latitude', 'longitude']

                # Step 2: Create pipelines for transformations
                # Standard scaling pipeline for lat/long
                standard_transformer = Pipeline([
                    ('scaler', StandardScaler()),  # Scale numeric features
                    ('kmeans', KMeans(n_clusters=10,
                                      random_state=42))
                    # Add K-Means clustering
                ])

                # Combine all transformers into a single ColumnTransformer
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('standard',
                         standard_transformer,
                         standard_cols)],
                    remainder='passthrough'
                    # Pass through any remaining columns
                )

                # Final pipeline
                prep_pipe_ll = preprocessor

                # Fitting the preprocessor
                X_transformed = prep_pipe_ll.fit_transform(X)

                self.prep_pipe_ll = prep_pipe_ll

                # RandomForest Classifier
                rf_model_ll = RandomForestClassifier(
                    n_estimators=50,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features=None,
                    random_state=42,
                    class_weight='balanced_subsample',
                )

                rf_model_ll.fit(X_transformed, y)

                self.rf_model_ll = rf_model_ll

                print("Model trained successfully.")

            # if update_hyperparameters:
            #     print(f'tuning {model} hyperparameters')
            #     # Do your hyperparameter tuning for the specified model
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
        # # INCOMPLETE: continue your work here
        # return pd.DataFrame(columns=['longitude', 'latitude'],
        #                     index=postcodes, dtype=dtype)

        # Ensure postcodes are in the format of an index
        postcodes = pd.Index(postcodes)

        # Copy the existing database and set index as postcode
        frame = self._postcodedb.copy()
        frame = frame.set_index('postcode')

        # Reindex to match input postcodes
        frame = frame.reindex(postcodes)

        # Ensure 'easting' and 'northing' columns are available
        if 'easting' not in frame.columns or 'northing' not in frame.columns:
            raise ValueError(
                "The database must contain 'easting' and 'northing' columns.")

        # Extract easting and northing
        eastings = frame['easting'].values
        northings = frame['northing'].values

        # Convert easting and northing to latitude and longitude
        latitudes, longitudes = get_gps_lat_long_from_easting_northing(
                                eastings, northings, dtype=dtype)

        # Create a DataFrame with latitude and longitude
        frame['latitude'] = latitudes
        frame['longitude'] = longitudes

        # Return only the latitude and longitude columns
        return frame[['latitude', 'longitude']]

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
            - 'mean': Use the mean for the labelled dataset.
            - 'constant': Use a constant value for imputation.
            - 'knn': Use k-nearest neighbours imputation from the
            labelled dataset.

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

        dataframe = dataframe.copy()

        # Ensure all expected columns exist in the dataframe
        for col in constant_values.keys():
            if col not in dataframe.columns:
                dataframe[col] = np.nan

        if method == 'mean':
            # Step 1: Fill columns based on `constant_values`
            for col, value in constant_values.items():
                if col in dataframe.columns and \
                        not pd.api.types.is_numeric_dtype(dataframe[col]):
                    dataframe[col] = dataframe[col].fillna(value)

            # Step 2: Fill remaining numerical columns with their mean values
            for col in dataframe.select_dtypes(
                                include=['float', 'int']).columns:
                dataframe[col] = dataframe[col].fillna(
                                dataframe[col].mean())

        elif method == 'constant':
            # Fill columns based on `IMPUTATION_CONSTANTS`
            for col, value in constant_values.items():
                if col in dataframe.columns:
                    dataframe[col] = dataframe[col].fillna(value)

        elif method == 'knn':

            # Step 1: Fill columns based on `constant_values`
            for col, value in constant_values.items():
                if col in dataframe.columns and \
                        not pd.api.types.is_numeric_dtype(dataframe[col]):
                    dataframe[col] = dataframe[col].fillna(value)

            # Step 2: Apply KNNImputer to the remaining numerical columns
            numeric_columns = dataframe.select_dtypes(
                                include=['float', 'int']).columns
            imputer = KNNImputer(n_neighbors=5)

            # Only impute numeric columns
            dataframe[numeric_columns] = imputer.fit_transform(
                                        dataframe[numeric_columns])

        else:
            raise ValueError(f"Unknown method '{method}'. "
                             f"Supported methods are 'mean', "
                             f"constant', and 'knn'.")

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

        if method == 'predicting_all_risks':
            # Step 1: Split postcodes into outward and inward parts
            print("Processing postcodes...")
            # outward, inward = zip(*[pc.split(" ") for pc in postcodes])

            # print(len(postcodes))

            labelled_drop_risk = self._postcodedb.drop(columns=["riskLabel"],
                                                       axis=1)

            Both_data_sets = pd.concat([labelled_drop_risk,
                                        self._unlabelled_unit_data], axis=0)
            # for pc in postcodes:
            # postcodes.loc[postcodes.index.max() + 1] = "test 123"
            # print(postcodes)
            Select_data = Both_data_sets[Both_data_sets['postcode']
                                         .isin(postcodes)]

            # postcodes_sele = pd.DataFrame(Select_data['postcode'])

            # print(postcodes_sele.columns)

            # postcodes_sele.set_index('postcode',inplace=True, drop=False)
            # Both_data_sets.set_index('postcode',inplace=True, drop=False)
            postcodes = pd.Series(postcodes)

            # postcodes_df = pd.DataFrame(postcodes, columns='postcode')

            Nan_data = postcodes[~postcodes.isin(Both_data_sets['postcode'])]

            # Nan_data.set_index('postcode', inplace=True, drop=False)
            # need remove dups in
            Select_data = Select_data.copy()
            Select_data.drop_duplicates(subset='postcode',
                                        keep="last",
                                        inplace=True)

            Select_post_code_list = Select_data['postcode'].to_list()

            Select_data.drop(columns=["nearestWatercourse"], axis=1)

            if not Select_data.empty:

                Select_data = Select_data.copy()
                Select_data[['outward_code', 'inward_code']] = \
                    Select_data['postcode'].str.split(" ", expand=True)

                Select_data.drop(columns=["postcode"], axis=1)
                # Step 2: Create a DataFrame for the input data
                input_data = Select_data

                Nan_data_list = list(Nan_data)
                # print(Select_post_code_list)
                # print(Nan_data_list)

                # Step 4: Predict flood classes using the trained model
                print("Predicting flood classes...")

                input_data = self.prep_pipe.transform(input_data)
                predictions = self.rf_model.predict(input_data)
                # print(f"nan data list{Nan_data_list}")
                fill_nan = np.empty(len(Nan_data_list))
                fill_nan.fill(np.nan)

                if len(Nan_data_list) != 0:

                    predictions = np.concatenate((predictions, fill_nan))
                else:
                    predictions = predictions
                # print(Select_post_code_list)
                postcodes = Select_post_code_list + Nan_data_list
                # predictions = predictions.reset_index(drop = True)

                # Step 5: Return the predictions as a Series

                return pd.Series(predictions,
                                 index=postcodes,
                                 name='Predicted Flood Class')

            else:

                Nan_data_list = list(Nan_data)
                fill_nan = np.empty(len(Nan_data_list))
                fill_nan.fill(np.nan)
                return pd.Series(data=fill_nan,
                                 index=Nan_data_list,
                                 name='Predicted Flood Class')

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

        if method == 'predicting_risk_from_easting_northing':
            input_data = pd.DataFrame(
                data=np.column_stack((eastings, northings)),
                columns=['easting', 'northing']
            )

            input_data = self.prep_pipe_en.transform(input_data)
            predictions = self.rf_model_en.predict(input_data)

            # Return the predictions as a Series
            location_index = [(e, n) for e, n in zip(eastings, northings)]
            return pd.Series(predictions,
                             index=location_index,
                             name='Predicted Flood Class')

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

        if method == 'predicting_risk_from_latitude_longitude':
            input_data = pd.DataFrame(
                data=np.column_stack((longitudes, latitudes)),
                columns=['latitude', 'longitude']
            )

            input_data = self.prep_pipe_ll.transform(input_data)
            predictions = self.rf_model_ll.predict(input_data)

            # Return the predictions as a Series
            location_index = [(long, lat) for long,
                              lat in zip(longitudes, latitudes)]
            return pd.Series(predictions,
                             index=location_index,
                             name='Predicted Flood Class')

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

        if method == 'house_price_rf':
            if isinstance(postcodes, list):
                postcodes = pd.Series(postcodes)
            elif isinstance(postcodes, np.ndarray):
                postcodes = pd.Series(postcodes.to_series())
            elif isinstance(postcodes, pd.Series):
                postcodes = postcodes
            elif isinstance(postcodes, pd.DataFrame):
                postcodes = pd.Series(postcodes[0])
            else:
                raise ValueError('postcodes must be a list, numpy array, '
                                 'pandas series or pandas dataframe')

            input_data = self.hp_data[self.hp_data['postcode'].isin(postcodes)]
            input_data = input_data.drop_duplicates(subset=['postcode'])

            valid_postcodes = input_data['postcode'].unique().tolist()
            input_data = input_data.drop(columns=['postcode'])

            features = pd.Series(
                data=self.house_model.predict(input_data),
                index=valid_postcodes,
                name='medianPrice',
            )

            return features

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
        method: str = 'local_authority'  # do_nothing before
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
        if method == 'local_authority':
            features = pd.DataFrame(
                data=np.column_stack((eastings, northings)),
                columns=['easting', 'northing']
            )

            predictions = self.rf_model.predict(features)

            idx = pd.MultiIndex.from_tuples([(est, nth) for est, nth in
                                             zip(eastings, northings)])
            return pd.Series(
                data=predictions,
                index=idx,
                name='localAuthority'
            )

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

        frame = self._postcodedb.copy()
        frame = frame.set_index('postcode')

        if method == 'historic_flooding':
            # split postcodes into outward and inward parts

            labelled_drop_hist = self._postcodedb\
                .drop(columns=['historicallyFlooded'])

            both_data_sets = pd.concat([labelled_drop_hist,
                                        self._unlabelled_unit_data])

            postcodes = pd.Series(postcodes)
            select_data = both_data_sets[both_data_sets["postcode"]
                                         .isin(postcodes)]
            select_data = select_data.drop(
                columns=['localAuthority',
                         "medianPrice",
                         "riskLabel",
                         "nearestWatercourse"], axis=1)
            select_data = select_data.drop_duplicates()
            select_post_code_list = select_data['postcode'].to_list()
            Nan_data = postcodes[~postcodes.isin(both_data_sets['postcode'])]
            print(f'nandatadf{Nan_data}')

            if not select_data.empty:

                input_data = select_data.copy()

                input_data[['outward', 'inward']] = \
                    input_data['postcode'].str.split(" ", expand=True)

                input_data = input_data.drop(columns=['postcode'], axis=1)

                input_data = self.preprocessor.transform(input_data)
                predictions = self.knn_model.predict(input_data)

                Nan_data_list = list(Nan_data)
                print(f'nandata{Nan_data_list}')
                fill_nan = np.empty(len(Nan_data_list))
                fill_nan.fill(np.nan)
                if len(Nan_data_list) != 0:
                    predictions = np.concatenate((predictions, fill_nan))

                else:
                    predictions = predictions

                postcodes = select_post_code_list + Nan_data_list
                print(f'predction{predictions}')
                print(f'fillnan{fill_nan}')
                print(len(predictions))
                return pd.Series(data=predictions,
                                 index=postcodes,
                                 name='historicallyFlooded')

            else:

                Nan_data_list = list(Nan_data)
                fill_nan = np.empty(len(Nan_data_list))
                fill_nan.fill(np.nan)
                return pd.Series(data=fill_nan,
                                 index=Nan_data_list,
                                 name='historicallyFlooded')

        if method == 'all_false':
            return pd.Series(
                data=np.full(len(postcodes), False),
                index=np.asarray(postcodes),
                name='historicallyFlooded',
            )
        else:
            raise NotImplementedError(f'method {method} not implemented')

    def estimate_total_value(self, postcodes: Sequence[str]) -> pd.Series:
        '''
        Return a series of estimates of the total property values
        of a sequence of postcode units or postcode sectors.

        The estimate is based on the median house price for the area and an
        estimate of the number of properties it contains.

        We assume that the total number of houses is number of households
        divided by the number of postcode units.

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

        if isinstance(postcodes, pd.Series):
            pass
        elif isinstance(postcodes, pd.DataFrame):
            postcodes = pd.Series(postcodes.iloc[:, 0])
        elif isinstance(postcodes, np.ndarray):
            postcodes = pd.Series(postcodes)
        else:
            raise TypeError('Must be Series, Dataframe or array')

        self._sector_data['postcodeSector'] = \
            self._sector_data['postcodeSector']\
            .str.replace('  ', ' ', regex=False).str.strip()
        sectors_check = (postcodes.str.split(' ', n=1)
                         .str[1].str.len() == 1).any()

        if sectors_check:
            part = postcodes
        else:
            part = (
                postcodes.str.split(' ', n=1).str[0]
                + ' '
                + postcodes.str.split(' ', n=1).str[1].str[0]
            )

        part = part.dropna()

        matched_rows = self._sector_data[
            self._sector_data['postcodeSector'].isin(part)]
        matched_postcodes = postcodes[
            part.isin(matched_rows['postcodeSector'])]

        matched_postcodes = matched_postcodes.reset_index(drop=True)
        matched_postcodes = pd.DataFrame(
            matched_postcodes, columns=['postcode'])
        matched_postcodes['postcodeSector'] = part

        if matched_postcodes.empty:
            raise ValueError('No data available for '
                             'the given postcode.')

        matched_postcodes = matched_postcodes.merge(
            self._sector_data, on='postcodeSector', how='left')

        matched_postcodes['num_houses'] = \
            matched_postcodes['households'] / \
            matched_postcodes['numberOfPostcodeUnits']

        combined_postcode = pd.concat(
            [self._postcodedb['postcode'],
             self._unlabelled_unit_data['postcode']], axis=0)

        def find_valid_postcode(part_value):
            try:
                valid_postcode = combined_postcode[
                    combined_postcode.str.startswith(part_value)].iloc[0]
                return valid_postcode
            except IndexError:
                return None

        if sectors_check:
            median_price_input = part.apply(find_valid_postcode)
        else:
            median_price_input = matched_postcodes['postcode']

        median_price = self.predict_median_house_price(
            median_price_input, method='house_price_rf')
        median_price = median_price.reset_index(drop=True)
        matched_postcodes['median_price'] = median_price

        matched_postcodes['totalvalue'] = (
            matched_postcodes['num_houses'] *
            matched_postcodes['median_price'])

        return pd.Series(
            matched_postcodes['totalvalue'].values,
            index=matched_postcodes['postcodeSector'])

    def estimate_annual_human_flood_risk(self, postcodes: Sequence[str],
                                         risk_labels: [pd.Series | None] = None
                                         ) -> pd.Series:
        '''
        Return a series of estimates of the risk to human life for a
        collection of postcodes.

        Risk is defined here as an impact coefficient multiplied by the
        estimated number of people under threat multiplied by the probability
        of an event.

        We assume that the total population is number of households
        divided by the number of postcode units.

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

        if isinstance(postcodes, pd.Series):
            pass
        elif isinstance(postcodes, pd.DataFrame):
            postcodes = pd.Series(postcodes.iloc[:, 0])
        elif isinstance(postcodes, np.ndarray):
            postcodes = pd.Series(postcodes)
        else:
            raise TypeError('Must be Series, '
                            'Dataframe or array')

        self._sector_data['postcodeSector'] = \
            self._sector_data['postcodeSector']\
            .str.replace('  ', ' ', regex=False).str.strip()

        part = (
            postcodes.str.split(' ', n=1).str[0]  # Outward code
            + ' '
            + postcodes.str.split(' ', n=1).str[1].str[0]
            # First character of inward code
        )

        part = part.dropna()

        matched_rows = self._sector_data[
            self._sector_data['postcodeSector'].isin(part)]
        matched_postcodes = postcodes[
            part.isin(matched_rows['postcodeSector'])]

        matched_postcodes = matched_postcodes.reset_index(drop=True)
        matched_postcodes = pd.DataFrame(
            matched_postcodes, columns=['postcode'])
        matched_postcodes['postcodeSector'] = part

        if matched_postcodes.empty:
            raise ValueError('No data available '
                             'for the given postcode.')

        matched_postcodes = matched_postcodes.merge(
            self._sector_data, on='postcodeSector', how='left')

        impact_coef = 0.1

        matched_postcodes['total_population'] = \
            matched_postcodes['headcount'] / \
            matched_postcodes['numberOfPostcodeUnits']

        flood_probability_initial = \
            self.predict_flood_class_from_postcode(
                matched_postcodes['postcode'],
                method='predicting_all_risks')

        flood_probability = flood_probability_initial\
            .reset_index(drop=True)

        matched_postcodes = matched_postcodes.reset_index(drop=True)
        matched_postcodes['flood_probability'] = flood_probability

        matched_postcodes['humanrisk'] = impact_coef \
            * matched_postcodes['total_population'] \
            * matched_postcodes['flood_probability']

        return pd.Series(
            matched_postcodes['humanrisk'].values,
            index=matched_postcodes['postcode'])

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

        if isinstance(postcodes, pd.Series):
            pass
        elif isinstance(postcodes, pd.DataFrame):
            postcodes = pd.Series(postcodes.iloc[:, 0])
        elif isinstance(postcodes, np.ndarray):
            postcodes = pd.Series(postcodes)
        else:
            raise TypeError('Must be Series, '
                            'Dataframe or array')

        self._sector_data['postcodeSector'] = \
            self._sector_data['postcodeSector']\
            .str.replace('  ', ' ', regex=False).str.strip()

        part = (
            postcodes.str.split(' ', n=1).str[0]  # Outward code
            + ' '
            + postcodes.str.split(' ', n=1).str[1].str[0]
            # First character of inward code
        )

        part = part.dropna()

        matched_rows = self._sector_data[
            self._sector_data['postcodeSector'].isin(part)]
        matched_postcodes = postcodes[
            part.isin(matched_rows['postcodeSector'])]

        matched_postcodes = matched_postcodes.reset_index(drop=True)
        matched_postcodes = pd.DataFrame(
            matched_postcodes, columns=['postcode'])
        matched_postcodes['postcodeSector'] = part

        if matched_postcodes.empty:
            raise ValueError('No data available '
                             'for the given postcode.')

        matched_postcodes = matched_postcodes.merge(
            self._sector_data, on='postcodeSector', how='left')

        damage_coef = 0.05
        totalpropvalue = self.estimate_total_value(
            matched_postcodes['postcode'])
        totalpropvalue = totalpropvalue.reset_index(drop=True)

        matched_postcodes['totalpropvalue'] = totalpropvalue

        flood_probability = \
            self.predict_flood_class_from_postcode(
                matched_postcodes['postcode'],
                method='predicting_all_risks')
        flood_probability = flood_probability.reset_index(drop=True)
        matched_postcodes['flood_probability'] = flood_probability

        matched_postcodes['economicrisk'] = (
            damage_coef * matched_postcodes['totalpropvalue']
            * matched_postcodes['flood_probability'])

        return pd.Series(
            matched_postcodes['economicrisk'].values,
            index=matched_postcodes['postcodeSector'])
