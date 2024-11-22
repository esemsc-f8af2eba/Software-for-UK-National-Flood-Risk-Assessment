====================================
Regression and Classification models
====================================

**Any papers or AI tools used to assist you in creating your models should be described here and referenced appropriately.**
**reference**

- Sphinx (https://www.sphinx-doc.org)
- LaTeX (https://www.latex-project.org/get/)

Median House Price Prediction
=============================

This function predicts the median house price for a collection of postcodes using either a regression model or a default median value for England.

**Description**

The `predict_median_house_price` function generates predictions for median house prices. It supports two methods:
1. **`house_price_rf`**: Uses a pre-trained regression model to predict house prices based on postcode-specific features.
2. **`all_england_median`**: Assigns the national median house price (£245,000) to all input postcodes, regardless of their specific features.

**Methodology**

1. **For `house_price_rf`**:
   - Validates the input `postcodes` to ensure they are provided as a list, NumPy array, Pandas Series, or DataFrame.
   - Filters the internal house price dataset (`hp_data`) for rows corresponding to the input postcodes.
   - Removes duplicate postcodes from the dataset.
   - Identifies and excludes invalid postcodes (those not present in `hp_data`).
   - Uses the regression model (`house_model`) to predict house prices for valid postcodes.
   - Returns predictions as a Pandas Series indexed by the valid postcodes.

2. **For `all_england_median`**:
   - Assigns a fixed median value (£245,000) to all postcodes.
   - Returns a Pandas Series indexed by the input postcodes.

**Parameters**

- **postcodes** (sequence of str): A list, NumPy array, Pandas Series, or DataFrame containing postcode units.
- **method** (str, optional): Specifies the prediction method:
  - `'house_price_rf'`: Uses the regression model.
  - `'all_england_median'`: Uses the national median house price.

**Returns**

- A `pandas.Series` indexed by postcodes with the predicted median house prices.

**Example**

.. code-block:: python

   tool = Tool()

   # Predict using the regression model
   predictions = tool.predict_median_house_price(
       postcodes=['AB12 3CD', 'EF45 6GH'], 
       method='house_price_rf'
   )
   print(predictions)

Expected Output:

.. code-block:: text

   AB12 3CD    300000.0
   EF45 6GH    275000.0
   dtype: float64

.. code-block:: python

   # Predict using the all-England median
   predictions = tool.predict_median_house_price(
       postcodes=['AB12 3CD', 'EF45 6GH'], 
       method='all_england_median'
   )
   print(predictions)

Expected Output:

.. code-block:: text

   AB12 3CD    245000.0
   EF45 6GH    245000.0
   dtype: float64

**Limitations**

- The `house_price_rf` method depends on the quality and completeness of the `hp_data` dataset.
- Invalid or missing postcodes will not be included in the predictions.
- The `all_england_median` method does not account for postcode-specific variations.

**Error Handling**

- If the input `postcodes` is not a valid type (list, NumPy array, Pandas Series, or DataFrame), a `ValueError` is raised.
- If an unsupported `method` is specified, a `NotImplementedError` is raised.

---





Risk Prediction Models
=======================

Postcode-based Risk Prediction
------------------------------

This model predicts flood risk levels based on features extracted from a database using input postcodes.

**Description**

The model uses a combination of numerical and categorical features to predict flood risk (`riskLabel`), applying a preprocessing pipeline for feature engineering and a `RandomForestClassifier` for prediction.

The features used by the model include:

- **Numerical features**:
  - ``elevation``
  - ``distanceToWatercourse``
  - ``medianPrice``
  - ``easting``
  - ``northing``
- **Categorical features**:
  - ``soilType``
  - ``localAuthority``
  - ``historicallyFlooded``
  - ``outward_code`` (derived from postcodes)
  - ``inward_code`` (derived from postcodes)

Missing values are filled with default values:
- Numerical values are replaced with ``0``.
- Categorical values are replaced with ``'unknown'``.

**Methodology**

1. The model extracts relevant features from the postcode database (``_postcodedb``), including columns like ``elevation``, ``soilType``, and ``riskLabel``.
2. Postcodes are split into ``outward_code`` and ``inward_code`` for better granularity.
3. A preprocessing pipeline is applied to transform the features:
   - **Log Transformation**: Reduces skewness in ``elevation`` and ``distanceToWatercourse``.
   - **Robust Scaling**: Handles outliers in ``medianPrice``.
   - **One-Hot Encoding**: Converts categorical features (``soilType``, ``localAuthority``, ``historicallyFlooded``) into binary vectors.
   - **Ordinal Encoding**: Assigns integer values to ``outward_code`` and ``inward_code``.
   - **Standard Scaling and Clustering**: Applies scaling to ``easting`` and ``northing`` while augmenting features using KMeans clustering.
4. A ``RandomForestClassifier`` is trained on the transformed features with the following parameters:
   - ``n_estimators=300``
   - ``min_samples_split=5``
   - ``max_depth=None``
   - ``class_weight='balanced'``
5. The trained model (``rf_model``) and preprocessing pipeline (``prep_pipe``) are saved for future use.

**Inputs**

- **postcodes**: A list of postcodes used to extract features from the database.
- **riskLabel**: The target variable representing flood risk levels.

**Outputs**

- A trained ``RandomForestClassifier`` is saved for predicting flood risk labels.
- The preprocessing pipeline (``prep_pipe``) is saved for future feature transformations.

**Example**

.. code-block:: python

   models = ['predicting_all_risks']

   for model in models:
       if model == 'predicting_all_risks':
           tool.train_model_for_postcode_risks()

Expected Output:

.. code-block:: text

   Training model for Postcode-based Prediction...
   Preprocessing data...
   Model trained successfully.

**Limitations**

- The model depends on the quality and completeness of the ``_postcodedb`` dataset. Missing or inaccurate data may reduce prediction accuracy.
- Default values for missing features might introduce bias.
- The models performance might vary on unbalanced datasets or unseen postcode patterns.



Easting and Northing-based Risk Prediction
------------------------------------------

This model predicts flood risk levels based on geographic features such as `easting` and `northing`.

**Description**

The `predicting_risk_from_easting_northing` model uses numerical geographic features to predict flood risk levels (`riskLabel`). The model applies a preprocessing pipeline to standardize the input data and uses a `RandomForestClassifier` for prediction.

The features used by the model include:

- **Numerical features**:
  - ``easting``
  - ``northing``

**Methodology**

1. The model extracts relevant features from the postcode database (`_postcodedb`), including:
   - Features: ``easting`` and ``northing``
   - Target: ``riskLabel``
2. A preprocessing pipeline is applied:
   - **Standard Scaling**: Normalizes ``easting`` and ``northing`` for consistent ranges.
   - **K-Means Clustering**: Adds contextual grouping by clustering geographic locations into 10 clusters.
3. A `RandomForestClassifier` is trained on the transformed data with the following parameters:
   - ``n_estimators=400``: Number of trees in the forest.
   - ``max_depth=20``: Maximum depth of each tree.
   - ``min_samples_split=2``: Minimum samples required to split an internal node.
   - ``min_samples_leaf=1``: Minimum samples required to form a leaf node.
   - ``max_features='log2'``: Maximum number of features considered at each split.
   - ``class_weight='balanced'``: Adjusts weights inversely proportional to class frequencies.
4. The trained model (`rf_model_en`) and preprocessing pipeline (`prep_pipe_en`) are saved for future use.

**Preprocessing Pipeline**

The preprocessing pipeline includes the following steps:

1. **Standard Scaling**:
   - Normalizes numerical features (`easting`, `northing`) to have zero mean and unit variance.
2. **K-Means Clustering**:
   - Groups geographic locations into 10 clusters using the `KMeans` algorithm, which augments the dataset with a cluster feature.

**Inputs**

- **X**: A dataset containing the features `easting` and `northing`.
- **y**: A dataset containing the target variable `riskLabel`.

**Outputs**

- A trained `RandomForestClassifier` is saved for predicting flood risk levels (`rf_model_en`).
- A preprocessing pipeline (`prep_pipe_en`) is saved for feature transformations.

**Example**

.. code-block:: python

   tool = Tool()
   tool.fit(models=['predicting_risk_from_easting_northing'])

Expected Output:

.. code-block:: text

   Training model for easting/northing-based prediction...
   Model trained successfully.

**Limitations**

- The model's performance depends heavily on the quality and completeness of the `easting` and `northing` data.
- Preprocessing relies on the effectiveness of clustering through K-Means, which may not capture all spatial relationships.
- The model assumes balanced or adjusted class weights for fair predictions.

---

Latitude and Longitude-based Risk Prediction
--------------------------------------------

This model predicts flood risk levels based on geographic features derived from latitude and longitude.

**Description**

The `predicting_risk_from_latitude_longitude` model uses latitude and longitude values, which are derived from `easting` and `northing` coordinates using the `get_gps_lat_long_from_easting_northing` function from `geo.py`. The model preprocesses these features and trains a `RandomForestClassifier` to predict flood risk levels (`riskLabel`).

The features used by the model include:

- **Numerical features**:
  - ``latitude``: Derived from `easting` and `northing`.
  - ``longitude``: Derived from `easting` and `northing`.

**Methodology**

1. **Coordinate Conversion**:
   - Extracts `easting` and `northing` from the postcode database (`_postcodedb`).
   - Converts `easting` and `northing` into latitude and longitude using the `get_gps_lat_long_from_easting_northing` function in `geo.py`.
   - Combines latitude and longitude into a single dataset.

2. **Data Preparation**:
   - Features: ``latitude`` and ``longitude``.
   - Target: ``riskLabel``.

3. **Preprocessing Pipeline**:
   - **Standard Scaling**: Normalizes `latitude` and `longitude`.
   - **K-Means Clustering**: Adds contextual grouping by dividing the geographic area into 10 clusters.

4. **Model Training**:
   - A `RandomForestClassifier` is trained on the transformed data with the following parameters:
   - `n_estimators=100`
   - `max_depth=20`
   - `min_samples_split=2`
   - `min_samples_leaf=1`
   - `max_features='log2'`
   - `class_weight='balanced'`

5. Saves the trained model (`rf_model_ll`) and preprocessing pipeline (`prep_pipe_ll`) for future use.

**Preprocessing Pipeline**

The preprocessing pipeline includes:

1. **Standard Scaling**:
   - Normalizes numerical features (`latitude`, `longitude`) to have zero mean and unit variance.
2. **K-Means Clustering**:
   - Groups geographic locations into 10 clusters using the `KMeans` algorithm, augmenting the dataset with a cluster feature.

**Inputs**

- **X**: A dataset containing the features `latitude` and `longitude` derived from `easting` and `northing`.
- **y**: A dataset containing the target variable `riskLabel`.

**Outputs**

- A trained `RandomForestClassifier` is saved for predicting flood risk levels (`rf_model_ll`).
- A preprocessing pipeline (`prep_pipe_ll`) is saved for feature transformations.

**Example**

.. code-block:: python

   tool = Tool()
   tool.fit(models=['predicting_risk_from_latitude_longitude'])

Expected Output:

.. code-block:: text

   Training model for latitude/longitude-based prediction...
   Model trained successfully.

**Limitations**

- The model's accuracy depends on the precision of the `get_gps_lat_long_from_easting_northing` function in `geo.py`.
- Preprocessing relies on clustering with K-Means, which may not fully capture the geographic context of all locations.
- The model assumes balanced or adjusted class weights for fair predictions.

---




Local Authority Prediction Models
=================================

Local Authority Prediction
---------------------------

This model predicts the local authority for a given location based on geographical features like `easting` and `northing`.

**Description**

The model uses a `RandomForestClassifier` to predict the local authority (`localAuthority`) based on geographical coordinates:

- **Features**:
  - ``easting``: The eastward coordinate of the location.
  - ``northing``: The northward coordinate of the location.
- **Target**:
  - ``localAuthority``: The local authority corresponding to the location.

**Methodology**

1. Extract relevant columns from the postcode database (`_postcodedb`):
   - Features: ``easting`` and ``northing``
   - Target: ``localAuthority``
2. Train a `RandomForestClassifier` using the extracted features (`X`) and target labels (`y`).
3. Save the trained model (`rf_model`) for future predictions.

**Random Forest Classifier Parameters**:
- ``n_estimators=100``: Number of trees in the forest.
- ``max_depth=None``: No limit on the depth of the trees.
- ``random_state=42``: Ensures reproducibility of results.
- ``class_weight='balanced'``: Handles imbalanced class distributions by weighting classes inversely proportional to their frequency.

**Inputs**

- **X**: A dataset containing the features ``easting`` and ``northing``.
- **y**: A dataset containing the target variable ``localAuthority``.

**Outputs**

- A trained `RandomForestClassifier` saved as `rf_model`, which predicts the local authority for a given location.

**Example**

.. code-block:: python

   models = ['local_authority']

   for model in models:
       if model == 'local_authority':
           tool.train_local_authority_model()

Expected Output:

.. code-block:: text

   Training model for Local Authority Prediction...
   Model trained successfully.

**Limitations**

- The model's accuracy depends on the quality and completeness of the `_postcodedb` dataset.
- Missing or inaccurate geographical features (`easting`, `northing`) may reduce prediction accuracy.
- The model assumes a balanced dataset or adjusts using `class_weight='balanced'`.

---

**Next Steps**

- Add hyperparameter tuning to optimize the models performance. For example, use `RandomizedSearchCV` or `GridSearchCV` to adjust parameters like `n_estimators` or `max_depth`.
- Evaluate the model using additional metrics such as precision, recall, or F1-score.





