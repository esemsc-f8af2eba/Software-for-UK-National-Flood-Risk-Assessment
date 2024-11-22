'''Analysis tools for flood data.'''

import os
import math

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ['plot_risk_map']

DEFAULT_POSTCODE_FILE = os.path.join(
    os.path.dirname(__file__), 'resources', 'postcodes_unlabelled.csv'
)


def eda_function(file_path, show_plot=True):
    """This function is used to perform
    exploratory data analysis on the data"""

    data = pd.read_csv(file_path)

    print("Showing you the first 5 rows of the data")

    print("Focusing now on any missing values in the data")
    missing = data.isnull().sum() / len(data) * 100
    print(missing)

    print("Checking for the correlation "
          "between different features in the data")

    num_cols = data.select_dtypes(include=[np.number]).columns
    data_corr = data[num_cols].corr()
    print(data_corr)

    print("\n ---Correlation Heatmap ---")
    sns.heatmap(data_corr, annot=True)
    plt.show()
    print("\n ---Pairplot for numerical features---")
    sns.pairplot(data[num_cols])
    plt.show()

    print("\n ---Checking numerical data distribution---")
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    ax = ax.flatten()
    for i in range(0, len(num_cols)):
        sns.histplot(data[num_cols[i]], ax=ax[i])
    plt.show()

    print("\n ---Checking the distribution of "
          "MedianPrice and impacts of outliers---")
    sns.histplot(data[data['medianPrice'] < 10e5]['medianPrice'])
    plt.show()

    print("\n ---Checking the description of the dataset---")
    data.describe()

    print("\n ---No outliers in the location data---")
    data[data['northing'].min() == data['northing']]
    data.sort_values(by='medianPrice', ascending=False)

    print("\n ---Distribution of data with BoxPlot---")
    sns.boxplot(data["medianPrice"], orient='h')

    return None


def plot_postcode_density(
    postcode_file=DEFAULT_POSTCODE_FILE,
    coordinate=['easting', 'northing'],
    dx=1000,
):
    '''Plot a postcode density map from a postcode file.'''

    pdb = pd.read_csv(postcode_file)

    bbox = (
        pdb[coordinate[0]].min() - 0.5 * dx,
        pdb[coordinate[0]].max() + 0.5 * dx,
        pdb[coordinate[1]].min() - 0.5 * dx,
        pdb[coordinate[1]].max() + 0.5 * dx,
    )

    nx = (
        math.ceil((bbox[1] - bbox[0]) / dx),
        math.ceil((bbox[3] - bbox[2]) / dx),
    )

    x = np.linspace(bbox[0] + 0.5 * dx, bbox[0] + (nx[0] - 0.5) * dx, nx[0])
    y = np.linspace(bbox[2] + 0.5 * dx, bbox[2] + (nx[1] - 0.5) * dx, nx[1])

    X, Y = np.meshgrid(x, y)

    Z = np.zeros(nx, int)

    for x, y in pdb[coordinate].values:
        Z[math.floor((x - bbox[0]) / dx), math.floor((y - bbox[2]) / dx)] += 1

    plt.pcolormesh(
        X, Y, np.where(Z > 0, Z, np.nan).T, norm=matplotlib.colors.LogNorm()
    )
    plt.axis('equal')
    plt.colorbar()


def plot_risk_map(risk_data, coordinate=['easting', 'northing'], dx=1000):
    '''Plot a risk map.'''

    bbox = (
        risk_data[coordinate[0]].min() - 0.5 * dx,
        risk_data[coordinate[0]].max() + 0.5 * dx,
        risk_data[coordinate[1]].min() - 0.5 * dx,
        risk_data[coordinate[1]].max() + 0.5 * dx,
    )

    nx = (
        math.ceil((bbox[1] - bbox[0]) / dx),
        math.ceil((bbox[3] - bbox[2]) / dx),
    )

    x = np.linspace(bbox[0] + 0.5 * dx, bbox[0] + (nx[0] - 0.5) * dx, nx[0])
    y = np.linspace(bbox[2] + 0.5 * dx, bbox[2] + (nx[1] - 0.5) * dx, nx[1])

    X, Y = np.meshgrid(x, y)

    Z = np.zeros(nx, int)

    for x, y, val in risk_data[coordinate + 'risk'].values:
        Z[
            math.floor((x - bbox[0]) / dx), math.floor((y - bbox[2]) / dx)
        ] += val

    plt.pcolormesh(
        X, Y, np.where(Z > 0, Z, np.nan).T, norm=matplotlib.colors.LogNorm()
    )
    plt.axis('equal')
    plt.colorbar()
