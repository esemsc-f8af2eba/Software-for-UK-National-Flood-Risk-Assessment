============
Data Formats
============

* Expand this section to describe any additional work on data formats you have done. *


Unit level Data
===============

Labelled data
-------------

Labelled postcode data is assumed to be in a tabular format, with the following columns:

- postcode: the full unit postcode of the area.
- easting: the OSGB36 easting coordinate of the centroid of the area.
- northing: the OSGB36 northing coordinate of the centroid of the area.
- soilType: the soil type of the area.
- elevation: the elevation (i.e. mean local height above mean sea level) of the area.
- nearestWaterway: the name of the nearest waterway to the postcode centroid.
- distanceToWaterway: the distance from the postcode centroid to the nearest waterway.
- localAuthority: the local authority in charge of the area.
- riskLabel: the flood risk label of the area on the 1-7 scale.
- medianPrice: the median house price for properties in the area.
- historicallyFlooded: a Boolean value indicating whether the area has a record of having been historically flooded.

Labelled data is assumed to have been cleaned and preprocessed to a limited extent, with no missing values, but
may require further processing before being used in a model.

Unlabelled postcode data
------------------------

Unlabelled unit level data is assumed to be in a tabular format, with many of the  columns as above, but without (at minimum)
the riskLabel, medianPrice, and historicallyFlooded columns. In some cases the soilType, elevation, nearestWaterway, and
distanceToWaterway and localAuthority data may also be missing for individual postcodes, and may need to be imputed, however
the easting and northing coordinates can be assumed to be present for all postcodes.

Sector Level Data
=================

Sector level data is assumed to be in a tabular format, with the following columns:

- postcodeSector: the sector of the postcode area.
- households: the number of households in the sector.
- numberOfPostcodeUnits: the number of postcode units in the sector.
- headcount: the total number of people in the sector based on the 2011 census.

Distrist level Data
===================

District level data is assumed to be in a tabular format, with the following columns:

- postcodeDistrict: the district of the postcode area.
- catsPerHousehold: the average number of cats per household in the district (based on government data).
- dogsPerHousehold: the average number of dogs per household in the district (based on government data.


Environment Agency Data
=======================

Station data
------------

Environment Agency station data is assumed to be provided in a tabular format, with the following columns:

- stationReference: the reference ID of the station.
- stationName: the given name of the station.
- latitude: the latitude of the station in decimal degrees (WGS84).
- longitude: the longitude of the station in decimal degrees (WGS84).
- maxOnRecord: the maximum water level on record at the station (for river levels only).
- minOnRecord: the minimum water level on record at the station (for river levels only).
- typicalRangeHigh: the typical high water level at the station (for river levels only).
- typicalRangeLow: the typical low water level at the station (for river levels only).

Reading data
------------

Environment Agency reading data is assumed to be provided in a tabular format, with the following columns:

- dateTime: the date and time of the reading.
- stationReference: the reference ID of the station (in the same format as the station data).
- parameter: the parameter being measured (e.g. rainfall water level, flow rate, etc.).
- qualifier: a qualifier for the reading (typically the measurement method).
- unitName: the unit of measurement for the reading. This is mm for rainfall and for water level readings usually one of:_[1]
    - mASD: metres above stage depth (i.e. height in comparison to a local fixed 0 height, which is usually close to, although not necessarily on the riverbed).
    - mAOD: metres above ordnance datum (i.e. height in comparison to the Ordnance Survey value for mean sea level, mostly used for tidal data).
- value: the value of the reading (in the specified unit).


_[1] https://check-for-flooding.service.gov.uk/how-we-measure-river-sea-groundwater-levels




