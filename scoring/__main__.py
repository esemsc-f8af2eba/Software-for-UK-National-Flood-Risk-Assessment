"""Score the flood tool based on provided sample data."""

import os
import sys
import time

import pandas as pd
import numpy as np

import flood_tool
from .scores import SCORES

DEFAULT_TEST_DATA = os.path.join(
    flood_tool._data_dir, "postcodes_labelled.csv"
)

if len(sys.argv) > 1:
    test_data = pd.read_csv(sys.argv[1])
    unlabelled_file = sys.argv[2]
    labelled_sample = sys.argv[3]
    household_file = sys.argv[4]
else:
    test_data = pd.read_csv(DEFAULT_TEST_DATA)
    unlabelled_file = ""
    labelled_sample = ""
    household_file = ""

tool = flood_tool.Tool(unlabelled_file, labelled_sample, household_file)


print("Scoring flood class from postcode methods")
print("=========================================\n")
methods = flood_tool.flood_class_from_postcode_methods

for method, name in list(methods.items())[:3]:
    t1 = time.time()
    tool.fit([method])
    t2 = time.time()
    print(f"{name}: training time {t2-t1:0.5f} s")
    t1 = time.time()
    prediction = tool.predict_flood_class_from_postcode(
        test_data.postcode, method
    )
    t2 = time.time()
    print(f"{name}: prediction time {t2-t1:0.5f} s")
    prediction.reindex(test_data.postcode)
    score = sum(
        [
            SCORES[_p - 1, _t - 1]
            for _p, _t in zip(prediction, test_data.riskLabel)
        ]
    )
    print(f"{name}: score {score}")


print("Scoring flood class from location methods")
print("=========================================\n")
methods = flood_tool.flood_class_from_location_methods

for method, name in list(methods.items())[:3]:
    t1 = time.time()
    tool.fit([method])
    t2 = time.time()
    print(f"{name}: training time {t2-t1:0.5f} s")
    t1 = time.time()
    prediction = tool.predict_flood_class_from_OSGB36_location(
        test_data.easting, test_data.northing, method
    )
    t2 = time.time()
    print(f"{name}: prediction time {t2-t1:0.5f} s")
    prediction.reindex([(east, north) for
                        east, north in zip(test_data.easting,
                                           test_data.northing)])
    score = sum(
        [
            SCORES[_p - 1, _t - 1]
            for _p, _t in zip(prediction, test_data.riskLabel)
        ]
    )
    print(f"{name}: score {score}")


print("Scoring historic flooding methods")
print("=================================\n")
methods = flood_tool.historic_flooding_methods

for method, name in list(methods.items())[:3]:
    t1 = time.time()
    tool.fit([method])
    t2 = time.time()
    print(f"{name}: training time {t2-t1:0.5f} s")
    t1 = time.time()
    prediction = tool.predict_historic_flooding(
        test_data.postcode.to_list(), method
    )
    t2 = time.time()
    print(f"{name}: prediction time {t2-t1:0.5f} s")

    prediction = prediction.reindex(test_data.postcode)

    truth = test_data.set_index("postcode")
    truth = truth.historicallyFlooded

    tps = sum(prediction & truth)
    fps = sum(prediction & ~truth)
    tns = sum(~prediction & ~truth)
    fns = sum(~prediction & truth)

    if tps+fps == 0:
        print(f"{name}: precision nan")
    else:
        print(f"{name}: precision {tps/(tps+fps):0.3f}")
    if tps+fns == 0:
        print(f"{name}: recall nan")
    else:
        print(f"{name}: recall {tps/(tps+fns):0.3f}")
    print(f"{name}: accuracy {(tps+tns)/(tps+tns+fps+fns):0.3f}")
    if 2*tps+fps+fns == 0:
        print(f"{name}: f1 score nan")
    else:
        print(f"{name}: f1 score {2*tps/(2*tps+fps+fns):0.3f}")


print("Scoring house price methods")
print("===========================\n")
methods = flood_tool.house_price_methods

for method, name in list(methods.items())[:3]:
    t1 = time.time()
    tool.fit([method])
    t2 = time.time()
    print(f"{name}: training time {t2-t1:0.5f} s")
    t1 = time.time()
    prediction = tool.predict_median_house_price(
        test_data.postcode.to_list(), method
    )
    t2 = time.time()
    print(f"{name}: prediction time {t2-t1:0.5f} s")

    prediction = prediction.reindex(test_data.postcode)
    truth = test_data.set_index("postcode")
    truth = truth.medianPrice

    valid = (prediction.notna() & truth.notna())

    score = np.sqrt(np.mean((prediction.loc[valid]
                             - truth.loc[valid])**2))

    fps = sum(prediction.isna() & ~truth.notna())
    fns = sum(~prediction.isna() & truth.notna())

    print(f"{name}: score {score:0.2f}")
    print(f"{name}: false positives {fps}")
    print(f"{name}: false negatives {fns}")


print("Scoring local authority methods")
print("===============================\n")
methods = flood_tool.local_authority_methods

for method, name in list(methods.items())[:3]:
    t1 = time.time()
    tool.fit([method])
    t2 = time.time()
    print(f"{name}: training time {t2-t1:0.5f} s")
    t1 = time.time()
    prediction = tool.predict_local_authority(
        test_data.easting.to_list(),
        test_data.northing.to_list(),
        method
    )
    t2 = time.time()
    print(f"{name}: prediction time {t2-t1:0.5f} s")

    locations = [(east, north) for
                 east, north in zip(test_data.easting,
                                    test_data.northing)]

    truth = test_data.set_index(pd.Series(locations,
                                          index=test_data.index))
    truth = truth.localAuthority

    tps = sum(prediction == truth)

    print(f"{name}: accuracy {tps/len(truth):0.3f}")
