"""Test flood tool."""

import numpy as np

from pytest import mark

import flood_tool.tool as tool


test_tool = tool.Tool()


def test_lookup_easting_northing():
    """Check"""

    data = test_tool.lookup_easting_northing(['RH16 2QE'])

    assert len(data.index) == 1
    assert 'RH16 2QE' in data.index

    assert np.isclose(data.loc['RH16 2QE', 'easting'], 535295).all()
    assert np.isclose(data.loc['RH16 2QE', 'northing'], 123643).all()


@mark.xfail  # We expect this test to fail until we write some code for it.
def test_lookup_lat_long():
    """Check"""

    data = test_tool.lookup_lat_long(["M34 7QL"])

    assert len(data.index) == 1
    assert 'RH16 2QE' in data.index

    assert np.isclose(data.loc['RH16 2QE', 'latitude'],
                      rtol=1.0e-3).all()
    assert np.isclose(data.loc['RH16 2QE', 'longitude'],
                      rtol=1.0e-3).all()


# Convenience implementation to be able to run tests directly.
if __name__ == "__main__":
    test_lookup_easting_northing()
    test_lookup_lat_long()
