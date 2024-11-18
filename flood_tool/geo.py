'''
Module file for coordinate transformations, principally
converting between GPS and OSGB36 coordinates.

Functions present in this module are:

* get_easting_northing_from_gps_lat_long
* get_gps_lat_long_from_easting_northing
* dms2rad
* rad2dms
* euclidean_distance
* haversine_distance
'''

from numpy import (
    array,
    asarray,
    mod,
    sin,
    cos,
    tan,
    sqrt,
    arctan2,
    floor,
    rad2deg,
    deg2rad,
    stack,
    float64,
    subtract,
)
from scipy.linalg import inv

__all__ = [
    'get_easting_northing_from_gps_lat_long',
    'get_gps_lat_long_from_easting_northing',
    'dms2rad',
    'rad2dms',
    'euclidean_distance',
    'haversine_distance',
]


class Ellipsoid(object):
    '''Data structure for a global ellipsoid.'''

    def __init__(self, a, b, F_0):
        self.a = a
        self.b = b
        self.n = (a - b) / (a + b)
        self.e2 = (a**2 - b**2) / a**2
        self.F_0 = F_0
        self.H = 0


class Datum(Ellipsoid):
    '''Data structure for a global datum.'''

    def __init__(self, a, b, F_0, phi_0, lam_0, E_0, N_0, H):
        super().__init__(a, b, F_0)
        self.phi_0 = phi_0
        self.lam_0 = lam_0
        self.E_0 = E_0
        self.N_0 = N_0
        self.H = H


def dms2rad(deg, min=0, sec=0):
    '''Convert degrees, minutes, seconds to radians.

    Parameters
    ----------
    deg: array_like
        Angle in degrees.
    min: array_like
        (optional) Angle component in minutes.
    sec: array_like
        (optional) Angle component in seconds.

    Returns
    -------
    numpy.ndarray
        Angle in radians.
    '''
    deg = asarray(deg)
    return deg2rad(deg + min / 60.0 + sec / 3600.0)


# We treat degrees, minutes, seconds as a 3 component vector
# for the purposes of this function.
def rad2dms(rad, dms=False):
    '''Convert radians to degrees or degrees, minutes, seconds.

    Parameters
    ----------

    rad: array_like
        Angle in radians.
    dms: bool
        Use degrees, minutes, seconds format. If False, use decimal degrees.

    Returns
    -------
    numpy.ndarray
        Angle in degrees, minutes, seconds or decimal degrees.
    '''

    rad = asarray(rad)
    deg = rad2deg(rad)
    if dms:
        min = 60.0 * mod(deg, 1.0)
        sec = 60.0 * mod(min, 1.0)
        return stack((floor(deg), floor(min), sec.round(4)), dtype=rad.dtype)
    else:
        return deg


def dms2deg(deg, min, sec):
    '''Convert degrees, minutes, seconds to decimal degrees.

    Parameters
    ----------
    deg: array_like
        Angle in degrees.
    min: array_like
        Angle component in minutes.
    sec: array_like
        Angle component in seconds.

    Returns
    -------
    numpy.ndarray
        Angle in decimal degrees.
    '''
    deg = asarray(deg)
    dtype = deg.dtype
    return (deg + min / 60.0 + sec / 3600.0).astype(dtype)


osgb36 = Datum(
    a=6377563.396,
    b=6356256.910,
    F_0=0.9996012717,
    phi_0=deg2rad(49.0),
    lam_0=deg2rad(-2.0),
    E_0=400000,
    N_0=-100000,
    H=24.7,
)

wgs84 = Ellipsoid(a=6378137, b=6356752.3142, F_0=0.9996)


def lat_long_to_xyz(phi, lam, rads=False, datum=osgb36):
    '''Convert input latitude/longitude in a given datum into
    Cartesian (x, y, z) coordinates.

    Parameters
    ----------

    phi: array_like
        Latitude in degrees (if radians=False) or radians (if radians=True).
    lam: array_like
        Longitude in degrees (if radians=False) or radians (if radians=True).
    rads: bool (optional)
        If True, input latitudes and longitudes are in radians.
    datum: Datum (optional)
        Datum to use for conversion.
    '''
    if not rads:
        phi = deg2rad(phi)
        lam = deg2rad(lam)

    nu = datum.a * datum.F_0 / sqrt(1 - datum.e2 * sin(phi) ** 2)

    return array(
        (
            (nu + datum.H) * cos(phi) * cos(lam),
            (nu + datum.H) * cos(phi) * sin(lam),
            ((1 - datum.e2) * nu + datum.H) * sin(phi),
        )
    )


def xyz_to_lat_long(x, y, z, rads=False, datum=osgb36):
    p = sqrt(x**2 + y**2)

    lam = arctan2(y, x)
    phi = arctan2(z, p * (1 - datum.e2))

    for _ in range(10):
        nu = datum.a * datum.F_0 / sqrt(1 - datum.e2 * sin(phi) ** 2)
        dnu = (
            -datum.a
            * datum.F_0
            * cos(phi)
            * sin(phi)
            / (1 - datum.e2 * sin(phi) ** 2) ** 1.5
        )

        f0 = (z + datum.e2 * nu * sin(phi)) / p - tan(phi)
        f1 = (
            datum.e2 * (nu ** cos(phi) + dnu * sin(phi)) / p
            - 1.0 / cos(phi) ** 2
        )
        phi -= f0 / f1

    if not rads:
        phi = rad2dms(phi)
        lam = rad2dms(lam)

    return phi, lam


def get_easting_northing_from_gps_lat_long(
    phi, lam, rads=False, dtype=float64
):
    '''Get OSGB36 easting/northing from GPS latitude and longitude pairs.

    Parameters
    ----------
    phi: float/arraylike
        GPS (i.e. WGS84 datum) latitude value(s)
    lam: float/arraylike
        GPS (i.e. WGS84 datum) longitude value(s).
    rads: bool (optional)
        If true, specifies input is is radians, otherwise
        degrees are assumed.
    dtype: numpy.dtype (optional)
        Data type of output arrays.

    Returns
    -------
    numpy.ndarray
        Easting values (in m)
    numpy.ndarray
        Northing values (in m)

    Examples
    --------
    >>> get_easting_northing_from_gps_lat_long([55.5], [-1.54], dtype=int)
    (array([429157]), array([623009]))

    References
    ----------
    Based on the formulas in 'A guide to coordinate systems in Great Britain'.
    See also https://webapps.bgs.ac.uk/data/webservices/convertForm.cfm
    '''

    if not rads:
        phi = deg2rad(phi)
        lam = deg2rad(lam)

    phi, lam = WGS84toOSGB36(phi, lam, rads=True)

    datum = osgb36

    nu = datum.a * datum.F_0 / sqrt(1 - datum.e2 * sin(phi) ** 2)

    rho = (
        datum.a
        * datum.F_0
        * (1 - datum.e2)
        / (1 - datum.e2 * sin(phi) ** 2) ** 1.5
    )

    eta = sqrt(nu / rho - 1)

    M = (
        datum.b
        * datum.F_0
        * (
            (1 + datum.n + 1.25 * datum.n**2 + 1.25 * datum.n**3)
            * (phi - datum.phi_0)
            - (3 * datum.n + 3 * datum.n**2 + 21.0 / 8.0 * datum.n**3)
            * sin(phi - datum.phi_0)
            * cos(phi + datum.phi_0)
            + (15.0 / 8.0 * datum.n**2 + 15.0 / 8.0 * datum.n**3)
            * sin(2 * (phi - datum.phi_0))
            * cos(2 * (phi + datum.phi_0))
            - 35.0
            / 24.0
            * datum.n**3
            * sin(3 * (phi - datum.phi_0))
            * cos(3 * (phi + datum.phi_0))
        )
    )

    const_I = M + datum.N_0
    const_II = nu / 2.0 * sin(phi) * cos(phi)
    const_III = (
        nu / 24.0 * sin(phi) * cos(phi) ** 3 * (5 - tan(phi) ** 2 + 9 * eta**2)
    )
    const_IIIA = (
        nu
        / 720.0
        * sin(phi)
        * cos(phi) ** 5
        * (61 - 58 * tan(phi) ** 2 + tan(phi) ** 4)
    )
    const_IV = nu * cos(phi)
    const_V = nu / 6.0 * cos(phi) ** 3 * (nu / rho - tan(phi) ** 2)
    const_VI = (
        nu
        / 120.0
        * cos(phi) ** 5
        * (
            5
            - 18 * tan(phi) ** 2
            + tan(phi) ** 4
            + 14.0 * eta**2
            - 58.0 * tan(phi) ** 2 * eta**2
        )
    )

    E = (
        datum.E_0
        + const_IV * (lam - datum.lam_0)
        + const_V * (lam - datum.lam_0) ** 3
        + const_VI * (lam - datum.lam_0) ** 5
    )
    N = (
        const_I
        + const_II * (lam - datum.lam_0) ** 2
        + const_III * (lam - datum.lam_0) ** 4
        + const_IIIA * (lam - datum.lam_0) ** 6
    )

    return E.astype(dtype), N.astype(dtype)


def get_gps_lat_long_from_easting_northing(
    east, north, rads=False, dms=False, dtype=float64
):
    '''Get OSGB36 easting/northing from GPS latitude and
    longitude pairs.

    Parameters
    ----------
    east: float/arraylike
        OSGB36 easting value(s) (in m).
    north: float/arrayling
        OSGB36 easting value(s) (in m).
    rads: bool (optional)
        If true, specifies ouput is is radians.
    dms: bool (optional)
        If true, output is in degrees/minutes/seconds. Incompatible
        with rads option.
    dtype: numpy.dtype (optional)
        Data type of output arrays.

    Returns
    -------
    numpy.ndarray
        GPS (i.e. WGS84 datum) latitude value(s).
    numpy.ndarray
        GPS (i.e. WGS84 datum) longitude value(s).
    Examples
    --------
    >>> from numpy import isclose, array
    >>> lat, long = get_gps_lat_long_from_easting_northing([429157], [623009])
    >>> bool(isclose(lat, array([55.5])).all())
    True
    >>> bool(isclose(long, array([-1.54])).all())
    True

    References
    ----------
    Based on the formulas in 'A guide to coordinate systems in Great Britain'.
    See also https://webapps.bgs.ac.uk/data/webservices/convertForm.cfm
    '''

    # Ensure input is a numpy array.
    # Calculate in float64 to avoid overflow.

    east = asarray(east, float64)
    north = asarray(north, float64)

    datum = osgb36

    phi_dash = datum.phi_0
    M = 0

    while ((north - datum.N_0 - M) ** 2 > 1.0e-10).all():
        phi_dash = phi_dash + (north - datum.N_0 - M) / (datum.a * datum.F_0)
        M = (
            datum.b
            * datum.F_0
            * (
                (1 + datum.n + 1.25 * datum.n**2 + 1.25 * datum.n**3)
                * (phi_dash - datum.phi_0)
                - (3 * datum.n + 3 * datum.n**2 + 21.0 / 8.0 * datum.n**3)
                * sin(phi_dash - datum.phi_0)
                * cos(phi_dash + datum.phi_0)
                + (15.0 / 8.0 * datum.n**2 + 15.0 / 8.0 * datum.n**3)
                * sin(2 * (phi_dash - datum.phi_0))
                * cos(2 * (phi_dash + datum.phi_0))
                - 35
                / 24.0
                * datum.n**3
                * sin(3 * (phi_dash - datum.phi_0))
                * cos(3 * (phi_dash + datum.phi_0))
            )
        )

    nu = datum.a * datum.F_0 / sqrt(1 - datum.e2 * sin(phi_dash) ** 2)
    rho = (
        datum.a
        * datum.F_0
        * (1 - datum.e2)
        / (1 - datum.e2 * sin(phi_dash) ** 2) ** 1.5
    )
    eta2 = nu / rho - 1

    tphi = tan(phi_dash)
    sphi = 1.0 / cos(phi_dash)

    VII = tphi / (2 * rho * nu)
    VIII = (
        tphi
        / (24 * rho * nu**3)
        * (5 + 3 * tphi**2 + eta2 - 9 * (tphi**2) * eta2)
    )
    IX = tphi / (720 * rho * nu**5) * (61 + 90 * tphi**2 + 45 * tphi**4)
    X = sphi / nu
    XI = sphi / (6 * nu**3) * (nu / rho + 2 * tphi**2)
    XII = sphi / (120 * nu**5) * (5 + 28 * tphi**2 + 24 * tphi**4)
    XIIA = (
        sphi
        / (5040 * nu**7)
        * (61 + 662 * tphi**2 + 1320 * tphi**4 + 720 * tphi**6)
    )

    d_east = east - datum.E_0
    phi = phi_dash - VII * d_east**2 + VIII * d_east**4 - IX * d_east**6
    lam = (
        datum.lam_0
        + X * d_east
        - XI * d_east**3
        + XII * d_east**5
        - XIIA * d_east**7
    )

    phi, lam = OSGB36toWGS84(phi, lam, rads=True)

    # Ensure output is in the correct format.
    phi = phi.astype(dtype)
    lam = lam.astype(dtype)

    if not rads:
        phi = rad2dms(phi, dms)
        lam = rad2dms(lam, dms)

    return phi, lam


class HelmertTransform(object):
    '''Callable class to perform a Helmert transform.'''

    def __init__(self, s, rx, ry, rz, T):
        self.T = T.reshape((3, 1))

        self.M = array([[1 + s, -rz, ry], [rz, 1 + s, -rx], [-ry, rx, 1 + s]])

    def __call__(self, X):
        X = X.reshape((3, -1))
        return self.T + self.M @ X


class HelmertInverseTransform(HelmertTransform):
    '''Callable class to perform the inverse of a Helmert transform.'''

    def __init__(self, s, rx, ry, rz, T):
        super().__init__(s, rx, ry, rz, T)

        self.M = inv(self.M)

    def __call__(self, X):
        X = X.reshape((3, -1))
        return self.M @ (X - self.T)


OSGB36transform = HelmertTransform(
    20.4894e-6,
    -dms2rad(0, 0, 0.1502),
    -dms2rad(0, 0, 0.2470),
    -dms2rad(0, 0, 0.8421),
    array([-446.448, 125.157, -542.060]),
)

WGS84transform = HelmertInverseTransform(
    20.4894e-6,
    -dms2rad(0, 0, 0.1502),
    -dms2rad(0, 0, 0.2470),
    -dms2rad(0, 0, 0.8421),
    array([-446.448, 125.157, -542.060]),
)


def WGS84toOSGB36(phi, lam, rads=False):
    '''Convert WGS84 latitude/longitude to OSGB36 latitude/longitude.

    Parameters
    ----------
    phi : array_like or float
        Latitude in degrees or radians on WGS84 datum.
    lam : array_like or float
        Longitude in degrees or radians on WGS84 datum.
    rads : bool, optional
        If True, phi and lam are in radians. If False,
        phi and lam are in degrees.

    Returns
    -------
    tuple of numpy.ndarrays
        Latitude and longitude on OSGB36 datum in degrees or radians.
    '''
    xyz = OSGB36transform(
        lat_long_to_xyz(asarray(phi), asarray(lam), rads=rads, datum=wgs84)
    )
    return xyz_to_lat_long(*xyz, rads=rads, datum=osgb36)


def OSGB36toWGS84(phi, lam, rads=False):
    '''Convert OSGB36 latitude/longitude to WGS84 latitude/longitude.

    Parameters
    ----------
    phi : array_like or float
        Latitude in degrees or radians on OSGB36 datum.
    lam : array_like or float
        Longitude in degrees or radians on OSGB36 datum.
    rads : bool, optional
        If True, phi and lam are in radians. If False,
        phi and lam are in degrees.

    Returns
    -------
    tuple of numpy.ndarrays
        Latitude and longitude on WGS84 datum in degrees or radians.
    '''
    xyz = WGS84transform(
        lat_long_to_xyz(asarray(phi), asarray(lam), rads=rads, datum=osgb36)
    )
    return xyz_to_lat_long(*xyz, rads=rads, datum=wgs84)


def euclidean_distance(easting1, northing1, easting2, northing2):

    '''Calculate the Euclidean distance points.

    Parameters
    ----------
    easting1 : array_like or float
        Easting of the first point.
    northing1 : array_like or float
        Northing of the first point.
    easting2 : array_like or float
        Easting of the second point.
    northing2 : array_like or float
        Northing of the second point.

    Returns
    -------
    numpy.ndarray
        Distance between the two points in metres. Shape is the
        broadcast shape of the input arrays.
    '''
    return sqrt(subtract.outer(easting2, easting1) ** 2
                + subtract.outer(northing2, northing1) ** 2)


def haversine_distance(lat1, long1, lat2, long2, deg=True, dms=False):
    '''Calculate the Haversine distance between points.

    This implementation uses a spherical Earth approximation
    using the mean radius of the Earth, which is 6371009 metres.

    Parameters
    ----------
    lat1 : array_like or float
        Latitude of the first point.
    long1 : array_like or float
        Longitude of the first point.
    lat2 : array_like or float
        Latitude of the second point.
    long2 : array_like or float
        Longitude of the second point.
    deg : bool, optional
        If True, input latitudes and longitudes are in degrees.
        If False, input latitudes and longitudes are in radians.
    dms : bool, optional
        If True, input latitudes and longitudes are in degrees/minutes/seconds.
        If False, input latitudes and longitudes are in decimal degrees.

    Returns
    -------
    numpy.ndarray
        Distance between the two points in metres. Shape is the
        broadcast shape of the input arrays.
    '''
    # Convert to radians
    if deg:
        lat1 = deg2rad(lat1, dms=dms)
        long1 = deg2rad(long1, dms=dms)
        lat2 = deg2rad(lat2, dms=dms)
        long2 = deg2rad(long2, dms=dms)
    else:
        lat1 = asarray(lat1)
        long1 = asarray(long1)
        lat2 = asarray(lat2)
        long2 = asarray(long2)

    dlat = subtract.outer(lat2, lat1)
    dlong = subtract(long2, long1)

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlong / 2) ** 2
    c = 2 * arctan2(sqrt(a), sqrt(1 - a))

    return 6371009 * c.squeeze()
