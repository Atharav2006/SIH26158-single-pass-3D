import math
from typing import Tuple, Union, List
import numpy as np

def wgs84_to_utm32n(
    lat_deg: float,
    lon_deg: float,
    alt_m: float = 0.0
) -> Tuple[float, float, float]:
    """
    Project WGS84 Geodetic coordinates (latitude, longitude, altitude) to UTM Zone 32N (EPSG:32632)
    using the standard Karney / Transverse Mercator formulation for the WGS84 ellipsoid.
    """
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)          # e^2 = (a^2 - b^2) / a^2
    e_prime2 = e2 / (1.0 - e2)    # e'^2 = (a^2 - b^2) / b^2
    k0 = 0.9996
    lon0_rad = math.radians(9.0)  # Zone 32 central meridian

    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)

    # Meridian distance M
    A0 = 1.0 - (e2 / 4.0) - (3.0 * e2**2 / 64.0) - (5.0 * e2**3 / 256.0)
    A2 = (3.0 * e2 / 8.0) + (3.0 * e2**2 / 32.0) + (45.0 * e2**3 / 1024.0)
    A4 = (15.0 * e2**2 / 256.0) + (45.0 * e2**3 / 1024.0)
    A6 = (35.0 * e2**3 / 3072.0)

    M = a * (A0 * lat_rad - A2 * math.sin(2.0 * lat_rad) + A4 * math.sin(4.0 * lat_rad) - A6 * math.sin(6.0 * lat_rad))

    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    tan_lat = math.tan(lat_rad)

    N = a / math.sqrt(1.0 - e2 * sin_lat**2) # Radius of curvature in prime vertical
    T = tan_lat**2
    C = e_prime2 * cos_lat**2
    A = (lon_rad - lon0_rad) * cos_lat

    # Easting
    easting = 500000.0 + k0 * N * (
        A +
        (1.0 - T + C) * (A**3) / 6.0 +
        (5.0 - 18.0 * T + T**2 + 72.0 * C - 58.0 * e_prime2) * (A**5) / 120.0
    )

    # Northing
    northing = k0 * (
        M + N * tan_lat * (
            (A**2) / 2.0 +
            (5.0 - T + 9.0 * C + 4.0 * C**2) * (A**4) / 24.0 +
            (61.0 - 58.0 * T + T**2 + 600.0 * C - 330.0 * e_prime2) * (A**6) / 720.0
        )
    )

    return float(easting), float(northing), float(alt_m)

def utm32n_to_wgs84(
    easting_m: float,
    northing_m: float,
    alt_m: float = 0.0
) -> Tuple[float, float, float]:
    """
    Inverse projection from UTM Zone 32N (EPSG:32632) back to WGS84 Geodetic coordinates (lat, lon, alt).
    """
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    e_prime2 = e2 / (1.0 - e2)
    k0 = 0.9996
    lon0_rad = math.radians(9.0)

    e1 = (1.0 - math.sqrt(1.0 - e2)) / (1.0 + math.sqrt(1.0 - e2))

    x = easting_m - 500000.0
    y = northing_m

    M = y / k0
    mu = M / (a * (1.0 - e2 / 4.0 - 3.0 * e2**2 / 64.0 - 5.0 * e2**3 / 256.0))

    phi1 = mu + (3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0) * math.sin(2.0 * mu) + \
           (21.0 * e1**2 / 16.0 - 55.0 * e1**4 / 32.0) * math.sin(4.0 * mu) + \
           (151.0 * e1**3 / 96.0) * math.sin(6.0 * mu) + \
           (1097.0 * e1**4 / 512.0) * math.sin(8.0 * mu)

    sin_phi1 = math.sin(phi1)
    cos_phi1 = math.cos(phi1)
    tan_phi1 = math.tan(phi1)

    N1 = a / math.sqrt(1.0 - e2 * sin_phi1**2)
    T1 = tan_phi1**2
    C1 = e_prime2 * cos_phi1**2
    R1 = a * (1.0 - e2) / ((1.0 - e2 * sin_phi1**2)**1.5)
    D = x / (N1 * k0)

    lat_rad = phi1 - (N1 * tan_phi1 / R1) * (
        D**2 / 2.0 -
        (5.0 + 3.0 * T1 + 10.0 * C1 - 4.0 * C1**2 - 9.0 * e_prime2) * D**4 / 24.0 +
        (61.0 + 90.0 * T1 + 298.0 * C1 + 45.0 * T1**2 - 252.0 * e_prime2 - 3.0 * C1**2) * D**6 / 720.0
    )

    lon_rad = lon0_rad + (
        D -
        (1.0 + 2.0 * T1 + C1) * D**3 / 6.0 +
        (5.0 - 2.0 * C1 + 28.0 * T1 - 3.0 * C1**2 + 8.0 * e_prime2 + 24.0 * T1**2) * D**5 / 120.0
    ) / cos_phi1

    return math.degrees(lat_rad), math.degrees(lon_rad), float(alt_m)

def utm32n_to_local_enu(
    easting_m: float,
    northing_m: float,
    altitude_m: float,
    origin_easting_m: float,
    origin_northing_m: float,
    origin_altitude_m: float
) -> Tuple[float, float, float]:
    """
    Convert UTM Zone 32N metric coordinates to a local centered ENU frame.
    """
    de = easting_m - origin_easting_m
    dn = northing_m - origin_northing_m
    du = altitude_m - origin_altitude_m
    return float(de), float(dn), float(du)
