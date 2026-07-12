"""Physics + parity tests for the round-7 air-density carry feature.

The load-bearing guarantees: the carry index has the right sign (thin air ->
higher), roof handling is keyed only on roof TYPE + closed_share (identical at
train and inference), and the elevation unit is feet (venues_v1 convention).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlblib import features as F  # noqa: E402


def test_carry_sign_and_magnitude():
    coors = float(F.air_carry_index(75, 30, 840)[0])       # thin, high altitude
    sea_cold = float(F.air_carry_index(55, 60, 1015)[0])   # dense
    sea_hot = float(F.air_carry_index(90, 70, 1010)[0])    # warm -> less dense
    assert coors > sea_hot > sea_cold                       # monotone in carry
    assert 1.18 < coors < 1.30                              # Coors realistic band
    assert 0.95 < sea_cold < 1.01
    # Missing inputs -> NaN (routed natively by HistGB).
    assert np.isnan(F.air_carry_index(np.nan, 50, 1000)[0])
    assert np.isnan(F.air_carry_index(70, 50, np.nan)[0])


def test_station_pressure_is_feet_and_coors_is_low():
    # Coors elevation ~5190 ft -> ~837 hPa (NOT treating feet as metres).
    p = float(F.station_pressure_hpa(5190)[0])
    assert 820 < p < 855, p
    # Sea level ~1013 hPa.
    assert abs(float(F.station_pressure_hpa(0)[0]) - 1013.25) < 0.5


def test_blend_carry_parity_rules():
    indoor, outdoor = 1.03, 1.10
    # Dome always indoor, regardless of outdoor/share.
    assert F.blend_carry("Dome", indoor, outdoor, 0.9) == indoor
    # Open always outdoor.
    assert F.blend_carry("Open", indoor, outdoor, 0.5) == outdoor
    # Retractable blends by closed_share.
    b = F.blend_carry("Retractable", indoor, outdoor, 0.25)
    assert abs(b - (0.25 * indoor + 0.75 * outdoor)) < 1e-12
    # Retractable with missing outdoor falls back to indoor and vice versa.
    assert F.blend_carry("Retractable", indoor, np.nan, 0.5) == indoor
    assert F.blend_carry("Retractable", np.nan, outdoor, 0.5) == outdoor


def test_indoor_carry_altitude_effect():
    # A closed roof at altitude is still thinner air than a sea-level dome.
    denver_dome = float(F.indoor_carry_index(5190)[0])
    sea_dome = float(F.indoor_carry_index(10)[0])
    assert denver_dome > sea_dome > 1.0
