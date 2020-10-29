import pytest
import numpy as np

from sampy.convert import country_to_code, country_to_eq_bbg, eq_bbg_to_country, show_all_conv

def test_if_country_converted_to_eq_bbg_ticker():
    assert np.array_equal(country_to_eq_bbg(['US', 'Europe']),
                          ['SPX Index', 'SXXP Index'])
    
def test_for_warning_with_invalid_country():
    with pytest.warns(UserWarning):
        results = country_to_eq_bbg(['US', 'Europe', "Venus"])
    assert np.array_equal(results,
                          ['SPX Index', 'SXXP Index', "Venus"])
