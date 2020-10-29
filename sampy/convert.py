"""
Conversion functions between country names, tickers and code etc.

Uses a data reference file for conversion, that needs to be updated to get access
to more conversion options
"""

import pandas as pd, numpy as np
import pkg_resources
import warnings

def _from_x_to_y(x_list, x, y):
    """Convert from x to y generically."""
    stream = pkg_resources.resource_stream(__name__, 'data/reference.csv')
    ref = pd.read_csv(stream)
    m = dict(zip(ref[x], ref[y]))
    
    x_list = pd.Series(x_list)
    x_missing = x_list[~x_list.isin(ref[x])]
    
    if x_missing.any():
          warnings.warn("At least one input not found: "+", ".join(x_missing))
            
    return x_list.replace(m).values

def country_to_code(x_list):
    """Convert from country name to country code."""
    return _from_x_to_y(x_list, "Country", "Country Code")

def country_to_eq_bbg(x_list):
    """Convert from country name to Bloomberg equity ticker.
    
    Examples:
        >>> country_to_eq_bbg(["US", "Europe", "China"])
        array(['SPX Index', 'SXXP Index', 'SHSZ300 Index'], dtype=object)
    """
    return _from_x_to_y(x_list, 'Country', 'Equity BBG Ticker')

def eq_bbg_to_country(x_list):
    """Convert from country name to Bloomberg equity ticker.
    
    Examples:
        >>> country_to_eq_bbg(["US", "Europe", "China"])
        array(['SPX Index', 'SXXP Index', 'SHSZ300 Index'], dtype=object)
    """
    return _from_x_to_y(x_list, 'Equity BBG Ticker', 'Country')

def show_all_conv():
    """Show all conversions currently available."""
    stream = pkg_resources.resource_stream(__name__, 'data/reference.csv')
    ref = pd.read_csv(stream)
    
    print(ref)
