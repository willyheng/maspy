import pytest
import numpy as np, pandas as pd
from pandas.testing import assert_series_equal

from sampy.stats import get_reg_slope, get_beta

y = [1.8,  5.7,  6.6, 10.1, 12.6, 12.9, 15.1, 18. , 18.7, 21.6]
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
x2 = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
y2 = [ -68.1,  -74.9,  -82.4,  -89.3,  -95.8, -101.8, -110.2, -117.3, -125. , -130. ]

df1 = pd.DataFrame({'x1':x, 'x2':x2})
df2 = pd.DataFrame({'x1':y, 'x2':y2})

def test_reg_slope():
    assert np.round(get_reg_slope(x, y), 4) == 2.0697
    assert np.round(get_reg_slope(x2, y2), 4) == -6.9758
    
def test_beta():
    results = get_beta(df1, df2)
    exp_results = pd.Series([2.06969696969697, -6.9757575757575765], index=['x1', 'x2'])
    assert_series_equal(results, exp_results)
    