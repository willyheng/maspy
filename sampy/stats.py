"""Tools for statistical analysis."""

import pandas as pd, numpy as np, datetime as dt
import scipy

############################################
###  Regressions
############################################

def get_reg_slope(x, y):
    """Get the regression slope of two Series.
    
    Recommend to use pd.Series for both x and y with date indices to ensure
    regression alignment is correct
    
    Args:
        x, y (pd.Series or list): x/y variable in the regression respectively.
    Returns:
        float: regression slope coefficient
    """
    x_series = pd.Series(x, name='x')
    y_series = pd.Series(y, name='y')
    joint_series = pd.concat([x_series, y_series], axis=1).dropna()

    slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(joint_series['x'], joint_series['y'])
    return slope

def get_beta(df_x, df_y, start_date=None, end_date=None):
    """Calculate the beta of the variables in df_y to that of df_x.
    
    Args:
        df_x, df_y (DataFrame): column names and indices of both DataFrames should match
        start_date, end_date (str or Timestamp, optional): the start or end dates 
    
    Returns:
        pd.Series: betas, index will be the same columns as df_x
    """
    start_date = start_date or df_x.index.min()
    end_date = end_date or df_x.index.max()
    x = df_x.loc[start_date:end_date]
    y = df_y.loc[start_date:end_date]
    
    return pd.Series([get_reg_slope(x[c], y[c]) for c in x], \
                     index = x.columns)

############################################
###  Correlations
############################################

def corr_with_lags(y, x, lags=[0]):
    """Calculate correlation between two DataFrames or Series across a list of lags.
    
    Args:
        y, x (DataFrame, Series): If DataFrame, column names and index should match
        lags (list or int): the lags to run regression over (positive lags means y lags x)
    Returns:
        DataFrame or Series: Returns Series only if both y and x are series
            Otherwise, DataFrame of size `lags x column labels`, positive lags means y lags x
    """
    # Returns Series if both are series
    if isinstance(y, pd.Series) and isinstance(x, pd.Series): 
        return pd.Series([y.corr(x.shift(l)) for l in lags], index=lags)
    
    # If y is series but x is DataFrame, run series against all columns of DataFrame
    elif isinstance(y, pd.Series) and isinstance(x, pd.DataFrame):
        return pd.concat([pd.Series([y.corr(x[col].shift(l)) for col in x.columns], index = x.columns, name = l) for l in lags], axis=1).T
    
    # Else run each column in y against a corresponding column in x
    elif isinstance(y, pd.DataFrame) and (isinstance(x, pd.DataFrame) or isinstance(x, pd.Series)):  
        return pd.concat([pd.Series(y.corrwith(x.shift(l)), name=l) for l in lags], axis=1).T
    raise Exception("Invalid y or x types, only DataFrames or Series allowed")

############################################
###  Data Manipulation
############################################
    
def remove_outliers(df, threshold_sd=3.0):
    """Remove outliers from a DataFrame exceeding the threshold standard deviation, currently only handles column-wise removal.
    
    Args:
        df (DataFrame): Each column should be one timeseries
        threshold_sd (float): number of standard deviations beyond which should be removed
    
    Returns:
        DataFrame: Same shape as df, with datapoints exceeding threshold replaced with NaN
    """
    df = df.copy()
    norm = df.subtract(df.mean()).divide(df.std())
    df[(norm.abs() > threshold_sd)] = np.NaN
    
    return df