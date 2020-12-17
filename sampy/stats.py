"""Tools for statistical analysis."""


import pandas as pd, numpy as np, datetime as dt
import scipy
from matplotlib.dates import num2date

############################################
###  Formatting Functions
############################################

def to_date(date):
    """Converts date of various formats into datetime.date, including float dates"""
    
    # Convert iterables
    if hasattr(date, '__iter__') and not isinstance(date, str):
        if not isinstance(date, pd.Series):
            if isinstance(date[0], np.float):
                return [x.date() for x in num2date(date)]
            elif isinstance(date[0], pd.Period):
                return [x.to_timestamp().date() for x in date]
            elif type(date) == pd.DatetimeIndex:
                return date.date
        results = pd.to_datetime(date)
        if isinstance(results, pd.DatetimeIndex):
            return results.date
        return results.dt.date
        
    # Convert non-iterables
    if isinstance(date, float):
        return num2date(date).date()
    elif isinstance(date, pd.Period):
        return date.to_timestamp().date()
    if date:
        return pd.to_datetime(date).date()
    return date

############################################
###  Time Series Manipulations
############################################

def get_last(df):
    """Get most recent entry by index. Useful when there are multiple entries of each date."""
    df = df.copy()
    return df[df.index == df.index.max()]

def return_on_dates(price_idx, dates, period=1, relative=False):
    """Calculate the periodic return based on the provided periods.
    
    Args:
        price_idx (DataFrame): with one set of prices in each column
        dates (list or pd.Series): index of dates which you require returns on
        period (int, optional): number of periods lag in calculating, defaults to 1 period, which means return is 1 period change
        relative (bool, optional): if true, subtract the mean return in each period from the assets returns (ie outperformance of average)
    
    Returns:
        DataFrame containing the `dates` as index and returns across each period
    """
    # Handle if price_idx is multiIndex, only able to handle 2 levels
    if isinstance(price_idx.columns, pd.MultiIndex):
        df_new = pd.DataFrame(index=np.unique(dates), columns=pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=price_idx.columns.names))
    else: 
        df_new = pd.DataFrame(index=np.unique(dates))
    df_new.index.rename("Date", inplace=True)
        
    df_new = df_new.join(price_idx).pct_change(period)
    if relative:
        df_new = df_new.subtract(df_new.mean(axis=1), axis=0)

    return df_new

def extend_ts_all_dates(df, min_date = None, max_date = None):
    """Extend a dataframe's index to fill missing dates."""
    min_date = min_date or df.index.min()
    max_date = max_date or df.index.max()
    df_new = pd.DataFrame(index=pd.date_range(min_date, max_date))
    return df_new.join(df).sort_index(ascending=True).fillna(method='ffill')

def get_wt_contrib(df, value_name, weight_name, group_name = None, return_wt = False):
    """Get weighted contribution to a value.
    
    Args:
        df (DataFrame): Input data
        value_name (str): column name of value
        weight_name (str): column to use as weights
        group_name (str): column to use as grouping, if None, calculates weight using entire set
        return_weight (bool, optional): if True, returns both weight and contribution
        
    Returns:
        DataFrame or Series: if return_weight is True, returns DataFrame with columns `weight` and `wt_contrib`
            else if return_weight is False, returns Series `wt_contrib`
            
    Example:
        >>> get_wt_contrib(df, 'Flow %', Total Net Assets Start', 'Subtype')
    
    """
    if isinstance(group_name, str):
        group_name = [group_name]
    index_name = df.index.name
    group_name = [index_name] + group_name
    
    df = df.copy()
    if group_name:
        df['weight'] = df.groupby(group_name)[weight_name].transform(lambda x: x/x.sum())
    else:
        df['weight'] = df[weight_name].transform(lambda x: x/x.sum())
    df['wt_contrib'] = df[value_name] * df['weight']
    
    if return_wt:
        return df[['weight', 'wt_contrib']]
    return df['wt_contrib']


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
        df_x, df_y (DataFrame): column names and indices of both DataFramesshould match
        start_date, end_date (str or Timestamp, optional): the start or end dates 
    
    Returns:
        pd.Series: betas, index will be the same columns as df_x
    """
    if not start_date:
        start_date = df_x.index.min()
    if not end_date:
        end_date = df_x.index.max()
    x = df_x[start_date:end_date]
    y = df_y[start_date:end_date]
    
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