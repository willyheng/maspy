"""Tools for data manipulation."""
import pkg_resources
import pandas as pd, numpy as np, datetime as dt
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

def extend_ts_all_dates(ts, min_date = None, max_date = None, method='ffill'):
    """Extend a dataframe's index to fill missing dates.
    
    Args:
        ts (Series or DataFrame): needs to have dates as index
        min_date, max_date (str or Date): minimum and maximum dates, if None, will use min and max from ts
        method (str): fill method for pd.DataFrame().fillna()
    
    Returns:
        Series or DataFrame: if ts is Series, returns series
    """
    min_date = min_date or ts.index.min()
    max_date = max_date or ts.index.max()
    df_new = pd.DataFrame(index=pd.date_range(min_date, max_date))
    df_new = df_new.join(ts).sort_index(ascending=True).fillna(method=method)
    # Try to convert types back to original types, does not work on `int` type if NA is present
    #   Skip if unable to convert types back
    try:
        df_new = df_new.astype(ts.dtypes)
    except:
        pass
    
    if isinstance(ts, pd.Series):
        return df_new.squeeze()
    
    return df_new

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

def partition_by(df, by, continuous=True):
    """Partition dataframe of values into a list of dataframe with each one containing continuous equal values.
    
    Arg:
        df (DataFrame): DataFrame with dates as index
        by (str): column to break the dataframe by
        continuous (bool): If True, adds the first row of the next partition to the previous, 
            this allows timeseries plots to be continuous
        
    Return:
        list: list of DataFrames
    """
    
    def _find_x(df_curr):
        curr_series = df_curr.loc[:, by]
        if len(df_curr) == 0:
            return []
        find_end = df_curr[curr_series != curr_series.iloc[0]]
        if len(find_end) > 0: 
            end_index = find_end.index[0]
            if continuous:
                ret_df = df_curr.loc[:end_index].copy()
                ret_df.iloc[-1, ret_df.columns.get_loc(by)] = curr_series.iloc[0]
            else:
                ret_df = df_curr.loc[:end_index].iloc[:-1]
            return [ret_df] + _find_x(df_curr.loc[end_index:])
        return [df_curr]

    arr = _find_x(df)
    
    return arr

def series_to_bands(series, drop_zero=True):
    """Partition timeseries of values into 'bands', with each continuous series of equal values in a single partition.
    
    Values are returned as dataframe containing `start`, `end` and the series name.
    Bands can be used with `add_events`
    
    Arg:
        series (Series): series with dates as index
        drop_zero (bool): If True, drops periods where value = 0 (ie no shading required)
        
    Return:
        DataFrame: columns of `start`, `end` and `value_name`
    """
    def _find_x(series):
        if len(series) == 0:
            return []
        find_end = series[series != series[0]]
        if len(find_end) > 0: 
            return [pd.DataFrame({'start':series.index[0], 
                               'end':find_end.index[0], 
                               (series.name or "value"):series[0]}, index=[0])] + _find_x(series[find_end.index[0]:])
        else:
            return [pd.DataFrame({'start':series.index[0], 
                               'end':series.index[-1], 
                               (series.name or "value"):series[0]}, index=[0])]

    arr = _find_x(series.dropna())
    results = pd.concat(arr, axis=0).reset_index(drop=True) 
    if drop_zero:
        results = results[results.iloc[:,-1] != 0]       
    
    return results

############################################
###  Get Data
############################################

def sample_spx():
    """Get sample OHLCV S&P500 data from 2011 to 2022."""
    stream = pkg_resources.resource_stream(__name__, 'data/spx.csv')
    return pd.read_csv(stream, encoding='latin-1', index_col=0, parse_dates=True)    