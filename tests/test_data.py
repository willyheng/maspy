import pytest
import numpy as np, datetime as dt, pandas as pd
from matplotlib.dates import datestr2num
from pandas.testing import assert_frame_equal

from sampy.data import to_date, partition_by, series_to_bands, extend_ts_all_dates
    
def test_to_date_convert_single():
    assert to_date("2020-01-01") == dt.date(2020, 1,1)
    assert to_date(dt.datetime(2020, 1, 1, 1, 1)) == dt.date(2020,1,1)
    assert to_date(pd.Timestamp(year = 2020, month=6, day=1, hour=6)) == dt.date(2020,6,1)    
    assert to_date(pd.Period(year=2020, month=6, day=1, freq='d')) == dt.date(2020,6,1)
    assert to_date(datestr2num("1950-01-15")) == dt.date(1950, 1, 15)
    
def test_to_date_convert_series():
    assert np.array_equal(to_date(pd.Series(["2018-06-10", "2019-01-09", "2020-01-25"])) , 
                          pd.Series([dt.date(2018,6,10), dt.date(2019,1,9), dt.date(2020,1,25)]))
    assert np.array_equal(to_date(pd.Series([dt.datetime(2018,6,10, 5, 5), 
                                             dt.datetime(2019,1,9, 10, 10), 
                                             dt.datetime(2020,1,25, 11, 11)])) , 
                          pd.Series([dt.date(2018,6,10), dt.date(2019,1,9), dt.date(2020,1,25)]))
    
def test_to_date_convert_array():
    assert np.array_equal(to_date(["2018-06-10", "2019-01-09", "2020-01-25"]) , 
                          [dt.date(2018,6,10), dt.date(2019,1,9), dt.date(2020,1,25)])
    assert np.array_equal(to_date([dt.datetime(2018,6,10, 5, 5), 
                                   dt.datetime(2019,1,9, 10, 10), 
                                   dt.datetime(2020,1,25, 11, 11)]) , 
                          [dt.date(2018,6,10), dt.date(2019,1,9), dt.date(2020,1,25)])
    
def test_to_date_convert_datetimeindex():
    assert np.array_equal(to_date(pd.date_range("2019-05-01", "2019-05-03")), 
                          [dt.date(2019, 5, 1), dt.date(2019, 5, 2), dt.date(2019, 5, 3)])


# Generate dataframe to partition (with date)
df = []
df.append(pd.DataFrame({'filler1':['a', 'b'], 'partition_by':[0,0], 'filler2':[1,2]}, 
                       index=pd.date_range("2020-01-05", "2020-01-06")))
df.append(pd.DataFrame({'filler1':['c','d','e'], 'partition_by':[2,2,2], 'filler2':[3,4,5]}, 
                       index=pd.date_range("2020-01-07", "2020-01-09")))
df.append(pd.DataFrame({'filler1':['f','g'], 'partition_by':[0,0], 'filler2':[6,7]}, 
                       index=pd.date_range("2020-01-10", "2020-01-11")))
df.append(pd.DataFrame({'filler1':['h'], 'partition_by':[-3], 'filler2':[8]}, 
                       index=pd.date_range("2020-01-15", "2020-01-15")))
df.append(pd.DataFrame({'filler1':['i','j'], 'partition_by':[0,0], 'filler2':[9,10]}, 
                       index=pd.date_range("2020-01-16", "2020-01-17")))
df_all = pd.concat(df)

# Generate expected results
df_new = []
for i in range(4):
    df_new.append(df[i].append(df[i+1].iloc[0]))
    df_new[i].iloc[-1, df_new[i].columns.get_loc('partition_by')] = df_new[i].iloc[0, df_new[i].columns.get_loc('partition_by')]
df_new.append(df[-1])

def test_partition_by_with_dateindex_continuous():
    results_cont = partition_by(df_all, by='partition_by', continuous=True)
    assert_frame_equal(results_cont[0], df_new[0], check_freq=False)
    assert_frame_equal(results_cont[1], df_new[1], check_freq=False)
    assert_frame_equal(results_cont[2], df_new[2], check_freq=False)
    assert_frame_equal(results_cont[3], df_new[3], check_freq=False)
    assert_frame_equal(results_cont[4], df_new[4], check_freq=False)
    
def test_partition_by_with_dateindex_noncontinuous():
    results_nonc = partition_by(df_all, by='partition_by', continuous=False)
    assert_frame_equal(results_nonc[0], df[0], check_freq=False)
    assert_frame_equal(results_nonc[1], df[1], check_freq=False)
    assert_frame_equal(results_nonc[2], df[2], check_freq=False)
    assert_frame_equal(results_nonc[3], df[3], check_freq=False)
    assert_frame_equal(results_nonc[4], df[4], check_freq=False)
    


# Generate dataframe to partition (without date)
df_nd = []
df_nd.append(pd.DataFrame({'filler1':['a', 'b'], 'partition_by':[1,1], 'filler2':[1,2]}, index=[5, 6]))
df_nd.append(pd.DataFrame({'filler1':['c','d','e'], 'partition_by':[2,2,2], 'filler2':[3,4,5]}, index=[7, 8, 9]))
df_nd.append(pd.DataFrame({'filler1':['f','g'], 'partition_by':[1,1], 'filler2':[6,7]}, index=[10, 11]))
df_nd.append(pd.DataFrame({'filler1':['h'], 'partition_by':[-3], 'filler2':[8]}, index=[12]))
df_nd.append(pd.DataFrame({'filler1':['i','j'], 'partition_by':[1,1], 'filler2':[9,10]}, index=[13, 14]))
df_nd_all = pd.concat(df_nd)

# Generate expected results
df_nd_new = []
for i in range(4):
    df_nd_new.append(df_nd[i].append(df_nd[i+1].iloc[0]))
    df_nd_new[i].iloc[-1, df_nd_new[i].columns.get_loc('partition_by')] = df_nd_new[i].iloc[0, df_nd_new[i].columns.get_loc('partition_by')]
df_nd_new.append(df_nd[-1])

def test_partition_by_without_dateindex_continuous():
    results_nd_c = partition_by(df_nd_all, by='partition_by', continuous=True)
    assert_frame_equal(results_nd_c[0], df_nd_new[0], check_freq=False)
    assert_frame_equal(results_nd_c[1], df_nd_new[1], check_freq=False)
    assert_frame_equal(results_nd_c[2], df_nd_new[2], check_freq=False)
    assert_frame_equal(results_nd_c[3], df_nd_new[3], check_freq=False)
    assert_frame_equal(results_nd_c[4], df_nd_new[4], check_freq=False)
    
def test_partition_by_without_dateindex_noncontinuous():
    results_nd_nc = partition_by(df_nd_all, by='partition_by', continuous=False)
    assert_frame_equal(results_nd_nc[0], df_nd[0], check_freq=False)
    assert_frame_equal(results_nd_nc[1], df_nd[1], check_freq=False)
    assert_frame_equal(results_nd_nc[2], df_nd[2], check_freq=False)
    assert_frame_equal(results_nd_nc[3], df_nd[3], check_freq=False)
    assert_frame_equal(results_nd_nc[4], df_nd[4], check_freq=False)
    

def test_series_to_bands_drop_zero():
    exp_results = pd.DataFrame({
        'start': pd.to_datetime(["2020-01-07", "2020-01-15"]),
        'end': pd.to_datetime(["2020-01-10", "2020-01-16"]),
        'partition_by' : [2,-3]}, index=[1,3])
    
    results = series_to_bands(df_all.partition_by, drop_zero=True)
    assert_frame_equal(results, exp_results)
    
def test_series_to_bands_not_drop():
    exp_results = pd.DataFrame({
        'start': pd.to_datetime(["2020-01-05", "2020-01-07", "2020-01-10", "2020-01-15", "2020-01-16"]),
        'end': pd.to_datetime(["2020-01-07", "2020-01-10", "2020-01-15", "2020-01-16", "2020-01-17"]),
        'partition_by' : [0,2,0,-3,0]})
    
    results = series_to_bands(df_all.partition_by, drop_zero=False)
    assert_frame_equal(results, exp_results)

def test_extend_series_all_dates_ffill():
    exp_results = pd.DataFrame({'filler1': list('abcdefgggghij'), 
                      'partition_by': [0,0,2,2,2,0,0,0,0,0, -3, 0, 0],
                      'filler2': [1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 8, 9, 10]}, 
                     index=pd.date_range("2020-01-05", "2020-01-17"))
    results = extend_ts_all_dates(df_all, method='ffill')
    assert_frame_equal(results, exp_results, check_freq=False)
    
def test_extend_series_all_dates_bfill():
    exp_results = pd.DataFrame({'filler1': list('abcdefghhhhij'), 
                      'partition_by': [0,0,2,2,2,0,0,-3,-3,-3, -3, 0, 0],
                      'filler2': [1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 9, 10]}, 
                     index=pd.date_range("2020-01-05", "2020-01-17"))
    results = extend_ts_all_dates(df_all, method='bfill')
    assert_frame_equal(results, exp_results, check_freq=False)
    
def test_extend_series_all_dates_min_max():
    exp_results = pd.DataFrame({'filler1': [np.nan] + list('abcdefgggghijjj'), 
                      'partition_by': [np.nan] + [0,0,2,2,2,0,0,0,0,0, -3, 0, 0, 0, 0],
                      'filler2': [np.nan] + [1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 8, 9, 10, 10, 10]}, 
                     index=pd.date_range("2020-01-04", "2020-01-19"))
    results = extend_ts_all_dates(df_all, min_date="2020-01-04", max_date="2020-01-19", method='ffill')
    assert_frame_equal(results, exp_results, check_freq=False)
    