"""
To deal with EPFR data.

Set of tools to work with EPFR data
"""

import pandas as pd, numpy as np
import re
from functools import reduce
import warnings, os

#####################################################
###    Reading and data cleaning
#####################################################

def split_asset_class(df, inplace=False):
    """Split EPFR asset class into Geography and Type.
    
    Args:
        df (DataFrame): DataFrame from reading in Excel EPFR data
    
    Returns:
        DataFrame: Same as input DataFrame with additional columns of Geography and Type
    """
    if not inplace:
        df = df.copy()
    df[['Geography', 'Type']] = df['Asset Class'].str.split("-FF-", expand=True)

    if not inplace:
        return df

def clean_geog(df, inplace=False):
    """Clean EPFR Geography into shorter more readable forms."""
    if not inplace:
        df = df.copy()
        
    df['Geography'].replace({'All DM Funds': 'DM',
                             'All EM Funds': 'EM',
                             'India-Asia Ex-Japan': 'India',
                             'China-Asia Ex-Japan': 'China',
                             'USA-North America': 'US',
                             'Europe ex-UK Regional-Western Europe': 'Europe',
                             'France-Western Europe': 'France',
                             'Germany-Western Europe': 'Germany',
                             'Belgium-Western Europe': 'Belgium',
                             'Spain-Western Europe': 'Spain',
                             'Netherlands-Western Europe': 'Netherlands',
                             'Italy-Western Europe': 'Italy',
                             'Japan-Asia Pacific': 'Japan',
                             'United Kingdom-Western Europe': 'UK',
                             'Australia-Asia Pacific': 'Australia', 
                             'Global ex-US-Global': 'Global ex-US',
                             'Global-Global': 'Global',
                             'Canada-North America': 'Canada',
                             
                            }, inplace=True)
    
    if not inplace: 
        return df
    
def clean_corp(df):
    """Clean corporate data: Extract out subtypes and removing subtype from filters."""
    corp_types = ["High Yield", "Investment Grade", 
                  "Mortgage Backed", "Total Return", 
                  "Inflation Protected", "Bank Loan"]
    type_regex = "|".join(corp_types)
    df = df.copy()
    df['Subtype'] = df.Filters.str.extract("({})".format(type_regex))

    df['Filters'] = df.Filters. \
        str.replace("({}), ".format(type_regex), "", regex=True). \
        str.replace(", ({})".format(type_regex), "", regex=True). \
        str.replace("{}".format(type_regex), "None", regex=True) 
    
    return df

def extract_filters(df, custom_types = None, custom_name = "Custom"):
    """Extract relevant filters from the Filters column, such as Subtype (Corporate, Mixed, Sovereign), Quality and Style.
    
    Args:
        df (DataFrame): As per provided from EPFR's exported Excel
        custom_types (list, optional): if nothing is provided, filters will be extracted based on "Quality", "Style" and "Subtype" 
        custom_name (str): name of custom column
    
    Returns:
        DataFrame: Same as input df, but with relevant filters extracted into own columns and removed from Filter column
    """
    df = df.copy()
    
    if custom_types:
        custom_regex = "|".join(custom_types)
        df[custom_name] = df.Filters.str.extract(r"\b({})\b".format(custom_regex))
        if (~df[custom_name].isna()).sum() == 0:
            warnings.warn("No custom types was found")
            df.drop(custom_name, axis=1, inplace=True)
        else:
            df.fillna({custom_name:'None'}, inplace=True)
        
    else:
        types = {'Quality': ["All Quality", "High Yield", "Investment Grade", "Unassigned Quality"],
                 'Style': ["Mortgage Backed", "Total Return", "Municipal", "Inflation Protected", "Bank Loan"],
                 'Subtype': ["Corporate", "Mixed", "Sovereign"]}
        # Extract types from each category into respective columns
        for c in types:
            custom_regex = "|".join(types[c])
            df[c] = df.Filters.str.extract(r"\b({})\b".format(custom_regex))
            
        # Remove columns with all na
        df.dropna(axis=1, how='all', inplace=True)
        
        # Fill NaN with None
        df.fillna({'Quality':'None', 'Subtype':'None', 'Style':'None'}, inplace=True)
        
        # Fill custom_regex to be deleted from Filters column
        flattened_types = [y for x in types.values() for y in x]
        custom_regex = "|".join(flattened_types)
        
    df['Filters'] = df.Filters. \
            str.replace(r"\b({}), ".format(custom_regex), "", regex=True). \
            str.replace(r", ({})\b".format(custom_regex), "", regex=True). \
            str.replace(r"\b({})\b".format(custom_regex), "None", regex=True) 

    return df
    
def read_excel(filepath, preprocess = True, corp = False):
    """Read excel generated by EPFR, needs to be in unpivoted form.
    
    Args: 
        filepath (str): path to the Excel
        preprocess (bool, optional): if True, runs split_asset_class and clean_geog preprocessors
        corp (bool, optional): if True, runs clean_corp preprocessor
    
    Returns:
        DataFrame: EPFR data
    """
    df = pd.read_excel(filepath, index_col=0, parse_dates=True).sort_index(ascending=True)
    if preprocess:
        split_asset_class(df, inplace=True)
        clean_geog(df, inplace=True)
        df.drop(['Asset Class'], axis=1, inplace=True)
    
    if corp:
        df = clean_corp(df)
        
    return df

#####################################################
###    Filters for EPFR
#####################################################

def filter_and(df, filters):
    """Filter a dataframe based on EPFR filters. Must fulfil ALL criteria under the filters.
    
    Args:
        filters (str or list): Required filters
        
    Returns:
        DataFrame: Filtered to include all filters 
        
    Examples:
        >>> filter_and(df, ["Institutional"])
        >>> filter_and(df, ["Institutional", "Active"])
    """
    if isinstance(filters, str): filters = [filters]
    bool_arr = map(lambda x: df.Filters.str.contains(x), filters)
    return df[reduce(lambda x, y: x & y, bool_arr)].copy().sort_index()
    
def filter_or(df, filters):
    """Filter a dataframe based on EPFR filters. Returns those that fulfil ANY criteria under the filters.
    
    Args:
        filters (str or list): Required filters
        
    Returns:
        DataFrame: Filtered to include any filters
        
    Examples:
        >>> filter_or(df, ["Institutional"])
        >>> filter_or(df, ["Institutional", "Active"])
    """
    if isinstance(filters, str): filters = [filters]
    bool_arr = map(lambda x: df.Filters.str.contains(x), filters)
    return df[reduce(lambda x, y: x | y, bool_arr)].copy().sort_index()

def filter_only(df, filters):
    """Filter a dataframe based on EPFR filters. Returns those that fulfil ALL AND ONLY IF ALL criteria under the filters.
    
    Args:
        filters (str or list): Required filters
        
    Returns:
        DataFrame: Filtered to include if and only if all filters are met
        
    Examples:
        >>> filter_only(df, ["Institutional", "Active"])
    """
    if isinstance(filters, str): filters = [filters]
    df = filter_and(df, filters)
    filter_str = ", ".join(filters)
    return df[df.Filters.str.len() == len(filter_str)].copy().sort_index()

def filter_only_multi(df, list_of_filters):
    """Filter a dataframe based on multiple sets of filters.
    
    Returns those that fulfil ALL AND ONLY IF ALL of each set of criteria under the filters.
    
    Args:
        list_of_filters (list): List of list e.g. [["Institutional", "Active"], ["Institutional", "Passive"]] 
        
    Returns:
        DataFrame: Concatenates all matches in a single DataFrame sorted by date
        
    Examples:
        >>> filter_only_multi(df, [["Institutional", "Active"], ["Institutional", "Passive"]])
    """
    return pd.concat([filter_only(df, l) for l in list_of_filters]).sort_index()

#####################################################
###    Calculations
#####################################################

def weighted_flow_over_range(df, start_date, end_date, category="Geography"):
    """Calculate weighted flow as % of AUM over date range, as grouped by Retail/Insti and Active/Passive.
    
    Args:
        df (DataFrame): Should be in long form, and columns Total Net Asset Start, Filters and the category (usually Geography or Subtype)
        start_date, end_date (str or Timestamp)
    
    Returns:
        DataFrame: pivot table with weighted flows
    """
    headline = filter_only(df, "None").reset_index().set_index(["Date", category])['Total Net Assets Start']
    headline.name = 'Category Total Assets'
    
    breakdown = pd.concat([filter_only(df, ["Retail", "Active"]), 
           filter_only(df, ["Retail", "Passive"]),
           filter_only(df, ["Institutional", "Active"]),
           filter_only(df, ["Institutional", "Passive"])]).reset_index().set_index(["Date", category])

    breakdown = breakdown.join(headline)
    breakdown['weight'] = breakdown['Total Net Assets Start'] / breakdown['Category Total Assets']
    breakdown['weighted_flow'] = breakdown['Flow %'] * breakdown['weight']

    breakdown.head()
    breakdown = breakdown.reset_index().set_index("Date")
    
    breakdown = breakdown.loc[start_date:end_date] # Filter by date range
    
    summ = breakdown.groupby([category, "Filters"])['weighted_flow'].sum().reset_index()
    return summ[[category, 'Filters', 'weighted_flow']].pivot_table(values='weighted_flow', index=category, columns='Filters')