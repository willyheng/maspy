"""
Custom plotting functions.

Quick functions to plot certain custom plots
"""

import pandas as pd
import plotly.express as px

def plot_treemap(df, name_col, parent_col, value_col, **kwargs):
    """Plot a plotly treemap.
    
    Args:
        df (DataFrame): Needs to have columns of name, parent and value
        name_col (str): column name for the name in treemap
        parent_col (str): column name of the parent node
        value_col (str): column name of the values
    """
    df1 = pd.DataFrame({name_col: df[parent_col].unique(), parent_col: "", value_col: 0})
    df2 = pd.DataFrame({name_col: df[name_col], parent_col: df[parent_col], value_col: df[value_col]})
    df_tree = pd.concat([df1, df2])

    fig = px.treemap(
        df_tree,
        names = name_col,
        parents = parent_col,
        values = value_col,
        **kwargs
    )
    return fig