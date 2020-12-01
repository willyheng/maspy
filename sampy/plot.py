"""
Custom plotting functions.

Quick functions to plot certain custom plots
"""

import pandas as pd
import plotly.express as px
from matplotlib.dates import date2num

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

def add_event(ax, text, date, end_date=None, color='gray', fontsize=12):
    """Add event to matplotlib axes. 
    
    Adds a vertical line with the text as label. 
    
    Args:
        ax (Axes): axes of the figure
        text (str): label text
        date (str or datetime or date): date to plot line on
        end_date (str or datetime or date, optional): end date of event, if provided, area will be drawn instead of line
        color (str): color of line, text, or area
        fontsize (int): size of label text
    Returns:
        None
    """
    # Set style
    linestyle = dict(linewidth=1, color=color)
    
    if end_date:
        # If is a range
        ax.axvspan(date, end_date, alpha=0.2, facecolor=color)
        
        (x1,y1), (x2,y2) = ax.transData.transform([(date2num(date), 0),
                                                   (date2num(end_date), 0)])
        if (x2 - x1) > fontsize:
        # Add vlabel inside if sufficiently wide
            add_vlabel(ax, text, date, color=color)
        else:
        # Otherwise put outside
            add_vlabel(ax, text, end_date, color=color)
        
    else:
        # Insert vertical line at date
        ax.axvline(date, **linestyle)
        add_vlabel(ax, text, date, color)
    
def add_vlabel(ax, text, date, color='gray', fontsize=12):
    """Add label to chart (use in conjunction with vline).
    
    Aligns text to top or bottom based on where the datapoint is. 
    
    Args:
        ax (Axes): axes of the figure
        text (str): label text
        date (str or datetime or date): date to plot line on
        color (str): color of line, text, or area
        fontsize (int): size of label text
    Returns:
        None    
    """
    fontstyle = dict(size=fontsize, color=color, weight='bold')
    
    # Get values and min/max
    miny, maxy = ax.get_yaxis().get_view_interval()
    x = ax.get_lines()[0].get_xdata()
    y = ax.get_lines()[0].get_ydata()
    df = pd.Series(y, index=x)
    
    # Inserting text
    rel_pos = (df[:date][-1] - miny) / (maxy - miny)
    if rel_pos > 0.5:  # Data is nearer max
        ax.text(date, miny * 1.05, text, ha='left', va='bottom', rotation=-90, **fontstyle)
    else:
        ax.text(date, maxy * 0.99, text, ha='left', va='top', rotation=-90, **fontstyle)