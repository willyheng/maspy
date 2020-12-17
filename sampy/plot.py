"""
Custom plotting functions.

Quick functions to plot certain custom plots
"""

import pandas as pd, numpy as np
import plotly.express as px
from matplotlib.dates import date2num
from .stats import to_date

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

def series_to_bands(series):
    """Partition timeseries of values into 'bands', with each continuous series of equal values in a single partition.
    
    Values are returned as dataframe containing `start`, `end` and the series name.
    Bands can be used with `add_events`
    
    Arg:
        series (Series): series with dates as index
        
    Return:
        DataFrame: columns of `start`, `end` and `value_name`
    """
    def _find_x(series):
        if len(series) == 0:
            return []
        find_end = series[series != series[0]]
        if len(find_end) > 0: 
            return [pd.Series({'start':series.index[0], 
                               'end':find_end.index[0], 
                               (series.name or "value"):series[0]})] + _find_x(series[find_end.index[0]:])
        else:
            return [pd.Series({'start':series.index[0], 
                               'end':series.index[0], 
                               (series.name or "value"):series[0]})]

    arr = _find_x(series.dropna())
    results = pd.concat(arr, axis=1).T
    
    return results

def add_events(axes, events, color='gray', alpha=0.5):
    """Add dataframe of events to chart.
    
    Args:
        axes (Axes or list): ax or axes to add events to. If multiple subplots require events, just pass list of Axes
        events (DataFrame): needs columns `start`, `end`. Optional columns are `label` and `color`.
            If no `color` is provided, gray is used
        color (str, optional): color of bands, used only if events does not have `color` column
        alpha (float, optional): alpha of bands
    
    Returns:
        Axes    
    """
    if not (isinstance(axes, np.ndarray) or isinstance(axes, list)):
        axes = [axes]
    min_date = min(to_date([l.get_xdata()[0] for ax in axes for l in ax.lines[:10]]))
    max_date = max(to_date([l.get_xdata()[-1] for ax in axes for l in ax.lines[:10]]))

    valid_events = events[(to_date(events.end) >= min_date) & (to_date(events.start) <= max_date)].copy()
    valid_events['start'] = np.where(to_date(valid_events.start) < min_date, min_date, to_date(valid_events.start))
    valid_events['end'] = np.where(to_date(valid_events.end) > max_date, max_date, to_date(valid_events.end))

    if "label" not in valid_events.columns:
        valid_events['label'] = ""
    if 'color' not in valid_events.columns:
        valid_events['color'] = color

    for idx, (start, end, label, c) in valid_events[['start', 'end', 'label', 'color']].iterrows():
        for ax in axes:
            add_event(ax, label, start, end, color=c, alpha=alpha)   
    return axes

def add_event(ax, text, date, end_date=None, color='gray', fontsize=12, alpha=0.5):
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
    
    date = to_date(date)
    end_date = to_date(end_date)
    min_date = to_date(ax.get_xlim()[0])
    max_date = to_date(ax.get_xlim()[1])
    
    if end_date:
        # If is a range
        if end_date < min_date or date > max_date:
            return ax
        date = max(date, min_date)
        end_date = min(end_date, max_date)
        ax.axvspan(date, end_date, alpha=alpha, facecolor=color)
        
        (x1,y1), (x2,y2) = ax.transData.transform([(date2num(date), 0),
                                                   (date2num(end_date), 0)])
        if text and len(text) > 0:
            if (x2 - x1) > fontsize:
            # Add vlabel inside if sufficiently wide
                add_vlabel(ax, text, date, color=color)
            else:
            # Otherwise put outside
                add_vlabel(ax, text, end_date, color=color)
        
    elif min_date <= date <= max_date:  # Only plot line if date of event is within range
        # Insert vertical line at date
        ax.axvline(date, **linestyle)
        add_vlabel(ax, text, date, color)
    return ax
    
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