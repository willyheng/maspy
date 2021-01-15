"""
Custom plotting functions.

Quick functions to plot certain custom plots
"""

import pandas as pd, numpy as np, seaborn as sns
import plotly.express as px
from matplotlib.dates import date2num
from matplotlib.lines import Line2D
import matplotlib
import warnings
import numbers
import matplotlib.pyplot as plt
from .data import to_date, partition_by

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

def lineplot_color(x, y, data, hue=None, color=None, label=None, legend_loc='best', ax=None, palette=None, **kwargs):
    """Plot a lineplot with multiple colors using seaborn.
    
    Either `hue` must be provided or both `color` and `label` must be provided, 
    
    Args:
        x (str): column name from data
        y (str): column name from data
        data (DataFrame): Data to plot
        hue (str): column name for hue, not required if `label` and `color` provided
        label (str): column name for labels (column should match that of colors), not used if `hue` provided
        color (str): column name for colors, not used if `hue` provided
        legend_loc (str): location of legend (as per matplotlib), hides legend if False. Defaults to `best`. 
            Other possible options for e.g. `upper left`, `lower right`, `center` etc
        ax (Axes, optional): ax to plot
        palette (str or seaborn.palettes._ColorPalette, optional): Only used if `hue` provided. Defaults to seaborn default
            If colorpalette is provided, number of colours must match number of unique hues
            For options, go to https://seaborn.pydata.org/tutorial/color_palettes.html
        **kwargs (optional): additional parameters to pass to `seaborn.lineplot` or `ax.set` 
            Parameters to be passed to lineplot includes `linewidth` and `linestyle`, 
            Other parameters will be passed to ax.set, for instance `title`, `xlabel`, `ylabel` etc
        
    Return:
        Axes
    """        
    # Break kwargs into for lineplot and for ax.set
    lineplot_options = ['linewidth', 'linestyle', 'palette']
    line_dict = {key:kwargs[key] for key in kwargs if key in lineplot_options}
    set_dict = {key:kwargs[key] for key in kwargs if key not in lineplot_options}
    
    # If hue is used
    if hue:
        data = data.copy()
        
        # If palette provided is a `str`, generate color_palette
        if palette and isinstance(palette, str):
            palette = sns.color_palette(palette, n_colors=len(np.unique(data[hue])))
        else:
            # Set palette as diverging green to red if hue is numeric, otherwise use seaborn default
            #palette = palette or (sns.diverging_palette(145, 15, s=100, l=50, center="dark", n=len(np.unique(data[hue]))) if isinstance(data[hue][0], numbers.Number) else None)
            # Set palette as `coolwarm` if hue is numeric, otherwise use seaborn default
            palette = palette or (sns.color_palette("coolwarm", n_colors=len(np.unique(data[hue]))) if isinstance(data[hue][0], numbers.Number) else None)
        line_dict['palette'] = palette

        # Set hue as categorical data
        data[hue] = data[hue].astype('category')
        # Break data into continuous parts and plot them
        broken = partition_by(data, hue, continuous=True)
        for d in broken:
            ax = sns.lineplot(x=x, y=y, hue=hue, data=d.reset_index(), **line_dict)

        # Display only unique legend labels
        hand, labl = ax.get_legend_handles_labels()
        uniq_labl = np.unique(labl, return_index=True)
        ax.legend(np.array(hand)[uniq_labl[1]], uniq_labl[0], loc=legend_loc)

        # Display only unique legend labels
        hand, labl = ax.get_legend_handles_labels()
        uniq_labl = np.unique(labl, return_index=True)
        ax.legend(np.array(hand)[uniq_labl[1]], uniq_labl[0], loc=legend_loc)
        
    # Otherwise both color and label must be provided
    elif color and label:
        if palette:
            warnings.warn("`Palette` ignored, as it is only used for `hue`. Manual colors have to be provided in `color` column")
        # Warn if label value and color value do not match
        color_df = data[[label, color]].reset_index(drop=True).drop_duplicates().sort_values(label)
        if len(color_df[label].drop_duplicates()) < len(color_df) or len(color_df[color].drop_duplicates()) < len(color_df):
            warnings.warn("Labels do not correspond to unique colors.")
        
        broken = partition_by(data, color, continuous=True)
        for d in broken:
            ax = sns.lineplot(x=x, y=y, color=d.color.iloc[0], data=d.reset_index(), ax=ax, **line_dict)
        # Add legend
        lw = ax.get_lines()[0].get_linewidth()

        custom_lines = [Line2D([0], [0], color=color, lw=lw) for idx, (label, color) in color_df.iterrows()]
        if legend_loc:
            ax.legend(custom_lines, color_df[label], loc=legend_loc)
    else:
        raise Exception("If `hue` is not provided, then both `color` and `label` needs to be provided")
        
    # Set stuff like title, xlabels etc
    if kwargs:
        ax.set(**set_dict)
    
    return ax

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
    min_date = min(to_date([l.get_xdata()[0] for ax in axes for l in ax.lines if len(l.get_xdata()) > 0] ))
    max_date = max(to_date([l.get_xdata()[-1] for ax in axes for l in ax.lines if len(l.get_xdata()) > 0]))

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

def plot_grid_heatmap(df, scalexy=True, palette=sns.color_palette("vlag_r", n_colors=50),
                      annot=True, fmt="0.1f", vmin=None, vmax=None, center=None, title=None,
                      figsize=(12,10)):
    """Plot a multiindex dataframe into heatmap by breaking into grids.

    Supports Multiindex on either column or row or both. Needs to have at least one axis to have multiindex.

    Args:
        df (DataFrame): dataframe of heatmap to plot
        scalexy (bool): If True, will resize subplots based on number of entries, otherwise all subplots equal size
        palette (seaborn.color_palette): Defaults to "vlag_r" (red to white to blue as value increases)
        annot (bool): If True, prints values as annotation
        fmt (str): format of annotation
        vmin, vmax (float, optional): values corresponding to extreme colors,
            defaults to extreme values in the dataframe
        center (float, optional): the value corresponding to the centre in colors
        title (str, optional): the suptitle to be displayed at the top of plot
        figsize (tuple): figsize of the plot

    Returns:
        Figure
    """
    df = df.copy()

    # Handling cases where columns or indices are not MultiIndex
    if not isinstance(df.columns, pd.MultiIndex) and not isinstance(df.index, pd.MultiIndex):
        raise Exception("Neither column nor index in df is MultiIndex, this function needs at least one to be MultiIndex")
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_product([[df.columns.name or ""], df.columns])
    if not isinstance(df.index, pd.MultiIndex):
        df.index = pd.MultiIndex.from_product([[df.index.name or ""], df.index])

    # Grid for heatmaps
    nrows = len(df.index.levels[0])
    ncols = len(df.columns.levels[0])

    # Set height and width ratios based on number of entries, used only if scalexy=True
    height_ratios = [len(df.loc[idx]) for idx in df.index.levels[0]] # Set height ratio to number of series in each group
    width_ratios = [df.loc[:,idx].shape[1] for idx in df.columns.levels[0]] # Set width ratio

    # Set subplot figures
    fig, axes = plt.subplots(figsize=figsize,
                             nrows=nrows, ncols=ncols,
                             sharex='col', sharey='row',
                             gridspec_kw={'height_ratios': height_ratios,
                                         'width_ratios': width_ratios} if scalexy else {})

    # Add colorbar axes
    cbar_ax = fig.add_axes([.91, .3, 0.03, .4])
    vmin = vmin or min(df.min().min(), -abs(df.max().max()))
    vmax = vmax or -vmin

    # Parameters for all plots
    params = {"vmin":vmin,
             "vmax":vmax,
             "center":center,
             "cmap":palette,
             "annot":annot,
             "fmt":fmt,
             "linewidths":0.5,
             "linecolor":"white"}

    # Plot heatmaps for every index
    for r_idx, r in enumerate(df.index.levels[0]):
        for c_idx, c in enumerate(df.columns.levels[0]):
            curr_ax = axes.flat[r_idx*ncols+c_idx]
            sns.heatmap(df.loc[r, c],
                        ax=curr_ax,
                        cbar_ax=cbar_ax if r_idx==c_idx==0 else None,
                        cbar=True if r_idx==c_idx==0 else False,
                        **params).set(ylabel="", xlabel="")
            # Add group labels for columns if last row
            if r_idx == nrows-1 and ncols > 1:
                curr_ax.set(xlabel=c)

        # Add group labels for rows
        axes.flat[r_idx*ncols].set(ylabel=r)

        # To fix alignment due to matplotlib bug
        axes.flat[(r_idx+1)*ncols-1].set_ylim([0,len(df.loc[r,c])])

    # add frames
    for ax in axes.flat:
        ax.tick_params(left=False, bottom=False)
        for _, spine in ax.spines.items():
            spine.set_visible(True)

    if title:
        fig.suptitle(title, y=1.01)

    with warnings.catch_warnings(): # Hide warnings from tight_layout
        warnings.simplefilter("ignore")
        fig.tight_layout(rect=[0,0,0.9,1])

    return fig