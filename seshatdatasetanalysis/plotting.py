import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seshatdatasetanalysis as sda
import seshatdatasetanalysis.utils as utils


def polity_bubble_plot(tsd, col_x, col_y, col_color, show_background_data = False, cmap = 'coolwarm', size_scale = 10, vmin = None, vmax = None, ax = None):
    """
    Create a bubble plot of polity data with specified x and y columns, color coding, and optional background plot where all x,y points are shown (useful for 
    visualizing how much data is available in col_color compared to col_x and col_y). Bubble sizes are scaled based on the number of observations per polity, and colors are determined by the specified color column.
    Inputs:
        tsd: TimeSeriesDataset or pandas DataFrame containing the data.
        col_x: Column name for the x-axis.
        col_y: Column name for the y-axis.
        col_color: Column name for the color coding of the bubbles.
        show_background_data: Boolean indicating whether to show a background plot with all data points.
        cmap: Colormap to use for coloring the bubbles.
        size_scale: Scale factor for bubble sizes based on number of observations per polity.
        vmin: Minimum value for color scaling (optional).
        vmax: Maximum value for color scaling (optional).
        ax: Optional matplotlib axes object to plot on (if None, will create a new figure and axes).
    Returns:
        fig: The matplotlib figure object.
        ax: The matplotlib axes object.
        scatter: The scatter plot object.
    """
    
    if isinstance(tsd, sda.TimeSeriesDataset):
        data = tsd.scv_imputed
    elif isinstance(tsd, pd.DataFrame):
        data = tsd
    else:
        raise TypeError("tsd must be a TimeSeriesDataset or a pandas DataFrame")
    
    if col_x not in data.columns or col_y not in data.columns or col_color not in data.columns or 'PolityName' not in data.columns:
        raise ValueError(f"Columns PolityName, {col_x}, {col_y}, or {col_color} not found in the dataset.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 5))
    else:
        fig = ax.figure

    if show_background_data:
        dataset_plot = data.copy().dropna(subset=[col_x, col_y])
        polity_group = dataset_plot.groupby('PolityName')
        mean_x = polity_group[col_x].mean()
        mean_y = polity_group[col_y].mean()
        num_observations = polity_group.size()

        scatter = plt.scatter(mean_x, mean_y, s=num_observations * size_scale, c = 'black', alpha=0.1)

    dataset_plot = data.dropna(subset=[col_x, col_y, col_color])
    polity_group = dataset_plot.groupby('PolityName')

    mean_x = polity_group[col_x].mean()
    mean_y = polity_group[col_y].mean()
    num_observations = polity_group.size()
    colors = polity_group[col_color].mean()

    vmin = vmin if vmin is not None else colors.min()
    vmax = vmax if vmax is not None else colors.max()

    scatter = plt.scatter(mean_x, mean_y, s=num_observations * size_scale, c=colors, cmap=cmap, alpha=0.7, vmin=vmin, vmax=vmax)

    plt.colorbar(scatter, label=col_color)
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    # plt.show()

    return fig, ax, scatter

def grid_bubble_plot(tsd, col_x, col_y, col_color, cmap = 'coolwarm', nbins = None, grid_size = 1, scale_size = 5, vmin = None, vmax = None, ax = None):
    """
    Create a grid bubble plot of polity data with specified x and y columns, color coding, and bubble sizes scaled based on the number of observations per polity.
    Inputs:
        tsd: TimeSeriesDataset or pandas DataFrame containing the data.
        col_x: Column name for the x-axis.
        col_y: Column name for the y-axis.
        col_color: Column name for the color coding of the bubbles.
        cmap: Colormap to use for coloring the bubbles.
        size_scale: Scale factor for bubble sizes based on number of observations per polity.
        nbins: Number of bins for the grid (optional, if None, will use grid_size).
        grid_size: Size of the grid for the bubble plot.
        scale_size: Scale factor for bubble sizes.
        vmin: Minimum value for color scaling (optional).
        vmax: Maximum value for color scaling (optional).
        ax: Optional matplotlib axes object to plot on (if None, will create a new figure and axes).
    Returns:
        fig: The matplotlib figure object.
        ax: The matplotlib axes object.
        scatter: The scatter plot object.
    """
        
    if isinstance(tsd, sda.TimeSeriesDataset):
        data = tsd.scv_imputed
    elif isinstance(tsd, pd.DataFrame):
        data = tsd
    else:
        raise TypeError("tsd must be a TimeSeriesDataset or a pandas DataFrame")
    
    if col_x not in data.columns or col_y not in data.columns or col_color not in data.columns or 'PolityName' not in data.columns:
        raise ValueError(f"Columns PolityName, {col_x}, {col_y}, or {col_color} not found in the dataset.")
    
    xlims = (data[col_x].min(), data[col_x].max())
    ylims = (data[col_y].min(), data[col_y].max())
    if (nbins is None) and (grid_size is None):
        nbins = (10, 10)
    elif (nbins is not None) and (grid_size is None):
        if isinstance(nbins, int):
            nbins = (nbins, nbins)
        elif len(nbins) == 1:
            nbins = (nbins[0], nbins[0])
        elif len(nbins) != 2:
            raise ValueError("nbins must be an int or a tuple of two ints.")
        grid_size = (x_bins[1] - x_bins[0], y_bins[1] - y_bins[0])
    elif grid_size is not None:
        nbins = ((xlims[1] - xlims[0]) // grid_size, (ylims[1] - ylims[0]) // grid_size)
    nbins = (int(nbins[0]), int(nbins[1]))
    x_bins = np.linspace(xlims[0], xlims[1], nbins[1] + 1)
    y_bins = np.linspace(ylims[0], ylims[1], nbins[0] + 1)

    data = data.dropna(subset=[col_x, col_y, col_color])
    x = data[col_x]
    y = data[col_y]
    z = data[col_color]

    df = pd.DataFrame({'x': x, 'y': y, 'z': z})

    # Bin the data
    df['x_bin'] = pd.cut(df['x'], bins=x_bins, labels=False, include_lowest=True)
    df['y_bin'] = pd.cut(df['y'], bins=y_bins, labels=False, include_lowest=True)

    x_bins = np.linspace(df.loc[df.x_bin == df.x_bin.min(),'x'].mean(),df.loc[df.x_bin == df.x_bin.max(),'x'].mean(), len(df.x_bin.unique()))
    y_bins = np.linspace(df.loc[df.y_bin == df.y_bin.min(),'y'].mean(),df.loc[df.y_bin == df.y_bin.max(),'y'].mean(), len(df.y_bin.unique()))

    # Group by the bins and calculate the mean of z
    grid = df.groupby(['x_bin', 'y_bin'])['z'].mean().unstack()
    grid_std = df.groupby(['x_bin', 'y_bin'])['z'].std().unstack()
    counts = df.groupby(['x_bin', 'y_bin'])['z'].count().unstack()
    x,y = np.meshgrid(x_bins, y_bins)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 5))
    else:
        fig = ax.figure

    if vmin is None:
        vmin = grid.min().min()
    if vmax is None:
        vmax = grid.max().max()

    ax.scatter(x.flatten(), y.flatten(), 
               c=grid.T.values.flatten(), 
               cmap= cmap, 
               s=counts.T.values.flatten()*scale_size,
               vmin=vmin, vmax=vmax)
    ax.grid(False)
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    plt.colorbar(ax.collections[0], label=col_color)
    # plt.show()
    return fig, ax


def plot_fit_coefficients(tsd, y_cols, x_cols, regression_type, pval_max = 0.05, cmap='coolwarm'):

    """
    Create a plot of linear or logistic fit coefficients for specified x and y columns 
    in a TimeSeriesDataset or pandas DataFrame.
    Inputs:
        tsd: TimeSeriesDataset or pandas DataFrame containing the data.
        y_cols: List of column names for the y-axis (dependent variables).
        x_cols: List of column names for the x-axis (independent variables).
        regression_type: Type of regression to perform ('logit' or 'linear').
        pval_max: Maximum p-value for including coefficients in the plot.
        cmap: Colormap to use for coloring the coefficients.
    Returns:
        fig: The matplotlib figure object.
        ax: The matplotlib axes object.
    """

    if isinstance(tsd, sda.TimeSeriesDataset):
        data = tsd.scv_imputed
    elif isinstance(tsd, pd.DataFrame):
        data = tsd
    else:
        raise TypeError("tsd must be a TimeSeriesDataset or a pandas DataFrame")
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i/len(y_cols)) for i in range(len(y_cols))]

    fig,ax = plt.subplots(1,len(x_cols), figsize=(3*len(x_cols), 3))
    # hide y tick labels
    x_lims = np.zeros((len(x_cols),2))
    for i, col in enumerate(x_cols):
        ax[i].set_title(col)
        ax[i].set_yticks([])
        ax[i].set_yticklabels([])

        ax[i].spines['right'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        plt.subplots_adjust(wspace=None, hspace=None)
        for j,y_col in enumerate(y_cols):
            fit_x_cols = x_cols.copy()
            Xy = data[[y_col]+fit_x_cols]
            if regression_type == 'logit':
                model = utils.fit_logit_to_variables(Xy, y_col, fit_x_cols, p_max = pval_max)   
            elif regression_type == 'linear':
                model = utils.fit_linear_to_variables(Xy, y_col, fit_x_cols, p_max = pval_max)
            else:   
                raise ValueError("regression_type must be 'logit' or 'linear'")
            ax[i].axhline(y=j+0.5, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            if pd.isna(model):
                continue 
            # check if the column is in the model
            if col in model.params:                                                               
                ax[i].plot([model.params[col]-model.bse[col], model.params[col]+model.bse[col]], [j,j], color=colors[j], linewidth=1)
                ax[i].scatter([model.params[col]-model.bse[col], model.params[col]+model.bse[col]], [j,j], s = 10, color=colors[j])

                x_lims[i,0] = min(x_lims[i,0], model.params[col]-model.bse[col])
                x_lims[i,1] = max(x_lims[i,1], model.params[col]+model.bse[col])
                
        # vertical line at 0
        ax[i].axvline(x=0, color='black', linestyle='--')
        # make x_lims symmetric around 0
        x_lims[i] = (-max(abs(x_lims[i]))*2.0, max(abs(x_lims[i]))*2.0)
        ax[i].set_xlim(x_lims[i,0], x_lims[i,1])
        ax[i].set_ylim([-0.5, len(y_cols)-0.5])
        ax[i].set_xlabel(col + '\ncoefficient')
    #add crisis vars to y axis in first subplot
    ax[0].set_yticks(range(len(y_cols)))
    ax[0].set_yticklabels(y_cols)
    ax[-1].spines['right'].set_visible(True)
    ax[0].spines['left'].set_visible(True)
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    return fig, ax

def band_plot(tsd, col_x, col_y, col_z = None, nbins = None, grid_size = 1, cmap = 'coolwarm', error = 'standard', ax = None):

    """
    Create a band plot of data with specified x and y columns, color coding, and optional error bands.
    Inputs:
        tsd: TimeSeriesDataset or pandas DataFrame containing the data.
        col_x: Column name for the x-axis.
        col_y: Column name for the y-axis.
        col_z: Optional column name for color coding (if None, will not use color).
        nbins: Number of bins for the grid (optional, if None, will use grid_size).
        grid_size: Size of the grid for the band plot.
        cmap: Colormap to use for coloring the bands.
        error: Type of error to calculate ('standard' or 'sem').
        ax: Optional matplotlib axes object to plot on (if None, will create a new figure and axes).
    Returns:
        fig: The matplotlib figure object.
        ax: The matplotlib axes object.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 5))
    else:
        fig = ax.figure

    if isinstance(tsd, sda.TimeSeriesDataset):
        df = tsd.scv_imputed
    elif isinstance(tsd, pd.DataFrame):
        df = tsd
    else:
        raise TypeError("tsd must be a TimeSeriesDataset or a pandas DataFrame")
    
    if col_x not in df.columns or col_y not in df.columns:
        raise ValueError(f"Columns {col_x} or {col_y} not found in the dataset.")
    if col_z is not None and col_z not in df.columns:
        raise ValueError(f"Column {col_z} not found in the dataset.")
    
    xlims = (df[col_x].min(), df[col_x].max())
    if (nbins is None) and (grid_size is None):
        nbins = 10
        grid_size = (xlims[1] - xlims[0]) // nbins
    elif (nbins is not None) and (grid_size is None):
        grid_size = (xlims[1] - xlims[0]) // nbins
    elif (nbins is None) and (grid_size is not None):
        nbins = (xlims[1] - xlims[0]) // grid_size
    nbins = int(nbins)
    bins = np.linspace(xlims[0], xlims[1], nbins + 1)

    fig, ax = plt.subplots(figsize=(4, 5))
    results = []
    cmap = plt.get_cmap(cmap)
    for i in range(len(bins) - 1):
        mask = (df[col_x] >= bins[i]) & (df[col_x] < bins[i + 1])
        y_values = df.loc[mask, col_y]
        if col_z is not None:
            z_values = df.loc[mask, col_z]
            color = z_values.mean() if not z_values.empty else np.nan
        else:
            color = 'royalblue'
        if not y_values.empty:
            mean = np.mean(y_values)
            if error == 'standard':
                err = np.std(y_values)
            elif error == 'sem':
                err = np.std(y_values) / np.sqrt(len(y_values))
            else:
                raise ValueError("error must be 'standard' or 'sem'")
            results.append({'x': (bins[i] + bins[i + 1]) / 2, 
                            'mean': mean, 
                            'err': err, 
                            'lower': mean - err,
                            'upper': mean + err,
                            'color': color})
            
            # if col_z is not None:
            #     ax.errorbar((bins[i] + bins[i + 1]) / 2, mean, yerr=err, fmt='o', color=color, ecolor=color, elinewidth=2, capsize=3)
    results = pd.DataFrame(results)
    if col_z is None:
        plt.plot(results['x'], results['mean'], color = 'firebrick', linewidth=2)
        plt.fill_between(results['x'], results['lower'], results['upper'], color='firebrick', alpha=0.2)
    elif col_z is not None:
        plt.plot(results['x'], results['mean'], color='black', linestyle='--', linewidth=2)
        results['color'] = (results.color-results.color.min())/(results.color.max()-results.color.min())
        colors = results['color'].apply(lambda x:  cmap(int(255*x))[:3] if isinstance(x, (int, float)) else 'black')
        for i, row in results.iterrows():
            plt.errorbar(row['x'], row['mean'], yerr=row['err'], fmt='o', 
                        color=colors.iloc[i], ecolor=colors.iloc[i], 
                        elinewidth=2, capsize=3) 
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    # plt.show()
    return fig, ax

if __name__ == "__main__":

    tsd = sda.TimeSeriesDataset(['sc'], file_path='test_dataset')
    tsd.build_social_complexity()
    grid_bubble_plot(tsd.scv, 'Hierarchy', 'Pop', 'Information', cmap='coolwarm', grid_size = 0.75, scale_size = 5)
    # plot_fit_coefficients(tsd.scv, x_cols=['Pop','Cap','Terr'], y_cols=['Information','Infrastructure','Money'], regression_type='linear', cmap='coolwarm', pval_max= 1)
    # band_plot(tsd.scv, 'Pop', 'Information', col_z = 'Cap', nbins=10, grid_size=0.75, cmap='coolwarm', error='sem')