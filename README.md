# SeshatDatasetAnalysis

SeshatDatasetAnalysis is a project for analyzing time series datasets. This project leverages various data science libraries to process and analyze historical datasets.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
  - [Template Data Structure](#template-data-structure)
  - [Sampling](#sampling)
  - [Creating the Final Database](#creating-the-final-database)
  - [Code References](#code-references)
  - [Example Usage](#example-usage)
  - [Plotting Module](#plotting-module)

## Installation

To install the dependencies, first install [Poetry](https://python-poetry.org/docs/#installation) and then run the following command:

```sh
pip install seshatdatasetanalysis
```

## Usage

To use the `TimeSeriesDataset` class, you can import it as follows:

```python
from seshatdatasetanalysis import TimeSeriesDataset as TSD
```

The notebook `create dataset` contains an example of how to use the class

Usage
To use the TimeSeriesDataset class, you can import it as follows:

## Documentation
Template Data Structure
The template data structure is a crucial component of the SeshatDatasetAnalysis project. It is designed to download information about polities from the SQL database and construct a template dataset. This template dataset is then used to derive analysis datasets as needed. The template does not make any assumptions about specific analysis timesteps.

## Structure of the Template
The template dataset has a single row for each polity, where each column contains a data structure that records, in a uniform manner, the different kinds of variable data from the SQL database. The data structure captures all the changed values for a variable in an ordered time sequence between the start and end of the polity. The data is represented using a Python dictionary to capture the values.

For example, consider the following representation:

| PolID | Start | End  | var1  | var2  | var3  | var4  |
|-------|-------|------|-------|-------|-------|-------|
|| 1700  | 1850  | valds1 | valds2 | valds3 | valds4 |


Each variable (var1, var2, etc.) is encoded as a dictionary. Here are some examples of how different types of data are encoded:

Single Value without Dates:
```python
valds1 = {'t': [1700, 1850], 'val': [[val1, val1]]}
```

Multiple Entries with Dates:
```python
valds2 = {'t': [1722, 1800, 1819], 'val': [[v21, (v22, v23), v24]]}
```

Single Value with an Explicit Date:
```python
valds3 = {'t': [1750], 'val': [[val3]]}
```

Disputed Values without Dates:
```python
valds4 = {'t': [1700, 1850], 'val': [[val41, val41], [val42, val42]]}
```

The t values are always ascending and within the start and end dates of the polity. The dictionary data structures encode the value and date assumptions from the SQL database in a uniform and time-ordered way.

## Sampling
Once all variables are constructed, a function is applied to 'sample' the variable dictionary at a specific time t. The function sample_var(var_dict, t, sampling_method_disputes, sampling_method_ranges, interpolation_method) performs the following steps:<br />

Ensures that t is between the start and end of the polity.<br />
Samples one of the entries in val using the sampling_method_disputes function.
Resolves any range uncertainties in that entry by applying the sampling_method_ranges function.<br />
Interpolates the resolved values using the interpolation_method at time t.
Returns the value at time t.<br />
Different sampling and interpolation methods can be chosen depending on the variable, allowing for flexibility in creating time series datasets.

## Creating the Final Database
Create a Template: The template is created as detailed above.
TimeSeriesDataset Module: The TimeSeriesDataset module creates a dataset based on a set of polities and years by sampling the template.<br />
Construct Social Complexity Variables and Perform Imputations: The sampled data is used to construct social complexity variables and perform imputations.<br />
The template serves as a snapshot of the database, allowing for resampling with different methods. The current sampling process involves resolving disputes by sampling one of the rows in values, sampling uniformly for each range variable, extending the data by adding the start and end of the polity, and taking the value for the closest time preceding the specified year.

## Code References
Template Data Structure: The template data structure is constructed in the Template class, which can be found in the src/Template.py file.<br />
Sampling Function: The sample_var function is used to sample the variable dictionary at a specific time t. This function is part of the Template class.<br />
TimeSeriesDataset Module: The TimeSeriesDataset class, which creates datasets based on the template, is located in the src/TimeSeriesDataset.py file.<br />
PCA Computation: The compute_PCA method in the TimeSeriesDataset class performs Principal Component Analysis on specified columns. This method is used to construct social complexity variables.

# Plotting Module

This module provides specialized plotting functions for visualizing Seshat dataset analysis results. It includes functions for creating bubble plots, grid visualizations, coefficient plots, and band plots with error bars.

## Functions Overview

### 1. `polity_bubble_plot()`
Creates a bubble plot where each bubble represents a polity, with bubble sizes based on the number of observations and colors determined by a specified variable.

**Parameters:**
- `tsd`: TimeSeriesDataset or pandas DataFrame
- `col_x`: Column name for x-axis
- `col_y`: Column name for y-axis  
- `col_color`: Column name for color coding
- `show_background_data`: Boolean to show background data points (default: False)
- `cmap`: Colormap name (default: 'coolwarm')
- `size_scale`: Scale factor for bubble sizes (default: 10)
- `vmin`, `vmax`: Color scale limits (optional)

**Example:**
```python
import seshatdatasetanalysis as sda
from seshatdatasetanalysis import plotting

# Load dataset
tsd = sda.TimeSeriesDataset(['sc'], file_path='test_dataset')

# Create polity bubble plot
fig, ax, scatter = plotting.polity_bubble_plot(
    tsd, 
    col_x='Pop', 
    col_y='Information', 
    col_color='Hierarchy',
    show_background_data=True,
    cmap='viridis',
    size_scale=15
)
```

### 2. `grid_bubble_plot()`
Creates a grid-based bubble plot that bins data spatially and shows averaged values within each grid cell.

**Parameters:**
- `tsd`: TimeSeriesDataset or pandas DataFrame
- `col_x`: Column name for x-axis
- `col_y`: Column name for y-axis
- `col_color`: Column name for color coding
- `cmap`: Colormap name (default: 'coolwarm')
- `nbins`: Number of bins (int or tuple, optional)
- `grid_size`: Size of grid cells (default: 1)
- `scale_size`: Scale factor for bubble sizes (default: 5)
- `vmin`, `vmax`: Color scale limits (optional)

**Example:**
```python
# Create grid bubble plot
fig, ax = plotting.grid_bubble_plot(
    tsd, 
    col_x='Hierarchy', 
    col_y='Pop', 
    col_color='Information',
    cmap='plasma',
    grid_size=0.75,
    scale_size=8
)
```

### 3. `plot_fit_coefficients()`
Visualizes regression coefficients with confidence intervals for multiple dependent and independent variables.

**Parameters:**
- `tsd`: TimeSeriesDataset or pandas DataFrame
- `y_cols`: List of dependent variable column names
- `x_cols`: List of independent variable column names
- `regression_type`: 'logit' or 'linear'
- `pval_max`: Maximum p-value for inclusion (default: 0.05)
- `cmap`: Colormap name (default: 'coolwarm')

**Example:**
```python
# Plot regression coefficients
fig, ax = plotting.plot_fit_coefficients(
    tsd,
    y_cols=['Information', 'Infrastructure', 'Money'],
    x_cols=['Pop', 'Cap', 'Terr'],
    regression_type='linear',
    pval_max=0.05,
    cmap='RdBu'
)
```

### 4. `band_plot()`
Creates a band plot showing mean values with error bands across binned x-values, optionally color-coded by a third variable.

**Parameters:**
- `tsd`: TimeSeriesDataset or pandas DataFrame
- `col_x`: Column name for x-axis
- `col_y`: Column name for y-axis
- `col_z`: Optional column name for color coding
- `nbins`: Number of bins (optional)
- `grid_size`: Size of bins (default: 1)
- `cmap`: Colormap name (default: 'coolwarm')
- `error`: Error type - 'standard' or 'sem' (default: 'standard')

**Example:**
```python
# Simple band plot without color coding
fig, ax = plotting.band_plot(
    tsd,
    col_x='Pop',
    col_y='Information',
    nbins=15,
    error='sem'
)

# Band plot with color coding
fig, ax = plotting.band_plot(
    tsd,
    col_x='Pop',
    col_y='Information',
    col_z='Cap',
    nbins=10,
    cmap='viridis',
    error='standard'
)
```

## Complete Example

```python
import seshatdatasetanalysis as sda
from seshatdatasetanalysis import plotting
import matplotlib.pyplot as plt

# Load your dataset
tsd = sda.TimeSeriesDataset(['sc'], file_path='your_dataset_path')

# Create multiple visualizations
plt.figure(figsize=(15, 10))

# 1. Polity bubble plot
plotting.polity_bubble_plot(
    tsd, 
    'Pop', 'Information', 'Hierarchy',
    show_background_data=True
)

# 2. Grid bubble plot  
plotting.grid_bubble_plot(
    tsd,
    'Hierarchy', 'Pop', 'Information',
    grid_size=0.5
)

# 3. Coefficient plot
plotting.plot_fit_coefficients(
    tsd,
    y_cols=['Information', 'Infrastructure'],
    x_cols=['Pop', 'Hierarchy'],
    regression_type='linear'
)

# 4. Band plot
plotting.band_plot(
    tsd,
    'Pop', 'Information', 'Cap',
    nbins=8,
    error='sem'
)

```

## Return Values

All plotting functions return:
- `fig`: matplotlib figure object
- `ax`: matplotlib axes object  
- Some functions also return additional objects (e.g., `scatter` object from bubble plots)

## Dependencies

- matplotlib
- numpy
- pandas