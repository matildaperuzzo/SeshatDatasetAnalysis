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

## Installation

To install the dependencies, first install [Poetry](https://python-poetry.org/docs/#installation) and then run the following command:

```sh
poetry install
```

## Usage

To use the `TimeSeriesDataset` class, you can import it as follows:

```python
from src.TimeSeriesDataset import TimeSeriesDataset as TSD
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

For example, consider the following representation:

| PolID | Start | End  | var1  | var2  | var3  | var4  |
|-------|-------|------|-------|-------|-------|-------|
| 1700  | 1850  | valds1 | valds2 | valds3 | valds4 |


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
Once all variables are constructed, a function is applied to 'sample' the variable dictionary at a specific time t. The function sample_var(var_dict, t, sampling_method_disputes, sampling_method_ranges, interpolation_method) performs the following steps:

Ensures that t is between the start and end of the polity.
Samples one of the entries in val using the sampling_method_disputes function.
Resolves any range uncertainties in that entry by applying the sampling_method_ranges function.
Interpolates the resolved values using the interpolation_method at time t.
Returns the value at time t.
Different sampling and interpolation methods can be chosen depending on the variable, allowing for flexibility in creating time series datasets.

## Creating the Final Database
Create a Template: The template is created as detailed above.
TimeSeriesDataset Module: The TimeSeriesDataset module creates a dataset based on a set of polities and years by sampling the template.
Construct Social Complexity Variables and Perform Imputations: The sampled data is used to construct social complexity variables and perform imputations.
The template serves as a snapshot of the database, allowing for resampling with different methods. The current sampling process involves resolving disputes by sampling one of the rows in values, sampling uniformly for each range variable, extending the data by adding the start and end of the polity, and taking the value for the closest time preceding the specified year.

## ßCode References
Template Data Structure: The template data structure is constructed in the Template class, which can be found in the src/Template.py file.
Sampling Function: The sample_var function is used to sample the variable dictionary at a specific time t. This function is part of the Template class.
TimeSeriesDataset Module: The TimeSeriesDataset class, which creates datasets based on the template, is located in the src/TimeSeriesDataset.py file.
PCA Computation: The compute_PCA method in the TimeSeriesDataset class performs Principal Component Analysis on specified columns. This method is used to construct social complexity variables.