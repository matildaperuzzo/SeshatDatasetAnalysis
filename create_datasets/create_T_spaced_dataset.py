import sys
import os
import numpy as np
import pandas as pd

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Now you can import the TimeSeriesDataset class
from src.TimeSeriesDataset import TimeSeriesDataset as TSD
from src.utils import download_data
from src.mappings import value_mapping, ideology_mapping

dt = 10
filename = f"{dt}_yr_dataset"
template_path = f"datasets/MSP_template.csv"

# initialize dataset by downloading dataset or downloading the data from polity_url
dataset = TSD(categories=['sc'], template_path=template_path)
dataset.initialize_dataset_grid(-10000,2000,dt)

dataset.download_all_categories()
for key in ideology_mapping['MSP'].keys():
    dataset.add_column('ideo/'+key.lower())

# remove all rows that have less than 30% of the columns filled in
# dataset.remove_incomplete_rows(nan_threshold=0.3)
# build the social complexity variables
dataset.build_social_complexity()
dataset.build_MSP()

imp_columns =  ['Pop','Cap','Terr','Hierarchy', 'Government', 'Infrastructure', 'Information', 'Money']
dataset.impute_missing_values()
sc_columns = ['Pop','Cap','Terr','Hierarchy', 'Government', 'Infrastructure', 'Information', 'Money']
# find rows in sc_columns with NaN values
nan_rows = dataset.scv_imputed[sc_columns].isnull().any(axis=1)
# drop rows with NaN values from scv and scv_imputed
dataset.scv = dataset.scv[~nan_rows]
dataset.scv_imputed = dataset.scv_imputed[~nan_rows]
pca = dataset.compute_PCA(sc_columns, 'PC', n_cols = 2, n_PCA = 8)
dataset.save_dataset(path='datasets/', name=filename)