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

dt = 100
filename = f"{dt}_yr_dataset"
template_path = f"datasets/template.csv"

# initialize dataset by downloading dataset or downloading the data from polity_url
dataset = TSD(categories=['sc','wf'], template_path=template_path)
dataset.initialize_dataset_grid(-10000,2024,dt)

error = 100
dataset.download_all_categories(polity_year_error=error, sampling_ranges='mean')
# for key in ideology_mapping['MSP'].keys():
#     dataset.add_column('ideo/'+key.lower(), polity_year_error=error)

# build the social complexity variables
dataset.build_social_complexity()
# dataset.build_MSP()
dataset.build_warfare()

# save the dataset
dataset.save_dataset(path='datasets/', name=filename)