import sys
import os
import numpy as np
import pandas as pd

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Now you can import the TimeSeriesDataset class
from seshatdatasetanalysis.TimeSeriesDataset import TimeSeriesDataset as TSD
from seshatdatasetanalysis.utils import download_data
from seshatdatasetanalysis.mappings import value_mapping

dt = 100
filename = f"{dt}_yr_dataset"
template_path = f"test_scripts/template.csv"

# initialize dataset by downloading dataset or downloading the data from polity_url
dataset = TSD(categories= ['sc','wf','id','rel'], template_path=template_path)
dataset.initialize_dataset_grid(-10000,1900,dt)

error = 0
dataset.download_all_categories()

dataset.build_social_complexity()
dataset.build_warfare()
dataset.build_MSP()

dataset.save_dataset(path='test_scripts', name=filename)