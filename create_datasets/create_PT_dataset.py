import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add the src directory to the Python path
from src.utils import download_data, weighted_mean
from src.mappings import PT_value_mapping, ideology_mapping
# Now you can import the TimeSeriesDataset class
from src.TimeSeriesDataset import TimeSeriesDataset as TSD

# initialize dataset by downloading dataset or downloading the data from polity_url
dataset = TSD(categories=['sc'], template_path='datasets/MSP_template_new.csv')
dataset.add_polities()

url = "https://seshat-db.com/api/crisisdb/power-transitions/"
pt_df = download_data(url)
pt_df.reset_index(drop=True, inplace=True)

PT_types = ['overturn', 'predecessor_assassination', 'intra_elite',
       'military_revolt', 'popular_uprising', 'separatist_rebellion',
       'external_invasion', 'external_interference']
for type in PT_types:
    pt_df[type] = pt_df[type].apply(lambda x: PT_value_mapping[x] if x in PT_value_mapping.keys() else np.nan)

for type in PT_types:
    dataset.raw[type] = np.nan

for idx, row in pt_df.iterrows():
    polity = row['polity_id']
    # check if polity is in dataset 
    if polity not in dataset.raw.PolityID.unique():
        print(f"Polity {row['polity_new_name']} in PT dataset but not in polity dataset")
        continue
    # get year
    year_from = row['year_from']
    year_to = row['year_to']
    if pd.notna(year_from) and pd.notna(year_to):
        year = year_to
    elif pd.notna(year_from) and pd.isna(year_to):
        year = year_to
    elif pd.isna(year_from) and pd.notna(year_to):
        year = year_from
    elif pd.isna(year_from) and pd.isna(year_to):
        year = np.nan

    if pd.isna(year):
        continue
    # add years to dataset
    dataset.add_years(polID=polity, year=year)
    # add PT types
    for col in PT_types:
        dataset.raw.loc[(dataset.raw.PolityID == polity)&(dataset.raw.Year==year), col] = row[col]

dataset.raw = dataset.raw.loc[(dataset.raw.Year.notna())]

# delete duplicates
dataset.raw.drop_duplicates(subset=['PolityID', 'Year'], inplace=True)

dataset.raw = dataset.raw.sort_values(by=['PolityID', 'Year'])
dataset.raw.reset_index(drop=True, inplace=True)

# download sc raw variables at PT years
dataset.download_all_categories(polity_year_error=20)

for key in ideology_mapping['MSP'].keys():
    dataset.add_column('ideo/'+key.lower())

# remove all rows that have less than 30% of the columns filled in
# dataset.remove_incomplete_rows(nan_threshold=0.3)
# build the social complexity variables
dataset.build_social_complexity()
dataset.build_MSP()

# add crisis to scv dataset
PT_mapping = {'PT_types':{}}
for col in PT_types:
    PT_mapping['PT_types'][col] = 1
dataset.scv['Crisis'] = dataset.raw.apply(lambda row: weighted_mean(row, PT_mapping, "PT_types", imputation='remove'), axis=1)
for type in PT_types:
    dataset.scv[type] = dataset.raw[type] 

# add 100 year dataset to PT dataset to increase the number of datapoints used
# in imputation and reduce bias
dataset_merged = TSD(categories=['sc'], file_path="datasets/100_yr_dataset.csv")
dataset_merged.scv['dataset'] = '100y'
pt_dat = dataset.scv.copy()
pt_dat['dataset'] = 'PT'
dataset_merged.scv = pd.concat([pt_dat, dataset_merged.scv])
dataset_merged.scv.reset_idatndex(drop=True, inplace=True)
# impute scale and non scale variables separately
scale_cols = ['Pop','Terr','Cap','Hierarchy']
dataset_merged.impute_missing_values(scale_cols)
non_scale_cols = ['Government', 'Infrastructure', 'Information', 'Money']
dataset_merged.impute_missing_values(non_scale_cols)
dataset_merged.scv_imputed['dataset'] = dataset_merged.scv['dataset']

dataset.scv = dataset_merged.scv.groupby('dataset').get_group('PT')
dataset.scv_imputed = dataset_merged.scv_imputed.groupby('dataset').get_group('PT')
dataset.scv.reset_index(drop=True, inplace=True)
dataset.scv_imputed.reset_index(drop=True, inplace=True)
dataset.save_dataset(path='datasets/', name='power_transitions')