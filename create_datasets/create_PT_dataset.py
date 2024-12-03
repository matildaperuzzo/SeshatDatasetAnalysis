import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add the src directory to the Python path
from src.utils import download_data
from src.mappings import value_mapping, ideology_mapping
# Now you can import the TimeSeriesDataset class
from src.TimeSeriesDataset import TimeSeriesDataset as TSD

# initialize dataset by downloading dataset or downloading the data from polity_url
dataset = TSD(categories=['sc'], template_path='datasets/MSP_template.csv')
dataset.add_polities()

url = "https://seshatdata.com/api/crisisdb/power-transitions/"
pt_df = download_data(url)

PT_types = ['overturn', 'predecessor_assassination', 'intra_elite',
       'military_revolt', 'popular_uprising', 'separatist_rebellion',
       'external_invasion', 'external_interference']
for type in PT_types:
    pt_df[type] = pt_df[type].apply(lambda x: value_mapping[x] if x in value_mapping.keys() else np.nan)

pt_df['Crisis'] = pt_df[PT_types].sum(axis=1)

# add crisis to dataset
dataset.raw['Crisis'] = np.nan
for type in PT_types:
    dataset.raw[type] = np.nan

for idx, row in pt_df.iterrows():
    polity = row['polity_id']
    # check if polity is in dataset 
    if polity not in dataset.raw.PolityID.unique():
        print(f"Polity {polity} in PT dataset but not in polity dataset")
        continue

    # get year
    year_from = row['year_from']
    year_to = row['year_to']
    if pd.notna(year_from) and pd.notna(year_to):
        year = np.mean([year_from,year_to])
    elif pd.notna(year_from) and pd.isna(year_to):
        year = year_to
    elif pd.isna(year_from) and pd.notna(year_to):
        year = year_from
    elif pd.isna(year_from) and pd.isna(year_to):
        year = np.nan

    # add years to dataset
    dataset.add_years(polID=polity, year=year)
    dataset.raw.loc[dataset.raw.Year == year,'Crisis'] = row.Crisis
    for type in PT_types:
        dataset.raw.loc[dataset.raw.Year == year,type] = row[type]

dataset.raw = dataset.raw.loc[(dataset.raw.Year.notna())&(dataset.raw.Year!=0)]

# delete duplicates
dataset.raw.drop_duplicates(subset=['PolityID', 'Year'], inplace=True)

dataset.raw = dataset.raw.sort_values(by=['PolityID', 'Year'])
dataset.raw.reset_index(drop=True, inplace=True)

# download sc raw variables at PT years
dataset.download_all_categories()

for key in ideology_mapping['MSP'].keys():
    dataset.add_column('ideo/'+key.lower())

# remove all rows that have less than 30% of the columns filled in
# dataset.remove_incomplete_rows(nan_threshold=0.3)
# build the social complexity variables
dataset.build_social_complexity()
dataset.build_MSP()

# add crisis to scv dataset
dataset.scv['Crisis'] = dataset.raw.Crisis
for type in PT_types:
    dataset.scv[type] = dataset.raw[type]

# impute missing data
imp_columns =  ['Pop','Cap','Terr','Hierarchy', 'Government', 'Infrastructure', 'Information', 'Money']
dataset.impute_missing_values(columns=imp_columns)

# add crisis to imputed dataset
dataset.scv_imputed['Crisis'] = dataset.scv['Crisis']
for type in PT_types:
    dataset.scv_imputed[type] = dataset.scv[type]

sc_columns = ['Pop','Cap','Terr','Hierarchy', 'Government', 'Infrastructure', 'Information', 'Money']
nan_rows = dataset.scv_imputed[sc_columns].isnull().any(axis=1)
dataset.scv = dataset.scv[~nan_rows]
dataset.scv_imputed = dataset.scv_imputed[~nan_rows]
# save dataset
dataset.save_dataset(path='datasets/', name='power_transitions')