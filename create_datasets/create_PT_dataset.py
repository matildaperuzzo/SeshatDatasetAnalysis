import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as LinearRegression
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add the src directory to the Python path
from src.utils import download_data, download_data_json, weighted_mean
from src.mappings import PT_value_mapping, ideology_mapping
# Now you can import the TimeSeriesDataset class
from src.TimeSeriesDataset import TimeSeriesDataset as TSD

# initialize dataset by downloading dataset or downloading the data from polity_url
dataset = TSD(categories=['sc','wf','rt'], template_path='datasets/template.csv')
dataset.add_polities()

url = "https://seshat-db.com/api/crisisdb/power-transitions/"
pt_df = download_data(url)
if len(pt_df) == 0:
    pt_df = download_data_json('datasets/crisisdb_power_transition_20250523_094154.json')

    new_col_names = {}
    for col in pt_df.columns:
        if col.startswith('coded_values_'):
            new_col_names[col] = col.replace('coded_values_', '')
    new_col_names['polity_new_ID'] = 'polity_name'
    new_col_names['polity_name'] = 'polity_long_name'
    pt_df.rename(columns=new_col_names, inplace=True)
    pt_df['polity_id'] = pt_df['polity_name'].apply(lambda x: dataset.raw.loc[dataset.raw.PolityName == x, 'PolityID'].values[0] if x in dataset.raw.PolityName.values else np.nan)
    polity_df = download_data("https://seshat-db.com/api/core/polities/")
    pt_df['polity_start_year'] = pt_df['polity_name'].apply(lambda x: polity_df.loc[polity_df['name'] == x, 'start_year'].values[0] if x in polity_df['name'].values else np.nan)
    pt_df['polity_end_year'] = pt_df['polity_name'].apply(lambda x: polity_df.loc[polity_df['name'] == x, 'end_year'].values[0] if x in polity_df['name'].values else np.nan)
    
pt_df.reset_index(drop=True, inplace=True)

PT_types = ['overturn', 'predecessor_assassination', 'intra_elite',
       'military_revolt', 'popular_uprising', 'separatist_rebellion',
       'external_invasion', 'external_interference']
for type in PT_types:
    pt_df[type] = pt_df[type].apply(lambda x: PT_value_mapping[x] if x in PT_value_mapping.keys() else np.nan)

for type in PT_types:
    dataset.raw[type] = np.nan

dataset.raw['duration'] = np.nan

for idx, row in pt_df.iterrows():
    polity = row['polity_id']
    # check if polity is in dataset 
    if polity not in dataset.raw.PolityID.unique():
        if row['polity_new_name'] in dataset.raw.PolityName.unique():
            polity = dataset.raw.loc[dataset.raw.PolityName == row['polity_new_name'], 'PolityID'].values[0]
        else:
            print(f"Polity {row['polity_new_name']} in PT dataset but not in polity dataset")
            continue
    # get year
    year_from = row['year_from']
    year_to = row['year_to']
    if pd.notna(year_from) and pd.notna(year_to):
        year = year_to
        duration = np.abs(year_to - year_from)
    elif pd.notna(year_from) and pd.isna(year_to):
        year = year_to
        duration = np.nan
    elif pd.isna(year_from) and pd.notna(year_to):
        year = year_from
        duration = np.nan
    elif pd.isna(year_from) and pd.isna(year_to):
        year = row[['polity_start_year','polity_end_year']].mean()
        duration = np.nan

    if pd.isna(year):
        continue
    # add years to dataset
    dataset.add_years(polID=polity, year=year)
    # add PT types
    for col in PT_types:
        dataset.raw.loc[(dataset.raw.PolityID == polity)&(dataset.raw.Year==year), col] = row[col]
    dataset.raw.loc[(dataset.raw.PolityID == polity)&(dataset.raw.Year==year), 'duration'] = duration

dataset.raw = dataset.raw.loc[(dataset.raw.Year.notna())]

# delete duplicates
dataset.raw.drop_duplicates(subset=['PolityID', 'Year'], inplace=True)

dataset.raw = dataset.raw.sort_values(by=['PolityID', 'Year'])
dataset.raw.reset_index(drop=True, inplace=True)

# download sc raw variables at PT years
error = 500
dataset.download_all_categories(polity_year_error=error)

# build the social complexity variables
dataset.build_social_complexity()
dataset.build_warfare()
# dataset.build_MSP()

# in imputation and reduce bias
dataset_100y = TSD(categories=['sc',"wf","rt"], file_path="datasets/100_yr_dataset.csv")
dataset_100y.raw["dataset"] = '100y'
dataset_100y.scv['dataset'] = '100y'
pt_dat = dataset.scv.copy()
pt_dat['dataset'] = 'PT'
dataset.raw['dataset'] = 'PT'
dataset.raw = pd.concat([dataset.raw, dataset_100y.raw])
dataset.scv = pd.concat([pt_dat, dataset_100y.scv])
dataset.scv.reset_index(drop=True, inplace=True)


dataset.scv = dataset.scv.loc[dataset.scv["Year"] <= 1800]
dataset.scv_imputed = pd.DataFrame([])
dataset.scv['Hierarchy_sq'] = dataset.scv['Hierarchy']**2

# impute scale and non scale variables separately
scale_cols = ['Pop','Terr','Cap','Hierarchy', 'Hierarchy_sq']
dataset.impute_missing_values(scale_cols, use_duplicates = False, r2_lim=0., add_resid=False)
non_scale_cols = ['Government', 'Infrastructure', 'Information', 'Money']
dataset.impute_missing_values(non_scale_cols, use_duplicates = False, r2_lim=0., add_resid=False)

# imp_cols = ['Pop','Terr','Cap','Hierarchy', 'Hierarchy_sq', 'Government', 'Infrastructure', 'Information', 'Money']
# dataset.impute_missing_values(imp_cols, use_duplicates = False, r2_lim=0., add_resid=False)

dataset.scv_imputed['dataset'] = dataset.scv['dataset']

dataset.scv.reset_index(drop=True, inplace=True)
dataset.scv_imputed.reset_index(drop=True, inplace=True)

# compute scale variable
scale_pca_cols = ['Pop','Terr','Cap']
scale_pca = dataset.compute_PCA(cols = scale_pca_cols, col_name = 'Scale', n_cols = 1, n_PCA= len(scale_pca_cols))
lm_df = dataset.scv_imputed[['Pop', 'Scale_1']].dropna()
X = lm_df[['Pop']]
y = lm_df['Scale_1']

# Normalize the Scale column to Pop
# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Extract the coefficients
intercept = model.intercept_
slope = model.coef_[0]
dataset.scv_imputed['Scale_1'] = (dataset.scv_imputed['Scale_1'] - intercept) / slope

# Calculate Comp variable
comp_mapping = {'Comp':{'Government': 11, 'Infrastructure': 12, 'Information':13, 'Money': 6}}
dataset.scv_imputed['Comp'] = dataset.scv_imputed.apply(lambda row: weighted_mean(row, comp_mapping,category = 'Comp',imputation = "remove", min_vals=0.5), axis=1)

# Move Miltech variables to imputed dataset
transfer_cols = ['Miltech','IronCav','Cavalry']
for col in transfer_cols:
    dataset.scv_imputed[col] = dataset.scv[col]

# save dataset
dataset.save_dataset(path='datasets/', name='power_transitions')