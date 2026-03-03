import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as LinearRegression
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add the src directory to the Python path
from seshatdatasetanalysis.utils import download_data, download_data_json, weighted_mean
from seshatdatasetanalysis.mappings import PT_value_mapping, ideology_mapping_polaris
# Now you can import the TimeSeriesDataset class
from seshatdatasetanalysis.TimeSeriesDataset import TimeSeriesDataset as TSD

# initialize dataset by downloading dataset or downloading the data from polity_url
dataset = TSD(categories=['sc','wf','rt','rel'], template_path='test_scripts/template.csv')
dataset.add_polities()

url = "https://seshat-db.com/api/core/polities/"
polity_df = download_data(url)

polities = dataset.raw.PolityID.unique()

for polity in polities:
    years = polity_df.loc[polity_df['id'] == polity, ['start_year', 'end_year']].values[0]
    dataset.add_years(polID = polity, year=years[0])
    dataset.add_years(polID = polity, year=years[1])

dataset.raw = dataset.raw.sort_values(by=['PolityID', 'Year'])
dataset.raw.reset_index(drop=True, inplace=True)
error = 1
dataset.download_all_categories(polity_year_error=error)

# in imputation and reduce bias
dataset_100y = TSD(categories=['sc',"wf","rt",'rel'], file_path="test_scripts/100_yr_dataset.csv")
dataset_100y.raw["dataset"] = '100y'
dataset_100y.scv['dataset'] = '100y'
pt_dat = dataset.scv.copy()
pt_dat['dataset'] = 'pol'
dataset.raw['dataset'] = 'pol'
dataset.raw = pd.concat([dataset.raw, dataset_100y.raw])
dataset.scv = pd.concat([pt_dat, dataset_100y.scv])

from seshatdatasetanalysis.utils import weighted_mean, get_max
# build the social complexity variables
dataset.scv = dataset.raw[['NGA', 'PolityID', 'PolityName', 'Year']].copy()

allow_missing = True

dataset.build_social_complexity(allow_missing=allow_missing, percentages=[0.5, 0.2, 0.5])
dataset.scv['dataset'] = dataset.raw['dataset']
dataset.build_warfare()
dataset.build_MSP()

dataset.scv_imputed = pd.DataFrame([])
dataset.scv['Hierarchy_sq'] = dataset.scv['Hierarchy']**2

scale_cols = ['Pop','Terr','Cap','Hierarchy', 'Hierarchy_sq']
non_scale_cols = ['Government', 'Infrastructure', 'Information', 'Money']
dataset.impute_missing_values(columns = scale_cols, add_resid = False)
dataset.impute_missing_values(columns = non_scale_cols, add_resid = False)

dataset.scv_imputed['dataset'] = dataset.scv['dataset']

dataset.raw = dataset.raw[dataset.raw.dataset == 'pol'].reset_index(drop=True)
dataset.scv = dataset.scv[dataset.scv.dataset == 'pol'].reset_index(drop=True) 
dataset.scv_imputed = dataset.scv_imputed[dataset.scv_imputed.dataset == 'pol'].reset_index(drop=True) 

dataset.raw.drop(columns=['dataset'], inplace=True)
dataset.scv.drop(columns=['dataset'], inplace=True)
dataset.scv_imputed.drop(columns=['dataset'], inplace=True)

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
dataset.scv_imputed['Comp'] = dataset.scv_imputed.apply(lambda row: weighted_mean(row, comp_mapping,category = 'Comp', nan_handling = "remove", min_vals=0.5), axis=1)

religion_columns = ['polity_religion_genus', 'polity_religion_family', 'polity_religion']
for item, row in dataset.raw.iterrows():

    for col in religion_columns:
        if pd.isna(row[col]):
            for col2 in religion_columns:
                if col2 != col and not pd.isna(row[col2]):
                    imp_df =dataset.raw[(dataset.raw[col2] == row[col2])&(dataset.raw[col].notna())][col]
                    if not imp_df.empty:
                        most_common_value = imp_df.mode()[0]
                        dataset.raw.at[item, col] = most_common_value
                        continue
dataset.raw = dataset.raw.reset_index(drop=True)       
dataset.scv_imputed['MSP'] = dataset.scv['MSP']
for ind, row in dataset.raw.iterrows():
    if pd.isna(dataset.scv.at[ind, 'MSP']):
        for col in religion_columns:
            temp_df = dataset.scv.loc[dataset.raw['religion-families'] == row['religion-families'], 'MSP'].dropna()
            if len(temp_df) > 0:
                dataset.scv_imputed.at[ind, 'MSP'] = temp_df.mode()[0]
                break

# Move Miltech variables to imputed dataset
transfer_cols = ['Miltech','IronCav','Cavalry']
for col in transfer_cols:
    dataset.scv_imputed[col] = dataset.scv[col]

dataset.save_dataset(path='datasets/', name='polities')