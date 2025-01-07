import pandas as pd
import numpy as np
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import download_data, fetch_urls, weighted_mean, get_max, is_same
from src.Template import Template

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset():
    def __init__(self, 
                 categories = list(['sc']),
                 polity_url = "https://seshat-db.com/api/core/polities/",
                 template_path = None,
                 file_path = None
                 ):

        self.categories = categories
        self.polity_url = polity_url
        self.raw = pd.DataFrame()
        self.scv = pd.DataFrame()
        self.scv_clean = pd.DataFrame()
        self.scv_imputed = pd.DataFrame()
        self.debug = pd.DataFrame(columns=["polity", "variable", "label", "issue"])

        if file_path is not None:
            path = os.path.dirname(file_path)
            filename = os.path.basename(file_path).split('.')[0]
            self.load_dataset(path=path, name=filename)
        elif (polity_url is not None ) and (template_path is None) and (file_path is None):
            self.template = Template(categories=categories, polity_url=polity_url)
        elif (template_path is not None) and (file_path is None):
            self.template = Template(categories=categories, file_path=template_path)
        else:
            print("Please provide either a polity_url or a template_path")
            sys.exit()
    
    def __len__(self):
        # check what datasets are available
        return len(self.raw)

    def __getitem__(self, cond):
        # check what datasets are available
        return self.raw.loc[cond]
    
    ########################## BUILDING FUNCTIONS ####################################

    def initialize_dataset_grid(self, start_year, end_year, dt):
        df = download_data(self.polity_url)
        # create the dattaframe staring with the polity data
        self.raw = pd.DataFrame(columns = ["NGA", "PolityID", "PolityName", "Year", "PolityActive", "Note"])
        # specify the columns data types
        self.raw['PolityID'] = self.raw['PolityID'].astype('int')
        self.raw['Year'] = self.raw['Year'].astype('int')
        self.raw['PolityActive'] = self.raw['PolityActive'].astype('bool')

        # polity_home_nga_id, polity_id, polity_name 
        polityIDs = df.id.unique()
        timeline = np.arange(start_year, end_year, dt)

        for polID in polityIDs:
            pol_df = df.loc[df.id == polID, ['home_nga_name', 'id', 'new_name','start_year','end_year']]
            # create a temporary dataframe with all data for current polity
            pol_df_new = pd.DataFrame(dict({"NGA" : pol_df.home_nga_name.values[0], 
                                            "PolityID": pol_df.id.values[0], 
                                            "PolityName": pol_df.new_name.values[0], 
                                            "Year": timeline, 
                                            "PolityActive": False}))
            row = pol_df.iloc[0]
            #Mark the years when the polity was active
            pol_df_new.loc[(pol_df_new.Year >= row.start_year) & (pol_df_new.Year <= row.end_year), 'PolityActive'] = True
            # Ensure the index is unique before concatenating
            if not pol_df_new.index.is_unique:
                pol_df_new = pol_df_new.reset_index(drop=True)
            self.raw = pd.concat([self.raw, pol_df_new])
        self.raw = self.raw.loc[self.raw.PolityActive == True]
        self.raw.drop(columns=['PolityActive'], inplace=True)
        self.raw.reset_index(drop=True, inplace=True)
    
    def add_polities(self):
        df = download_data(self.polity_url)
        # create the dattaframe staring with the polity data
        self.raw = pd.DataFrame(columns = ["NGA", "PolityID", "PolityName", "Year"])
        # specify the columns data types
        self.raw['PolityID'] = self.raw['PolityID'].astype('int')
        self.raw['Year'] = self.raw['Year'].astype('int')

        # polity_home_nga_id, polity_id, polity_name 
        polityIDs = df.id.unique()

        for polID in polityIDs:
            pol_df = df.loc[df.id == polID, ['home_nga_name', 'id', 'new_name','start_year','end_year']]
            # create a temporary dataframe with all data for current polity
            pol_df_new = pd.DataFrame(dict({"NGA" : pol_df.home_nga_name.values[0], 
                                            "PolityID": pol_df.id.values[0], 
                                            "PolityName": pol_df.new_name.values[0], 
                                            "Year": np.nan}), index=[0])
            self.raw = pd.concat([self.raw, pol_df_new])
        self.raw.reset_index(drop=True, inplace=True)

    def add_years(self,polID, year):

        pol_df = self.raw.loc[self.raw.PolityID == polID]
        pol_df_new = pd.DataFrame(dict({"NGA" : pol_df.NGA.values[0], 
                                        "PolityID": pol_df.PolityID.values[0], 
                                        "PolityName": pol_df.PolityName.values[0], 
                                        "Year": year}), index=[self.raw.index.max()+1])
        self.raw = pd.concat([self.raw, pol_df_new])
        row = self.raw.loc[self.raw.Year.isna()&(self.raw.PolityID == polID)]
        if len(row) > 0:
            self.raw.drop(row.index, inplace=True)
        self.raw.reset_index(drop=True, inplace=True)

    def download_all_categories(self, polity_year_error = 0):
        urls = {}
        for category in self.categories:
            urls.update(fetch_urls(category))
        for key in urls.keys():
            self.add_column(key, polity_year_error)
    
    def add_column(self, key, polity_year_error = 0):
        variable_name = key.split('/')[-1]
        grouped_variables = self.raw.groupby('PolityName').apply(lambda group: self.sample_from_template(group, variable_name, polity_year_error=polity_year_error))
        for polity in grouped_variables.index:
            self.raw.loc[self.raw.PolityName == polity, variable_name] = grouped_variables[polity]

    def sample_from_template(self, rows, variable, label = 'pt', polity_year_error = 0):
        pol = rows.PolityID.values[0]
        years = rows.Year.values
        entry = self.template.template.loc[(self.template.template.PolityID == pol), variable]
        if len(entry) == 0:
            return [np.nan]*len(years)
        if pd.isna(entry.values[0]):
            return [np.nan]*len(years)
        
        _dict = eval(entry.values[0])
        results = self.template.sample_dict(_dict, years, polity_year_error)

        # check if any of the years are out of bounds
        not_in_bounds = np.array([r == "Out of bounds" for r in results])
        if np.any(not_in_bounds):
            debug_row = {"polity" : rows.PolityName.values[0],
                        "variable": variable, 
                        "label": label,
                        "issue": f"{years[not_in_bounds]} ouside of polity years"}
            self.debug = pd.concat([self.debug, pd.DataFrame([debug_row], columns=self.debug.columns)], ignore_index=True)
            # substitute the out of bounds years with np.nan
            results = [np.nan if result == "Out of bounds" else result for result in results]
        return results

    ##################################### ANALYSIS FUNCTIONS ############################################

    def remove_incomplete_rows(self, nan_threshold = 0.3):
        # add all columns from sc_mapping
        from src.mappings import social_complexity_mapping
        cols = []
        for key in social_complexity_mapping.keys():
            cols += [key for key in list(social_complexity_mapping[key].keys())]
        
        # remove rows with less than 30% of the columns filled in
        self.raw = self.raw.loc[self.raw[cols].notna().sum(axis=1)/len(cols)>0.3]
        self.raw.reset_index(drop=True, inplace=True)

    def build_social_complexity(self):
        from src.mappings import social_complexity_mapping
        # create dataframe for social complexity
        self.scv = self.raw[['NGA', 'PolityID', 'PolityName', 'Year']].copy()

        # add population variables
        self.scv['Pop'] = (self.raw['polity-populations']).apply(np.log10)
        self.scv['Terr'] = (self.raw['polity-territories']).apply(np.log10)
        self.scv['Cap'] = (self.raw['population-of-the-largest-settlements']).apply(np.log10)

        # add hierarchy variables
        self.scv['Hierarchy'] = self.raw.apply(lambda row: weighted_mean(row, social_complexity_mapping, "Hierarchy", imputation='remove'), axis=1)
        self.scv['Government'] = self.raw.apply(lambda row: weighted_mean(row, social_complexity_mapping, "Government", imputation = 'zero'), axis=1)
        self.scv['Infrastructure'] = self.raw.apply(lambda row: weighted_mean(row, social_complexity_mapping, "Infrastructure", imputation= 'zero'), axis=1)
        self.scv['Information'] = self.raw.apply(lambda row: weighted_mean(row, social_complexity_mapping, "Information", imputation='zero'), axis=1)
        # find the maximum weight for money
        max_money = max(social_complexity_mapping['Money'].items(), key=lambda item: item[1])[1]
        self.scv['Money'] = self.raw.apply(lambda row: get_max(row, social_complexity_mapping, "Money"), axis=1)/max_money
        
    def build_MSP(self):
        from src.mappings import ideology_mapping
        self.scv['MSP'] = self.raw.apply(lambda row: weighted_mean(row, ideology_mapping, "MSP", imputation='remove'), axis=1)

    def impute_missing_values(self, columns = ['Pop','Cap','Terr','Hierarchy', 'Government', 'Infrastructure', 'Information', 'Money'], use_duplicates = False):

        if self.scv_imputed.empty:
            polity_cols = ['NGA', 'PolityID', 'PolityName', 'Year']
            self.scv_imputed = self.scv[polity_cols].copy()
        scv = self.scv[columns]
        if not use_duplicates:
            # remove duplicates
            scv = scv.drop_duplicates()

        self.scv_imputed[columns] = self.scv[columns].copy()

        df_fits = pd.DataFrame(columns=["Y column", "X columns", "fit", "num_rows","p-values"])
        df_fits['X columns'] = df_fits['X columns'].astype(object)

        for index, row in scv.iterrows():
            # find positions of nans
            nan_cols = row[row.isna()].index
            non_nan_cols = row[row.notna()].index
            if len(non_nan_cols) == 0:
                continue
            for col in nan_cols:

                fit_cols = [col] + list(non_nan_cols)
                # find entries in scv where fit_cols are not nan
                mask = scv[fit_cols].notna().all(axis=1)
                # fit a linear regression
                X = scv[fit_cols][mask].drop(columns=col)
                y = scv[fit_cols][mask][col]
                # print(f'fitting for {col} with {len(X)} rows' )
                reg = LinearRegression().fit(X, y)
                # extract p-values for each coefficient
                X2 = sm.add_constant(X)
                est = sm.OLS(y, X2)
                est2 = est.fit()
                p_values = est2.summary2().tables[1]['P>|t|'][1:]
                if all(p_values>0.05):
                    print('Not enough significant variables')
                    print(f'p-values for {col} are {p_values}')
                else:

                    relevant_cols = p_values[p_values<0.001].index
                    # check if the amount of relevant columns is greater than 1
                    if len(relevant_cols) < 1:
                        continue
                    # check if the fit is already in the dataframe
                    if len(df_fits.loc[(df_fits["Y column"] == col) & df_fits['X columns'].apply(lambda x: is_same(x, relevant_cols))]) > 0:
                        continue
                    relevant_cols = [col] + list(relevant_cols)
                    # fit a linear regression with only significant variables
                    mask = scv[relevant_cols].notna().all(axis=1)
                    X = scv[relevant_cols][mask].drop(columns=col)
                    y = scv[relevant_cols][mask][col]
                    try:
                        reg = LinearRegression().fit(X, y)
                        # impute the missing values
                        fit_row_dict = {"Y column": col, 
                                        "X columns": relevant_cols[1:], 
                                        "fit": reg,
                                        "num_rows": len(X),
                                        "p-values": p_values}
                        df_fits = pd.concat([df_fits, pd.DataFrame([fit_row_dict], columns=df_fits.columns)], ignore_index=True)
                    except Exception as e:
                        print(f"Error fitting {col} with {relevant_cols[1:]}")
                        print(e)

        for index, row in self.scv[columns].iterrows():
            # find positions of nans
            nan_cols = row[row.isna()].index
            non_nan_cols = row[row.notna()].index
            # check if non_nan_cols is greater than 1
            if len(non_nan_cols) < 2:
                continue
            for col in nan_cols:
                col_df = df_fits.loc[df_fits['Y column'] == col]
                overlap_rows = (col_df['X columns'].apply(lambda x: len(x)*set(x).issubset(set(non_nan_cols))))
                # find positions of best overlap
                best_overlap = np.where(overlap_rows == overlap_rows.max())[0]
                try:
                    if len(best_overlap) > 1:
                        # if there are multiple best overlaps, choose the one with the highest number of rows
                        sorted_col_df = col_df.iloc[best_overlap].copy()
                        sorted_col_df.sort_values('num_rows', ascending=False)
                        if sorted_col_df is None:
                            print('Oh no')
                        best_overlap =  sorted_col_df.index[0]
                    else:
                        best_overlap = col_df.iloc[best_overlap].index[0]
                except Exception as e:
                    print(f"Error finding best overlap for {col}")
                    print(e)
                    continue

                feature_columns = col_df.loc[best_overlap]['X columns']
                input_data = pd.DataFrame([row[feature_columns].values], columns=feature_columns)
                self.scv_imputed.loc[index, col] = col_df.loc[best_overlap]['fit'].predict(input_data)[0]

    ######################################## PCA FUNCTIONS #################################################

    def compute_PCA(self, cols, col_name, n_cols, n_PCA, pca_func = None, rescale = False, contributions = False):

        if len(self.scv_imputed) == 0:
            print("No imputed data found")
            return

        if self.scv_imputed[cols].isna().any().any():
            print("there are some NaNs in the imputed dataset")
        
        scaler = StandardScaler()
        clean_data = self.scv_imputed[cols].dropna()
        df_scaled = scaler.fit_transform(clean_data)

        if pca_func is None:
            pca = PCA(n_components=n_PCA)
            pca.fit(df_scaled)
        else:
            pca = pca_func
        
        if len(self.scv_clean) == 0:
            scv_clean = self.scv.copy()
            self.scv_clean = scv_clean
        self.scv_clean.dropna(subset=cols, inplace=True)
        scv_scaled = scaler.transform(self.scv_clean[cols])
        # Quantify variance explained by each PC
        explained_variance = pca.explained_variance_ratio_

        print("Explained variance by each PC:")
        for i, variance in enumerate(explained_variance):
            print(f"PC{i+1}: {variance*100:.2f}%")

        # calculate the PCA components for each dataset row
        for col in range(n_cols):
            self.scv_clean[f"{col_name}_{col+1}"] = pca.transform(scv_scaled)[:,col]
            self.scv_imputed.loc[self.scv_imputed.index.isin(clean_data.index), f"{col_name}_{col+1}"] = pca.transform(df_scaled)[:, col]
        if rescale:
            # rescale the PCA components to be between 1 and 10
            for col in range(n_cols):
                self.scv_clean[f"{col_name}_{col+1}"] = self.scv_clean[f"{col_name}_{col+1}"].apply(lambda x: (x - min(self.scv_imputed[f"{col_name}_{col+1}"]))/(max(self.scv_imputed[f"{col_name}_{col+1}"]) - min(self.scv_imputed[f"{col_name}_{col+1}"]))*9 + 1)
                self.scv_imputed[f"{col_name}_{col+1}"] = self.scv_imputed[f"{col_name}_{col+1}"].apply(lambda x: (x - min(self.scv_imputed[f"{col_name}_{col+1}"]))/(max(self.scv_imputed[f"{col_name}_{col+1}"]) - min(self.scv_imputed[f"{col_name}_{col+1}"]))*9 + 1)
            
        if contributions:
            # print how much each variable contributes to each PC
            for col in range(n_cols):
                print(f"PC{1} contributions:")
                for i, variable in enumerate(cols):
                    print(f"{variable}: {pca.components_[col][i]:.2f}")
        return pca

    ######################################## SAVE AND LOAD FUNCTIONS ###########################################

    def save_dataset(self, path='', name = 'dataset'):
        if path == '':
            path = os.getcwd()
        file_path = os.path.join(path, name + ".xlsx")
        
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            self.raw.to_excel(writer, sheet_name='Raw', index=False)
            self.scv.to_excel(writer, sheet_name='SCV', index=False)
            self.scv_imputed.to_excel(writer, sheet_name='SCV_Imputed', index=False)
            self.scv_clean.to_excel(writer, sheet_name='SCV_Clean', index=False)
            self.debug.to_excel(writer, sheet_name='Debug', index=False)

        print(f"Dataset saved to {file_path}")

    def load_dataset(self, path='', name = 'dataset'):
        if path == '':
            path = os.getcwd()
        file_path = os.path.join(path, name + ".xlsx")
        
        with pd.ExcelFile(file_path) as reader:
            self.raw = pd.read_excel(reader, sheet_name='Raw')
            self.scv = pd.read_excel(reader, sheet_name='SCV')
            self.scv_imputed = pd.read_excel(reader, sheet_name='SCV_Imputed')
            self.scv_clean = pd.read_excel(reader, sheet_name='SCV_Clean')
            self.debug = pd.read_excel(reader, sheet_name='Debug')

        print(f"Dataset loaded from {file_path}")
        

if __name__ == "__main__":

    # Add the src directory to the Python path
    sys.path.append(os.path.abspath(os.path.join('..')))
    from utils import download_data
    from src.mappings import value_mapping
    dataset_100y = TimeSeriesDataset(categories=['sc'], file_path="datasets/100_yr_dataset.csv")
    scale_cols = ['Pop','Terr','Cap','Hierarchy']
    scale_pca = dataset_100y.compute_PCA(cols = scale_cols, col_name = 'Scale', n_cols = 1, n_PCA= len(scale_cols))