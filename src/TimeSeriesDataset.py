import pandas as pd
import numpy as np
import time
import sys
import os

from utils import download_data, fetch_urls, weighted_mean, get_max, is_same
from Template import Template

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


class TimeSeriesDataset():
    def __init__(self, 
                 categories = list(['sc']),
                 polity_url = "https://seshatdata.com/api/core/polities/?page_size=1000",
                 template_path = None,
                 file_path = None
                 ):

        self.categories = categories
        self.polity_url = polity_url
        self.raw = pd.DataFrame()
        self.scv = pd.DataFrame()
        self.scv_imputed = pd.DataFrame()
        self.debug = pd.DataFrame(columns=["polity", "variable", "label", "issue"])

        if (polity_url is not None ) and (template_path is None):
            self.template = Template(categories=categories, polity_url=polity_url)
        elif (template_path is not None):
            self.template = Template(categories=categories, file_path=template_path)
        else:
            print("Please provide either a polity_url or a template_path")
            sys.exit()
    
    def __len__(self):
        # check what datasets are available
        return len(self.raw)

    def __getitem__(self, idx):
        # check what datasets are available
        return self.raw.iloc[idx]
    
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
                                        "Year": year}), index=[0])
        self.raw = pd.concat([self.raw, pol_df_new])
        row = self.raw.loc[self.raw.Year.isna()&(self.raw.PolityID == polID)]
        if len(row) > 0:
            self.raw.drop(row.index, inplace=True)
        self.raw.reset_index(drop=True, inplace=True)

    def download_all_categories(self):
        urls = {}
        for category in self.categories:
            urls.update(fetch_urls(category))
        for key in urls.keys():
            self.add_column(key)
    
    def add_column(self, key):
        variable_name = key.split('/')[-1]
        self.raw[variable_name] = self.raw.apply(lambda row: self.sample_from_template(row, variable_name), axis=1)
    
    def sample_from_template(self, row, variable, label = 'pt'):
        pol = row.PolityID
        year = row.Year
        entry = self.template.template.loc[(self.template.template.PolityID == pol), variable]
        if len(entry) == 0:
            return np.nan
        
        if pd.isna(entry.values[0]):
            return np.nan
        _dict = eval(entry.values[0])
        result =  self.template.sample_dict(_dict, year)
        if result == "Out of bounds":
            debug_row = {"polity" : row.PolityName,
                        "variable": variable, 
                        "label": label,
                        "issue": f"{year} ouside of polity years"}
            self.debug = pd.concat([self.debug, pd.DataFrame([debug_row], columns=self.debug.columns)], ignore_index=True)
            return np.nan
        return result

    def remove_incomplete_rows(self, nan_threshold = 0.3):
        # add all columns from sc_mapping
        from mappings import social_complexity_mapping
        cols = []
        for key in social_complexity_mapping.keys():
            cols += [key for key in list(social_complexity_mapping[key].keys())]
        
        # remove rows with less than 30% of the columns filled in
        self.raw = self.raw.loc[self.raw[cols].notna().sum(axis=1)/len(cols)>0.3]
        self.raw.reset_index(drop=True, inplace=True)

    def build_social_complexity(self):
        from mappings import social_complexity_mapping
        # create dataframe for social complexity
        self.scv = self.raw[['NGA', 'PolityID', 'PolityName', 'Year']].copy()

        # add population variables
        self.scv['Pop'] = (self.raw['polity-populations']).apply(np.log10)
        self.scv['Terr'] = (self.raw['polity-territories']).apply(np.log10)
        self.scv['Cap'] = (self.raw['population-of-the-largest-settlements']).apply(np.log10)

        # add hierarchy variables
        self.scv['Hierarchy'] = self.raw.apply(lambda row: weighted_mean(row, social_complexity_mapping, "Hierarchy", imputation='mean'), axis=1)
        self.scv['Government'] = self.raw.apply(lambda row: weighted_mean(row, social_complexity_mapping, "Government", imputation = 'zero'), axis=1)
        self.scv['Infrastructure'] = self.raw.apply(lambda row: weighted_mean(row, social_complexity_mapping, "Infrastructure", imputation= 'zero'), axis=1)
        self.scv['Information'] = self.raw.apply(lambda row: weighted_mean(row, social_complexity_mapping, "Information", imputation='zero'), axis=1)

        self.scv['Money'] = self.raw.apply(lambda row: get_max(row, social_complexity_mapping, "Money"), axis=1)

    def impute_missing_values(self):

        sc_columns = ['Pop','Cap','Terr','Hierarchy', 'Government', 'Infrastructure', 'Information', 'Money']
        scv = self.scv[sc_columns]
        self.scv_imputed = scv.copy()

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

        for index, row in scv.iterrows():
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

    def save_dataset(self, path=''):
        if path == '':
            path = os.getcwd()
        raw_path = os.path.join(path, "raw.csv")
        self.raw.to_csv(raw_path, index=False)
        debug_path = os.path.join(path, "debug.csv")
        
        #TODO: check if other datasets are empty and if not save them as well

    def load_dataset(self, path, dataset = 'raw'):
        
        if dataset == 'raw':
            path = os.path.join(path, "raw.csv")
            self.raw = pd.read_csv(path, low_memory=False)

        else:
            print("Dataset not found")
            return
        

if __name__ == "__main__":
    import sys
    import os
    import numpy as np

    # Add the src directory to the Python path
    sys.path.append(os.path.abspath(os.path.join('..', 'src')))
    from utils import download_data
    from mappings import value_mapping
    # initialize dataset by downloading dataset or downloading the data from polity_url
    dataset = TimeSeriesDataset(categories=['sc'], template_path='/Users/mperuzzo/Documents/repos/SeshatDatasetAnalysis/datasets/test.csv')
    dataset.add_polities()
    url = "https://seshatdata.com/api/crisisdb/power-transitions/"
    pt_df = download_data(url)
    PT_types = ['overturn', 'predecessor_assassination', 'intra_elite',
        'military_revolt', 'popular_uprising', 'separatist_rebellion',
        'external_invasion', 'external_interference']
    for type in PT_types:
        pt_df[type] = pt_df[type].apply(lambda x: value_mapping[x] if x in value_mapping.keys() else np.nan)

    # set nan values to 0
    pt_df.fillna(0, inplace=True)
    for idx, row in pt_df.iterrows():
        polity = row['polity_id']
        if polity not in dataset.raw.PolityID.unique():
            continue
        year = np.mean([row['year_from'], row['year_to']])
        dataset.add_years(polID=polity, year=year)
    dataset.raw = dataset.raw.loc[(dataset.raw.Year.notna())&(dataset.raw.Year!=0)]

    # delete duplicates
    dataset.raw.drop_duplicates(subset=['PolityID', 'Year'], inplace=True)

    dataset.raw = dataset.raw.sort_values(by=['PolityID', 'Year'])
    dataset.raw.reset_index(drop=True, inplace=True)
    dataset.download_all_categories()
    print(dataset.debug)