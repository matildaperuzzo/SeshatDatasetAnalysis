import pandas as pd
import numpy as np
import time
import sys
import os

from utils import download_data, fetch_urls, weighted_mean, get_max, is_same
import requests

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


class TimeSeriesDataset():
    def __init__(self, 
                 categories = list(['sc']),
                 dt = 100, 
                 start_year = -10000, 
                 end_year = 2000, 
                 polity_url = "https://seshatdata.com/api/core/polities/?page_size=1000",
                 file_path = None
                 ):
        self.raw = pd.DataFrame()
        self.scv = pd.DataFrame()
        self.categories = categories
        self.polity_url = polity_url

        self.debug_unknowns = pd.DataFrame(columns = ["category", "unknown label"])

        self.timeline = np.arange(start_year, end_year, dt)
        if (polity_url is not None ) and (file_path is None):
            self.initialize_dataset(polity_url)
        elif (file_path is not None):
            self.load_dataset(file_path, dataset='raw')
        else:
            print("Please provide either a polity_url or a file_path")
            sys.exit()
        
    def __len__(self):
        # check what datasets are available
        return len(self.raw)

    def __getitem__(self, idx):
        # check what datasets are available
        return self.raw.iloc[idx]
    
    def initialize_dataset(self, url):
        df = download_data(url)
        # create the dattaframe staring with the polity data
        self.raw = pd.DataFrame(columns = ["NGA", "PolityID", "PolityName", "Year", "PolityActive", "Note"])
        # specify the columns data types
        self.raw['PolityID'] = self.raw['PolityID'].astype('int')
        self.raw['Year'] = self.raw['Year'].astype('int')
        self.raw['PolityActive'] = self.raw['PolityActive'].astype('bool')
        self.raw['Note'] = self.raw['Note'].astype('object')

        # polity_home_nga_id, polity_id, polity_name 
        polityIDs = df.id.unique()

        for polID in polityIDs:
            pol_df = df.loc[df.id == polID, ['home_nga_name', 'id', 'new_name','start_year','end_year']]
            # create a temporary dataframe with all data for current polity
            pol_df_new = pd.DataFrame(dict({"NGA" : pol_df.home_nga_name.values[0], 
                                            "PolityID": pol_df.id.values[0], 
                                            "PolityName": pol_df.new_name.values[0], 
                                            "Year": self.timeline, 
                                            "PolityActive": False, 
                                            "Note": ""}))
            row = pol_df.iloc[0]
            #Mark the years when the polity was active
            pol_df_new.loc[(pol_df_new.Year >= row.start_year) & (pol_df_new.Year <= row.end_year), 'PolityActive'] = True
            # Ensure the index is unique before concatenating
            if not pol_df_new.index.is_unique:
                pol_df_new = pol_df_new.reset_index(drop=True)
            self.raw = pd.concat([self.raw, pol_df_new])

    def download_all_categories(self):
        urls = {}
        for category in self.categories:
            urls.update(fetch_urls(category))
        for key in urls.keys():
            self.add_dataset(key,urls[key])
    
    def add_dataset(self, key, url):

        # check if the dataset is already in the dataframe
        if key in self.raw.columns:
            print(f"Dataset {key} already in dataframe")
            return
        
        # download the data
        tic = time.time()
        df = download_data(url)
        df.reset_index(drop=True, inplace=True)
        toc = time.time()
        print(f"Downloaded {key} dataset with {len(df)} rows in {toc-tic} seconds")
        if len(df) == 0:
            print(f"Empty dataset for {key}")
            return
        
        # fill in the year variables
        df['years_absent'] = (df.year_from.isna())&(df.year_to.isna()) # check which rows have no years, in these cases polity start and end year are used
        df.loc[(df.year_from.notna())&(df.year_to.isna()), 'year_to'] = df.loc[(df.year_from.notna())&(df.year_to.isna()), 'year_from']
        df.loc[(df.year_from.isna())&(df.year_to.isna()), 'year_from'] = df.loc[(df.year_from.isna())&(df.year_to.isna()), 'polity_start_year']
        df.loc[(df.year_to.isna()), 'year_to'] = df.loc[(df.year_to.isna()), 'polity_end_year']
        df.loc[(df.year_from.isna())&(df.year_to.notna()), 'year_from'] = df.loc[(df.year_from.isna())&(df.year_to.notna()), 'year_to']

        # create new columns for the variable
        variable_name = df.name.unique()[0].lower()
        # check if it is a range variable
        range_var =  variable_name + "_from" in df.columns
        new_keys = [key, key + '_note'] 

        if range_var: # if it is a range variable create new columns for the range
            new_keys = [key, key + '_from', key + '_to', key + '_note']

        new_columns = pd.DataFrame(columns = new_keys)
        # add new columns to the dataframe
        if not new_columns.index.is_unique:
            new_columns = new_columns.reset_index(drop=True)
        try:
            self.raw = pd.concat([self.raw, new_columns])
        except Exception as e:
            print(f"Error adding {key} to dataset")
            print(e)
            return
        
        # define data type for the new columns (done to avoid warnings)
        for k in new_keys[:-1]: #don't change note column
            self.raw[k] = np.nan
            self.raw[k] = self.raw[k].astype('float')

        self.raw[key + '_note'] = np.nan
        self.raw[key + '_note'] = self.raw [key + '_note'].astype('object')  # Explicitly cast to object dtype

        for index, row in df.iterrows():
            # add event from df to self.raw
            self.add_event(index, df, range_var, key, variable_name)

    def add_event(self, event, df, range_var, key, variable_name):
        # import mappings between variable values and numerical values
        from mappings import value_mapping

        row = df.loc[event]
        polityID = row.polity_id
        start_year = row.year_from
        end_year = row.year_to
        # remove polities that have no data at he specified time points
        if self.raw.loc[(self.raw.PolityID == polityID) & (self.raw.Year >= start_year) & (self.raw.Year <= end_year)].empty:
            return
        
        if range_var: # in the case of range variables add columns for from, to and mean
            key_from = key + '_from'
            key_to = key + '_to'
            value_from = row[variable_name + "_from"]
            value_to = row[variable_name + "_to"]

            # for disputed and unceratin rows, take the mean of the values
            if row.is_disputed:
                disputed_rows = df.loc[(df.polity_id == row.polity_id)&(df.year_from == row.year_from)&(df.year_to==row.year_to)]
                value_from = disputed_rows[variable_name + "_from"].mean()
                value_to = disputed_rows[variable_name + "_to"].mean()
            elif row.is_uncertain:
                uncertain_rows = df.loc[(df.polity_id == row.polity_id)&(df.year_from == row.year_from)&(df.year_to==row.year_to)]
                value_from = uncertain_rows[variable_name + "_from"].mean()
                value_to = uncertain_rows[variable_name + "_to"].mean()

            # if one of the values is missing, fill it in with the other value
            if pd.isnull(value_from) and pd.notnull(value_to):
                value_from = value_to
            elif pd.isnull(value_to) and pd.notnull(value_from):
                value_to = value_from
            elif pd.isnull(value_to) and pd.isnull(value_from):
                value_from = np.nan
                value_to = np.nan

            # fill in the values in raw dataframe

            # sometimes there is a value for the entire year range and a value for a part of the range, in that case chose the value from the part of the range
            if row.years_absent: # if the years are absent, only fill out the values if they are not already filled outs
                self.raw.loc[(self.raw.PolityID == polityID) & (self.raw.Year >= start_year) & (self.raw.Year <= end_year) & (self.raw[key_from]).isna(), key_from] = value_from
                self.raw.loc[(self.raw.PolityID == polityID) & (self.raw.Year >= start_year) & (self.raw.Year <= end_year) & (self.raw[key_from]).isna(), key_to] = value_to
                self.raw.loc[(self.raw.PolityID == polityID) & (self.raw.Year >= start_year) & (self.raw.Year <= end_year) & (self.raw[key_from]).isna(), key] = np.mean([value_from, value_to])
                pass
            else:
                self.raw.loc[(self.raw.PolityID == polityID) & (self.raw.Year >= start_year) & (self.raw.Year <= end_year), key_from] = value_from
                self.raw.loc[(self.raw.PolityID == polityID) & (self.raw.Year >= start_year) & (self.raw.Year <= end_year), key_to] = value_to
                self.raw.loc[(self.raw.PolityID == polityID) & (self.raw.Year >= start_year) & (self.raw.Year <= end_year), key] = np.mean([value_from, value_to])
            
        else: # if it is a binary variable
            # find the numerical value from value mapping
            df_key = variable_name
            value = value_mapping.get(row[df_key], -np.inf)

            # if the value is not in the mapping, add it to the debug_unknowns dataframe
            if np.isinf(value):
                self.debug_unknowns = pd.concat([self.debug_unknowns, pd.DataFrame(dict({"category": key, "unknown label": row[df_key]}), index=[0])], axis=0)
                value  = np.nan

            # for disputed and unceratin rows, take the mean of the values
            if row.is_disputed:
                disputed_rows = df.loc[(df.polity_id == row.polity_id) & (df.year_from == row.year_from) & (df.year_to == row.year_to)]
                values = disputed_rows[df_key].map(value_mapping)
                value = values.mean()
            elif row.is_uncertain:
                uncertain_rows = df.loc[(df.polity_id == row.polity_id) & (df.year_from == row.year_from) & (df.year_to == row.year_to)]
                values = uncertain_rows[df_key].map(value_mapping)
                value = values.mean()

            # fill in the values in raw dataframe
            # sometimes there is a value for the entire year range and a value for a part of the range, in that case chose the value from the part of the range
            if row.years_absent: # if the years are absent, only fill out the values if they are not already filled outs
                self.raw.loc[(self.raw.PolityID == polityID) & (self.raw.Year >= start_year) & (self.raw.Year <= end_year) & (self.raw[key]).isna(), key] = value
                pass
            else:   
                self.raw.loc[(self.raw.PolityID == polityID) & (self.raw.Year >= start_year) & (self.raw.Year <= end_year), key] = value

        # add note
        note = ""
        if row.is_disputed:
            note = "disputed"
        if row.is_uncertain:
            if note == "disputed":
                note += ", uncertain"
            else:   
                note = "uncertain"
            
        self.raw.loc[(self.raw.PolityID == polityID) & (self.raw.Year >= start_year) & (self.raw.Year <= end_year), key+'_note'] = note


    def debug_clean_dataset(self):

        if "PolityActive" not in self.raw.columns:
            print("Dataset already cleaned")
            return

        variable_cols = self.raw.columns[6:]
        polity_inactive_data = self.raw.loc[self.raw.PolityActive == False]
        for col in variable_cols:
            if "note" not in col:
                if len(polity_inactive_data.loc[polity_inactive_data[col].notna()]):
                    data = polity_inactive_data.loc[polity_inactive_data[col].notna()]
                    for iter,row in data.iterrows():
                        polityID = row.PolityID
                        polityName = row.PolityName
                        polityStart = self.raw.loc[(self.raw.PolityID == polityID)&(self.raw.PolityActive), 'Year'].min()
                        polityEnd = self.raw.loc[(self.raw.PolityActive == True) & (self.raw.PolityID == polityID), 'Year'].max()
                        issue = f"Polity {polityName} ({polityStart} - {polityEnd}) is inactive but has data for {col} from {row.Year} to {row.Year}"
                        print(issue)
                        new_row = pd.DataFrame({
                            "PolityID": [polityID],
                            "PolityName": [polityName],
                            "Variable": [col],
                            "Year": [row.Year],
                            "Issue": [issue]
                        })
                        self.debug_unknowns = pd.concat([self.debug_unknowns, new_row], ignore_index=True)
        for polityID in self.raw.PolityID.unique():
            duration = self.raw.loc[(self.raw.PolityActive == True) & (self.raw.PolityID == polityID), 'Year'].max() - self.raw.loc[(self.raw.PolityActive == True) & (self.raw.PolityID == polityID), 'Year'].min()
            
            if duration > 8000:
                start_year = self.raw.loc[(self.raw.PolityID == polityID)&(self.raw.PolityActive), 'Year'].min()
                end_year = self.raw.loc[(self.raw.PolityID == polityID)&(self.raw.PolityActive), 'Year'].max()
                polityName = self.raw.loc[self.raw.PolityID == polityID, 'PolityName'].values[0]
                print(start_year, end_year)
                new_row = pd.DataFrame({
                            "PolityID": [polityID],
                            "PolityName": [polityName],
                            "Variable": "Polity duration",
                            "Year": f"{start_year} - {end_year}",
                            "Issue": "unexpectedly long duration for polity"
                        })
                self.debug_unknowns = pd.concat([self.debug_unknowns, new_row], ignore_index=True)
        
        # remove rows where the polity is inactive
        self.raw = self.raw.loc[self.raw.PolityActive == True]
        # remove PolityActive row
        self.raw.drop(columns = ['PolityActive'], inplace=True)
        self.raw.reset_index(drop=True, inplace=True)

    def remove_incomplete_rows(self, nan_threshold = 0.3):
        # add all columns from sc_mapping
        from mappings import social_complexity_mapping
        cols = []
        for key in social_complexity_mapping.keys():
            cols += ['sc/'+ key for key in list(social_complexity_mapping[key].keys())]
        
        # remove rows with less than 30% of the columns filled in
        self.raw = self.raw.loc[self.raw[cols].notna().sum(axis=1)/len(cols)>0.3]
        self.raw.reset_index(drop=True, inplace=True)

    def build_social_complexity(self):
        from mappings import social_complexity_mapping
        # create dataframe for social complexity
        self.scv = self.raw[['NGA', 'PolityID', 'PolityName', 'Year']].copy()

        # add population variables
        self.scv['Pop'] = (self.raw['sc/polity-populations']).apply(np.log10)
        self.scv['Terr'] = (self.raw['sc/polity-territories']).apply(np.log10)
        self.scv['Cap'] = (self.raw['sc/population-of-the-largest-settlements']).apply(np.log10)

        # add hierarchy variables
        self.scv['Hierarchy'] = self.raw.apply(lambda row: weighted_mean(row, social_complexity_mapping, 'sc', "Hierarchy", imputation='remove'), axis=1)
        self.scv['Government'] = self.raw.apply(lambda row: weighted_mean(row, social_complexity_mapping, 'sc', "Government", imputation = 'zero'), axis=1)
        self.scv['Infrastructure'] = self.raw.apply(lambda row: weighted_mean(row, social_complexity_mapping, 'sc', "Infrastructure", imputation= 'zero'), axis=1)
        self.scv['Information'] = self.raw.apply(lambda row: weighted_mean(row, social_complexity_mapping, 'sc', "Information", imputation='zero'), axis=1)

        self.scv['Money'] = self.raw.apply(lambda row: get_max(row, social_complexity_mapping, 'sc', "Money"), axis=1)

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
                if all(p_values>0.001):
                    print('Not enough significant variables')
                    print(f'p-values for {col} are {p_values}')
                else:

                    relevant_cols = p_values[p_values<0.001].index
                    # check if the amount of relevant columns is greater than 1
                    if len(relevant_cols) < 2:
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
                
                if len(best_overlap) > 1:
                    # if there are multiple best overlaps, choose the one with the highest number of rows
                    sorted_col_df = col_df.iloc[best_overlap].copy()
                    sorted_col_df.sort_values('num_rows', ascending=False)
                    if sorted_col_df is None:
                        print('Oh no')
                    best_overlap =  sorted_col_df.index[0]
                else:
                    best_overlap = col_df.iloc[best_overlap].index[0]

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
        elif dataset == 'debug':
            path = os.path.join(path, "debug_unknowns.csv")
            self.debug_unknowns = pd.read_csv(path)
        else:
            print("Dataset not found")
            return
        

if __name__ == "__main__":
    import sys
    import os

    # Add the src directory to the Python path
    sys.path.append(os.path.abspath(os.path.join('..', 'src')))
    # initialize dataset by downloading dataset or downloading the data from polity_url
    dataset = TimeSeriesDataset(categories=['sc'])
    # download all datasets
    dataset.download_all_categories()
    dataset.save_dataset(path = "datasets")
    # remove all rows that have less than 30% of the columns filled in
    # dataset.remove_incomplete_rows(nan_threshold=0.3)
    # build the social complexity variables
    dataset.build_social_complexity()
    # dataset.impute_missing_values()
    print("Done")
