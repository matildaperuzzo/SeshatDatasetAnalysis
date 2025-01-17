import pandas as pd
import numpy as np
import time
import sys
import os
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import download_data, fetch_urls, weighted_mean, get_max
from src.mappings import value_mapping, social_complexity_mapping, miltech_mapping, ideology_mapping


class Template():
    def __init__(self, 
                 categories = list(['sc']),
                 polity_url = "https://seshat-db.com/api/core/polities/",
                 file_path = None
                 ):
        self.template = pd.DataFrame()
        self.categories = categories
        self.polity_url = polity_url

        self.debug = pd.DataFrame(columns=["polity", "variable", "label", "issue"])

        if (polity_url is not None ) and (file_path is None):
            self.initialize_dataset(polity_url)
        elif (file_path is not None):
            self.load_dataset(file_path)
        else:
            print("Please provide either a polity_url or a file_path")
            sys.exit()
        
    def __len__(self):
        return len(self.template)

    def __getitem__(self, idx):
        return self.template.iloc[idx]
    
    # ---------------------- HELPER FUNCTIONS ---------------------- #
    def compare_dicts(self, dict1, dict2):
        """Compare whether two dictionaries are the same entry by entry."""
        differences = {}
        
        # Get all keys from both dictionaries
        all_keys = set(dict1.keys()).union(set(dict2.keys()))
        
        for key in all_keys:
            value1 = dict1.get(key, None)
            value2 = dict2.get(key, None)
            
            if value1 != value2:
                if pd.isnull(value1) and pd.isnull(value2):
                    continue
                differences[key] = (value1, value2)
        
        return differences

    def compare_rows(self, row1, row2):
        """Compare whether two rows are the same entry by entry. Returns a dictionary of differences."""
        differences = self.compare_dicts(dict(row1), dict(row2))
        return differences

    def is_same(self, row1, row2):
        """Check if two rows are the same entry by entry. Returns a boolean."""
        return self.compare_rows(row1, row2) == {}

    def check_for_nans(self,d):
        """Check if a variable dictionary contains NaN values."""

        if not isinstance(d, dict):
            if np.isnan(d):
                return False
            else:
                print(d)
            return False
        
        def contains_nan(values):
            # Check if the values are numeric and contain NaNs
            if isinstance(values, (list, np.ndarray)):
                return any(isinstance(v, (int, float)) and np.isnan(v) for v in values)
            return False
        
        ts = d.get('t', [])
        if contains_nan(ts):
            return True
        
        vals = d.get('value', [])
        for val_row in vals:
            for (x, y) in val_row:
                if (isinstance(x, (int, float)) and np.isnan(x)) or (isinstance(y, (int, float)) and np.isnan(y)):
                    return True
        
        years = d.get('polity_years', [])
        if contains_nan(years):
            return True
        
        return False

    def check_nan_polities(self, pol, df, variable_name):
        """Check if a polity has all NaN values for a given variable."""
        pol_df = df.loc[df.polity_id == pol]
        if pol_df.empty:
            return True
        if pol_df[variable_name].isnull().all():
            return True
        return False

    def get_values(self, val_from, val_to):
        """Clean up the values for a range variable."""
        if (val_from is None) and (val_to is None):
            return None
        elif (val_from is not None) and (val_to is None):
            val_to = val_from
        elif (val_from is None) and (val_to is not None):
            val_from = val_to
        return (val_from, val_to)

    def add_empty_col(self, variable_name):
        self.template[variable_name] = np.nan
        self.template[variable_name] = self.template[variable_name].astype('object')

    # ---------------------- BUILDING FUNCTIONS ---------------------- #

    def initialize_dataset(self, url):
        # set up empty template
        self.template = pd.DataFrame(columns = ["NGA", "PolityID", "PolityName"])
        # specify the columns data types
        self.template['PolityID'] = self.template['PolityID'].astype('int')
        # download the polity data
        df = download_data(url)

        polityIDs = df.id.unique()
        # iterate over all polities
        for polID in polityIDs:
            pol_df = df.loc[df.id == polID, ['home_nga_name', 'id', 'new_name', 'name','start_year','end_year']]
            # create a temporary dataframe with all data for current polity
            pol_df_new = pd.DataFrame(dict({"NGA" : pol_df.home_nga_name.values[0], 
                                            "PolityID": pol_df.id.values[0], 
                                            "PolityName": pol_df.new_name.values[0], 
                                            "PolityOldName": pol_df.name.values[0],
                                            "StartYear": pol_df.start_year.values[0],
                                            "EndYear": pol_df.end_year.values[0]}), index = [0])
            # add the temporary dataframe to the template
            self.template = pd.concat([self.template, pol_df_new])
        self.template.reset_index(drop=True, inplace=True)

    def download_all_categories(self):
        urls = {}
        for category in self.categories:
            urls.update(fetch_urls(category))
        for key in urls.keys():
            self.add_dataset_from_url(key,urls[key])
    
    def add_dataset_from_url(self, key, url):

        # check if the dataset is already in the dataframe
        if key in self.template.columns:
            print(f"Dataset {key} already in dataframe")
            return
        
        # download the data
        tic = time.time()
        df = download_data(url)
        toc = time.time()
        print(f"Downloaded {key} dataset with {len(df)} rows in {toc-tic} seconds")
        if len(df) == 0:
            print(f"Empty dataset for {key}")
            return
        self.add_to_template(df, key)


    def add_to_template(self, df, key):
        variable_name = df.name.unique()[0].lower()
        range_var =  variable_name + "_from" in df.columns
        col_name = key.split('/')[-1]
        self.add_empty_col(col_name)
        polities = self.template.PolityName.unique()
        
        for pol in polities:
            if pol not in df.polity_new_name.values:
                pol_old_name = self.template.loc[self.template.PolityName == pol, 'PolityOldName'].values[0]
                pol_df = download_data("https://seshat-db.com/api/"+f"{key}/?polity__name__icontains={pol_old_name}",size = None)
                if pol_df.empty:
                    continue
                else:
                    print(f"Found {pol_old_name} in {key} dataset")
            else:
                pol_df = df.loc[df.polity_new_name == pol]
            
            self.add_polity(pol_df, range_var, variable_name, col_name)
        
        self.perform_tests(df, variable_name, range_var, col_name)
        print(f"Added {key} dataset to template")

    def add_polity(self, pol_df, range_var, variable_name, col_name):
        
        # create a dataframe with only the data for the current polity and sort it by year
        # this allows to assume entries are dealth with in chronological order
        pol = pol_df.polity_id.values[0]

        pol_df = pol_df.sort_values(by = 'year_from')
        pol_df = pol_df.reset_index(drop=True)

        polity_years = [self.template.loc[self.template.PolityID == pol, 'StartYear'].values[0], self.template.loc[self.template.PolityID == pol, 'EndYear'].values[0]]
        
        # reset variable dict variables
        times = [[]]
        values = [[]]

        for ind,row in pol_df.iterrows():
            # reset variables
            disp = False
            unc = False
            t = []
            value = []
            # check if the polity has multiple rows
            if ind > 0:
                if range_var:
                    relevant_columns = ['polity_id','year_from', 'year_to', 'is_disputed', 'is_uncertain', variable_name+'_from', variable_name +'_to']
                else:
                    relevant_columns = ['polity_id','year_from', 'year_to', 'is_disputed', 'is_uncertain', variable_name]
                # if the row is a duplicate of the previous row, skip it
                if pol_df.loc[:ind-1, relevant_columns].apply(lambda x: self.is_same(x, pol_df.loc[ind,relevant_columns]), axis=1).any():
                    print("Duplicate rows found")
                    continue
                elif pol_df.loc[ind,'is_disputed']:
                    # check if the disputed row has the same year as a previous row
                    if pol_df.loc[:ind-1,'year_from'].apply(lambda x: x == pol_df.loc[ind,'year_from']).any():
                        disp = True
                    # check if the disputed row doesn't have a year, this is here because NaN != NaN so need to check separately
                    elif pol_df.loc[:ind-1,'year_from'].isna().any() and pol_df.loc[ind,'year_from'].isna():
                        disp = True
                elif pol_df.loc[ind,'is_uncertain']:
                    if pol_df.loc[:ind-1,'year_from'].apply(lambda x: x == pol_df.loc[ind,'year_from']).any():
                        unc = True
                    elif pol_df.loc[:ind-1,'year_from'].isna().any() and pd.isna(pol_df.loc[ind,'year_from']):
                        unc = True

            if ind < len(pol_df)-1:
                # in the case of the year to being the same as the year from of the next row, subtract one year to the year from to remove overlap
                if (pol_df.loc[ind,'year_from'] is not None):
                    if (pol_df.loc[ind,'year_to'] == pol_df.loc[ind+1,'year_from']) and (pol_df.loc[ind,'year_from'] != pol_df.loc[ind+1,'year_from']):
                        if row.year_to == row.year_from:
                            sys.exit(7)
                        row.year_to = row.year_to - 1
                    
            # check if polity has no year data and in that case use the polity start and end year
            if (row.year_from is None or pd.isna(row.year_from)) and (row.year_to is None or pd.isna(row.year_to)):
                # if the variable is a range variable, check if the range is defined
                if range_var:
                    val_from = row[variable_name + "_from"]
                    val_to = row[variable_name + "_to"]
                    # if no range variables are defined skip the row
                    val = self.get_values(val_from, val_to)
                    if val is None:
                        continue
                else:
                    v = value_mapping.get(row[variable_name], -1)
                    if (v is None) or pd.isna(v):
                        continue
                    elif v == -1:
                        debug_row = pd.DataFrame({"polity": pol, "variable": variable_name, "label": 'template', "issue": f"value {row[variable_name]} is not in mapping"}, index = [0])
                        self.debug = pd.concat([self.debug, debug_row])
                        continue
                    val = (value_mapping[row[variable_name]], value_mapping[row[variable_name]])

                # append the values and times to the lists
                value.append(val)
                value.append(val)
                t.append(self.template.loc[self.template.PolityID == pol, 'StartYear'].values[0])
                t.append(self.template.loc[self.template.PolityID == pol, 'EndYear'].values[0])
                
            # check if only one year is defined, either because the year_from and year_to are the same or one of them is None
            elif (row.year_from == row.year_to) or ((row.year_from is None) and (row.year_to is not None)) or ((row.year_from is not None) and (row.year_to is None)):
                # if variable is a range variable, check if the range is defined
                if range_var:
                    val_from = row[variable_name + "_from"]
                    val_to = row[variable_name + "_to"]
                    # if no range variables are defined skip the row
                    val = self.get_values(val_from, val_to)
                    if val is None:
                        continue
                    
                else:
                    v = value_mapping.get(row[variable_name], -1)
                    if (v is None) or pd.isna(v):
                        continue
                    elif v == -1:
                        debug_row = pd.DataFrame({"polity": pol, "variable": variable_name, "label": 'template', "issue": f"value {row[variable_name]} is not in mapping"}, index = [0])
                        self.debug = pd.concat([self.debug, debug_row])
                        continue
                    val = (v, v)

                value.append(val)
                year = row.year_from if row.year_from is not None else row.year_to
                
                if year < self.template.loc[self.template.PolityID == pol, 'StartYear'].values[0]:
                    print("Error: The year is outside the polity's start and end year")
                    debug_row = pd.DataFrame({"polity": pol, "variable": variable_name, "label": 'template', "issue": f"year {year} outside polity years"}, index = [0])
                    self.debug = pd.concat([self.debug, debug_row])
                    continue
                elif year > self.template.loc[self.template.PolityID == pol, 'EndYear'].values[0]:
                    print("Error: The year is outside the polity's start and end year")
                    debug_row = pd.DataFrame({"polity": pol, "variable": variable_name, "label": 'template', "issue": f"year {year} outside polity years"}, index = [0])
                    self.debug = pd.concat([self.debug, debug_row])
                    continue
                    
                t.append(year)

            # elif (row.year_from != row.year_to) and ((row.year_from is not None) or (pd.isnan(row.year_from))) and ((row.year_to is not None) or (pd.isnan(row.year_to))):
            elif (row.year_from != row.year_to) and pd.notna(row.year_from) and pd.notna(row.year_to):
                
                if range_var:
                    val_from = row[variable_name + "_from"]
                    val_to = row[variable_name + "_to"]
                    # if no range variables are defined skip the row
                    val = self.get_values(val_from, val_to)
                    if val is None:
                        continue
                else:
                    # check if row[variable_name] is a finite number
                    if isinstance(row[variable_name], (int, float)) and pd.notna(row[variable_name]):
                        v = row[variable_name]
                    else:
                        v = value_mapping.get(row[variable_name], -1)
                        if (v is None) or pd.isna(v):
                            continue
                        elif v == -1:
                            debug_row = pd.DataFrame({"polity": pol, "variable": variable_name, "label": 'template', "issue": f"value {row[variable_name]} is not in mapping"}, index = [0])
                            self.debug = pd.concat([self.debug, debug_row])
                            continue
                    val = (v, v)

                value.append(val)
                value.append(val)
                t_from = row.year_from
                t_to = row.year_to

                if isinstance(t_from, (str)):
                    t_from = t_from.replace('CE', '').replace('BCE','')
                    t_from = int(t_from)
                if isinstance(t_to, (str)):
                    t_to = t_to.replace('CE', '').replace('BCE','')
                    t_to = int(t_to)

                if t_from<self.template.loc[self.template.PolityID == pol, 'StartYear'].values[0]:
                    print("Error: The year is outside the polity's start and end year")
                    debug_row = pd.DataFrame({"polity": pol, "variable": variable_name, "label": "template", "issue": f"year {t_from} outside polity years"}, index = [0])
                    self.debug = pd.concat([self.debug, debug_row])
                    continue
                elif t_to > self.template.loc[self.template.PolityID == pol, 'EndYear'].values[0]:
                    print("Error: The year is outside the polity's start and end year")
                    debug_row = pd.DataFrame({"polity": pol, "variable": variable_name, "label": "template", "issue": f"{t_to} outside polity years"}, index = [0])
                    self.debug = pd.concat([self.debug, debug_row])
                    continue
                    
                t.append(t_from)
                t.append(t_to)
            else:
                print('new')
                sys.exit(1) 
                
            if disp or unc:

                new_vals = []
                new_t = []
                for val_row,time_row in zip(values,times):
                    # find the closest year to t
                    for ti in t:
                        time_diff = np.abs(np.array(time_row)-np.array(ti))
                        ind = np.argmin(time_diff)
                        new_t_row = time_row.copy()
                        new_t_row[ind] = ti
                        new_t.append(new_t_row)
                        new_row = val_row.copy()
                        new_row[ind] = val
                        new_vals.append(new_row)
                #  append new timeline to the value entry of the dictionary
                values = values + new_vals
                times = times + new_t
            else:
                if len(values[0]) == 0:
                    values = list([value])
                else:
                    for val_row in range(len(values)):
                        values[val_row] = values[val_row] + value
                if len(times[0]) == 0:
                    times = list([t])
                else:
                    for time_row in range(len(times)):
                        times[time_row] = list(times[time_row]) + t

        variable_dict = {"t": times, "value": values, "polity_years": polity_years}

        for dict_row,t_row in zip(variable_dict['value'],variable_dict['t']):
            if len(t_row) != len(dict_row):
                # add to debug dataframe
                debug_row = pd.DataFrame({"polity": pol, "variable": variable_name, "label": "template", "issue": "mismatched lengths"}, index = [0])
                self.debug = pd.concat([self.debug, debug_row])
                return "Error: The length of the time and value arrays are not the same"
    
        if len(variable_dict['t'][0]) == 0:
            return "Error: No data for polity"
            
        self.template.loc[self.template.PolityID == pol, col_name] = [variable_dict]

    def perform_tests(self, df, variable_name, range_var, col_name):
        if self.template[col_name].apply(lambda x: self.check_for_nans(x)).any():
            print("Error: NaNs found in the data")
            sys.exit(4)
        if range_var:
            var_name = variable_name + "_from"
        else:
            var_name = variable_name
        if (self.template['PolityID'].apply(lambda x: self.check_nan_polities(x, df, var_name)) > self.template[col_name].isna()).all():
            print("Nans in template that are not in the template")
            sys.exit(5)
        elif (self.template['PolityID'].apply(lambda x: self.check_nan_polities(x, df, var_name)) < self.template[col_name].isna()).all():
            print("Extra entries in the template")
            sys.exit(6)

        return "Passed tests"
    
    # ---------------------- SAMPLING FUNCTIONS ---------------------- #

    def sample_dict(self, variable_dict, t, error = 0):
        if variable_dict is None or pd.isna(variable_dict):
            return None
        if len(variable_dict['t'][0]) == 0:
            return None
        if len(variable_dict['value'][0]) == 0:
            return None
        if len(variable_dict['polity_years']) == 0:
            return None
        
        n_timelines = len(variable_dict['value'])
        s = random.randint(0, n_timelines-1)
        times = variable_dict['t'][s]
        values = variable_dict['value'][s]
        polity_years = variable_dict['polity_years']
        error = abs(error)
        polity_years = [polity_years[0] - error, polity_years[1] + error]

        if polity_years[0] not in times:
            times = [polity_years[0]] + times
            values = [values[0]] + values

        if polity_years[1] not in times:
            times = times + [polity_years[1]]
            values = values + [values[-1]]

        times = np.array(times)
        
        random_number = random.random()
        if isinstance(t, (list, np.ndarray)):
            vals = [None] * len(t)
            for i, time in enumerate(t):
                if time < polity_years[0] or time > polity_years[1]:
                    print(f"Error: The year {time} is outside the polity years {polity_years}")
                    vals[i] = "Out of bounds"
                    continue
                time_selection = times[times<=time]
                ind = np.argmin(np.abs(np.array(time_selection) - time))
                val = values[ind][0] + random_number * (values[ind][1] - values[ind][0])
                vals[i] = val
            return vals
        elif isinstance(t, (int, float, np.int64, np.int32, np.float64, np.float32)):
            if t < polity_years[0] or t > polity_years[1]:
                print(f"Error: The year {t} is outside the polity years {polity_years}")
                return "Out of bounds"
            # find the closest year to t
            times = times[times<=t]
            ind = np.argmin(np.abs(np.array(times) - t))
            # sample the values
            val = values[ind][0] + random.random() * (values[ind][1] - values[ind][0])
            return val
        else:
            print("Error: The year is not a number")
            return "Error: The year is not a number"
        
    # ---------------------- DEBUG FUNCTIONS ---------------------- #

    def is_in_range(self, variable_dict, t, value):
        if variable_dict is None and pd.notna(value):
            return False
        elif variable_dict is None and pd.isna(value):
            return True
        
        if len(variable_dict['t'][0]) == 0:
            return np.nan
        if len(variable_dict['value'][0]) == 0:
            return np.nan
        if len(variable_dict['polity_years']) == 0:
            return np.nan

        times = variable_dict['t'][0]
        values = self.reduce_to_largest_ranges(variable_dict['value'])
        polity_years = variable_dict['polity_years']

        if polity_years[0] not in times:
            times = [polity_years[0]] + times
            values = [values[0]] + values

        if polity_years[1] not in times:
            times = times + [polity_years[1]]
            values = values + [values[-1]]

        if t < polity_years[0] or t > polity_years[1]:
            print(f"Error: The year {t} is outside the polity years {polity_years}")
            return "Out of bounds"
        # find the closest year to t
        times = np.array(times)[times<=t]
        ind = np.argmin(np.abs(times - t))
        # sample the values
        val = values[ind]
        if min(val) <= value <= max(val):
            return True
        else:
            return False

    def reduce_to_largest_ranges(self, values):
        # Initialize a list to store the (min, max) tuples
        result = []
        
        # Get the length of the inner lists (assuming all inner lists have the same length)
        num_points = len(values[0])
        
        # Iterate through the indices of the inner lists
        for i in range(num_points):
            # Initialize min and max values for the current index
            min_value = float('inf')
            max_value = float('-inf')
            
            # Iterate through the outer list
            for inner_list in values:
                # Get the tuple at the current index
                x1, x2 = inner_list[i]
                
                # Update min and max values
                min_value = min(min_value, x1)
                max_value = max(max_value, x2)
            
            # Append the (min, max) tuple to the result list
            result.append((min_value, max_value))
        
        # Return the result list
        return result

    # ---------------------- SAVING FUNCTIONS ---------------------- #
    def save_dataset(self, file_path):
        self.template.to_csv(file_path, index = False)
        print(f"Saved template to {file_path}")
    # ---------------------- LOADING FUNCTIONS ---------------------- #
    def load_dataset(self, file_path):
        self.template = pd.read_csv(file_path)
        print(f"Loaded template from {file_path}")


# ---------------------- TESTING ---------------------- #
if __name__ == "__main__":
    # Test the Template class
    template = Template(categories = ['sc'])
    template.download_all_categories()
    template.save_dataset("datasets/test.csv")
    template.debug.to_csv("datasets/template_debug.csv", index = False)

    import sys
    import os


    from mappings import ideology_mapping
    template = Template(categories = ['sc','wf'], file_path= 'datasets/test.csv')
    df = pd.read_csv('datasets/mr_dataset.04.2021.csv')
    polity_df = download_data("https://seshat-db.com/api/core/polities/")
    column_mappings = {}
    for column in df.columns:
        column_mappings[column] = column.lower().replace('.', '_').replace('-', '_')
    column_mappings['Date.From'] = 'year_from'
    column_mappings['Date.To'] = 'year_to'
    column_mappings['Variable'] = 'name'
    df.rename(columns=column_mappings, inplace=True)

    df['polity_id'] = df.apply(lambda x: int(polity_df.loc[polity_df['name'] == x['polid'], 'id'].values[0]) if x['polid'] in polity_df['name'].values else None, axis=1)

    df = df[df.polity_id.notna()]
    df.polity_id = df.polity_id.astype(int)

    df['polity_new_name'] = df.apply(lambda x: polity_df.loc[polity_df['id'] == x['polity_id'], 'new_name'].values[0] if x['polity_id'] in polity_df['id'].values else None, axis=1)
    col_replace = {'primary' : 'moral-concern-is-primary',
               'certain' : 'moralizing-enforcement-is-certain',
               'broad' : 'moralizing-norms-are-broad',
               'targeted' : 'moralizing-enforcement-is-targeted',
               'ruler': 'moralizing-enforcement-of-rulers',
               'elite' : 'moralizing-religion-adopted-by-elites',
               'commoners' : 'moralizing-religion-adopted-by-commoners',
               'afterlife' : 'moralizing-enforcement-in-afterlife',
               'thislife' : 'moralizing-enforcement-in-this-life',
               'agency' : 'moralizing-enforcement-is-agentic',
               'start' : 'year_from',
                'end' : 'year_to',
               }

    df.rename(columns=col_replace, inplace=True)    
    df['is_disputed'] = False
    df['is_uncertain'] = False

    variables = [col for col in col_replace.values()]
    for variable in variables:
        if (variable in ideology_mapping['MSP'].keys()):
            print(variable)
            var_df = df.copy()
            var_df['name'] = variable
            # make sure that if an element of the col variable has more than one entry, the first one is taken
            template.add_to_template(var_df, variable)

    template.save_dataset("datasets/SC_WF_MSP_template.csv")