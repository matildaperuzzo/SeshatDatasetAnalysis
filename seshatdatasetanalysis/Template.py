import pandas as pd
import numpy as np
import time
import sys
import os
import random
import itertools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from seshatdatasetanalysis.utils import download_data, fetch_urls, weighted_mean, get_max
from seshatdatasetanalysis.mappings import value_mapping


class Template():
    def __init__(self, 
                 categories = list(['sc']),
                 polity_url = "https://seshat-db.com/api/core/polities/",
                 file_path = None,
                 keep_raw_data = True
                 ):
        self.template = pd.DataFrame()
        self.categories = categories
        self.polity_url = polity_url
        self.keep_raw_data = keep_raw_data
        self.vars_in_template = set()

        self.debug = pd.DataFrame(columns=["polity", "variable", "label", "issue"])
        self.full_dataset = pd.DataFrame()
        self.polity_df = pd.DataFrame()

        if (polity_url is not None ) and (file_path is None):
            self.initialize_dataset(polity_url)
        elif (file_path is not None):
            self.load_dataset(file_path)
#        else: -- Equinox case: pol_df will be created when reading the data
#            print("Please provide either a polity_url or a file_path")
#            sys.exit()
        
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

    @staticmethod
    def is_none_value(val):
        return val is None or pd.isna(val)

    @staticmethod
    def get_values(val_from, val_to):
        """Clean up the values for a range variable."""
        if Template.is_none_value(val_from) and Template.is_none_value(val_to):
            return None
        elif Template.is_none_value(val_to):
            val_to = val_from # here val_from is valid
        elif Template.is_none_value(val_from):
            val_from = val_to # here val_to is valid
        return (val_from, val_to)

    def add_empty_col(self, variable_name):
        self.template[variable_name] = np.nan
        self.template[variable_name] = self.template[variable_name].astype('object')

    # ---------------------- BUILDING FUNCTIONS ---------------------- #

    def initialize_dataset(self, url):
        """
        Initializes the dataset by downloading polity data from the given URL and populating a template DataFrame.
        Args:
            url (str): The URL from which to download the polity data.
        Returns:
            None
        This function performs the following steps:
        1. Sets up an empty template DataFrame with columns ["NGA", "PolityID", "PolityName"].
        2. Specifies the data type for the "PolityID" column as integer.
        3. Downloads the polity data from the provided URL.
        4. Iterates over all unique polity IDs in the downloaded data.
        5. For each polity ID, creates a temporary DataFrame with relevant data.
        6. Adds the temporary DataFrame to the template.
        7. Resets the index of the template DataFrame.
        """

        # download the polity data if needed
        if self.polity_df.empty:
            if url is None:
                raise BaseException('No download URL!')
            self.polity_df = download_data(url)

        # set up empty template
        self.template = self.polity_df[['home_nga_name', 'id', 'name','start_year','end_year']].copy()
        self.template.rename({
            'home_nga_name': 'NGA',
            'id': 'PolityID',
            'name': 'PolityName',
            'start_year': 'StartYear',
            'end_year': 'EndYear'},
            axis = 1, inplace = True)
        
        self.template.reset_index(drop=True, inplace=True)
        
        self.vars_in_template = set()

    def download_all_categories(self, check_polities : bool = False, add_to_template : bool = True):
        """
        Downloads datasets for all categories in the attribute self.categories.
        This method iterates over all categories, fetches URLs for each category,
        and then adds the datasets from the fetched URLs to the instance.
        Parameters:
            check_polities (bool): Whether to check if all polities are included in the downloads, and try to separately download missing ones. This will make the download much slower.
            add_to_template (bool): Whether to add the newly downloaded data to the template. If False, the standalone function add_data_to_template() should be used after the download finished.
        Returns:
            A dict with key-value pairs of variable names and URLs that failed to download.
        """

        urls = {}
        errors = {}
        for category in self.categories:
            urls.update(fetch_urls(category))
        for key in urls.keys():
            if not self.add_dataset_from_url(key, urls[key], check_polities, add_to_template):
                errors[key] = urls[key]
        return errors
    
    def add_dataset_from_url(self, key, url, check_polities : bool = False, add_to_template : bool = False):
        """
        Adds a dataset to the template from a given URL.
        This method checks if the dataset identified by the given key is already present in the template's dataframe.
        If the dataset is not present, it downloads the data from the specified URL, measures the download time,
        and adds the dataset to the template.
        Parameters:
            key (str): The key to identify the dataset in the dataframe.
            url (str): The URL from which to download the dataset.
            check_polities (bool): Whether to check if all polities are included in the download, and try to separately download missing ones.
            add_to_template (bool): Whether to actually add the newly downloaded data to the template.
        Returns:
            Whether the dataset could be successfully downloaded.
        """

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
            return False
        self.process_download(df, key, check_polities, add_to_template)
        return True


    def process_download(self, df, key, check_polities : bool = False, add_to_template : bool = False):
        """
        Process newly downloaded data, optionally storing it in self.full_dataset and / or
        adding it to the template. It checks for the presence of specific columns and handles
        the addition of data for each polity in the template or in full_dataset. If a polity
        is not found in the DataFrame, it can attempt to download it from a specified URL
        (this is controlled by the check_polities parameter). If data is added to the
        template, then this function also performs tests to ensure consistency.
        Args:
            df (pandas.DataFrame): The DataFrame containing the data to be added.
            key (str): The key used to identify the dataset and construct the URL for downloading missing data if check_polities == True.
            check_polities (bool): Whether to check if all polities are included in df, and try to download missing ones.
            add_to_template (bool): Whether to actually add the newly downloaded data to the template.
        Returns:
            None
        """

        if not (add_to_template or self.keep_raw_data):
            # better to throw an exception than silently throw away data if the user
            # called this function with the wrong parameters
            raise BaseException('Nothing to do!')

        variable_name = df.name.unique()[0].lower()
        row_variable_name = variable_name
        if (variable_name not in df.columns) and (variable_name + "_from" not in df.columns):
            row_variable_name = 'coded_value'
        range_var = False
        if variable_name + "_from" in df.columns:
            range_var = True
        elif ('religion' in variable_name) and ('polity' in variable_name):
            row_variable_name = variable_name.replace('polity_', '')
        
        self.add_empty_col(variable_name)
        polities = self.template.PolityName.unique()
        df.columns = df.columns.str.lower()
        n_added = 0
        
        for pol in polities:
            if check_polities and pol not in df.polity_name.values:
                # pol_old_name = self.template.loc[self.template.PolityName == pol, 'PolityOldName'].values[0]
                pol_df = download_data("https://seshat-db.com/api/"+f"{key}/?polity__new_name__icontains={pol}",size = None)
                if not pol_df.empty:
                    print(f"Downloaded {pol} for {key} dataset")
            else:
                pol_df = df.loc[df.polity_name == pol]
            
            if pol_df.empty:
                continue
            
            if add_to_template:
                n_added += self.add_polity(pol_df.polity_name.values[0], pol_df, range_var, variable_name, variable_name)
        
            if self.keep_raw_data:
                if range_var:
                    new_df = pd.DataFrame({
                        "seshat_region": self.template.loc[self.template.PolityID == pol_df.polity_id.iloc[0],'NGA'].values[0],
                        "polity_number": pol_df['polity_id'],
                        "polity_id": pol_df['polity_name'],
                        "section": key.split('/')[0],
                        "variable_name": variable_name,
                        "value_from": pol_df[row_variable_name + '_from'],
                        "value_to": pol_df[row_variable_name + '_to'],
                        "year_from": pol_df['year_from'],
                        "year_to": pol_df['year_to'],
                        "is_disputed": pol_df['is_disputed'],
                        "is_uncertain": pol_df['is_uncertain'],
                        "api_name": key.split('/')[1],
                        "range_var": range_var
                    })
                else:
                    new_df = pd.DataFrame({
                        "seshat_region": self.template.loc[self.template.PolityID == pol_df.polity_id.iloc[0],'NGA'].values[0],
                        "polity_number": pol_df['polity_id'],
                        "polity_id": pol_df['polity_name'],
                        "section": key.split('/')[0],
                        "variable_name": variable_name,
                        "value_from": pol_df[row_variable_name],
                        "value_to": np.nan,
                        "year_from": pol_df['year_from'],
                        "year_to": pol_df['year_to'],
                        "is_disputed": pol_df['is_disputed'],
                        "is_uncertain": pol_df['is_uncertain'],
                        "api_name": key.split('/')[1],
                        "range_var": range_var
                    })
                self.full_dataset = pd.concat([self.full_dataset, new_df], ignore_index=True)
        if add_to_template:
            if n_added == 0:
                print(f"No valid data added for {key}")
            else:
                self.perform_tests(df, row_variable_name, range_var, variable_name)
                self.vars_in_template.add(variable_name)
                print(f"Added {key} dataset to template")
        
    
    def template_from_dataset(self, use_new_method : bool = False):
        """
        (Re-)create the template from self.full_dataset. This should be called after the
        data has been downloaded with download_all_categories(add_to_template = False) or
        read from a polaris Excel file.
        """
        self.initialize_dataset(None)
        polities = self.full_dataset.polity_id.unique()
        variables = self.full_dataset.variable_name.unique()
        for var in variables:
            n_added = 0
            self.add_empty_col(var)
            tmp1 = self.full_dataset.loc[self.full_dataset.variable_name == var]
            # We need to: (1) detect whether this is a range variable; (2) give the correct column names
            range_var = False
            if 'range_var' in tmp1.columns:
                # data downloaded by us, it should have a consistent marking for range_var
                range_var = tmp1.range_var.iloc[0]
                if range_var:
                    tmp1.value_from = tmp1.value_from.astype(np.float64) # this will throw an exception on non-numeric values
            else:
                # check if all are numeric
                is_numeric = tmp1.value_from.apply(
                    lambda x: isinstance(x, int) or isinstance(x, float) or
                    isinstance(x, np.float64) or isinstance(x, np.float32)).all()
                have_to = not tmp1.value_to.isna().all()
                if not is_numeric:
                    try:
                        tmp2 = tmp1.value_from.astype(np.float64)
                        tmp1.value_from = tmp2
                        is_numeric = True
                    except:
                        pass
                if is_numeric:
                    range_var = True
                    ##!! TODO: check that value_to is all numeric as well (although this never has been a problem)
                elif have_to:
                    raise BaseException(f"Expected numeric values for {var}!")
            
            row_variable_name = 'value' if range_var else 'value_from'
            
            for pol in polities:
                tmp2 = tmp1.loc[tmp1.polity_id == pol]
                if not tmp2.empty:
                    n_added += (self.add_polity2(pol, tmp2, range_var, row_variable_name, var) if
                        use_new_method else self.add_polity(pol, tmp2, range_var, row_variable_name, var))
            
            if n_added == 0:
                print(f"No valid data added for {var}")
            else:
                self.perform_tests(tmp1, row_variable_name, range_var, var)
                self.vars_in_template.add(var)
                print(f"Added {var} dataset to template")
        

    def add_polity(self, polity_id : str, pol_df, range_var, variable_name, col_name):
        """
        Adds polity data to the template.
        This function processes a DataFrame containing polity data, checks for duplicates, handles disputed and uncertain entries, 
        and appends the processed data to the template. It ensures that the data is in chronological order and handles various 
        cases where year data might be missing or overlapping.
        Parameters:
        polity_id (str): polity unique (textual) ID
        pol_df (pd.DataFrame): DataFrame containing polity data with columns such as 'year_from', 'year_to', 
                               'is_disputed', 'is_uncertain', and the variable of interest.
        range_var (bool): Indicates whether the variable of interest is a range variable.
        variable_name (str): The name of the variable to be processed.
        col_name (str): The name of the column in the template where the processed data will be stored.
        Returns:
        Number of variant time series added. Can be zero if all coded values are unknown or if an
        error occurs while processing the data (in this case, the polity ID and variable name are added
        to self.debug)
        Raises:
        BaseException: If there is an unexpected error when processing time ranges.
        """
        
        # create a dataframe with only the data for the current polity and sort it by year
        # this allows to assume entries are dealth with in chronological order
   #     pol = pol_df.polity_id.values[0]

        pol_df = pol_df.sort_values(by = 'year_from')
        pol_df = pol_df.reset_index(drop=True)

        polity_years = [self.template.loc[self.template.PolityName == polity_id, 'StartYear'].values[0],
                        self.template.loc[self.template.PolityName == polity_id, 'EndYear'].values[0]]
        
        # reset variable dict variables
        times = [[]]
        values = [[]]
        
        for ind,row in pol_df.iterrows():
            # reset variables
            disp = False
            unc = False
            row_variable_name = variable_name
            if 'polity_religion' in variable_name:
                row_variable_name = variable_name.replace('polity_', '')
            if (row_variable_name not in row) and (row_variable_name + "_from" not in row):
                row_variable_name = 'coded_value'

            t = []
            value = []
            # check if the polity has multiple rows
            if ind > 0:
                if range_var:
                    relevant_columns = ['polity_id','year_from', 'year_to', 'is_disputed', 'is_uncertain', row_variable_name+'_from', row_variable_name +'_to']
                else:
                    relevant_columns = ['polity_id','year_from', 'year_to', 'is_disputed', 'is_uncertain', row_variable_name]
               
                # if the row is a duplicate of the previous row, skip it
                if pol_df.loc[:ind-1, relevant_columns].apply(lambda x: self.is_same(x, pol_df.loc[ind,relevant_columns]), axis=1).any():
                    print("Duplicate rows found")
                    continue
                elif pol_df.loc[ind,'is_disputed']:
                    # check if the disputed row has the same year as a previous row
                    if pol_df.loc[:ind-1,'year_from'].apply(lambda x: x == pol_df.loc[ind,'year_from']).any():
                        disp = True
                    # check if the disputed row doesn't have a year, this is here because NaN != NaN so need to check separately
                    elif pol_df.loc[:ind-1,'year_from'].isna().any() and pd.isna(pol_df.loc[ind,'year_from']):
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
                    
            # check if this value has no year data and in that case use the polity start and end year
            if Template.is_none_value(row.year_from) and Template.is_none_value(row.year_to):
                # if the variable is a range variable, check if the range is defined
                if range_var:

                    val_from = row[row_variable_name + "_from"]
                    val_to = row[row_variable_name + "_to"]
                    # if no range variables are defined skip the row
                    val = Template.get_values(val_from, val_to)
                    if val is None:
                        continue
                elif isinstance(row[row_variable_name], str) and row_variable_name.startswith('religion'):
                    v = row[row_variable_name].lower()
                    val = (v,v)
                else:
                    v = value_mapping.get(row[row_variable_name], -1)
                    if (v is None) or pd.isna(v):
                        continue
                    elif v == -1:
                        debug_row = pd.DataFrame({"polity": polity_id, "variable": variable_name, "label": 'template', "issue": f"value {row[row_variable_name]} is not in mapping"}, index = [0])
                        self.debug = pd.concat([self.debug, debug_row])
                        continue

                    val = (value_mapping[row[row_variable_name]], value_mapping[row[row_variable_name]])

                # append the values and times to the lists
                value.append(val)
                value.append(val)
                t += polity_years
                
            # check if only one year is defined, either because the year_from and year_to are the same or one of them is None
            elif (row.year_from == row.year_to) or (Template.is_none_value(row.year_from) != Template.is_none_value(row.year_to)):
                # if variable is a range variable, check if the range is defined
                if range_var:
                    val_from = row[row_variable_name + "_from"]
                    val_to = row[row_variable_name + "_to"]
                    # if no range variables are defined skip the row
                    val = Template.get_values(val_from, val_to)
                    if val is None:
                        continue
                elif isinstance(row[row_variable_name], str) and row_variable_name.startswith('religion'):
                    v = row[row_variable_name].lower()
                    val = (v,v)
                else:
                    v = value_mapping.get(row[row_variable_name], -1)
                    if (v is None) or pd.isna(v):
                        continue
                    elif v == -1:
                        debug_row = pd.DataFrame({"polity": polity_id, "variable": variable_name, "label": 'template', "issue": f"value {row[row_variable_name]} is not in mapping"}, index = [0])
                        self.debug = pd.concat([self.debug, debug_row])
                        continue
                    val = (v, v)

                value.append(val)
                year = row.year_from if Template.is_none_value(row.year_to) else row.year_to # at least one of year_from and year_to are valid
                
                if year < polity_years[0]:
                    print("Error: The year is outside the polity's start and end year")
                    debug_row = pd.DataFrame({"polity": polity_id, "variable": variable_name, "label": 'template', "issue": f"year {year} outside polity years"}, index = [0])
                    self.debug = pd.concat([self.debug, debug_row])
                    continue
                elif year > polity_years[1]:
                    print("Error: The year is outside the polity's start and end year")
                    debug_row = pd.DataFrame({"polity": polity_id, "variable": variable_name, "label": 'template', "issue": f"year {year} outside polity years"}, index = [0])
                    self.debug = pd.concat([self.debug, debug_row])
                    continue
                    
                t.append(year)

            # check if both years are defined
            # note: these checks are redundant since we know that both cannot be None (first if),
            # and only one of them cannot be None (elif) and that the two values cannot be equal
            # (elif), so they must be two valud and distinct values
            elif (row.year_from != row.year_to) and not Template.is_none_value(row.year_from) and (
                    not Template.is_none_value(row.year_to)):
                if range_var:
                    val_from = row[row_variable_name + "_from"]
                    val_to = row[row_variable_name + "_to"]
                    # if no range variables are defined skip the row
                    val = Template.get_values(val_from, val_to)
                    if val is None:
                        continue
                elif isinstance(row[row_variable_name], str) and row_variable_name.startswith('religion'):
                    v = row[row_variable_name].lower()
                    val = (v,v)
                else:
                    
                    # check if row[variable_name] is a finite number
                    if isinstance(row[row_variable_name], (int, float)) and pd.notna(row[row_variable_name]):
                        v = row[row_variable_name]
                    else:
                        v = value_mapping.get(row[row_variable_name], -1)
                        if (v is None) or pd.isna(v):
                            continue
                        elif v == -1:
                            debug_row = pd.DataFrame({"polity": polity_id, "variable": variable_name, "label": 'template', "issue": f"value {row[row_variable_name]} is not in mapping"}, index = [0])
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

                if t_from < polity_years[0]:
                    print("Error: The year is outside the polity's start and end year")
                    debug_row = pd.DataFrame({"polity": polity_id, "variable": variable_name, "label": "template", "issue": f"year {t_from} outside polity years"}, index = [0])
                    self.debug = pd.concat([self.debug, debug_row])
                    continue
                elif t_to > polity_years[1]:
                    print("Error: The year is outside the polity's start and end year")
                    debug_row = pd.DataFrame({"polity": polity_id, "variable": variable_name, "label": "template", "issue": f"{t_to} outside polity years"}, index = [0])
                    self.debug = pd.concat([self.debug, debug_row])
                    continue
                    
                t.append(t_from)
                t.append(t_to)
            else:
                # should not be reached (see above)
                raise BaseException(f"Error processing {col_name} variable for polity {polity_id}")
                
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
                debug_row = pd.DataFrame({"polity": polity_id, "variable": variable_name, "label": "template", "issue": "mismatched lengths"}, index = [0])
                self.debug = pd.concat([self.debug, debug_row])
                return 0
    
        if len(variable_dict['t'][0]) == 0:
            # this is normal if the value is coded as "unknown" for this polity
            return 0
        
        self.template.loc[self.template.PolityName == polity_id, col_name] = [variable_dict]
        return len(variable_dict['value'])

    @staticmethod
    def check_polity_dates(pol_df : pd.DataFrame, polity_years : list):
        """
        Helper function to check that data for a polity has proper time ranges. This includes:
            - check that either all or no values have an associated date range
            - filter out values that are outside the polity date range; if a coded value has a date
                range partially outside, it is cut to the overlap; if a coded value has a date range
                that completely falls outside the polity date range, it is removed
            - check that date ranges do not overlap
        
        Parameters:
            pol_df (pd.DataFrame): DataFrame containing data for one polity and one variable
            polity_year (list of int): date range corresponding to this polity
        
        Returns:
            A tuple with:
                the first element is pol_df after filtering or None if a fatal error was detected
                a list of strings with the errors detected
        """
        
        # check that either all or no values have dates
        have_any_time = pol_df.year_from.notna().any() or pol_df.year_to.notna().any()
        if have_any_time and not np.all(pol_df.year_from.notna() | pol_df.year_to.notna()):
            return (None, ["Values both with and without time ranges"])
        
        range_error = False
        
        if have_any_time:
            # fill in "missing" values -- if we are here, either year_from or year_to are given
            ix1 = np.where(pol_df.year_from.isna())
            pol_df.loc[ix1[0], 'year_from'] = pol_df.loc[ix1[0], 'year_to']
            ix1 = np.where(pol_df.year_to.isna())
            pol_df.loc[ix1[0], 'year_to'] = pol_df.loc[ix1[0], 'year_from']
            
            # leave out data that is outside the polity time range
            if (np.any(pol_df.year_from < polity_years[0]) or np.any(pol_df.year_to < polity_years[0]) or
                      np.any(pol_df.year_from > polity_years[1]) or np.any(pol_df.year_to > polity_years[1])):
                range_error = True
            pol_df = pol_df[(pol_df.year_to >= polity_years[0]) & (pol_df.year_from <= polity_years[1])]
            pol_df.reset_index(inplace = True, drop = True)

        elif not have_any_time and len(pol_df) == 1:
            # if no time ranges are given and there is only one value, assign the polity date range
            pol_df['year_from'] = polity_years[0]
            pol_df['year_to'] = polity_years[1]
            
        elif not have_any_time and len(pol_df) > 1:
            if pol_df.variable_name[0].startswith('polity_religion'):
                # special case: polity religion data without time ranges can be multiple entries
                # (this is not ideal, but it is how the data is currently coded)
                pol_df['year_from'] = polity_years[0]
                pol_df['year_to'] = polity_years[1]
                
            elif (pol_df.is_uncertain.any() or pol_df.is_disputed.any()):
                pol_df['year_from'] = polity_years[0]
                pol_df['year_to'] = polity_years[1]
            
        if pol_df.shape[0] > 0:
            # potentially cut remaining ranges to be inside the polity date range
            pol_df.year_from = np.maximum(pol_df.year_from, polity_years[0])
            pol_df.year_to = np.minimum(pol_df.year_to, polity_years[1])
        
            # check for overlaps
            group1 = pol_df.groupby(['year_from', 'year_to'], dropna = False)
            
            tmp_dates = sorted(group1.groups.keys())
            for i in range(len(tmp_dates) - 1):
                if tmp_dates[i+1][0] < tmp_dates[i][1]:
                    return (None, ["Overlapping time ranges"])
                    
        errors1 = list()
        if range_error:
            errors1.append("Year range outside of polity time range")
        
        return (pol_df, errors1)
        
    
    @staticmethod
    def check_polity_disp_unc(pol_df : pd.DataFrame):
        """
        Helper function to check whether disputed and uncertain values are tagged correctly for one
        polity / variable combination. This includes checking the following after grouping by date ranges:
            - Each group should have either all or no values marked as disputed
            - Each group should have either all or no values marked as uncertain
            - If a group has multiple values, these should be either all marked as disputed or as uncertain
            - If a value is marked as disputed, it should be part of a group with multiple values
        All of the above are considered fatal errors. Furthermore, it is also checked that uncertain
        values are part of a group with multiple values, but this does not result in errors.
        
        Parameters:
        pol_df (pd.DataFrame): DataFrame containing data for one polity and one variable

        Returns:
        A tuple of a boolean indicator whether a fatal error was found and a string detailing any error found.

        """
        
        # group by dates
        group1 = pol_df.groupby(['year_from', 'year_to'], dropna = False)
        
        # check whether disputes and uncertain values are marked consistently
        if np.any(group1.is_disputed.any() != group1.is_disputed.all()):
            return (False, "Inconsistent marking of disputed data")
        if np.any(group1.is_uncertain.any() != group1.is_uncertain.all()):
            return (False, "Inconsistent marking of uncertain data")
        
        # check that groups with multiple values are marked as either disputed or uncertain
        have_multiple = (group1.is_disputed.count() > 1)
        if np.any(have_multiple & ~(group1.is_disputed.any() | group1.is_uncertain.any())):
            return (False, "Multiple values without variable being marked disputed or uncertain")
        # check that disputed groups actually have multiple values
        if np.any(group1.is_disputed.any() & ~have_multiple):
            return (False, "Data marked as disputed, but only one value")
        # check that uncertain groups actually have multiple values (this is not a fatal error)
        if np.any(group1.is_uncertain.any() & ~have_multiple):
            # This is not a fatal error, we will just ignore this
            return (True, "Data marked as uncertain, but only one value")
            
        # all is OK
        return (True, None)
    

    def add_polity2(self, polity_id : str, pol_df, range_var, variable_name, col_name):
        """
        Adds polity data to the template.
        This function processes a DataFrame containing polity data, checks for duplicates, handles disputed and uncertain entries, 
        and appends the processed data to the template. It ensures that the data is in chronological order and handles various 
        cases where year data might be missing or overlapping.
        Parameters:
        pol_df (pd.DataFrame): DataFrame containing polity data with columns such as 'polity_id', 'year_from', 'year_to', 
                               'is_disputed', 'is_uncertain', and the variable of interest.
        range_var (bool): Indicates whether the variable of interest is a range variable.
        variable_name (str): The name of the variable to be processed.
        col_name (str): The name of the column in the template where the processed data will be stored.
        Returns:
        None
        Raises:
        SystemExit: If there is an unexpected overlap in year data or other critical issues.
        
        Simplified version with the following changes:
            - throw exception on fatal errors instead of exiting
            - "missing" values (when parsing returns None) are silently ignored
        """
    
        # 1. get rid of duplicates (this happens in some cases)
        len1 = pol_df.shape[0]
        pol_df = pol_df.drop_duplicates()
        if len1 != pol_df.shape[0]:
            print(f"Duplicate values for {polity_id} -- {col_name}")
            debug_row = pd.DataFrame({"polity": polity_id, "variable": col_name, "label": "template", "issue": "Duplicate values"}, index = [0])
            self.debug = pd.concat([self.debug, debug_row])
        
        # ensure we have a sane index (this is needed since pol_df is typically a subset)
        pol_df.reset_index(drop = True, inplace = True)

        # 2. Check whether date ranges are consistent and cut them to the polity date ranges
        polity_years = [self.template.loc[self.template.PolityName == polity_id, 'StartYear'].values[0],
                        self.template.loc[self.template.PolityName == polity_id, 'EndYear'].values[0]]
    
        pol_df, tmp_errors = Template.check_polity_dates(pol_df, polity_years)
        
        if len(tmp_errors) > 0:
            for a in tmp_errors:
                print(f"{a} (polity: {polity_id}, variable: {col_name}")
            debug_df = pd.DataFrame({"polity": polity_id, "variable": col_name, "label": "template", "issue": tmp_errors}, index = [0])
            self.debug = pd.concat([self.debug, debug_df])
        
        if pol_df is None or pol_df.shape[0] == 0:
            return 0
        
        # 3. Check whether uncertain and disputed values are marked consistently
        check_ok, tmp_error_str = Template.check_polity_disp_unc(pol_df)
        if tmp_error_str is not None:
            if pol_df['variable_name'].values[0].startswith('polity_religion') and  ("Multiple values without variable being marked disputed or uncertain" in tmp_error_str):
                # religion data has often several entries without time ranges, we collect them in a list
                religions = list(pol_df[variable_name].unique())
                pol_df = pol_df.drop_duplicates(subset = [variable_name])
                pol_df = pol_df.reset_index(drop = True)
                pol_df[variable_name] = [religions for _ in range(pol_df.shape[0])]
                check_ok = True
            else:
                print(f"{tmp_error_str} (polity: {polity_id}, variable: {col_name}")
                debug_row = pd.DataFrame({"polity": polity_id, "variable": col_name, "label": "template", "issue": tmp_error_str}, index = [0])
                self.debug = pd.concat([self.debug, debug_row])
        if not check_ok:
            return 0
        
        # 4. Preprocess values
        row_variable_name = variable_name
        if 'polity_religion' in variable_name:
            row_variable_name = variable_name.replace('polity_', '')
        if (row_variable_name not in pol_df.columns) and (row_variable_name + '_from' not in pol_df.columns):
            row_variable_name = 'coded_value'
        
        if range_var:
            # numeric values should already be parsed, we just need to verify that they are not NaN
            pol_df['val'] = pol_df.apply(lambda x: Template.get_values(
                x[row_variable_name + "_from"], x[row_variable_name + "_to"]),
                axis = 1)
        elif pol_df['variable_name'].values[0].startswith('polity_religion'):

            # religion data has often several entries without time ranges, we collect them in a list    
            def process_value(row):
                v = row[row_variable_name]
                return (v,v) if (isinstance(v, list) or (isinstance(v, str) and v != '')) else None 
            pol_df['val'] = pol_df.apply(process_value, axis = 1)
        else:
            # for binary values, we look up the corresponding numeric value
            def process_value(row):
                v = value_mapping.get(row[row_variable_name], None)
                return None if Template.is_none_value(v) else (v, v)
            pol_df['val'] = pol_df.apply(process_value, axis = 1)
    
        # 5. Filter out missing values (explicit NaN for numeric and binary values coded as unknown)
        pol_df = pol_df[pol_df.val.notna()]
        if pol_df.shape[0] == 0:
            # no valid data values -- have to return here, as the below code does not work for empty dataframes
            # (this is not an error, as data can be explicitly coded as unknown / missing)
            return 0
    
        # 6. Main processing
        # 6.1. group by time, calculate all combinations for each group    
        tmp1 = pol_df.groupby(['year_from', 'year_to'], dropna = False).apply(lambda x: list(x.val))
        times = list(tmp1.index)
        tmp2 = list(tmp1)
        values = list(np.array(x, dtype = object) for x in itertools.product(*tmp2))
        
        # 6.2. process times -- take care of overlaps
        # (cases when the last year of a range is the same as the beginning of the next range;
        # larger overlaps were detected above and resulted in an error)
        for i in range(len(times) - 1):
            if times[i][1] == times[i+1][0]:
                if times[i][0] != times[i][1]:
                    times[i] = (times[i][0], times[i][1] - 1)
                else:
                    times[i+1] = (times[i+1][0] + 1, times[i+1][1])
        
        # 6.3. process times -- replace mising cases with the polity duration
        # and duplicate values that correspond to time ranges
        times2 = list()
        ixs = list()
        for i in range(len(times)):
            x = times[i]
            if Template.is_none_value(x[0]):
                x = tuple(polity_years)
            if Template.is_none_value(x[1]) or x[0] == x[1]:
                ixs.append(i)
                times2.append(x[0])
            else:
                ixs.append(i)
                ixs.append(i)
                times2.append(x[0])
                times2.append(x[1])
        for i in range(len(values)):
            values[i] = values[i][ixs]
        times = list(times2 for _ in range(len(values)))
        
        # 6.4. construct the final result
        variable_dict = {"t": list(times), "value": list(values), "polity_years": list(polity_years)}
    
        # 6.5. final sanity check
        if len(variable_dict['t'][0]) == 0:
            # this should not happen, since we already threw out cases with no data above
            raise BaseException('Unexpected empty result!')
        
        # 6.6. save the results
        self.template.loc[self.template.PolityName == polity_id, col_name] = [variable_dict]
        return len(variable_dict['value'])


    def perform_tests(self, df, variable_name, range_var, col_name):
        if self.template[col_name].apply(lambda x: self.check_for_nans(x)).any():
            raise BaseException(f"Error: NaNs found in the data for variable {col_name}")
        if range_var:
            var_name = variable_name + "_from"
        else:
            var_name = variable_name
        if (self.template['PolityName'].apply(lambda x: self.check_nan_polities(x, df, var_name)) > self.template[col_name].isna()).all():
            raise BaseException(f"Nans in template that are not in the template for variable {col_name}")
        elif (self.template['PolityName'].apply(lambda x: self.check_nan_polities(x, df, var_name)) < self.template[col_name].isna()).all():
            raise BaseException(f"Extra entries in the template for variable {col_name}")

        return "Passed tests"

    
    def read_polaris(self, filename : str):
        """
        Read data from Polaris dataset in xlsx format. Data is stored in self.full_dataset
        without further checks.

        Parameters
        ----------
        filename : str
            Path to xlsx file to read. It is expected to be in the format of the official
            Polaris release.

        Returns
        -------
        None. Will throw exception if the input file is not in the expected format.

        """
        # 1. read the list of polities
        self.polity_df = pd.read_excel(filename, 'Polities')
        self.polity_df.rename({
            'polity_number': 'id',
            'polity_id': 'name',
            'seshat_region': 'home_nga_name'
        }, inplace = True, axis = 1)
        self.polity_df.drop('long_name', inplace = True, axis = 1)
        
        # 2. read the main data
        tmp1 = list()
        sheets_needed = ["Social complexity", "Warfare", "Luxury goods", "Religion"]
        for a in sheets_needed:
            tmp1.append(pd.read_excel(filename, a))
        self.full_dataset = pd.concat(tmp1)
        ##!! TODO: do some basic checks, e.g. column names, data types, etc.
    
    def read_equinox(self, equinox_spreadsheet : str, variable_mapping : str, polity_mapping : str):
        """
        Read data from the Equinox dataset in xlsx format. Data is stored in self.full_dataset
        after some basic preprocessing, but without further checks. To work correctly, you
        need to provide additional mappings between the old and new variable names and polity IDs.

        Parameters
        ----------
        equinox_spreadsheet : str
            Path to the Equinox data release spreadsheet in the xlsx format of the official
            data release. Currently available at:
                https://seshat-db.com/download_oldcsv/Equinox2020.05.2023.xlsx/
        variable_mapping : str
            Path to a CSV file that maps variable names in Equinox to the ones used in Polaris.

        Returns
        -------
        None.

        """
        
        # 0. load data and mapping files
        equinox = pd.read_excel(equinox_spreadsheet)
        # 47477 rows
        variable_mapping = pd.read_csv(variable_mapping)
        
        
        # 1. Recreate pol_df based on the polities in Equinox
        
        # 1.1. extract coded durations + NGAs
        polity_duration = equinox[(equinox.Section == 'General variables') & (equinox.Variable == 'Duration')]
        polity_duration = polity_duration[['NGA', 'Polity', 'Value.From']]
        
        # 1.2. use the expected column names
        polity_duration.rename({'Polity': 'name', 'Value.From': 'Duration', 'NGA': 'home_nga_name'},
                               inplace = True, axis = 1)
        
        # 1.3. helpers to parse dates
        def parse_one_date(x : str, bce_hint):
            if x.endswith('BCE') or x.endswith('bce'):
                return -1 * float(x[:-3])
            if x.endswith('CE') or x.endswith('ce'):
                return float(x[:-2])
            if not isinstance(bce_hint, bool):
                return None
            return float(x) * (-1 if bce_hint else 1)
            
        def parse_duration(x : str):
            if x is None or pd.isna(x):
                return (None, None)
            y = x.split('-')
            if len(y) != 2:
                return (None, None)
            y[0] = y[0].strip()
            y[1] = y[1].strip()
            end = parse_one_date(y[1], None)
            if end is None:
                return (None, None)
            start = parse_one_date(y[0], end <= 0)
            return (start, end)
         
        # 1.4. parse all dates
        tmp1 = polity_duration.Duration.apply(parse_duration)
        polity_duration['start_year'] = list(x[0] for x in tmp1)
        polity_duration['end_year'] = list(x[1] for x in tmp1)
         
        # 1.5. manually correct some errors and missing values
        polity_duration.loc[polity_duration.name == 'GrCrEmr', 'start_year'] = 824
        polity_duration.loc[polity_duration.name == 'GrCrEmr', 'end_year'] = 961
        polity_duration.loc[polity_duration.name == 'InGupta', 'start_year'] = 320
        polity_duration.loc[polity_duration.name == 'InGupta', 'end_year'] = 515
        polity_duration.loc[polity_duration.name == 'JpJomo4', 'start_year'] = -3500
        polity_duration.loc[polity_duration.name == 'JpJomo4', 'end_year'] = -2500
        polity_duration.loc[polity_duration.name == 'PeCuzLF', 'start_year'] = -500
        polity_duration.loc[polity_duration.name == 'PeCuzLF', 'end_year'] = 200
        polity_duration.loc[polity_duration.name == 'PkUrbn2', 'start_year'] = -2100
        polity_duration.loc[polity_duration.name == 'PkUrbn2', 'end_year'] = -1800
        polity_duration.loc[polity_duration.name == 'JpJomo6', 'start_year'] = -1200
        polity_duration.loc[polity_duration.name == 'JpJomo6', 'end_year'] = -300
         
        # 1.6. sanity check (we need this to be available for all polities)
        if polity_duration.start_year.isnull().any() or polity_duration.end_year.isnull().any():
            raise BaseException('Missing polity durations!')
        
        # 1.7. manually add two more missing values and save the result
        self.polity_df = pd.concat([polity_duration[['home_nga_name', 'name', 'start_year','end_year']],
                         pd.DataFrame({
                            'home_nga_name': None,
                            'name': ['InAyodE', 'JpJomo6'],
                            'start_year': [-64, -1200],
                            'end_year': [34, -300]
                         })], ignore_index=True)
         
        # 1.8. convert name and id to new format based on the provided mapping
        polity_table = pd.read_csv(polity_mapping)
        self.polity_df['name'] = self.polity_df['name'].apply(lambda x: polity_table[polity_table.polity_id_old == x].polity_id.values[0] if not polity_table[polity_table.polity_id_old == x].empty else x)
        self.polity_df['id'] = self.polity_df['name'].apply(lambda x: polity_table[polity_table.polity_id == x].polity_number.values[0] if not polity_table[polity_table.polity_id == x].empty else np.nan)
        

        # 2. Keep only the interesting variables
        
        # 2.1. filter based on the variable mapping
        # note: in theory we could also match Section and Subsection, but these do not
        # matter for the variables we are interested in
        equinox = equinox.merge(variable_mapping, on = 'Variable')
        # 34392 rows remain
        
        # 2.2. get rid of the useless colums now
        equinox = equinox[['Polity', 'variable_name', 'Value.From',
               'Value.To', 'Date.From', 'Date.To', 'Value.Note', 'Type']]
        
        # 2.3. Add some columns in the correct format
        equinox['is_disputed'] = (equinox['Value.Note'] == 'disputed')
        equinox['is_uncertain'] = (equinox['Value.Note'] == 'uncertain')
        equinox['range_var'] = (equinox['Type'] == 'numeric')
        
        
        # 3. Correct some errors manually
        
        # 3.0 rename some columns to be more convenient to use in this step
        equinox.rename({'Value.From': 'value_from', 'Value.To': 'value_to', 'Polity': 'polity_id'}, inplace = True, axis = 1)
        
        # 3.1. long wall (+ other numeric variables) code unknown as NaN, absent as 0
        equinox.loc[(equinox.variable_name == 'long_wall') & (
            equinox.value_from.isin(['unknown', 'suspected unknown'])), 'value_from'] = np.nan
        equinox.loc[(equinox.variable_name == 'long_wall') & (
            equinox.value_from.isin(['absent', 'inferred absent'])), 'value_from'] = 0
        equinox.loc[(equinox.Type == 'numeric') & (
            equinox.value_from.isin(['unknown', 'suspected unknown'])), 'value_from'] = np.nan
        
        # 3.2. wrong formatting for Eastern Han Empire population
        ix1 = np.where(equinox.value_from == '480000-53869588')
        equinox.loc[ix1[0], 'value_from'] = 48000000
        equinox.loc[ix1[0], 'value_to'] = 53869588
        ix1 = np.where(equinox.value_from == '49150220-49730550')
        equinox.loc[ix1[0], 'value_from'] = 49150220
        equinox.loc[ix1[0], 'value_to'] = 49730550
        
        # 3.3. wrong formatting for MnRourn Specialized government buildings
        ix1 = np.where(equinox.value_from == 'inferred present 501-551 CE')
        equinox.loc[ix1[0], 'value_from'] = 'inferred present'
        equinox.loc[ix1[0], 'Date.From'] = '501CE'
        equinox.loc[ix1[0], 'Date.To'] = '551CE'
        
        # 6. Process dates
        def process_dates(dt):
            is_bce = dt.apply(lambda x: (pd.notnull(x) and x.endswith('BCE')))
            ix_bce = np.where(is_bce)
            ix_ce = np.where(~is_bce & dt.apply(lambda x: (pd.notnull(x) and x.endswith('CE'))))
            
            res = np.full_like(dt, np.nan, dtype=float)
            res[ix_bce[0]] = dt[ix_bce[0]].apply(lambda x : -1 * float(x[:-3]))
            res[ix_ce[0]] = dt[ix_ce[0]].apply(lambda x : float(x[:-2]))
            return res
        
        equinox['year_from'] = process_dates(equinox['Date.From'])
        equinox['year_to'] = process_dates(equinox['Date.To'])

        equinox['polity_id'] = equinox['polity_id'].apply(lambda x: polity_table[polity_table.polity_id_old == x].polity_id.values[0] if not polity_table[polity_table.polity_id_old == x].empty else x)

        # 6. store the final result, keeping only the useful columns
        self.full_dataset = equinox[['polity_id', 'variable_name', 'value_from', 'value_to',
                           'year_from', 'year_to', 'is_disputed', 'is_uncertain', 'range_var']]
            

    
    # ---------------------- SAMPLING FUNCTIONS ---------------------- #

    def sample_dict(self, variable_dict, t, error, interpolation = 'zero', sampling = 'uniform'):
        """
        Samples values from a given dictionary of timelines based on the provided time(s) and error margin.
        Parameters:
        variable_dict (dict): A dictionary containing 't', 'value', and 'polity_years' keys.
                              't' is a list of time points, 'value' is a list of value ranges corresponding to the time points,
                              and 'polity_years' is a list of years defining the polity period.
        t (int, float, list, np.ndarray): The time or list of times at which to sample the values.
        error (int, float): The error margin to extend the polity years.
        Returns:
        list or float: The sampled value(s) at the given time(s). If the time is out of bounds, returns "Out of bounds".
                       If the input time is not a number, returns "Error: The year is not a number".
                       If the input dictionary is None or invalid, returns None.
        """

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
        polity_years = [min(polity_years) - error, max(polity_years) + error]

        if polity_years[0] not in times:
            times = [polity_years[0]] + times
            values = [values[0]] + values

        if polity_years[1] not in times:
            times = times + [polity_years[1]]
            values = values + [values[-1]]

        times = np.array(times)
        random_number = random.random()
        if interpolation == 'zero':
            pass
        elif (interpolation == 'linear') or (interpolation == 'smooth'):
            # create a smoothing effect on the data with a smoothing window of 50 years
            import scipy.interpolate as spi
            x = np.array(times)
            if sampling == 'uniform':
                y = np.array([v[0] + random_number*(v[1]-v[0]) for v in values])
            elif sampling == 'mean':
                y = np.array([np.mean([v[0],v[1]]) for v in values])
            smooth_window = 50
            if interpolation == 'linear':
                smoothing = np.ones(smooth_window)
                smoothing = smoothing / smoothing.sum()
            elif interpolation == 'smooth':
                smoothing = np.exp(-np.linspace(-3, 3, smooth_window)**2)
                smoothing /= smoothing.sum()
            x_new = np.arange(min(x), max(x), smooth_window // 5)
            y_new = spi.interp1d(x, y)(x_new)
            y_new = np.pad(y_new, smooth_window, mode='edge')
            y_new = np.convolve(y_new, smoothing, mode='same')[smooth_window:-smooth_window]
            
        if isinstance(t, (list, np.ndarray)):
            vals = [None] * len(t)
            for i, time in enumerate(t):
                if time < polity_years[0] or time > polity_years[1]:
                    print(f"Error: The year {time} is outside the polity years {polity_years}")
                    vals[i] = "Out of bounds"
                    continue
                if interpolation == 'zero':
                    time_selection = times[times<=time]
                    ind = np.argmin(np.abs(np.array(time_selection) - time))
                    if ind >= len(values):
                        ind = len(values) - 1
                    if isinstance(values[ind][0], str):
                        vals[i] = values[ind][0]
                        continue
                    if isinstance(values[0][0], str) or isinstance(values[0][0], np.ndarray):
                        val = random.choice(values[ind][0])         
                    elif sampling == 'uniform':
                        val = values[ind][0] + random_number * (values[ind][1] - values[ind][0])
                    elif sampling == 'mean':
                        val = np.mean(values[ind])
                    vals[i] = val
                elif (interpolation == 'linear') or (interpolation == 'smooth'):
                    if isinstance(values[ind][0], str):
                        print(f"Error: String column must use 'zero' interpolation")
                        vals[i] = np.nan
                        continue
                    vals[i] = y_new[np.argmin(np.abs(x_new - time))]
            return vals
        elif isinstance(t, (int, float, np.int64, np.int32, np.float64, np.float32)):
            if t < polity_years[0] or t > polity_years[1]:
                print(f"Error: The year {t} is outside the polity years {polity_years}")
                return "Out of bounds"
            # find the closest year to t
            if interpolation == 'zero':
                times = times[times<=t]
                ind = np.argmin(np.abs(np.array(times) - t))
                # sample the values
                val = values[ind][0] + random.random() * (values[ind][1] - values[ind][0])
            elif (interpolation == 'linear') or (interpolation == 'smooth'):
                val = y_new[np.argmin(np.abs(x_new - t))]
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
    template = Template(categories = ['rt'], keep_raw_data=True)
    template.download_all_categories(add_to_template=False)
    template.template_from_dataset(use_new_method = True)
    template.save_dataset("template.csv")
