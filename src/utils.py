import pandas as pd
import requests
from pandas import json_normalize
import json
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def download_data(url,size = 1000):
    
    if pd.isna(size):
        url = url
    elif isinstance(size, int):
        url = url+"?page_size="+str(size)
    df = pd.DataFrame()
    
    while True:
        try:
            try:
                response = requests.get(url, timeout=5)
            except requests.exceptions.Timeout:
                # print("Timeout occurred")
                continue
            data = response.json()
            df_temp = pd.DataFrame(data)

            for polity_dict in df_temp.results:

                # unpack polity_dict
                flattened_dict = json_normalize(polity_dict, sep='_')
                df = pd.concat([df, flattened_dict], axis=0)

            url = df_temp.next.values[0]

        except:
            if len(df) > 0:
                print(f"Downloaded {len(df)} rows")
            return df

def download_data_json(filepath):

    data = json.load(open(filepath))
    df = pd.DataFrame()
    for row in data:
        # unpack polity_dict
        flattened_dict = json_normalize(row, sep='_')
        df = pd.concat([df, flattened_dict], axis=0)
    return df

def fetch_urls(category):
    url = "https://seshat-db.com/api/"
    response = requests.get(url)
    data = response.json()
    variable_urs = dict()
    import src.mappings as mappings
    if category == 'wf':
        mapping = mappings.miltech_mapping
    elif category == 'sc':
        mapping = mappings.social_complexity_mapping
    elif category == 'id':
        mapping = mappings.ideology_mapping
    
    used_keys = []
    for key in mapping.keys():
        used_keys.append(mapping[key].keys())
    used_keys = [category+'/'+key for sublist in used_keys for key in sublist]
    for key in data.keys():
        if key.split('/')[0] == category:
            if key in used_keys:
                variable_urs[key] = data[key]
    return variable_urs


def weighted_mean(row, mappings, category = "Metal", imputation = 'remove', min_vals = 0.):
    weights = 0
    result = 0

    keys = mappings[category].keys()
    entries = [mappings[category][key] for key in mappings[category].keys()]

    for key in keys:
        if key not in row:
            if key + "_from" in row:
                row[key] = (row[key + "_from"] + row[key + "_to"]) / 2
            else:
                print(key, "not in row")
                continue
    
    values = row[keys]
    if values.isna().sum() >= len(values)*(1-min_vals):
        return np.nan
    
    if imputation == 'remove':
        entries = [entry for entry, value in zip(entries, values) if not np.isnan(value)]
        values = values.dropna()
    elif imputation == 'mean':
        values = values.infer_objects()
        values = values.fillna(values.mean())
        entries = [entry for entry, value in zip(entries, values) if not np.isnan(value)]
    elif imputation == 'zero':
        values = values.infer_objects()
        values = values.fillna(0)
        entries = [entry for entry, value in zip(entries, values) if not np.isnan(value)]
    elif imputation == 'half':
        values = values.infer_objects()
        values = values.fillna(0.5)
    
        entries = [entry for entry, value in zip(entries, values) if not np.isnan(value)]
    
    return np.average(values, weights = entries)


def get_max(row, mappings, category):

    result = -1
    for key, entry in mappings[category].items():
        if key not in row:
            if key + "_from" in row:
                if np.isnan(row[key + "_from"]):
                    continue
                value = (row[key + "_from"] + row[key + "_to"]) / 2
            else:
                print(key, "not in row")
                continue
        else:
            if np.isnan(row[key]):
                continue
            value = row[key]
        if entry * value > result:
            result = entry * value

    if result == -1:
        result = np.nan
    return result

def convert_to_year(year_str):
    """Convert string of the type '1000CE' or '1000BCE' to integer, any non string is returned as is"""
    # check if str
    if type(year_str) != str:
        return year_str
    if 'BCE' in year_str:
        return -int(year_str.split('B')[0])
    elif 'CE' in year_str:
        return int(year_str.split('C')[0])
    
def is_same(list1,list2):
    if len(list1) != len(list2):
        return False
    for i in range(len(list1)):
        if list1[i] not in list2:
            return False
    return True

def convert_to_year(year_str):
    """Convert string of the type '1000CE' or '1000BCE' to integer, any non string is returned as is"""
    # check if str
    if type(year_str) != str:
        return year_str
    if 'BCE' in year_str:
        return -int(year_str.split('B')[0])
    elif 'CE' in year_str:
        return int(year_str.split('C')[0])


def compare(old, new, common_columns):
    # check if two have same entries
    for col in common_columns:
        if col == 'polityname' or col == 'year':
            continue
            # remove nan values
        old_col = old[col].dropna()
        new_col = new[col].dropna()
        if len(old_col) == 0 and len(new_col) == 0:
            print("no values for", col)
            continue
        if len(old_col) != len(new_col):
            print("different lengths for", col)
            print("old data")
            print(old_col)
            print("new data")
            print(new_col)
            print("\n\n")
            continue
        if not (old_col.values == new_col.values).all():
            print("same values for", col)

            continue