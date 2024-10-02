import pandas as pd
import requests
from pandas import json_normalize
import numpy as np

def download_data(url):
    df = pd.DataFrame()
    while True:
        try:
            try:
                response = requests.get(url, timeout=0.5)
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
            print(f"Downloaded {len(df)} rows")
            return df

def fetch_urls(category):
    url = "https://seshatdata.com/api/"
    response = requests.get(url)
    data = response.json()
    variable_urs = dict()
    import mappings
    if category == 'wf':
        mapping = mappings.miltech_mapping
    elif category == 'sc':
        mapping = mappings.social_complexity_mapping
    elif category == 'id':
        mapping = mappings.ideology_mapping
    
    used_keys = []
    for key in mapping.keys():
        used_keys.append(mapping[key].keys())
    used_keys = ['sc/'+key for sublist in used_keys for key in sublist]
    for key in data.keys():
        if key.split('/')[0] == category:
            if key in used_keys:
                variable_urs[key] = data[key]
    return variable_urs


def weighted_mean(row, mappings, variable = 'wf', category = "Metal", imputation = 'remove'):
    weights = 0
    result = 0

    keys = mappings[category].keys()
    keys = [str(variable) + '/' + key for key in keys]
    entries = [mappings[category][key] for key in mappings[category].keys()]

    for key in keys:
        if key not in row:
            if key + "_from" in row:
                row[key] = (row[key + "_from"] + row[key + "_to"]) / 2
            else:
                print(key, "not in row")
                continue
    
    values = row[keys]
    if values.isna().sum() == len(values):
        return np.nan
    
    if imputation == 'remove':
        entries = [entry for entry, value in zip(entries, values) if not np.isnan(value)]
        values = values.dropna()
    elif imputation == 'mean':
        values = values.infer_objects(copy=False)
        values = values.fillna(values.mean())
        entries = [entry for entry, value in zip(entries, values) if not np.isnan(value)]
    elif imputation == 'zero':
        values = values.infer_objects(copy=False)
        values = values.fillna(0)
        entries = [entry for entry, value in zip(entries, values) if not np.isnan(value)]
    elif imputation == 'half':
        values = values.infer_objects(copy=False)
        values = values.fillna(0.5)
    
        entries = [entry for entry, value in zip(entries, values) if not np.isnan(value)]
    
    return np.average(values, weights = entries)


def get_max(row, mappings, variable = 'wf', category = "Metal"):

    result = -1
    for key, entry in mappings[category].items():
        key = str(variable) + '/' + key
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