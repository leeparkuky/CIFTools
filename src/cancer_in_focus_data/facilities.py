from typing import Union, List, Dict
import pandas as pd
import numpy as np
import re
import urllib.request
from zipfile import ZipFile
from io import BytesIO
import requests
from csv import DictReader
from joblib import Parallel, delayed
from itertools import product
import os
from glob import glob
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import shutil

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # add src to path
from src.utils.facilities_utils import *
from src.utils.states import stateDf



def gen_facility_data(location: Union[List[str], str], taxonomy: List[str] = ['Gastroenterology','colon','obstetrics']):
    """
    Generates facility data by parallelizing multiple data retrieval functions.
    - Ensures multiprocessing is properly leveraged.
    """
    data_dict = {}

    # Run `nppes` sequentially (as it may require different processing)
    # data_dict['nppes'] = nppes(location)

    # List of functions and corresponding dataset names
    functions = [nppes, mammography, hpsa, fqhc, lung_cancer_screening, toxRel_data]
    dataset_names = ['nppes','mammography', 'hpsa', 'fqhc', 'lung_cancer_screening', 'tri_facility']

    # Execute functions in parallel using multiprocessing backend
    datasets = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(f)(location) for f in functions)

    # Store results in dictionary
    for name, df in zip(dataset_names, datasets):
        data_dict[name] = df

    return data_dict



def mammography(location: Union[str, List[str]]):
    """Fetches mammography facilities and filters by state or states."""
    df = fetch_mammography_data()

    # Normalize and filter by location
    location = [location.upper()] if isinstance(location, str) else [l.upper() for l in location]
    df = df[df["State"].isin(location)].reset_index(drop=True)

    # Extract facility name (handles cases where 'B.' might not exist)
    df["Name"] = df["Name"].str.extract(r'[bB]\.\s*(.*)', expand=False).fillna(df["Name"])

    # Construct Address
    df["Address"] = df[["Street", "City", "State", "Zip_code"]].astype(str).agg(", ".join, axis=1)

    # Format phone numbers
    df["Phone_number"] = df["Phone_number"].apply(format_phone_number)

    # Define relevant columns
    df = df[["Name", "Address", "State", "Phone_number"]]
    df.insert(0, "Type", "Mammography")
    df["Notes"] = ""

    return df



def hpsa(location: Union[str, List[str]]) -> pd.DataFrame:
    """Fetches, cleans, and filters HPSA facilities based on the provided location."""
    df = download_hpsa_data()  # Load raw data

    df = filter_hpsa_data(df, location)  # Apply filtering

    df = clean_hpsa_data(df)  # Clean and transform data

    # Ensure valid entries (avoid NaN in important fields)
    df = df.loc[df["longitude"].notnull() | df["Address"].notnull()].reset_index(drop=True)

    # Add missing columns
    df["Phone_number"] = None
    df["Notes"] = ""

    # Final column selection
    return df[["Type", "Name", "Address", "State", "Phone_number", "Notes", "latitude", "longitude"]]



def fqhc(location: Union[str, List[str]]) -> pd.DataFrame:
    """
    Main function to fetch, process, and return FQHC data for a given location.
    """
    fname = download_fqhc_data()
    df = process_fqhc_data(fname, location)

    # Add Type column
    df["Type"] = "FQHC"

    # Construct full address
    df["Address"] = df["Site_Address"] + ", " + df["Site_City"] + ", " + df["Site_State_Abbreviation"] + " " + df["Site_Postal_Code"]

    # Rename columns for clarity
    df.rename(columns={
        "Site_Name": "Name",
        "Site_Telephone_Number": "Phone_number",
        "Site_State_Abbreviation": "State",
        "Health_Center_Service_Delivery_Site_Location_Setting_Description": "Notes",
        "Geocoding_Artifact_Address_Primary_X_Coordinate": "longitude",
        "Geocoding_Artifact_Address_Primary_Y_Coordinate": "latitude"
    }, inplace=True)

    # Remove rows where Address is null
    df = df[df["Address"].notnull()].reset_index(drop=True)

    # Select final columns
    return df[["Type", "Name", "Address", "State", "Phone_number", "Notes", "latitude", "longitude"]]




def nppes(location: Union[str, List[str]], taxonomy: List[str] = None) -> pd.DataFrame:
    """
    Main entry function to fetch provider data asynchronously.
    """
    if taxonomy is None:
        taxonomy = list(taxonomy_names.keys())

    return asyncio.run(fetch_nppes_data(location, taxonomy))

    
def lung_cancer_screening(location: Union[str, List[str]]) -> pd.DataFrame:
    selenium_used = lung_cancer_screening_file_download()
    downloads = glob.glob('./ACRLCSDownload*.csv')

    df = process_lcs_data(downloads[0], location)

    if selenium_used:
        chrome_driver_path = setup_chrome_driver()
        remove_chromedriver(chrome_driver_path)

    for file in downloads:
        os.remove(file)

    return df.reset_index(drop=True)
    

def uscs_incidence():
    """
    Finds and loads the 'uscs_cancer_incidence_county_2017-2021.csv' file dynamically from the repo.
    """
    repo_root = find_repo_root()  # Get repo root dynamically
    search_pattern = os.path.join(repo_root, "**", "uscs_cancer_incidence_county_2017-2021.csv")
    
    # Search recursively for the file inside the repository
    file_path = glob.glob(search_pattern, recursive=True)

    if not file_path:
        raise FileNotFoundError("CSV file not found in the repository.")

    # Load the CSV file
    df = pd.read_csv(file_path[0])
    return df


def toxRel_data(location: Union[str, List[str]]) -> pd.DataFrame:
    """
    Main function to:
    - Download TRI dataset.
    - Extract relevant columns.
    - Filter and return cleaned data.
    """
    file_path = download_toxRel_data()
    csv_keys, temp_fields = extract_columns(file_path)
    df_filtered = filter_toxRel_data(file_path, location, csv_keys, temp_fields)

    # Cleanup
    os.remove(file_path)

    return df_filtered


if __name__ == "__main__":
    data = gen_facility_data(['CA', 'TX', 'GA', "CO", "FL"])
    print(data)