from typing import Union, List, Dict, Optional
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
from tqdm.auto import tqdm
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import shutil
from functools import partial

import sys
import os

from ..utils.facilities_utils import *
from ..utils.states import stateDf
from ..utils.ciftools_logger import logger


def gen_facility_data(
    location: Union[List[str], str],
    facility_type: Union[str, List[str]] = "all",
    taxonomy: List[str] = ["Gastroenterology", "colon", "obstetrics"],
):
    """
    Generates facility data by parallelizing multiple data retrieval functions.
    - Ensures multiprocessing is properly leveraged.
    """
    data_dict = {}

    if taxonomy is not None:
        nppes_func = partial(nppes, taxonomy=taxonomy)
    else:
        nppes_func = nppes

    # List of functions and corresponding dataset names
    functions = [
        nppes_func,
        mammography,
        hpsa,
        fqhc,
        lung_cancer_screening,
        toxRel_data,
        superfund,
    ]
    dataset_names = [
        "nppes",
        "mammography",
        "hpsa",
        "fqhc",
        "lung_cancer_screening",
        "tri_facility",
        "superfund",
    ]

    if facility_type != "all":
        if isinstance(facility_type, str):
            # Filter functions and dataset names based on facility_type
            functions = [
                f
                for f, name in zip(functions, dataset_names)
                if name.lower() == facility_type.lower()
            ]
            dataset_names = [
                name for name in dataset_names if name.lower() == facility_type.lower()
            ]
            if len(functions) == 0:
                raise ValueError(
                    f"Facility type '{facility_type}' not recognized. Available types: {dataset_names}"
                )
        elif isinstance(facility_type, list):
            # Filter functions and dataset names based on facility_type
            functions = [
                f
                for f, name in zip(functions, dataset_names)
                if name.lower() in [ftype.lower() for ftype in facility_type]
            ]
            dataset_names = [
                name
                for name in dataset_names
                if name.lower() in [ftype.lower() for ftype in facility_type]
            ]
            if len(functions) == 0:
                raise ValueError(
                    f"Facility types '{facility_type}' not recognized. Available types: {dataset_names}"
                )
        else:
            raise ValueError(
                f"Invalid facility_type: {facility_type}. Expected str or list of str."
            )
    # Check if location is a string and convert to list
    if isinstance(location, str):
        location = [location]
    elif not isinstance(location, list):
        raise ValueError(
            f"Invalid location type: {type(location)}. Expected str or list of str."
        )
    # Execute functions in parallel using multiprocessing backend
    datasets = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(f)(location) for f in functions
    )
    # Store results in dictionary
    for name, df in zip(dataset_names, datasets):
        data_dict[name] = df

    dfs_to_concat = [df for df in data_dict.values() if isinstance(df, pd.DataFrame)]
    if not dfs_to_concat:
        logger.info("No valid facility dataframes to concatenate in data_dict['all'], resulting in empty data.")
        data_dict["all"] = pd.DataFrame()
    else:
        data_dict["all"] = pd.concat(
            dfs_to_concat,
            ignore_index=True,
        )
    data_dict["all"] = data_dict["all"].reset_index(drop=True)
    return data_dict


def mammography(location: Union[str, List[str]]):
    """Fetches mammography facilities and filters by state or states."""
    logger.info("Fetching mammography data...")
    df = fetch_mammography_data()

    # Normalize and filter by location
    location = (
        [location.upper()]
        if isinstance(location, str)
        else [l.upper() for l in location]
    )
    df = df[df["State"].isin(location)].reset_index(drop=True)

    # Extract facility name (handles cases where 'B.' might not exist)
    df["Name"] = (
        df["Name"].str.extract(r"[bB]\.\s*(.*)", expand=False).fillna(df["Name"])
    )

    # Construct Address
    df["Address"] = (
        df[["Street", "City", "State", "Zip_code"]].astype(str).agg(", ".join, axis=1)
    )

    # Format phone numbers
    df["Phone_number"] = df["Phone_number"].apply(format_phone_number)

    # Define relevant columns
    df = df[["Name", "Address", "State", "Phone_number"]]
    df.insert(0, "Type", "Mammography")
    df["Notes"] = ""

    return df


def hpsa(location: Union[str, List[str]]) -> pd.DataFrame:
    """Fetches, cleans, and filters HPSA facilities based on the provided location."""
    logger.info("Fetching HPSA data...")
    df = download_hpsa_data()  # Load raw data

    df = filter_hpsa_data(df, location)  # Apply filtering

    df = clean_hpsa_data(df)  # Clean and transform data

    # Ensure valid entries (avoid NaN in important fields)
    df = df.loc[df["longitude"].notnull() | df["Address"].notnull()].reset_index(
        drop=True
    )

    # Add missing columns
    df["Phone_number"] = None
    df["Notes"] = ""

    # Final column selection
    return df[
        [
            "Type",
            "Name",
            "Address",
            "State",
            "Phone_number",
            "Notes",
            "latitude",
            "longitude",
        ]
    ]


def fqhc(location: Union[str, List[str]]) -> pd.DataFrame:
    """
    Main function to fetch, process, and return FQHC data for a given location.
    """
    logger.info("Fetching FQHC data...")
    fname = download_fqhc_data()
    df = process_fqhc_data(fname, location)

    # Add Type column
    df["Type"] = "FQHC"

    # Construct full address
    df["Address"] = (
        df["Site_Address"]
        + ", "
        + df["Site_City"]
        + ", "
        + df["Site_State_Abbreviation"]
        + " "
        + df["Site_Postal_Code"]
    )

    # Rename columns for clarity
    df.rename(
        columns={
            "Site_Name": "Name",
            "Site_Telephone_Number": "Phone_number",
            "Site_State_Abbreviation": "State",
            "Health_Center_Service_Delivery_Site_Location_Setting_Description": "Notes",
            "Geocoding_Artifact_Address_Primary_X_Coordinate": "longitude",
            "Geocoding_Artifact_Address_Primary_Y_Coordinate": "latitude",
        },
        inplace=True,
    )

    # Remove rows where Address is null
    df = df[df["Address"].notnull()].reset_index(drop=True)

    # Select final columns
    return df[
        [
            "Type",
            "Name",
            "Address",
            "State",
            "Phone_number",
            "Notes",
            "latitude",
            "longitude",
        ]
    ]


def nppes(location: Union[str, List[str]], taxonomy: List[str] = None) -> pd.DataFrame:
    """
    Main entry function to fetch provider data asynchronously.
    """
    logger.info("Fetching NPPES data...")
    if taxonomy is None:
        taxonomy = list(taxonomy_names.keys())

    return asyncio.run(fetch_nppes_data(location, taxonomy))


def lung_cancer_screening(location: Union[str, List[str]]) -> pd.DataFrame:
    logger.info("Fetching lung cancer screening data...")
    selenium_used = lung_cancer_screening_file_download()
    downloads = glob.glob("./ACRLCSDownload*.csv")

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
    logger.info("Fetching USCS cancer incidence data...")
    repo_root = find_repo_root()  # Get repo root dynamically
    search_pattern = os.path.join(
        repo_root, "**", "uscs_cancer_incidence_county_2017-2021.csv"
    )

    # Search recursively for the file inside the repository
    file_path = glob.glob(search_pattern, recursive=True)

    if not file_path:
        logger.error(
            "USCS cancer incidence file not found in the repository. Please check the file path."
        )
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
    logger.info("Fetching ToxRel data...")
    file_path = download_toxRel_data()
    csv_keys, temp_fields = extract_columns(file_path)
    df_filtered = filter_toxRel_data(file_path, location, csv_keys, temp_fields)

    # Cleanup
    os.remove(file_path)

    return df_filtered


def gen_single_superfund(location: str) -> Optional[pd.DataFrame]:
    """
    Fetches and processes superfund site data for a single state.

    Args:
        location: A 2-letter state abbreviation.

    Returns:
        A pandas DataFrame with superfund site data, or None if an error occurs.
    """
    if not (isinstance(location, str) and len(location) == 2 and location.isalpha()):
        logger.warning(
            f"Invalid location format provided to gen_single_superfund: {location}. Expected 2-letter state code."
        )
        return None
    location = location.upper()  # Ensure consistency
    url = f"https://data.epa.gov/efservice/ENVIROFACTS_SITE/FK_REF_STATE_CODE/{location}/JSON"

    try:
        # Fetch data
        try:
            sf = pd.read_json(url)
            if (
                sf.empty
                or "fips_code" not in sf.columns
                or "zip_code" not in sf.columns
                or "npl_status_name" not in sf.columns
            ):
                return None
        except Exception as e:
            logger.error(f"Error fetching data from {url}: {e}")
            return None

        # Basic check if data seems valid (e.g., has expected columns)
        if (
            "fips_code" not in sf.columns
            or "zip_code" not in sf.columns
            or "npl_status_name" not in sf.columns
        ):
            logger.warning(f"Unexpected data format received for {location}. Skipping.")
            return None

        # Process data
        sf["fips_code"] = sf["fips_code"].astype(str).str.slice(0, 5)
        sf["zip_code"] = sf["zip_code"].astype(str).str.slice(0, 5)

        # Filter by NPL status
        valid_statuses = [
            "Currently on the Final NPL",
            "Deleted from the Final NPL",
            "Site is Part of NPL Site",
        ]
        sf2 = sf.loc[
            sf["npl_status_name"].isin(valid_statuses)
        ]  # Filter without using .copy()

        if sf2.empty:
            logger.warning(
                f"No relevant superfund sites found for {location} with statuses: {valid_statuses}"
            )
            return None  # Return None if no sites match the criteria

        # Select and rename columns
        columns_to_keep = {
            "name": "Name",
            "street_addr_txt": "street_addr_txt",
            "city_name": "city_name",
            "fk_ref_state_code": "State",
            "zip_code": "zip_code",
            "fips_code": "FIPS",
            "npl_status_name": "Notes",
            "primary_latitude_decimal_val": "latitude",
            "primary_longitude_decimal_val": "longitude",
        }
        sf3 = sf2[list(columns_to_keep.keys())].rename(columns=columns_to_keep)

        # Create Address column
        sf3["Address"] = (
            sf3["street_addr_txt"].fillna("")
            + ", "
            + sf3["city_name"].fillna("")
            + ", "
            + sf3["State"].fillna("")
            + " "
            + sf3["zip_code"].fillna("").astype(str)
        )
        # Clean up address (remove leading/trailing commas/spaces)
        sf3["Address"] = (
            sf3["Address"]
            .str.replace(r"(^[, ]+|[, ]+$)", "", regex=True)
            .str.replace(r", ,", ",", regex=True)
        )

        # Add standard columns
        sf3["Type"] = "Superfund Site"
        sf3["Phone_number"] = None  # Explicitly set as NaN for consistency

        # Final column selection and order
        final_columns = [
            "Type",
            "Name",
            "Address",
            "State",
            "Phone_number",
            "Notes",
            "latitude",
            "longitude",
            "FIPS",
        ]
        sf_final = sf3[final_columns]

        return sf_final.reset_index(drop=True)

    except Exception as e:
        logger.error(f"Error processing superfund data for {location}: {e}")
        return None


# --- Modified superfund ---
def superfund(location: Union[str, List[str]]) -> Optional[pd.DataFrame]:
    """
    Fetches superfund site data for one or more locations (states or FIPS codes).
    Uses parallel processing for multiple locations.

    Args:
        location: A single state abbreviation/FIPS code (str) or a list of them.

    Returns:
        A combined pandas DataFrame for all locations, or None if no data is found/processed.
    """
    logger.info("Fetching superfund data...")
    state_codes = []

    # --- Helper function to convert location input to state code ---
    def get_state_code(loc_input: str) -> Optional[str]:
        if isinstance(loc_input, str):
            if loc_input.isalpha() and len(loc_input) == 2:
                return loc_input.upper()
            elif loc_input.isnumeric():
                try:
                    # Ensure FIPS code is treated correctly (e.g., '06' vs '6')
                    fips_lookup = loc_input.zfill(2)
                    state_abbrev = stateDf.loc[
                        stateDf["FIPS2"] == fips_lookup, "StateAbbrev"
                    ].values
                    if len(state_abbrev) > 0:
                        return state_abbrev[0]
                    else:
                        logger.warning(f"Invalid location format or FIPS code '{loc_input}' not found. Skipping.")
                        return None
                except Exception as e:
                    logger.warning(f"Error looking up FIPS code '{loc_input}': {e}")
                    return None
            else:
                logger.warning(f"Invalid location format or FIPS code '{loc_input}' not found. Skipping.")
                return None
        else:
            logger.warning(f"Invalid location format or FIPS code '{loc_input}' not found. Skipping.")
            return None

    # --- Process single location ---
    if isinstance(location, str):
        code = get_state_code(location)
        if code:
            state_codes.append(code)
        else:
            return None  # Return None if single location is invalid

    # --- Process list of locations ---
    elif isinstance(location, list):
        valid_codes = [get_state_code(loc) for loc in location]
        state_codes = [
            code for code in valid_codes if code is not None
        ]  # Filter out None values
        if not state_codes:
            logger.error("No valid state codes found in the input list.")
            return None  # Return None if no valid locations in the list
    else:
        logger.error(
            f"Invalid type for location: {type(location)}. Expected str or List[str]."
        )
        return None

    logger.info(f"Processing superfund data for state codes: {state_codes}")

    # --- Fetch data (sequentially for one, parallel for multiple) ---
    if len(state_codes) == 1:
        df = gen_single_superfund(state_codes[0])
    else:
        # logger.info(f"Fetching superfund data in parallel for: {state_codes}") # Already logged above
        # Use threading backend for I/O-bound tasks like network requests
        datasets = Parallel(n_jobs=-1, backend="threading")(
            delayed(gen_single_superfund)(code)
            for code in tqdm(state_codes, desc="Fetching superfund data")
        )

        # Filter out None results (from errors) and concatenate
        valid_datasets = [d for d in datasets if d is not None and not d.empty]
        if not valid_datasets:
            logger.info("No valid superfund datasets collected, returning None.")
            return None
        df = pd.concat(valid_datasets, ignore_index=True)

    if df is None or df.empty:
        logger.info(
            "Superfund data query resulted in no data for the provided location(s)."
        )
        return None

    return df.reset_index(drop=True)


if __name__ == "__main__":
    data = gen_facility_data(["CA", "TX", "GA", "CO", "FL"])
    print(data)
