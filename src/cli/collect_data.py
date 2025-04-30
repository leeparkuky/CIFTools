# %%
import sys
import os
import dotenv
from tqdm.auto import tqdm
import pandas as pd
from glob import glob
import joblib

import fire
from ..config.socrata_config import SocrataConfig
from .config import read_yaml_config
from ..utils.ciftools_logger import logger
from ..utils.states import stateDf

# %%
dotenv.load_dotenv()
api_key = os.getenv("CENSUS_API_KEY")

# config = read_yaml_config("../../example_config/kentucky.yaml")
"""
{'area': {'type': 'catchement', 'area_file': 'uky_ca.csv'}, 
'acs': {'query_level': ['county', 'tract'], 'acs_year': [2021]}, 
'facilities': {'facility_types': ['nppes', 'mammography', 'hpsa', 'fqhc', 'lung_cancer_screening', 'tri_facility', 'superfund']}, 
'cancer_data': True, 
'bls': True, 
'food_desert': True, 
'fcc': True, 
'urban_rural': True, 
'download_dir': '/root/CIFTools/example_config'}
"""
# config = {
#     "area": {"type": "catchement", "area_file": "uky_ca.csv"},
#     "acs": {"query_level": ["county", "tract"], "acs_year": 2021},
#     "facilities": {
#         "facility_types": [
#             "nppes",
#             "mammography",
#             "hpsa",
#             "fqhc",
#             "lung_cancer_screening",
#             "tri_facility",
#             "superfund",
#         ]
#     },
#     "cancer": True,
#     "bls": True,
#     "food_desert": True,
#     "fcc": True,
#     "urban_rural": True,
#     "ejscreen": True,
#     "cdc_places": True,
#     "cdc_svi": True,
#     "download_dir": "/root/CIFTools/example_config",
# }


def check_ca_file(ca_file_path):
    if os.path.split(ca_file_path)[0] == "":
        if len(ca_file_path.split(".")) == 1:  # if no extension is provided
            ca_file_path = ca_file_path + ".csv"

        # Search in the entire repository by going up to root directory
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        glob_result = glob(os.path.join(repo_root, "**", ca_file_path), recursive=True)

        if len(glob_result) == 0:
            raise ValueError(
                "Please check if your file exists in the repository or \n upload your own catchment_area file using 'import_custom_ca_file' function from utils package"
            )
        return glob_result[0]
    else:
        glob_result = glob(ca_file_path)
        if len(glob_result):
            return glob_result[0]
        raise ValueError(f"File not found: {ca_file_path}")


def fetch_data(config: dict):
    """
    Collects various datasets based on the provided configuration.

    Args:
        config (dict): A dictionary containing configuration options for data collection.
            - area: Information about the area (e.g., type and area file).
            - acs: Configuration for ACS data (e.g., query level and year).
            - facilities: Types of facilities to collect data for.
            - cancer, bls, food_desert, fcc, urban_rural, ejscreen, cdc_places, cdc_svi:
              Boolean flags indicating whether to collect respective datasets.
            - download_dir: Directory to save downloaded data.

    Returns:
        dict: A dictionary containing the collected data organized by data source and geographic level.
    """
    # Filter out any top-level config keys explicitly set to False
    if any(val is False for val in config.values()):
        config = {key: val for key, val in config.items() if val is not False}

    # Calculate the number of data sources to fetch (excluding 'area' and 'download_dir')
    num_data = len(config) - 2
    logger.info(f"Number of data sources to collect: {num_data}")

    # Get the list of data source keys to iterate over
    data_to_collect = [
        key for key in config.keys() if key not in ["area", "download_dir"]
    ]

    # Find and validate the path to the catchment area file
    ca_file_path = check_ca_file(config["area"]["area_file"])

    # Read the catchment area file, ensuring FIPS codes are treated as strings
    try:
        ca = pd.read_csv(ca_file_path, dtype={"FIPS": str})
    except Exception as e:
        logger.error(f"Failed to read the file at {ca_file_path}: {e}")
        raise ValueError(f"Unable to read the catchment area file: {ca_file_path}")

    # Ensure FIPS codes are zero-padded to 5 digits
    ca["FIPS"] = ca.FIPS.str.zfill(5)

    # Extract unique state FIPS codes from the catchment area
    state_FIPS = ca.FIPS.apply(lambda x: x[:2]).unique().tolist()

    # Determine if working with a single state or multiple states
    if len(state_FIPS) == 1:
        state_fips = state_FIPS[0]  # Use the single FIPS string
    else:
        state_fips = state_FIPS  # Use the list of FIPS strings

    # Get state abbreviations corresponding to the FIPS codes for location-based lookups
    if isinstance(state_fips, str):
        # Lookup abbreviation for a single state
        location = stateDf.loc[stateDf.FIPS2.eq(state_fips), "StateAbbrev"].values[0]
    else:
        # Lookup abbreviations for multiple states
        location = stateDf.loc[
            stateDf.FIPS2.isin(state_fips), "StateAbbrev"
        ].values.tolist()

    # Initialize the progress bar
    pbar = tqdm(total=num_data, desc="Collecting Data", leave=False)

    # Initialize the dictionary to store the collected data, pre-populating keys for levels
    output_data = {}
    output_data["county"] = {}
    output_data["tract"] = {}
    # 'facility' data doesn't have a standard level, added directly later

    # Iterate through each requested data source
    for dataname in data_to_collect:
        pbar.set_description(
            f"Collecting {dataname} data"
        )  # Update progress bar description

        # --- ACS Data ---
        if dataname == "acs":
            from ..cancer_in_focus_data.acs import acs_sdoh  # Local import

            # Loop through specified query levels (e.g., 'county', 'tract')
            for query_level in config["acs"]["query_level"]:
                # Instantiate ACS data fetcher
                acs_data = acs_sdoh(
                    year=config["acs"]["acs_year"],
                    state_fips=state_fips,
                    query_level=query_level,
                    key=api_key,  # Pass API key
                )
                # Attempt to download and store data, handle potential errors
                try:
                    output_data[query_level][dataname] = (
                        acs_data.cancer_infocus_download()
                    )
                except Exception as e:
                    logger.error(f"Failed to download ACS data for {query_level}: {e}")
                    output_data[query_level][dataname] = None  # Store None on failure

        # --- Facilities Data ---
        elif dataname == "facilities":
            from ..cancer_in_focus_data.facilities import (
                gen_facility_data,
            )  # Local import

            # Fetch facility data using state abbreviations and config options
            output_data["facility"] = gen_facility_data(
                location=location,
                facility_type=config["facilities"].get(
                    "facility_types", "all"
                ),  # Use specified types or 'all'
                taxonomy=config["facilities"].get(
                    "taxonomy", None
                ),  # Optional taxonomy filter
            )

        # --- State Cancer Profile Data ---
        elif dataname == "cancer":
            from ..cancer_in_focus_data.state_cancer_profile import (
                scp_cancer_data,
            )  # Local import

            # Fetch state cancer profile data
            output_data["cancer"] = scp_cancer_data(
                state_fips=state_fips,
                folder_name="cancer_data",  # Specify subfolder if needed
            ).cancer_data

        # --- Bureau of Labor Statistics (BLS) Unemployment Data ---
        elif dataname == "bls":
            from ..cancer_in_focus_data.bls import BLS  # Local import

            # Fetch BLS data (defaults to most recent)
            bls = BLS(state_fips=state_fips)
            # Store the resulting DataFrame under the 'county' level
            output_data["county"]["bls_unemployment"] = bls.bls_data

        # --- Food Desert Data ---
        elif dataname == "food_desert":
            from ..cancer_in_focus_data.food_deserts import FoodDesert  # Local import

            # Fetch food desert data
            food_desert = FoodDesert(state_fips=state_fips)
            data = (
                food_desert.food_desert_data
            )  # Returns a dict with 'County' and 'Tract' keys
            # Store data at respective levels
            output_data["county"]["food_desert"] = data["County"]
            output_data["tract"]["food_desert"] = data["Tract"]

        # --- FCC Broadband Availability Data ---
        elif dataname == "fcc":
            from ..cancer_in_focus_data.fcc import FCCAvailability  # Local import

            # Fetch FCC data
            fcc_availability = FCCAvailability(state_fips=state_fips)
            df = fcc_availability.fcc_avail_data
            # Store county-level FCC data
            output_data["county"]["fcc"] = df

        # --- Urban/Rural Classification Data ---
        elif dataname == "urban_rural":
            from ..cancer_in_focus_data.urban_rural import (
                urban_rural_counties,
            )  # Local import

            # Fetch urban/rural county classifications
            urban_rural_data = urban_rural_counties(state_fips=state_fips)
            # Store county-level urban/rural data
            output_data["county"]["urban_rural"] = urban_rural_data

        # --- EPA EJScreen Data ---
        elif dataname == "ejscreen":
            from ..cancer_in_focus_data.ejscreen import EJScreen  # Local import

            # Fetch EJScreen data
            ejscreen = EJScreen(state_fips=state_fips)
            data = ejscreen.ejscreen_data  # Returns dict, likely with 'Tract' level
            # Store tract-level EJScreen data
            output_data["tract"]["ejscreen"] = data[
                "Tract"
            ]  # Assuming 'Tract' key exists

        # --- CDC PLACES Data ---
        elif dataname == "cdc_places":
            from ..cancer_in_focus_data.cdc_places import places_data  # Local import
            from ..config.socrata_config import SocrataConfig  # Local import for config

            # Configure Socrata API access
            socrata_config = SocrataConfig(
                domain="data.cdc.gov"
            )  # Renamed variable to avoid conflict

            # Fetch CDC PLACES data
            places_data_instance = places_data(
                state_fips=state_fips, config=socrata_config
            )
            # Store county and tract level data under a descriptive key
            output_data["county"]["risk_and_screening"] = (
                places_data_instance.places_data["county"]
            )
            output_data["tract"]["risk_and_screening"] = (
                places_data_instance.places_data["tract"]
            )

        # --- CDC Social Vulnerability Index (SVI) Data ---
        elif dataname == "cdc_svi":
            from ..cancer_in_focus_data.cdc_svi import svi_data  # Local import

            # Fetch SVI data
            svi = svi_data(state_fips=state_fips)
            # Store county and tract level SVI data
            output_data["county"]["svi"] = svi.svi_data["county"]
            output_data["tract"]["svi"] = svi.svi_data["tract"]

        # --- Handle Unrecognized Data Sources ---
        else:
            logger.warning(f"Data source '{dataname}' not recognized. Skipping.")
            continue  # Skip to the next data source

        # Update the progress bar after successfully processing a data source
        pbar.update(1)

    # Finalize progress bar
    pbar.set_description("Data collection complete")
    pbar.close()

    # Return the dictionary containing all collected data
    return output_data


def main(config: str):
    """
    Main function to collect data based on the provided configuration.

    Args:
        config (str): Path to the configuration file.
        output_file_path (str): Path to save the collected data. Make sure the extension is .pickle (it will be saved using joblib.dump)
    """
    # Load the configuration
    config = read_yaml_config(config)
    # Collect the data based on the configuration
    output_data = fetch_data(config)
    # Save the collected data to a file
    output_file_path = os.path.join(
        config["download_dir"], "ciftool_data.pickle"
    )  # Ensure the file has a .pickle extension
    joblib.dump(output_data, output_file_path)
    logger.info(f"Data collected and saved to {output_file_path}")


if __name__ == "__main__":
    fire.Fire(main)
# %%
