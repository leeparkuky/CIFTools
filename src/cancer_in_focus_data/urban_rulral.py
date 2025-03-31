from typing import Union, List
import pandas as pd
import requests
from tqdm.auto import tqdm
from tempfile import NamedTemporaryFile
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # add src to path

from utils.ciftools_logger import logger


def urban_rural_counties(state_fips: Union[str, List[str]]):
    url = 'https://www2.census.gov/geo/docs/reference/ua/2020_UA_COUNTY.xlsx'
    logger.info(f"Starting download of urban-rural counties data from {url}...")

    # Stream the file while downloading
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1 MB chunks

    with NamedTemporaryFile(delete=True, suffix=".xlsx") as temp_file:
        temp_filename = temp_file.name
        with open(temp_filename, "wb") as file, tqdm(
            desc="Downloading urban-rural counties data",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=True
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
                bar.update(len(chunk))

        logger.info("Download complete. Processing dataset...")

        # Read the dataset
        urban_rural_counties = pd.read_excel(temp_filename, dtype={'STATE': str, 'COUNTY': str})

    logger.info("Cleaning and filtering urban-rural counties data...")

    # Filter by state FIPS
    if isinstance(state_fips, str):
        urban_rural_counties = urban_rural_counties.loc[
            urban_rural_counties.STATE.eq(state_fips),
            ['STATE', 'COUNTY', 'STATE_NAME', 'COUNTY_NAME', 'POPPCT_URB']
        ]
    else:
        urban_rural_counties = urban_rural_counties.loc[
            urban_rural_counties.STATE.isin(state_fips),
            ['STATE', 'COUNTY', 'STATE_NAME', 'COUNTY_NAME', 'POPPCT_URB']
        ]

    # Process the data
    urban_rural_counties['FIPS'] = urban_rural_counties.STATE + urban_rural_counties.COUNTY
    urban_rural_counties = urban_rural_counties[['FIPS', 'COUNTY_NAME', 'STATE_NAME', 'POPPCT_URB']]
    urban_rural_counties.rename(
        columns={
            'COUNTY_NAME': 'County',
            'STATE_NAME': 'State',
            'POPPCT_URB': 'Urban_Percentage'
        },
        inplace=True
    )
    urban_rural_counties['Urban_Percentage'] = urban_rural_counties.Urban_Percentage
    urban_rural_counties['County'] = urban_rural_counties.County + ' County'
    urban_rural_counties = urban_rural_counties.sort_values('FIPS').reset_index(drop=True)

    logger.info("Urban-rural counties data processing complete.")
    return urban_rural_counties

if __name__=="__main__":
    """
    Example usage of the urban_rural_counties function.
    You can run this script directly to see the output for a specific state or list of states.
    """
    # Example state FIPS codes
    state_fips = ['12', '13']  # Florida and Georgia

    # Get urban-rural counties data for the specified state(s)
    urban_rural_data = urban_rural_counties(state_fips)

    # Display the resulting DataFrame
    print(urban_rural_data)