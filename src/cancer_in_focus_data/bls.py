import logging
import os
import sys
import requests
import pandas as pd
from tqdm.auto import tqdm
from functools import cached_property
from dataclasses import dataclass
from typing import Union, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # add src to path

from utils.ciftools_logger import logger
from dotenv import load_dotenv

load_dotenv()
bls_user_agent = os.getenv("BLS_USER_AGENT")


@dataclass
class BLS:
    """
    A class to fetch and process Bureau of Labor Statistics (BLS) county-level unemployment data.
    """

    state_fips: Union[str, List[str]]
    most_recent: bool = True
    _cached_data: pd.DataFrame = None  # Store the cleaned DataFrame in memory

    def _fetch_and_clean_data(self) -> pd.DataFrame:
        """
        Fetches raw BLS data, processes it, and caches the result in memory.
        Ensures the file is only downloaded once per execution.
        """
        if self._cached_data is not None:
            logger.info("Using cached BLS data.")
            return self._cached_data

        url = "https://www.bls.gov/web/metro/laucntycur14.txt"
        user_agent = {'User-agent': bls_user_agent}
        temp_filename = "bls_data.txt"

        logger.info("Downloading BLS data from %s", url)

        try:
            with requests.get(url, headers=user_agent, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))

                with open(temp_filename, "wb") as file, tqdm(
                    desc="Downloading BLS data file",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    leave=True
                ) as bar:
                    for chunk in response.iter_content(chunk_size=1024 * 10):  # 10 KB chunks
                        file.write(chunk)
                        bar.update(len(chunk))

            logger.info("Successfully downloaded BLS data.")
        except requests.RequestException as e:
            logger.error("Failed to fetch BLS data: %s", e)
            raise RuntimeError(f"Failed to fetch BLS data: {e}")

        # Read the file and process data
        try:
            with open(temp_filename, "r") as f:
                lines = f.readlines()[6:-7]  # Skip headers and footers

            df = pd.DataFrame(
                [x.strip().split('|') for x in lines],
                columns=['LAUS Area Code', 'State', 'County', 'Area', 'Period', 
                         'Civilian Labor Force', 'Employed', 'Unemployed', 'Unemployment Rate']
            )

            # Cleanup data
            df['State'] = df['State'].str.strip().astype(str)
            df['County'] = df['County'].str.strip().str.zfill(3)
            df['FIPS'] = df['State'] + df['County']

            for col in ['Civilian Labor Force', 'Employed', 'Unemployed', 'Unemployment Rate']:
                df[col] = pd.to_numeric(df[col].str.replace(',', '').str.strip(), errors='coerce')

            df['Period'] = df['Period'].str.strip().str.replace(r'\(p\)', '', regex=True)
            df['Period'] = df['Period'].apply(lambda x: x[:-2] + '20' + x[-2:])

            logger.info("Successfully processed BLS data. Records: %d", len(df))
        finally:
            # Delete the file after reading it
            os.remove(temp_filename)
            logger.info("Deleted temporary BLS data file.")

        self._cached_data = df  # Store in memory for reuse
        return df

    @cached_property
    def bls_data(self) -> pd.DataFrame:
        """
        Fetches and processes the most recent unemployment rate data.
        """
        logger.info("Processing most recent BLS data for state(s): %s", self.state_fips)
        df = self._fetch_and_clean_data()

        # Filter for state(s)
        if isinstance(self.state_fips, str):
            df = df[df['State'] == self.state_fips]
        else:
            df = df[df['State'].isin(self.state_fips)]

        # Select most recent period if applicable
        if self.most_recent:
            latest_period = df['Period'].unique()[0]
            df = df[df['Period'] == latest_period]
            logger.info("Filtered for the most recent period: %s", latest_period)

        # Rename columns
        df = df[['FIPS', 'Unemployment Rate', 'Period']].sort_values('FIPS').reset_index(drop=True)
        df[f'Monthly Unemployment Rate ({df.Period.unique()[0]})'] = df['Unemployment Rate'] * 0.01
        df.drop(columns=['Unemployment Rate', 'Period'], inplace=True)

        logger.info("Successfully processed BLS data. Records returned: %d", len(df))
        return df

    @cached_property
    def bls_data_timeseries(self) -> pd.DataFrame:
        """
        Fetches and processes time series unemployment rate data.
        """
        logger.info("Processing BLS time series data for state(s): %s", self.state_fips)
        df = self._fetch_and_clean_data()

        # Filter for state(s) (use .copy() to avoid SettingWithCopyWarning)
        if isinstance(self.state_fips, str):
            df = df[df['State'] == self.state_fips].copy()
        else:
            df = df[df['State'].isin(self.state_fips)].copy()

        # Prepare time-series sorting column
        df['period_for_ordering'] = pd.to_datetime(df['Period'], format='%b-%Y')
        df = df.sort_values(['FIPS', 'period_for_ordering'])

        # Select and rename columns
        df = df[['FIPS', 'Civilian Labor Force', 'Unemployment Rate', 'Period']].reset_index(drop=True)
        df.rename(columns={'Unemployment Rate': 'Monthly Unemployment Rate'}, inplace=True)
        df['Monthly Unemployment Rate'] *= 0.01

        logger.info("Successfully processed BLS time series data. Records returned: %d", len(df))
        return df


