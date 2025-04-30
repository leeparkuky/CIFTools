import logging
import requests
import pandas as pd
import json
from tqdm.auto import tqdm
from typing import Union, List, Dict
from zipfile import ZipFile
from io import BytesIO
import os

# Import logger
from ..utils.ciftools_logger import logger
from ..utils.states import stateDf
from dotenv import load_dotenv

load_dotenv()


fcc_username = os.getenv("FCC_USERNAME")
fcc_hash_value = os.getenv("FCC_HASH_VALUE")
fcc_user_agent = os.getenv("FCC_USER_AGENT")


class FCCAvailability:
    """
    A class to fetch and process FCC broadband availability data.

    Attributes:
        state_fips (Union[str, List[str]]): State FIPS code(s) to filter the dataset.
    """

    def __init__(self, state_fips: Union[str, List[str]]):
        """
        Initializes the FCCAvailability class.

        Args:
            state_fips (Union[str, List[str]]): A single state FIPS code or a list of them.
        """
        self.state_fips = state_fips

        # Configure persistent session with necessary headers
        self.session = requests.Session()
        self.session.headers = {
            "username": fcc_username,
            "hash_value": fcc_hash_value,
            "user-agent": fcc_user_agent,
        }

    @property
    def fcc_avail_data(self) -> pd.DataFrame:
        """
        Retrieves FCC broadband availability data, downloading it if not already available.

        Returns:
            pd.DataFrame: Processed broadband availability data.
        """
        if not hasattr(self, "_fcc_avail_data"):
            logger.info("Fetching FCC broadband availability data...")
            self._fcc_avail_data = self._download_data()
            logger.info("FCC broadband availability data processing complete.")
        return self._fcc_avail_data

    def _download_data(self) -> pd.DataFrame:
        """
        Downloads, extracts, and processes the FCC broadband availability data.

        Returns:
            pd.DataFrame: Processed data with broadband availability statistics.
        """
        try:
            # State mapping dictionary
            state_dict = stateDf.set_index("StateAbbrev")["State"].to_dict()

            # Fetch the latest available data date
            date_url = "https://broadbandmap.fcc.gov/api/public/map/listAsOfDates"
            response = self.session.get(date_url, timeout=10)
            response.raise_for_status()

            dates = pd.DataFrame(json.loads(response.text)["data"])
            dates["date"] = dates["as_of_date"].str[:10]
            as_of_date = dates.loc[dates["data_type"] == "availability", "date"].max()

            logger.info(f"Latest available FCC data date: {as_of_date}")

            # Fetch file metadata
            data_url = f"https://broadbandmap.fcc.gov/api/public/map/downloads/listAvailabilityData/{as_of_date}"
            response = self.session.get(data_url, timeout=10)
            response.raise_for_status()

            filing_data = pd.DataFrame(json.loads(response.text)["data"])

            # Get file details for fixed broadband
            fb_data = filing_data[
                (filing_data["category"] == "Summary")
                & (
                    filing_data["subcategory"]
                    == "Summary by Geography Type - Other Geographies"
                )
                & (filing_data["technology_type"] == "Fixed Broadband")
            ].reset_index()

            # Get file details for mobile broadband
            mb_data = filing_data[
                (filing_data["category"] == "Summary")
                & (
                    filing_data["subcategory"]
                    == "Summary by Geography Type - Other Geographies"
                )
                & (filing_data["technology_type"] == "Mobile Broadband")
            ].reset_index()

            # Define base URL for file download
            base_url = "https://broadbandmap.fcc.gov/api/public/map/downloads//downloadFile/availability/"

            # Download Fixed Broadband data
            logger.info("Downloading Fixed Broadband data: %s", fb_data["file_name"][0])
            fb_url = f"{base_url}{fb_data['file_id'][0]}/1"
            fixed_broadband = self._download_and_extract(fb_url)

            # Filter and rename Fixed Broadband columns
            fixed_broadband = fixed_broadband[
                (fixed_broadband["geography_type"] == "County")
                & (fixed_broadband["area_data_type"] == "Total")
                & (fixed_broadband["biz_res"] == "R")
                & (fixed_broadband["technology"] == "Any Technology")
            ][["geography_id", "geography_desc_full", "speed_100_20", "speed_1000_100"]]

            # Download Mobile Broadband data
            logger.info(
                "Downloading Mobile Broadband data: %s", mb_data["file_name"][0]
            )
            mb_url = f"{base_url}{mb_data['file_id'][0]}/1"
            mobile_broadband = self._download_and_extract(mb_url)

            # Filter and rename Mobile Broadband columns
            mobile_broadband = mobile_broadband[
                (mobile_broadband["geography_type"] == "County")
                & (mobile_broadband["area_data_type"] == "Total")
            ][
                [
                    "geography_id",
                    "geography_desc",
                    "mobilebb_5g_spd1_area_st_pct",
                    "mobilebb_5g_spd2_area_st_pct",
                ]
            ]

            mobile_broadband = mobile_broadband.rename(
                columns={"geography_desc": "geography_desc_full"}
            )

            # Merge datasets
            df = pd.merge(
                fixed_broadband,
                mobile_broadband,
                on=["geography_id", "geography_desc_full"],
            )

            # Extract county and state names
            df[["County", "State"]] = df["geography_desc_full"].str.split(
                ", ", expand=True
            )
            df["State"] = df["State"].replace(
                state_dict
            )  # Convert state abbreviations to full names

            # Rename columns for clarity
            df = df.rename(
                columns={
                    "geography_id": "FIPS",
                    "speed_100_20": "pctBB_100_20",
                    "speed_1000_100": "pctBB_1000_100",
                    "mobilebb_5g_spd1_area_st_pct": "pct5G_7_1",
                    "mobilebb_5g_spd2_area_st_pct": "pct5G_35_3",
                }
            )

            # Final column order
            df = df[
                [
                    "FIPS",
                    "County",
                    "State",
                    "pctBB_100_20",
                    "pctBB_1000_100",
                    "pct5G_7_1",
                    "pct5G_35_3",
                ]
            ]

            # Filter by state
            df["FIPS2"] = df["FIPS"].str[:2]
            df = df[
                df["FIPS2"].isin(
                    [self.state_fips]
                    if isinstance(self.state_fips, str)
                    else self.state_fips
                )
            ].reset_index(drop=True)
            df.drop(columns=["FIPS2"], inplace=True)

            logger.info(
                "FCC broadband data successfully processed. Records: %d", len(df)
            )
            return df

        except Exception as e:
            logger.error("Failed to process FCC broadband data: %s", e)
            raise RuntimeError(f"Error processing FCC broadband data: {e}")

    def _download_and_extract(self, url: str) -> pd.DataFrame:
        """
        Downloads and extracts a ZIP file from a given URL with a progress bar.

        Args:
            url (str): The download URL.

        Returns:
            pd.DataFrame: Extracted data.
        """
        logger.info(f"Downloading data from {url}...")

        # Stream the request with tqdm progress bar
        response = self.session.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))  # Total file size
        chunk_size = 1024 * 1024  # 1MB chunk size
        buffer = BytesIO()

        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading ZIP file",
            leave=True,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                buffer.write(chunk)
                progress_bar.update(len(chunk))

        logger.info("Download complete. Extracting ZIP file...")

        # Extract ZIP from the in-memory buffer
        with ZipFile(buffer, "r") as zip_ref:
            file_name = zip_ref.namelist()[0]  # Get first file in ZIP
            logger.info(f"Extracting file: {file_name}")
            df = pd.read_csv(zip_ref.open(file_name), dtype={"geography_id": str})

        logger.info("Extraction complete. Data loaded into memory.")
        return df


if __name__ == "__main__":
    # Example usage
    state_fips = "12"  # Florida
    fcc_availability = FCCAvailability(state_fips)
    df = fcc_availability.fcc_avail_data
    print(df.head())
