import logging
import urllib.request
import pandas as pd
import os
from tqdm import tqdm
from typing import Union, List, Dict
from io import BytesIO
from zipfile import ZipFile

# Import logger
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # add src to path
from utils.ciftools_logger import logger


class EJScreen:
    """
    A class to fetch and process EJScreen data from the EPA.
    
    Attributes:
        state_fips (Union[str, List[str]]): State FIPS code(s) to filter the dataset.
    """

    def __init__(self, state_fips: Union[str, List[str]]):
        """
        Initializes the EJScreen class with a predefined URL to fetch data.
        """
        self.url = "https://web.archive.org/web/20250205214353if_/https://gaftp.epa.gov/ejscreen/2024/2.32_August_UseMe/EJScreen_2024_Tract_with_AS_CNMI_GU_VI.csv.zip"
        self.state_fips = state_fips

    @property
    def ejscreen_data(self) -> Dict[str, pd.DataFrame]:
        """
        Retrieves EJScreen data, downloading it if not already available.
        Returns a dictionary with the processed data.
        """
        if not hasattr(self, "_ejscreen_data"):
            logger.info("Fetching EJScreen data...")
            self._ejscreen_data = self._download_data()
            logger.info("EJScreen data processing complete.")
        return self._ejscreen_data

    def _download_data(self) -> Dict[str, pd.DataFrame]:
        """
        Downloads, extracts, and processes the EJScreen data file.

        Returns:
            Dict[str, pd.DataFrame]: Processed data dictionary with 'Tract' level data.
        """
        try:
            # Open the URL and get the file size for tqdm
            with urllib.request.urlopen(self.url) as response:
                total_size = int(response.headers.get("content-length", 0))
                logger.info("Starting EJScreen data download. File size: %.2f MB", total_size / (1024 * 1024))

                # Download file with progress tracking
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc="Downloading EJScreen data",
                ) as bar:
                    file_data = BytesIO()
                    while True:
                        chunk = response.read(1024 * 10)  # Read in 10KB chunks
                        if not chunk:
                            break
                        file_data.write(chunk)
                        bar.update(len(chunk))

            # Extract the ZIP file
            with ZipFile(file_data, "r") as zip_ref:
                file_name = zip_ref.namelist()[0]  # Get the first file in the ZIP
                logger.info("Extracting and loading data from: %s", file_name)
                with zip_ref.open(file_name) as file:
                    df = pd.read_csv(file, dtype={"ID": str})

            logger.info("Data successfully loaded. Cleaning and filtering...")

            # Keep relevant columns and rename them
            column_mapping = {
                "ID": "CensusTract",
                "PRE1960PCT": "Lead Paint",
                "DSLPM": "Diesel PM",
                "PTRAF": "Traffic Proximity",
                "PWDIS": "Water Discharge",
                "PNPL": "Superfund Proximity",
                "PRMP": "RMP Proximity",
                "PTSDF": "Hazardous Waste Proximity",
                "OZONE": "Ozone",
                "UST": "Underground Storage Tanks",
                "RSEI_AIR": "Toxics Release to Air",
                "NO2": "Nitrogen Dioxide",
                "DWATER": "Drinking Water Noncompliance",
            }
            df = df[list(column_mapping.keys())].rename(columns=column_mapping)

            # Standardize Census Tract format (FIPS code)
            df["CensusTract"] = df["CensusTract"].str.zfill(11)  # 11-digit FIPS
            df["State"] = df["CensusTract"].str[:2]  # Extract state FIPS

            # Filter by state(s)
            if isinstance(self.state_fips, str):
                assert len(self.state_fips) == 2, "State FIPS should be a 2-digit string."
                df = df.loc[df["State"] == self.state_fips].reset_index(drop=True)
            else:
                for state in self.state_fips:
                    assert len(state) == 2, "State FIPS should be a 2-digit string."
                df = df.loc[df["State"].isin(self.state_fips)].reset_index(drop=True)

            # Store processed data
            df.rename(columns={"CensusTract": "FIPS"}, inplace=True)
            df["FIPS"] = df["FIPS"].astype(str)

            logger.info("EJScreen data successfully processed. Records: %d", len(df))
            return {"Tract": df}

        except Exception as e:
            logger.error("Failed to process EJScreen data: %s", e)
            raise RuntimeError(f"Error processing EJScreen data: {e}")


# if __name__ == "__main__":
#     # Example usage
#     ejscreen = EJScreen(state_fips=["06", "12"])  # California and Florida
#     data = ejscreen.ejscreen_data
#     print(data["Tract"].head())