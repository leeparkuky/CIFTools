import requests
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
from typing import Union, List, Dict
from tempfile import NamedTemporaryFile

import sys
from ..utils.ciftools_logger import logger


class FoodDesert:
    """
    A class to download and process USDA food desert data for given state FIPS codes.

    Attributes:
        state_fips (Union[str, List[str]]): A single FIPS code or a list of state FIPS codes.
        var_name (str): The variable name from the dataset (default: 'LILATracts_Vehicle').
    """

    def __init__(
        self, state_fips: Union[str, List[str]], var_name: str = "LILATracts_Vehicle"
    ):
        """
        Initializes the FoodDesert class by fetching the most recent dataset URL.

        Args:
            state_fips (Union[str, List[str]]): State FIPS codes.
            var_name (str): The variable of interest in the dataset.
        """
        self.var_name = var_name
        self.state_fips = state_fips
        self.url = self._fetch_dataset_url()

    def _fetch_dataset_url(self) -> str:
        """
        Scrapes the USDA website to get the latest food desert dataset URL.

        Returns:
            str: The dataset download URL.
        """
        logger.info("Fetching the USDA food desert dataset URL...")

        try:
            response = requests.get(
                "https://www.ers.usda.gov/data-products/food-access-research-atlas/download-the-data/"
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            hrefs = [a["href"] for a in soup.find_all("a", href=True)]
            dataset_url = next(
                (
                    url
                    for url in hrefs
                    if re.search(r"FoodAccessResearchAtlasData.*", url, re.I)
                ),
                None,
            )

            if not dataset_url:
                logger.error("No dataset URL found on the page.")
                raise ValueError("Dataset URL could not be found.")

            logger.info("Dataset URL successfully retrieved.")
            return dataset_url

        except requests.RequestException as e:
            logger.error("Failed to fetch dataset URL: %s", e)
            raise RuntimeError(f"Failed to fetch dataset URL: {e}")

    @property
    def food_desert_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetches and processes the USDA food desert dataset.

        Returns:
            Dict[str, pd.DataFrame]: Processed data at both Tract and County levels.
        """
        if not hasattr(self, "_food_desert_data"):
            self._food_desert_data = self._download_and_process_data()
        return self._food_desert_data

    def _download_and_process_data(self) -> Dict[str, pd.DataFrame]:
        """
        Downloads and processes the USDA food desert dataset.

        Returns:
            Dict[str, pd.DataFrame]: Processed data at both Tract and County levels.
        """
        logger.info("Starting download of food desert dataset...")

        # Stream the file while downloading
        response = requests.get(self.url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024 * 1024  # 1 MB chunks

        with NamedTemporaryFile(delete=True, suffix=".xlsx") as temp_file_obj:
            with tqdm(
                desc="Downloading food desert data file",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=True,
            ) as bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    temp_file_obj.write(chunk)
                    bar.update(len(chunk))

            temp_file_obj.flush() # Ensure all data is written to disk
            logger.info("Download complete. Processing dataset...")

            # Read dataset (assuming sheet index 2 contains the data)
            df = pd.read_excel(
                temp_file_obj.name, # Use the name attribute of the temp file object
                engine="openpyxl",
                sheet_name=2,
                dtype={"CensusTract": str},
            )

        logger.info("Cleaning and filtering dataset...")

        # Standardize FIPS codes
        df["CensusTract"] = df["CensusTract"].str.zfill(11)
        df["State"] = df["CensusTract"].str[:2]

        # Filter by state FIPS
        if isinstance(self.state_fips, str):
            assert len(self.state_fips) == 2, "State FIPS should be a 2-digit string."
            df = df[df["State"] == self.state_fips].reset_index(drop=True)
        else:
            for fips in self.state_fips:
                assert len(fips) == 2, f"Invalid FIPS: {fips}"
            df = df[df["State"].isin(self.state_fips)].reset_index(drop=True)

        # Keep relevant columns
        df = df[["CensusTract", self.var_name, "OHU2010"]]

        return self._process_levels(df)

    def _process_levels(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Processes data at the Tract and County levels.

        Args:
            df (pd.DataFrame): Filtered dataset.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing 'Tract' and 'County' level data.
        """
        logger.info("Processing data at Tract and County levels...")

        # **Tract-Level Data**
        df_tract = df[["CensusTract", self.var_name]].rename(
            columns={"CensusTract": "FIPS"}
        )
        df_tract["FIPS"] = df_tract["FIPS"].astype(str)

        # **County-Level Data**
        df_county = df[df["OHU2010"] > 0].copy()
        df_county["FIPS"] = df_county["CensusTract"].str[:5]

        df_county = (
            df_county.groupby("FIPS")
            .apply(
                lambda x: pd.Series(np.average(x[self.var_name], weights=x["OHU2010"]))
            )
            .reset_index()
        )
        df_county.columns = ["FIPS", self.var_name]
        df_county["FIPS"] = df_county["FIPS"].astype(str)

        logger.info("Data processing complete.")
        return {"Tract": df_tract, "County": df_county}


# **🔍 Example Usage**
if __name__ == "__main__":
    food_desert = FoodDesert(state_fips="21")
    data = food_desert.food_desert_data
    print(data["Tract"].head())
    print(data["County"].head())
