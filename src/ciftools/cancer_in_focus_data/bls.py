import logging
import os
import sys
import requests
import pandas as pd
from tqdm.auto import tqdm
from functools import cached_property
from dataclasses import dataclass
from typing import Union, List
import zipfile  # Added for zip file handling
import io  # Added for handling bytes stream

from ..utils.ciftools_logger import logger
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
        Fetches raw BLS data from a zip file containing an Excel spreadsheet,
        processes it, and caches the result in memory.
        Ensures the file is only downloaded once per execution.
        """
        if self._cached_data is not None:
            logger.info("Using cached BLS data.")
            return self._cached_data

        url = "https://www.bls.gov/web/metro/laucntycur14.zip"
        user_agent = {"User-agent": bls_user_agent}
        # Define expected Excel filename within the zip (adjust if needed)
        # Usually it's consistent, e.g., 'laucntycur14.xlsx'
        excel_filename_in_zip = "laucntycur14.xlsx"
        temp_zip_filename = "bls_data.zip"  # Save as zip

        logger.info("Downloading BLS data from %s", url)

        try:
            with requests.get(url, headers=user_agent, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))

                with (
                    open(temp_zip_filename, "wb") as file,  # Write to zip file
                    tqdm(
                        desc="Downloading BLS data file",
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        leave=True,
                    ) as bar,
                ):
                    for chunk in response.iter_content(
                        chunk_size=1024 * 10
                    ):  # 10 KB chunks
                        if chunk:  # filter out keep-alive new chunks
                            file.write(chunk)
                            bar.update(len(chunk))

            logger.info("Successfully downloaded BLS data zip file.")

            # Extract and read the Excel file from the zip
            logger.info(f"Extracting and reading {excel_filename_in_zip} from zip.")
            with zipfile.ZipFile(temp_zip_filename, "r") as zip_ref:
                # Check if the expected excel file exists
                if excel_filename_in_zip not in zip_ref.namelist():
                    logger.error(
                        f"Expected file '{excel_filename_in_zip}' not found in the downloaded zip."
                    )
                    # You might want to list available files for debugging:
                    # logger.error(f"Files found in zip: {zip_ref.namelist()}")
                    raise FileNotFoundError(
                        f"'{excel_filename_in_zip}' not found in {temp_zip_filename}"
                    )

                with zip_ref.open(excel_filename_in_zip) as excel_file:
                    # Read directly from the file object within the zip
                    # Skip 6 header rows (0-5), skip 3 footer rows
                    # Use header=None and specify names
                    df = pd.read_excel(
                        excel_file,
                        skiprows=6,
                        skipfooter=3,
                        header=None,  # No header row in the data section we're reading
                        names=[  # Provide column names explicitly
                            "LAUS Area Code",
                            "State FIPS",  # Excel file has separate FIPS codes
                            "County FIPS",
                            "Area Name",
                            "Period",
                            "Civilian Labor Force",
                            "Employed",
                            "Unemployed",
                            "Unemployment Rate",
                        ],
                    )

            logger.info("Successfully read data from Excel file.")

        except requests.RequestException as e:
            logger.error("Failed to fetch BLS data: %s", e)
            raise RuntimeError(f"Failed to fetch BLS data: {e}")
        except (
            zipfile.BadZipFile,
            FileNotFoundError,
            KeyError,
        ) as e:  # Added specific exceptions
            logger.error("Failed to process BLS zip file: %s", e)
            raise RuntimeError(f"Failed to process BLS zip file: {e}")
        except ImportError:
            logger.error(
                "The 'openpyxl' library is required to read Excel files. Please install it (`pip install openpyxl`)."
            )
            raise ImportError("The 'openpyxl' library is required. Please install it.")
        except Exception as e:  # Catch other potential pandas or processing errors
            logger.error("An error occurred during BLS data processing: %s", e)
            raise RuntimeError(f"An error occurred during BLS data processing: {e}")
        finally:
            # Delete the temporary zip file
            if os.path.exists(temp_zip_filename):
                os.remove(temp_zip_filename)
                logger.info("Deleted temporary BLS data zip file.")

        # --- Data Cleaning (Adjusted for Excel structure) ---
        logger.info("Cleaning BLS data...")

        # Convert FIPS codes to string and pad county FIPS
        df["State"] = (
            df["State FIPS"].astype(str).str.zfill(2)
        )  # Assuming State FIPS needs padding
        df["County"] = df["County FIPS"].astype(str).str.zfill(3)
        df["FIPS"] = df["State"] + df["County"]

        # Clean numeric columns (remove commas, strip whitespace, convert)
        for col in [
            "Civilian Labor Force",
            "Employed",
            "Unemployed",
            "Unemployment Rate",
        ]:
            # Ensure the column exists before processing
            if col in df.columns:
                # Convert to string first to handle potential non-numeric entries safely
                df[col] = (
                    df[col].astype(str).str.replace(",", "", regex=False).str.strip()
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                logger.warning(f"Column '{col}' not found during cleaning.")

        # Clean Period column (remove '(p)', format to Mon-YYYY)
        if "Period" in df.columns:
            df["Period"] = (
                df["Period"]
                .astype(str)
                .str.strip()
                .str.replace(r"\s*\(p\)", "", regex=True)
            )
            # Assuming the format is like 'Jan-24', convert to 'Jan-2024'
            # This lambda might need adjustment if the format changes
            df["Period"] = df["Period"].apply(
                lambda x: x[:-2] + "20" + x[-2:]
                if isinstance(x, str) and len(x) > 2
                else x
            )
        else:
            logger.warning("Column 'Period' not found during cleaning.")

        # Drop intermediate FIPS columns if they exist
        df.drop(columns=["State FIPS", "County FIPS"], errors="ignore", inplace=True)

        logger.info("Successfully processed BLS data. Records: %d", len(df))

        self._cached_data = df  # Store in memory for reuse
        return df

    # --- bls_data and bls_data_timeseries methods remain the same ---
    # (Make sure they use the correct column names after cleaning,
    #  e.g., 'State', 'County', 'FIPS', 'Unemployment Rate', 'Period')

    @cached_property
    def bls_data(self) -> pd.DataFrame:
        """
        Fetches and processes the most recent unemployment rate data.
        """
        logger.info("Processing most recent BLS data for state(s): %s", self.state_fips)
        df = self._fetch_and_clean_data()

        # Filter for state(s)
        if isinstance(self.state_fips, str):
            # Ensure state_fips is padded if necessary for comparison
            state_fips_padded = self.state_fips.zfill(2)
            df_filtered = df[df["State"] == state_fips_padded].copy()  # Use .copy()
        else:
            # Ensure all state_fips in the list are padded
            state_fips_padded = [s.zfill(2) for s in self.state_fips]
            df_filtered = df[df["State"].isin(state_fips_padded)].copy()  # Use .copy()

        if df_filtered.empty:
            logger.warning(
                f"No data found for state(s): {self.state_fips}. Returning empty DataFrame."
            )
            return pd.DataFrame(
                columns=["FIPS", f"Monthly Unemployment Rate (Unknown Period)"]
            )

        # Select most recent period if applicable
        if self.most_recent:
            # Convert Period to datetime to find the latest correctly
            try:
                df_filtered["period_dt"] = pd.to_datetime(
                    df_filtered["Period"], format="%b-%Y", errors="coerce"
                )
                latest_period_dt = df_filtered["period_dt"].max()
                if pd.isna(latest_period_dt):
                    logger.warning(
                        "Could not determine the latest period. Using the first unique period found."
                    )
                    # Fallback to original method if conversion fails widely
                    latest_period_str = df_filtered["Period"].unique()[0]
                else:
                    latest_period_str = latest_period_dt.strftime("%b-%Y")
                    df_filtered = df_filtered[
                        df_filtered["period_dt"] == latest_period_dt
                    ]

                df_filtered.drop(
                    columns=["period_dt"], inplace=True, errors="ignore"
                )  # Clean up helper column
                logger.info(
                    "Filtered for the most recent period: %s", latest_period_str
                )

            except Exception as e:
                logger.error(
                    f"Error processing period for 'most_recent': {e}. Using first unique period as fallback."
                )
                latest_period_str = df_filtered["Period"].unique()[0]  # Fallback
                df_filtered = df_filtered[df_filtered["Period"] == latest_period_str]

        else:
            latest_period_str = "Multiple Periods"  # Placeholder if not filtering

        # Check if 'Unemployment Rate' column exists before proceeding
        if "Unemployment Rate" not in df_filtered.columns:
            logger.error(
                "'Unemployment Rate' column not found after filtering. Cannot proceed."
            )
            # Return an empty df with expected columns
            return pd.DataFrame(
                columns=["FIPS", f"Monthly Unemployment Rate ({latest_period_str})"]
            )

        # Rename columns and calculate rate
        df_result = (
            df_filtered[["FIPS", "Unemployment Rate", "Period"]]
            .sort_values("FIPS")
            .reset_index(drop=True)
        )
        # Use the determined latest period string in the column name
        rate_col_name = f"Monthly Unemployment Rate ({latest_period_str})"
        df_result[rate_col_name] = df_result["Unemployment Rate"] * 0.01
        df_result.drop(columns=["Unemployment Rate", "Period"], inplace=True)

        logger.info(
            "Successfully processed BLS data. Records returned: %d", len(df_result)
        )
        return df_result

    @cached_property
    def bls_data_timeseries(self) -> pd.DataFrame:
        """
        Fetches and processes time series unemployment rate data.
        """
        logger.info("Processing BLS time series data for state(s): %s", self.state_fips)
        df = self._fetch_and_clean_data()

        # Filter for state(s) (use .copy() to avoid SettingWithCopyWarning)
        if isinstance(self.state_fips, str):
            # Ensure state_fips is padded if necessary for comparison
            state_fips_padded = self.state_fips.zfill(2)
            df_filtered = df[df["State"] == state_fips_padded].copy()
        else:
            # Ensure all state_fips in the list are padded
            state_fips_padded = [s.zfill(2) for s in self.state_fips]
            df_filtered = df[df["State"].isin(state_fips_padded)].copy()

        if df_filtered.empty:
            logger.warning(
                f"No time series data found for state(s): {self.state_fips}. Returning empty DataFrame."
            )
            return pd.DataFrame(
                columns=[
                    "FIPS",
                    "Civilian Labor Force",
                    "Monthly Unemployment Rate",
                    "Period",
                ]
            )

        # Prepare time-series sorting column
        # Add error handling for date conversion
        try:
            df_filtered["period_for_ordering"] = pd.to_datetime(
                df_filtered["Period"], format="%b-%Y", errors="coerce"
            )
            # Drop rows where conversion failed if necessary, or handle them
            df_filtered.dropna(subset=["period_for_ordering"], inplace=True)
        except Exception as e:
            logger.error(
                f"Error converting 'Period' to datetime for sorting: {e}. Sorting may be incorrect."
            )
            # Add a fallback or log more details if needed
            df_filtered["period_for_ordering"] = (
                pd.NaT
            )  # Assign NaT if conversion fails

        df_sorted = df_filtered.sort_values(["FIPS", "period_for_ordering"])

        # Select and rename columns
        # Ensure required columns exist
        required_cols = ["FIPS", "Civilian Labor Force", "Unemployment Rate", "Period"]
        available_cols = [col for col in required_cols if col in df_sorted.columns]
        missing_cols = [col for col in required_cols if col not in available_cols]
        if missing_cols:
            logger.warning(f"Missing expected columns for time series: {missing_cols}")

        df_result = df_sorted[available_cols].reset_index(drop=True)

        if "Unemployment Rate" in df_result.columns:
            df_result.rename(
                columns={"Unemployment Rate": "Monthly Unemployment Rate"}, inplace=True
            )
            df_result["Monthly Unemployment Rate"] *= 0.01
        else:
            # Add the column with NaNs if it was missing but expected
            if "Monthly Unemployment Rate" not in df_result.columns:
                df_result["Monthly Unemployment Rate"] = pd.NA

        logger.info(
            "Successfully processed BLS time series data. Records returned: %d",
            len(df_result),
        )
        return df_result


# Example Usage (assuming the rest of your setup is correct)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     # Example: Get latest data for California (FIPS 06)
#     bls_instance_latest = BLS(state_fips="06", most_recent=True)
#     latest_data = bls_instance_latest.bls_data
#     print("Latest Data for CA:")
#     print(latest_data.head())
#
#     # Example: Get time series for California and Nevada (FIPS 32)
#     bls_instance_ts = BLS(state_fips=["06", "32"], most_recent=False)
#     ts_data = bls_instance_ts.bls_data_timeseries
#     print("\nTime Series Data for CA & NV:")
#     print(ts_data.head())
#     print(ts_data.tail())
