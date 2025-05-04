# %% import statements

import pandas as pd
import numpy as np
from typing import Union, List
from glob import glob
from dataclasses import dataclass
from requests.adapters import HTTPAdapter
import os
import re
import shutil
import requests
from csv import DictReader
from dotenv import load_dotenv
from functools import partial
from itertools import product
from joblib import Parallel, delayed
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm.auto import tqdm

from ..utils.ciftools_logger import logger
from ..utils.general_utils import source_folder
from ..utils.states import stateDf

# Load environment variables
load_dotenv()


# %%
@dataclass
class scp_cancer_data:
    state_fips: Union[str, List[str]]
    folder_name: str = "cancer_data"

    def __post_init__(self):
        # Create a temporary directory for storing data
        self.temp_dir = os.path.join(source_folder, "tmp_file")
        create_directory(self.temp_dir)
        self.folder_path = os.path.join(self.temp_dir, self.folder_name)
        create_directory(self.folder_path)  # Ensure the directory exists

    @property
    def cancer_data(self):
        if hasattr(self, "_cancer_data"):
            pass
        else:
            data_dict = {}
            data_dict["incidence"] = self.uscs_incidence()
            data_dict["mortality"] = self.scp_cancer_mor()
            data_dict["incidence"]["AAR"] = self.convert_dtype(
                data_dict["incidence"].AAR
            )
            data_dict["mortality"]["AAR"] = self.convert_dtype(
                data_dict["mortality"].AAR
            )
            self._cancer_data = data_dict
        return self._cancer_data

    def __del__(self):
        # Clean up the temporary directory when the object is deleted
        try:
            shutil.rmtree(self.temp_dir)  # Remove the directory and all its contents
            logger.info(f"Temporary directory {self.temp_dir} removed successfully.")
        except Exception as e:
            logger.error(f"Failed to remove temporary directory {self.temp_dir}: {e}")

    @staticmethod
    def convert_dtype(series):
        series = series.copy()
        series = series.astype(str).str.strip()
        series[~series.str.contains(r"\d+\.?\d*")] = np.nan
        series = series.astype(float)
        return series

    def uscs_incidence(self):
        logger.info("Searching for USCS cancer incidence file.")
        try:
            # Locate the file
            cancer_incidence_file_finds = list(
                source_folder.rglob("*/*/uscs_cancer_incidence_county_2017-2021.csv")
            )
            if not cancer_incidence_file_finds:
                logger.error("USCS cancer incidence file not found.")
                raise FileNotFoundError("USCS cancer incidence file not found.")

            file_path = cancer_incidence_file_finds[0]
            logger.info(f"USCS cancer incidence file found: {file_path}")

            # Load the data
            df = pd.read_csv(file_path)
            logger.info("USCS cancer incidence data loaded successfully.")

            # Process the data
            df.FIPS = df.FIPS.astype(str).str.zfill(5)
            logger.info("Processed FIPS codes in USCS cancer incidence data.")
            return df
        except Exception as e:
            logger.error(f"Error in uscs_incidence: {e}")
            raise

    def scp_cancer_inc(self):
        sites = {
            "001": "All Site",
            "071": "Bladder",
            "076": "Brain & ONS",
            "020": "Colon & Rectum",
            "017": "Esophagus",
            "072": "Kidney & Renal Pelvis",
            "090": "Leukemia",
            "035": "Liver & IBD",
            "047": "Lung & Bronchus",
            "053": "Melanoma of the Skin",
            "086": "Non-Hodgkin Lymphoma",
            "003": "Oral Cavity & Pharynx",
            "040": "Pancreas",
            "018": "Stomach",
            "080": "Thyroid",
        }

        sitesf = {
            "001": "All Site",
            "071": "Bladder",
            "076": "Brain & ONS",
            "020": "Colon & Rectum",
            "017": "Esophagus",
            "072": "Kidney & Renal Pelvis",
            "090": "Leukemia",
            "035": "Liver & IBD",
            "047": "Lung & Bronchus",
            "053": "Melanoma of the Skin",
            "086": "Non-Hodgkin Lymphoma",
            "003": "Oral Cavity & Pharynx",
            "040": "Pancreas",
            "018": "Stomach",
            "080": "Thyroid",
            "055": "Female Breast",
            "057": "Cervix",
            "061": "Ovary",
            "058": "Corpus Uteri & Uterus, NOS",
        }

        sitesm = {
            "001": "All Site",
            "071": "Bladder",
            "076": "Brain & ONS",
            "020": "Colon & Rectum",
            "017": "Esophagus",
            "072": "Kidney & Renal Pelvis",
            "090": "Leukemia",
            "035": "Liver & IBD",
            "047": "Lung & Bronchus",
            "053": "Melanoma of the Skin",
            "086": "Non-Hodgkin Lymphoma",
            "003": "Oral Cavity & Pharynx",
            "040": "Pancreas",
            "018": "Stomach",
            "080": "Thyroid",
            "066": "Prostate",
        }

        # re = {'00': 'All', '07': 'White NH', '28': 'Black NH', '05': 'Hispanic'} #, '38': 'AIAN', '48': 'API'
        re = {
            "00": "All",
            "07": "White NH",
            "28": "Black NH",
            "05": "Hispanic",
            "38": "American Indian/Alaska Native NH",
            "48": "Asian/Pacific Islander NH",
        }

        gen_single_cancer_inc_all = partial(
            self.gen_single_cancer_inc, sex="0", folder_name=self.folder_name
        )
        gen_single_cancer_inc_male = partial(
            self.gen_single_cancer_inc, sex="1", folder_name=self.folder_name
        )
        gen_single_cancer_inc_female = partial(
            self.gen_single_cancer_inc, sex="2", folder_name=self.folder_name
        )

        # Create a list of tasks
        tasks_all = list(product(sites.items(), re.items()))
        tasks_female = list(product(sites.items(), re.items()))
        tasks_male = list(product(sites.items(), re.items()))

        # Use tqdm to monitor progress
        incidence_all = Parallel(n_jobs=-1)(
            delayed(gen_single_cancer_inc_all)(site[0], site[1], re[0], re[1])
            for site, re in tqdm(tasks_all, desc="Processing All Incidence")
        )
        incidence_female = Parallel(n_jobs=-1)(
            delayed(gen_single_cancer_inc_female)(site[0], site[1], re[0], re[1])
            for site, re in tqdm(tasks_female, desc="Processing Female Incidence")
        )
        incidence_male = Parallel(n_jobs=-1)(
            delayed(gen_single_cancer_inc_male)(site[0], site[1], re[0], re[1])
            for site, re in tqdm(tasks_male, desc="Processing Male Incidence")
        )

        df = (
            pd.concat(incidence_all + incidence_female + incidence_male, axis=0)
            .sort_values(["FIPS", "Site"])
            .reset_index(drop=True)
        )
        if df.FIPS.eq("51917").sum():  # if we find 51917 in FIPS
            vaFix = {"51917": "51019", "Bedford City and County": "Bedford County"}
            df = df.replace(vaFix)
        return df

    @staticmethod
    def gen_single_cancer_inc(
        cancer_site_id: str,
        cancer_site: str,
        re_id: str,
        re_g: str,
        sex: int,
        folder_name: str,
    ):
        validate_inputs(cancer_site_id, re_id, sex)

        retry_strategy = Retry(total=3, backoff_factor=1)
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)
        http.mount("http://", adapter)

        path = f"{BASE_URL}/incidencerates/index.php?stateFIPS=00&areatype=county&cancer={cancer_site_id}&race={re_id}&sex={sex}&age=001&stage=999&year=0&type=incd&sortVariableName=rate&sortOrder=desc&output=1"

        try:
            resp = http.get(path, timeout=10)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch data from {path}: {e}")
            return pd.DataFrame(
                columns=[
                    "FIPS",
                    "County",
                    "State",
                    "Type",
                    "Site",
                    "RE",
                    "Sex",
                    "AAR",
                    "AAC",
                ]
            )

        fname = os.path.join(
            folder_name, f"incidence_us_{cancer_site_id}_{sex}_{re_id}.csv"
        )
        try:
            with open(fname, "w") as f:
                flag = False
                for row in resp.iter_lines(decode_unicode=True):
                    if row[:6] in ["County", "Area, "]:
                        flag = True
                        row = (
                            row.replace("Area", "County")
                            .replace(", ", ",")
                            .replace(" ,", "")
                            .replace(" ", "")
                        )
                    elif flag and row == "":
                        flag = False
                    if flag:
                        f.write(row + "\n")
        except Exception as e:
            logger.error(f"Error writing to file {fname}: {e}")
            return pd.DataFrame(
                columns=[
                    "FIPS",
                    "County",
                    "State",
                    "Type",
                    "Site",
                    "RE",
                    "Sex",
                    "AAR",
                    "AAC",
                ]
            )

        # Process CSV file
        try:
            with open(fname, "r") as f:
                reader = DictReader(f)
                if not reader.fieldnames:
                    raise ValueError("CSV file has no fieldnames.")
                # Process data...
        except Exception as e:
            logger.error(f"Error processing file {fname}: {e}")
            return pd.DataFrame(
                columns=[
                    "FIPS",
                    "County",
                    "State",
                    "Type",
                    "Site",
                    "RE",
                    "Sex",
                    "AAR",
                    "AAC",
                ]
            )

        os.remove(fname)  # Clean up
        # return df

    def scp_cancer_mor(self):
        logger.info("Starting SCP cancer mortality data collection.")
        try:
            # Define cancer sites and race/ethnicity groups
            sites = {
                "001": "All Site",
                "071": "Bladder",
                "076": "Brain & ONS",
                "020": "Colon & Rectum",
                "017": "Esophagus",
                "072": "Kidney & Renal Pelvis",
                "090": "Leukemia",
                "035": "Liver & IBD",
                "047": "Lung & Bronchus",
                "053": "Melanoma of the Skin",
                "086": "Non-Hodgkin Lymphoma",
                "003": "Oral Cavity & Pharynx",
                "040": "Pancreas",
                "018": "Stomach",
                "080": "Thyroid",
            }

            sitesf = {
                "001": "All Site",
                "071": "Bladder",
                "076": "Brain & ONS",
                "020": "Colon & Rectum",
                "017": "Esophagus",
                "072": "Kidney & Renal Pelvis",
                "090": "Leukemia",
                "035": "Liver & IBD",
                "047": "Lung & Bronchus",
                "053": "Melanoma of the Skin",
                "086": "Non-Hodgkin Lymphoma",
                "003": "Oral Cavity & Pharynx",
                "040": "Pancreas",
                "018": "Stomach",
                "080": "Thyroid",
                "055": "Female Breast",
                "057": "Cervix",
                "061": "Ovary",
                "058": "Corpus Uteri & Uterus, NOS",
            }

            sitesm = {
                "001": "All Site",
                "071": "Bladder",
                "076": "Brain & ONS",
                "020": "Colon & Rectum",
                "017": "Esophagus",
                "072": "Kidney & Renal Pelvis",
                "090": "Leukemia",
                "035": "Liver & IBD",
                "047": "Lung & Bronchus",
                "053": "Melanoma of the Skin",
                "086": "Non-Hodgkin Lymphoma",
                "003": "Oral Cavity & Pharynx",
                "040": "Pancreas",
                "018": "Stomach",
                "080": "Thyroid",
                "066": "Prostate",
            }

            re = {
                "00": "All",
                "07": "White NH",
                "28": "Black NH",
                "05": "Hispanic",
                "38": "American Indian/Alaska Native NH",
                "48": "Asian/Pacific Islander NH",
            }

            logger.info("Creating tasks for mortality data collection.")
            gen_single_cancer_mor_all = partial(
                self.gen_single_cancer_mor, sex="0", folder_name=self.folder_path
            )
            gen_single_cancer_mor_male = partial(
                self.gen_single_cancer_mor, sex="1", folder_name=self.folder_path
            )
            gen_single_cancer_mor_female = partial(
                self.gen_single_cancer_mor, sex="2", folder_name=self.folder_path
            )

            # Create tasks
            tasks_all = list(product(sites.items(), re.items()))
            tasks_female = list(product(sitesf.items(), re.items()))
            tasks_male = list(product(sitesm.items(), re.items()))

            logger.info("Starting parallel processing for mortality data.")
            # Use tqdm to monitor progress
            mortality_all = Parallel(n_jobs=-1)(
                delayed(gen_single_cancer_mor_all)(site[0], site[1], re[0], re[1])
                for site, re in tqdm(tasks_all, desc="Processing All Mortality")
            )
            mortality_female = Parallel(n_jobs=-1)(
                delayed(gen_single_cancer_mor_female)(site[0], site[1], re[0], re[1])
                for site, re in tqdm(tasks_female, desc="Processing Female Mortality")
            )
            mortality_male = Parallel(n_jobs=-1)(
                delayed(gen_single_cancer_mor_male)(site[0], site[1], re[0], re[1])
                for site, re in tqdm(tasks_male, desc="Processing Male Mortality")
            )

            # Combine results
            logger.info("Combining mortality data results.")
            df = (
                pd.concat(mortality_all + mortality_female + mortality_male, axis=0)
                .sort_values(["FIPS", "Site"])
                .reset_index(drop=True)
            )
            if df.FIPS.eq("51917").sum():  # if we find 51917 in FIPS
                vaFix = {"51917": "51019", "Bedford City and County": "Bedford County"}
                df = df.replace(vaFix)
                logger.info("Applied fixes for Bedford City and County.")
            return df
        except Exception as e:
            logger.error(f"Error in scp_cancer_mor: {e}")
            raise

    @staticmethod
    def gen_single_cancer_mor(
        cancer_site_id: str,
        cancer_site: str,
        re_id: str,
        re_g: str,
        sex: int,
        folder_name: str,
    ):
        retry_strategy = Retry(total=3, backoff_factor=1)
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)
        http.mount("http://", adapter)

        assert sex in list("012")
        assert len(cancer_site_id) == 3
        assert len(re_id) == 2
        sex_n = {"0": "All", "1": "Male", "2": "Female"}
        path = f"https://www.statecancerprofiles.cancer.gov/deathrates/index.php?stateFIPS=00&areatype=county&cancer={cancer_site_id}&race={re_id}&sex={sex}&age=001&year=0&type=death&sortVariableName=rate&sortOrder=desc&output=1"
        resp = http.get(path)
        resp.raise_for_status()

        # first we will create "cancer_data" directory and download the csv file

        folder_dir = os.path.join(os.getcwd(), folder_name)
        # first we will create "cancer_data" directory and download the csv file
        if len(glob(folder_dir)) == 0:  # if we don't yet have 'cancer_data' directory
            os.mkdir(folder_dir)
        # We then will select row that are relevant
        flag = False
        fname = os.path.join(
            folder_dir, f"mortality_us_{cancer_site_id}_{sex}_{re_id}.csv"
        )
        # file name will be unique for each query
        with open(fname, "w") as f:
            for row in resp.iter_lines(decode_unicode=True):  # go through response
                if row[:6] in ["County", "Area, "]:
                    flag = True
                    row = (
                        row.replace("Area", "County")
                        .replace(", ", ",")
                        .replace(" ,", "")
                        .replace(" ", "")
                    )
                elif flag & (row == ""):
                    flag = False
                if flag:
                    f.write(row)
                    f.write("\n")
        # read the file with csv.DictReader
        reader = DictReader(open(fname, "r"))
        if reader.fieldnames:
            # find relevant field name (AAR and AAC)
            fieldnames = pd.Series(reader.fieldnames)
            AAR_field_name = fieldnames[
                fieldnames.str.contains("^Age-Adjusted", flags=re.I)
            ].values[0]
            AAC_field_name = fieldnames[
                fieldnames.str.contains("Count$", flags=re.I)
            ].values[0]
            # Go through reader
            stateDict = stateDf.set_index("FIPS2")["State"].to_dict()

            FIPS = []
            County = []
            State = []
            RE = []
            Sex = []
            AAR = []
            AAC = []
            colname = [
                "FIPS",
                "County",
                "State",
                "Type",
                "Site",
                "RE",
                "Sex",
                "AAR",
                "AAC",
            ]
            for row in reader:
                if row["FIPS"][:2] in stateDf.FIPS2.tolist():
                    FIPS.append(row["FIPS"])
                    # County.append(row['County'].rstrip('\(0123456789\)'))
                    County.append(row["County"].split(", ")[0])
                    State.append(stateDict[row["FIPS"][:2]])
                    try:
                        row[AAR_field_name] = float(row[AAR_field_name])
                    except:  # noqa: E722
                        row[AAR_field_name] = None
                    AAR.append(row[AAR_field_name])
                    try:
                        row[AAC_field_name] = int(row[AAC_field_name])
                    except:  # noqa: E722
                        row[AAC_field_name] = None
                    AAC.append(row[AAC_field_name])
            Type = ["Mortality" for _ in range(len(FIPS))]
            Site = [cancer_site for _ in range(len(FIPS))]
            RE = [re_g for _ in range(len(FIPS))]
            Sex = [sex_n[sex] for _ in range(len(FIPS))]
            df = pd.DataFrame(
                zip(FIPS, County, State, Type, Site, RE, Sex, AAR, AAC), columns=colname
            )
            df = df.sort_values("FIPS").reset_index(drop=True)
            del FIPS, County, State, Type, Site, RE, Sex, AAR, AAC, reader, resp
            os.remove(fname)
            return df

        else:
            return pd.DataFrame(
                None,
                columns=[
                    "FIPS",
                    "County",
                    "State",
                    "Type",
                    "Site",
                    "RE",
                    "Sex",
                    "AAR",
                    "AAC",
                ],
            )


def validate_inputs(cancer_site_id, re_id, sex):
    if not (len(cancer_site_id) == 3 and cancer_site_id.isdigit()):
        raise ValueError(f"Invalid cancer_site_id: {cancer_site_id}")
    if not (len(re_id) == 2 and re_id.isdigit()):
        raise ValueError(f"Invalid re_id: {re_id}")
    if sex not in ["0", "1", "2"]:
        raise ValueError(f"Invalid sex: {sex}")


def create_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise


# %% testing with 01 and 02 FIPS
if __name__ == "__main__":
    # Example usage|
    state_fips = ["01", "02"]  # Example state FIPS codes
    cancer_data_instance = scp_cancer_data(
        state_fips=state_fips, folder_name="cancer_data"
    )

    # Access the cancer data
    cancer_data = cancer_data_instance.cancer_data

    # Print the incidence and mortality data
    print("Cancer Incidence Data:")
    print(cancer_data["incidence"].head())

    print("\nCancer Mortality Data:")
    print(cancer_data["mortality"].head())

# %%
