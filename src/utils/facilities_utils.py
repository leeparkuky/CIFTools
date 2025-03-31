import urllib.request
from io import BytesIO
from zipfile import ZipFile
import pandas as pd
import re
from tqdm.auto import tqdm
import requests
from typing import Union, List
from csv import DictReader
import os
import asyncio
import aiohttp
import glob
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
from .chrome_driver_utils import setup_chrome_driver, remove_chromedriver
import datetime
from itertools import product


def download_hpsa_data():
    """Downloads and loads HPSA dataset from HRSA into a pandas DataFrame."""
    
    url = "https://data.hrsa.gov/DataDownload/DD_Files/BCD_HPSA_FCT_DET_PC.xlsx"
    
    # Stream download with progress bar
    try:
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch HPSA data: {e}")

    total_size = int(resp.headers.get("content-length", 0))
    chunk_size = 1024 * 10  # 10 KB chunks
    buffer = BytesIO()

    with tqdm(
        desc="Downloading HPSA data file",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            buffer.write(chunk)
            bar.update(len(chunk))

    buffer.seek(0)  # Reset buffer position

    # Read Excel file directly from memory
    df = pd.read_excel(buffer, engine="openpyxl")
    
    # Ensure columns have no spaces
    df.columns = df.columns.str.replace(" ", "_", regex=False)
    
    return df


def clean_hpsa_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and standardizes HPSA dataset in a memory-efficient way."""
    
    # Normalize column names
    df.columns = df.columns.str.replace(" ", "_")

    # Select relevant columns using .loc[] (avoids full DataFrame copy)
    columns_to_keep = [
        "HPSA_Name", "HPSA_ID", "Designation_Type", "HPSA_Score", "HPSA_Address",
        "HPSA_City", "State_Abbreviation", "Common_State_County_FIPS_Code",
        "HPSA_Postal_Code", "Longitude", "Latitude"
    ]
    df = df.loc[:, columns_to_keep]

    # Extract ZIP codes (handling "XXXXX-XXXX" formats)
    df["HPSA_Postal_Code"] = df["HPSA_Postal_Code"].astype(str, copy=False).str.extract(r"(\d+)-?\d*", expand=False)

    # Convert address components to strings without unnecessary copies
    for col in ["HPSA_Address", "HPSA_City", "State_Abbreviation", "HPSA_Postal_Code"]:
        df[col] = df[col].fillna("").astype(str, copy=False)

    # Construct full address efficiently
    df["Address"] = df.apply(lambda x: f"{x.HPSA_Address}, {x.HPSA_City}, {x.State_Abbreviation} {x.HPSA_Postal_Code}".strip(", "), axis=1)

    # Remove duplicates **in-place** to save memory
    df.drop_duplicates(inplace=True)

    # Assign HPSA type
    df["Type"] = "HPSA " + df["Designation_Type"]

    # Rename columns with inplace=True to avoid unnecessary copies
    df.rename(columns={
        "HPSA_Name": "Name",
        "Common_State_County_FIPS_Code": "FIPS",
        "State_Abbreviation": "State",
        "Longitude": "longitude",
        "Latitude": "latitude"
    }, inplace=True)

    return df

def filter_hpsa_data(df: pd.DataFrame, location: Union[str, List[str]]) -> pd.DataFrame:
    """Filters the HPSA dataset based on location and designation criteria in a memory-efficient way."""
    
    # Ensure location is a list
    location = [location] if isinstance(location, str) else location

    # Use `.query()` for filtering, but avoid unnecessary `.copy()` except where necessary
    return df.query(
        "Primary_State_Abbreviation in @location & "
        "HPSA_Status == 'Designated' & "
        "Designation_Type != 'Federally Qualified Health Center'"
    ).reset_index(drop=True)


def fetch_mammography_data():
    """Downloads and extracts the mammography facility dataset from the FDA website with a progress bar."""
    
    url = "http://www.accessdata.fda.gov/premarket/ftparea/public.zip"

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
        response.raise_for_status()  # Raises an error if the request fails

        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024 * 10  # 10 KB chunks
        buffer = BytesIO()

        # Download with tqdm progress bar
        with tqdm(
            desc="Downloading Mammography data file",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                buffer.write(chunk)
                bar.update(len(chunk))

        buffer.seek(0)  # Reset buffer position

        # Extract the ZIP file from memory
        with ZipFile(buffer) as zip_file:
            file_name = zip_file.namelist()[0]  # Get the first file in the ZIP
            with zip_file.open(file_name) as file:
                df = pd.read_csv(file, delimiter="|", header=None, dtype=str)

        # Ensure proper column naming
        df.columns = ['Name', 'Street', 'Street2', 'Street3', 'City', 'State', 'Zip_code', 'Phone_number', 'Fax']

        return df
    
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch or process the file: {e}")

def format_phone_number(phone: str) -> str:
    """
    Formats a phone number into (XXX) XXX-XXXX with an optional extension.
    
    - Handles separators: `-`, `.`, `space`
    - Handles country code `+1`, `1`
    - Handles extensions: `x`, `ext`, `#`
    - Leaves invalid numbers unchanged
    """
    phone = phone.strip()  # Remove leading/trailing spaces

    # Remove country code if present (e.g., "+1", "1-")
    phone = re.sub(r"^(?:\+1\s*|1-?)", "", phone)

    # Extract extension (if any)
    match_ext = re.search(r"(?:x|ext\.?|#)\s*(\d+)$", phone, re.IGNORECASE)
    ext = match_ext.group(1) if match_ext else None
    phone = re.sub(r"(?:x|ext\.?|#)\s*\d+$", "", phone, re.IGNORECASE)  # Remove extension from main number

    # Remove all non-numeric characters
    phone = re.sub(r"[^\d]", "", phone)

    # Ensure valid 10-digit number
    if len(phone) == 10:
        formatted = f"({phone[:3]}) {phone[3:6]}-{phone[6:]}"
        if ext:
            formatted += f" ext. {ext}"
        return formatted

    return phone  # Return as-is if no match

def download_fqhc_data() -> str:
    """
    Downloads the FQHC dataset and saves it as a temporary CSV file.
    Returns the filename of the downloaded CSV.
    """
    url = "https://data.hrsa.gov/DataDownload/DD_Files/Health_Center_Service_Delivery_and_LookAlike_Sites.csv"
    fname = os.path.join(os.getcwd(), "fqhc.csv")

    try:
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download FQHC data: {e}")

    total_size = int(resp.headers.get("content-length", 0))
    chunk_size = 1024 * 10  # 10 KB chunks

    with open(fname, "wb") as file, tqdm(
        desc="Downloading FQHC data file",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=True
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            size = file.write(chunk)
            bar.update(size)

    return fname

def process_fqhc_data(fname: str, location: Union[str, List[str]]) -> pd.DataFrame:
    """
    Reads the downloaded FQHC CSV file, filters the data, and returns a cleaned DataFrame.
    """
    col_mapping = {
        "Health_Center_Type": "Health Center Type",
        "Site_Name": "Site Name",
        "Site_Address": "Site Address",
        "Site_City": "Site City",
        "Site_State_Abbreviation": "Site State Abbreviation",
        "Site_Postal_Code": "Site Postal Code",
        "Site_Telephone_Number": "Site Telephone Number",
        "Health_Center_Service_Delivery_Site_Location_Setting_Description": "Health Center Service Delivery Site Location Setting Description",
        "Geocoding_Artifact_Address_Primary_X_Coordinate": "Geocoding Artifact Address Primary X Coordinate",
        "Geocoding_Artifact_Address_Primary_Y_Coordinate": "Geocoding Artifact Address Primary Y Coordinate"
    }

    # Convert location to a list of uppercase values
    location = [location.upper()] if isinstance(location, str) else [loc.upper() for loc in location]

    # Read CSV file efficiently
    data_dict = {col: [] for col in col_mapping.keys()}
    
    with open(fname, newline='', encoding='utf8') as csvfile:
        reader = DictReader(csvfile)

        for row in reader:
            if (
                row["Site State Abbreviation"].upper() in location and
                row["Health Center Type"] == "Federally Qualified Health Center (FQHC)" and
                row["Site Status Description"] == "Active"
            ):
                for dict_key, row_key in col_mapping.items():
                    data_dict[dict_key].append(row[row_key])

    # Convert to DataFrame
    df = pd.DataFrame(data_dict)

    # Delete dictionary to free memory
    del data_dict

    # Remove temporary file
    os.remove(fname)

    return df


def parse_basic(basic):
    if 'organization_name' in basic.keys():
        name = basic['organization_name'].title()
    else:
        if 'middle_name' in basic.keys():
            name = basic['first_name'].title() + ' ' + basic['middle_name'][0].upper() + ' ' + basic['last_name'].title()
        else:
            name = basic['first_name'].title() + ' ' + basic['last_name'].title()
        if 'credential' in basic.keys():
            name = name + ' ' + basic['credential'].upper()
    return name


def parse_address(address):
    address_dict = [x for x in address if x['address_purpose'] == 'LOCATION'][0]
    if 'address_2' in address_dict.keys():
        street = address_dict['address_1'].title() + ', ' + address_dict['address_2'].title() + ', ' + address_dict['city'].title() + ', ' + address_dict['state'].upper() 
    else:
        street = address_dict['address_1'].title() + ', ' + address_dict['city'].title() + ', ' + address_dict['state'].upper()
    if 'postal_code' in address_dict.keys():
        street += ' '
        street += address_dict['postal_code'][:5]
    if 'telephone_number' in address_dict.keys():
        phone_number = address_dict['telephone_number']
    else:
        phone_number = None
    state = address_dict['state']
    return street, phone_number, state



# Taxonomy mappings
taxonomy_names = {
    "Gastroenterology": "Gastroenterology",
    "colon": "Colon & Rectal Surgeon",
    "obstetrics": "Obstetrics & Gynecology",
    "Hematology%20&%20Oncology": "Hematology & Oncology",
    "Medical%20Oncology": "Medical Oncology",
    "Gynecologic%20Oncology": "Gynecologic Oncology",
    "Pediatric%20Hematology-Oncology": "Pediatric Hematology-Oncology",
    "Radiation%20Oncology": "Radiation Oncology",
    "Surgical%20Oncology": "Surgical Oncology",
}

async def fetch_nppes(session, taxonomy: str, location: str, skip: int) -> dict:
    """
    Asynchronously fetch provider data from the NPI Registry API.
    """
    url = f"https://npiregistry.cms.hhs.gov/api/?version=2.1&address_purpose=LOCATION&number=&state={location}&taxonomy_description={taxonomy}&skip={skip}&limit=200"

    async with session.get(url) as response:
        return await response.json()

async def parse_nppes_response(output: dict, taxonomy: str, location: str) -> pd.DataFrame:
    """
    Parses the API response into a structured DataFrame.
    """
    if 'result_count' not in output or output['result_count'] == 0:
        return pd.DataFrame(columns=["Type", "Name", "Address", "State", "Phone_number", "Notes"])

    df = pd.DataFrame(output["results"])

    # Parse name, address, phone, and state
    df["Name"] = df["basic"].apply(parse_basic)
    df["Phone_number"] = df["addresses"].apply(lambda x: parse_address(x)[1])
    df["Address"] = df["addresses"].apply(lambda x: parse_address(x)[0])
    df["State"] = df["addresses"].apply(lambda x: parse_address(x)[2])

    # Keep only rows that match the requested state
    df = df[df["State"] == location]

    # Assign taxonomy type
    df["Type"] = taxonomy_names.get(taxonomy, taxonomy)

    # Handle missing ZIP codes
    df["Notes"] = df["Address"].apply(lambda x: "" if x[-5:].isnumeric() else "missing zip code")

    return df[["Type", "Name", "Address", "State", "Phone_number", "Notes"]]

async def gen_nppes_by_taxonomy(session, taxonomy: str, location: str) -> pd.DataFrame:
    """
    Fetches and processes provider data for a given taxonomy and location.
    """
    count, result_count = 0, 200
    datasets = []

    while result_count == 200:
        count += 1
        output = await fetch_nppes(session, taxonomy, location, skip=200 * (count - 1))

        if "result_count" in output:
            result_count = output["result_count"]

            if result_count:
                df = await parse_nppes_response(output, taxonomy, location)
                datasets.append(df)

            # Stop fetching if results repeat
            if count > 1 and datasets[-1].equals(datasets[-2]):
                break

    if datasets:
        return pd.concat(datasets, axis=0).drop_duplicates().reset_index(drop=True)
    return pd.DataFrame(columns=["Type", "Name", "Address", "State", "Phone_number", "Notes"])



async def fetch_nppes_data(location: Union[str, List[str]], taxonomy_list: List[str]) -> pd.DataFrame:
    """
    Orchestrates multiple API requests using asyncio for all taxonomies and locations.
    """
    async with aiohttp.ClientSession() as session:
        tasks = []

        if isinstance(location, str):
            tasks = [gen_nppes_by_taxonomy(session, taxonomy, location) for taxonomy in taxonomy_list]
        else:
            tasks = [gen_nppes_by_taxonomy(session, taxonomy, loc) for taxonomy in taxonomy_list for loc in location]

        results = await asyncio.gather(*tasks)

    return pd.concat(results, axis=0).reset_index(drop=True)


def lung_cancer_screening_file_download(chrome_driver_path=None) -> bool:
    """
    Attempts to download the lung cancer screening dataset.
    - First, tries using `requests`.
    - If it fails, falls back to Selenium.
    """
    url = 'https://report.acr.org/t/PUBLIC/views/NRDRLCSLocator/ACRLCSDownload.csv'
    filename = './ACRLCSDownload.csv'

    # Try downloading via requests first
    try:
        resp = requests.get(url, stream=True, timeout=10)
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        chunk_size = 1024

        with open(filename, 'wb') as f, tqdm(
            desc="Downloading LCS data file",
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=True
        ) as bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = f.write(data)
                bar.update(size)

        return False  # No Selenium needed

    except requests.RequestException as e:
        print(f"Failed to download via requests, falling back to Selenium: {e}")

    # Use Selenium as a backup method
    chromeOptions = ChromeOptions()
    prefs = {"download.default_directory": os.getcwd()}
    chromeOptions.add_experimental_option("prefs", prefs)
    chromeOptions.add_argument(f"download.default_directory={os.getcwd()}")

    if chrome_driver_path is None:
        chrome_driver_path = setup_chrome_driver()
    else:
        chrome_driver_path = None
    driver = webdriver.Chrome(service=Service(chrome_driver_path), options=chromeOptions)

    # Open webpage
    driver.get("https://report.acr.org/t/PUBLIC/views/NRDRLCSLocator/LCSLocator?:embed=y&:showVizHome=no")
    time.sleep(10)

    try:
        # Locate and click download buttons
        state_dropdown = driver.find_elements(By.CLASS_NAME, 'tabComboBoxButtonHolder')[2]
        state_dropdown.click()
        time.sleep(5)

        select_all_option = driver.find_elements(By.CLASS_NAME, 'tabMenuItemNameArea')[1]
        select_all_option.click()
        time.sleep(5)

        download_button = driver.find_element(By.ID, 'tabZoneId422')
        download_button.click()

        # Wait for the file to appear
        for _ in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            if glob.glob('./ACRLCSDownload*.csv'):
                print("LCSR data ready")
                break
        else:
            raise TimeoutError("Download failed.")

    except Exception as e:
        print(f"Selenium failed: {e}")
    finally:
        driver.quit()

    if chrome_driver_path is not None:
        remove_chromedriver(chrome_driver_path)


    return True  # Selenium was used

def process_lcs_data(file_path: str, location: Union[str, List[str]]) -> pd.DataFrame:
    """
    Processes the downloaded lung cancer screening dataset.
    - Reads the CSV efficiently with `pandas.read_csv()`.
    - Filters by location and ensures column names match the expected format.
    """
    df = pd.read_csv(file_path, dtype=str)

    # Preprocess column names (remove leading numbers and spaces)
    df.columns = [re.sub(r"^\d+\s*", "", col).strip() for col in df.columns]

    # Rename columns dynamically
    column_mapping = {}
    for col in df.columns:
        if re.search(r"name", col, re.I):
            column_mapping[col] = "Name"
        elif re.search(r"street|address", col, re.I):
            column_mapping[col] = "Street"
        elif re.search(r"city", col, re.I):
            column_mapping[col] = "City"
        elif re.search(r"state", col, re.I):
            column_mapping[col] = "State"
        elif re.search(r"zip", col, re.I):
            column_mapping[col] = "Zip_code"
        elif re.search(r"phone|contact", col, re.I):
            column_mapping[col] = "Phone"

    # Apply renaming
    df.rename(columns=column_mapping, inplace=True)

    # Ensure all required columns exist before proceeding
    required_columns = ["Name", "Street", "City", "State", "Zip_code", "Phone"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""  # Create empty column if missing

    # Construct Address column
    df["Address"] = df["Street"].fillna("").str.title() + ", " + df["City"].fillna("").str.title() + ", " + df["State"].fillna("") + " " + df["Zip_code"].fillna("").str[:5]
    
    # Filter by location
    if isinstance(location, str):
        df = df[df["State"] == location]
    else:
        df = df[df["State"].isin(location)]

    # Select and rename final columns to match the old version
    df = df[["Name", "Address", "State", "Phone"]].rename(columns={"Phone": "Phone_number"})
    df["Type"] = "Lung Cancer Screening"
    df["Notes"] = ""

    # Ensure final column order matches the original version
    df = df[['Type', 'Name', 'Address', 'State', 'Phone_number', 'Notes']]

    return df


def find_repo_root():
    """
    Dynamically finds the repository root by searching for a common marker file (e.g., .git).
    """
    current_dir = os.getcwd()
    while current_dir != os.path.dirname(current_dir):  # Stop at root ("/" or "C:/")
        if os.path.exists(os.path.join(current_dir, ".git")):
            return current_dir
        current_dir = os.path.dirname(current_dir)  # Move one level up
    raise FileNotFoundError("Repository root not found. Ensure you are inside a Git repository.")


def download_toxRel_data() -> str:
    """
    Downloads the most recent Toxics Release Inventory (TRI) dataset from the EPA website.
    - Ensures we get the latest available year.
    - Returns the path to the downloaded CSV file.
    """
    today = datetime.date.today()
    year = today.year
    file_path = os.path.join(os.getcwd(), "toxRel.csv")

    # Try downloading the most recent dataset
    while year >= 2000:  # Limit to reasonable years to prevent infinite loop
        url = f"https://data.epa.gov/efservice/downloads/tri/mv_tri_basic_download/{year}_US/csv"
        try:
            resp = requests.get(url, stream=True, timeout=10)
            resp.raise_for_status()
            break  # Exit loop if download succeeds
        except requests.RequestException:
            year -= 1  # Try previous year

    # Download file with a progress bar
    total_size = int(resp.headers.get("content-length", 0))
    chunk_size = 512 * 1024  # 512 KB

    with open(file_path, "wb") as file, tqdm(
        desc="Downloading toxRel data file",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=True
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            file.write(chunk)
            bar.update(len(chunk))

    return file_path

def extract_columns(file_path: str) -> tuple:
    """
    Reads the CSV file and extracts relevant column mappings dynamically.
    - Uses regex to match column names without hardcoding.
    - Returns a tuple of (column mappings, additional address mappings).
    """
    with open(file_path, newline='', encoding="utf-8") as csvfile:
        reader = DictReader(csvfile)
        # logging: print column names
        colnames = ['FRS ID', 'FACILITY NAME', 'LATITUDE', 'LONGITUDE', 'COUNTY', 'CHEMICAL', 'ST']
        temp_col = ['STREET ADDRESS', 'CITY', 'ST', 'ZIP', 'CARCINOGEN']

        # Find actual column names dynamically
        csv_keys = {col: field for col, field in product(colnames, reader.fieldnames) if re.match(r"\d+\.\s" + col + "$", field, flags=re.I)}
        temp_fields = {col: field for col, field in product(temp_col, reader.fieldnames) if re.match(r"\d+\.\s" + col + "$", field, flags=re.I)}

    return csv_keys, temp_fields

def filter_toxRel_data(file_path: str, location: Union[str, List[str]], csv_keys: dict, temp_fields: dict) -> pd.DataFrame:
    """
    Filters the TRI dataset based on location and carcinogenic chemicals.
    - Constructs full address fields.
    - Returns a cleaned Pandas DataFrame.
    """
    with open(file_path, newline='', encoding="utf-8") as csvfile:
        reader = DictReader(csvfile)

        # Data dictionary for storing extracted values
        data_dict = {key: [] for key in csv_keys}
        data_dict["Address"] = []

        # Ensure location is a list
        location_list = [location.upper()] if isinstance(location, str) else [loc.upper() for loc in location]

        for row in reader:
            if row[temp_fields["ST"]].upper() in location_list and row[temp_fields["CARCINOGEN"]] == "YES":
                address = f"{row[temp_fields['STREET ADDRESS']].title()}, {row[temp_fields['CITY']].title()}, {row[temp_fields['ST']].upper()} {row[temp_fields['ZIP']]}"
                data_dict["Address"].append(address)

                for dict_key, row_key in csv_keys.items():
                    data_dict[dict_key].append(row[row_key])

    df = pd.DataFrame(data_dict)
    # Group by facility and aggregate chemicals
    df = df.groupby(
        ["FRS ID", "FACILITY NAME", "Address", "LATITUDE", "LONGITUDE", "COUNTY", "ST"]
    )["CHEMICAL"].agg(lambda col: ", ".join(col)).reset_index()

    # Add notes and rename columns
    df["Notes"] = "Chemicals released: " + df["CHEMICAL"]
    df = df.rename(columns={
        "FACILITY NAME": "Name",
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
        "ST": "State"
    })

    # Standard output structure
    df["Type"] = "Toxic Release Inventory Facility"
    df["Phone_number"] = None

    return df[["Type", "Name", "Address", "State", "Phone_number", "Notes", "latitude", "longitude"]]

