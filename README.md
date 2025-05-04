# Cancer In Focus Tools (CIFTools) - Data Processing

This tool collects, processes, and integrates various publicly available datasets relevant to cancer research and public health, focusing on specific geographic areas. It uses a YAML configuration file to specify the desired datasets, geographic scope, and other parameters.

## Overview

The primary script, `process_data.py` (assumed entry point), orchestrates the data collection pipeline. It reads a configuration file, identifies the target geographic area (counties based on state FIPS codes), and fetches data from multiple sources, including:

*   American Community Survey (ACS)
*   Healthcare Facilities (NPPES, Mammography, HPSA, FQHC, etc.)
*   State Cancer Profiles (SCP)
*   Bureau of Labor Statistics (BLS) Unemployment Data
*   USDA Food Desert Atlas
*   FCC Broadband Availability
*   NCHS Urban-Rural Classification
*   EPA EJScreen Environmental Justice Data
*   CDC PLACES Health Data
*   CDC Social Vulnerability Index (SVI)

The collected data is then processed and potentially merged, organized by geographic level (county, tract, facility).

## Prerequisites

1.  **Python Environment:** Ensure you have Python 3.8+ installed. It's recommended to use a virtual environment.
2.  **Dependencies:** Install the required Python packages. Navigate to the root directory of the project (containing `setup.py` or `requirements.txt`) and run:
    ```bash
    pip install .
    # or if a requirements.txt file exists:
    # pip install -r requirements.txt
    ```
    *Note: You might need to install `openpyxl` specifically if not included: `pip install openpyxl`*
3.  **Environment Variables:** Certain data sources require API keys or specific user agents/tokens. Create a `.env` file in the project's root directory and set the following environment variables. **All listed variables are necessary if you intend to collect data from the corresponding sources.**
    *   `CENSUS_API_KEY`: Your Census API key (Required for ACS data). Get one here.
    *   `BLS_USER_AGENT`: An email address or unique identifier for BLS API requests (Required for BLS data). Example: `your.email@example.com`.
    *   `FCC_USERNAME`: Username for the FCC Broadband Map data (visit https://broadbandmap.fcc.gov/home)
    *   `FCC_USER_AGENT`: FCC User agent for the FCC Broadband Map data 
    *   `FCC_HASH_VALUE`: Has Value for the FCC Broadband Map data
    *   `SOCRATA_APP_TOKEN`: Your Socrata App Token (Required for CDC PLACES and CDC SVI data). Get one by signing up on the respective data portal (e.g., [socrata](https://opendata.socrata.com/login)).
    *   `SOCRATA_CDC_USERNAME`: Your Socrata username (Required for CDC PLACES and CDC SVI data).
    *   `SOCRATA_CDC_PASSWORD`: Your Socrata password (Required for CDC PLACES and CDC SVI data).
    *   `SOCRATA_CDC_COUNTY`: Socrata dataset id for cdc places in county level
    *   `SOCRATA_CDC_TRACT`: Socrata dataset id for cdc places in tract level


## Configuration (YAML File)

The data collection process is controlled by a YAML configuration file. This file specifies the target area, the datasets to fetch, and any specific parameters for those datasets.

**YAML Structure Example (`config.yaml`):**

```yaml
# 1. Define the Catchment Area
area:
  type: catchment # Or other types if supported
  # Path to a CSV file containing a 'FIPS' column with 5-digit county FIPS codes
  area_file: path/to/your/county_fips_list.csv

# 2. Configure ACS Data Collection
acs:
  # List the geographic levels ('county', 'tract')
  query_level: 
    - county
    - tract
  # Specify the ACS 5-Year estimate year
  acs_year: 2022

# 3. Configure Facility Data Collection
facilities:
  # List desired facility types or use 'all'
  facility_types:
    - nppes
    - mammography
    - fqhc
    # - hpsa
    # - lung_cancer_screening
    # - tri_facility
    # - superfund
  # Optional: Specify taxonomy codes for NPPES filtering
  # taxonomy: [ "208D00000X", "..." ]

# 4. Enable/Disable other data sources (use true/false)
cancer: true       # State Cancer Profiles data
bls: true          # BLS County Unemployment (most recent month)
food_desert: true  # USDA Food Access Research Atlas data (County & Tract)
fcc: true          # FCC Broadband Availability (County)
urban_rural: true  # NCHS Urban-Rural Classification (County)
ejscreen: true     # EPA EJScreen Data (Tract)
cdc_places: true   # CDC PLACES Local Data for Better Health (County & Tract)
cdc_svi: true      # CDC Social Vulnerability Index (County & Tract)

# 5. Specify Output/Download Directory (Optional - behavior depends on script implementation)
# download_dir: /path/to/save/outputs

## Usage

The primary way to use this tool is via the command-line interface provided by `src/cli/collect_data.py`.

1.  **Prepare your Configuration:** Create or modify a YAML configuration file (e.g., `config.yaml`) as described in the "Configuration (YAML File)" section above. Ensure the `area_file` path is correct and you have set the necessary environment variables in your `.env` file.
2.  **Run the Script:** Open your terminal, navigate to the root directory of the `CIFTools` project (the directory containing the `src` folder), and execute the following command:

    ```bash
    python -m src.cli.collect_data --config path/to/your/config.yaml
    ```

    *   Replace `path/to/your/config.yaml` with the actual path to your configuration file.
    *   The `-m` flag tells Python to run the specified module (`src.cli.collect_data`) as a script.
    *   The `--config` argument passes the path of your YAML file to the script.

3.  **Execution Process:** The script will:
    *   Read the specified YAML configuration.
    *   Determine the state(s) involved from the `area_file`.
    *   Iterate through the data sources enabled (`true`) in the config file.
    *   Fetch data using the appropriate modules (like `BLS`, `FoodDesert`, `acs_sdoh`, etc.), displaying a progress bar.
    *   Log progress and potential errors to the console.
    *   Save the collected data.

## Output

By default, the `collect_data.py` script saves the collected data as a Python dictionary into a single file using `joblib`.

*   **File Name:** `ciftool_data.pickle`
*   **Location:** The directory specified by the `download_dir` key in your YAML configuration file. If `download_dir` is not specified or the path is invalid, saving might fail or default to the current working directory (behavior depends on implementation details).
*   **Format:** The `.pickle` file contains a Python dictionary where keys often represent geographic levels (`county`, `tract`, `facility`) and data source names (e.g., `bls_unemployment`, `food_desert`, `svi`). You can load this file back into Python using `joblib.load()`.

```python
# Example of loading the output data in Python
import joblib

output_file = "/path/to/your/download_dir/ciftool_data.pickle"
collected_data = joblib.load(output_file)

# Access county-level BLS data
# bls_df = collected_data['county']['bls_unemployment']
# print(bls_df.head())
