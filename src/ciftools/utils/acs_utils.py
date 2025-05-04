import asyncio
import pandas as pd
import functools
import re
import sys
import os
from functools import partial
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional
from aiohttp import ClientSession, TCPConnector

# Ensure parent directories are in system path for module access
from ..config.acs_config import ACSConfig  # Import ACSConfig for configuration handling
from .ciftools_logger import logger
# -------------------------------------
# 📌 AGE GROUPING UTILITIES
# -------------------------------------

# Define standard age groups
TEN_YEAR_AGE_GROUPS: Dict[str, Tuple[int, int]] = {
    "Under 5 years": (0, 4),
    "5 to 14 years": (5, 14),
    "15 to 24 years": (15, 24),
    "25 to 34 years": (25, 34),
    "35 to 44 years": (35, 44),
    "45 to 54 years": (45, 54),
    "55 to 64 years": (55, 64),
    "65 to 74 years": (65, 74),
    "75 to 84 years": (75, 84),
    "85 years and over": (85, 100),
}

LARGE_AGE_GROUPS: Dict[str, Tuple[int, int]] = {
    "Under 18": (0, 17),
    "18 to 64": (18, 64),
    "Over 64": (65, 100),
}


def extract_age_range(text: str) -> Tuple[int, int]:
    """
    Extracts the numerical age range from a given ACS text label.

    Parameters:
        text (str): Age label from ACS data.

    Returns:
        Tuple[int, int]: Minimum and maximum age from the label.
    """
    numbers: List[int] = list(map(int, re.findall(r"\d+", text)))
    if "Under" in text:
        return (0, numbers[0])
    elif "over" in text or "and over" in text:
        return (numbers[0], 100)
    elif len(numbers) == 1:
        return (numbers[0], numbers[0])
    return tuple(numbers)


def find_index_for_age_group(
    age_group_dict: Dict[str, Tuple[int, int]],
    config: Optional[ACSConfig] = None,
    **kwargs,
) -> Dict[str, List[int]]:
    """
    Maps ACS age labels to predefined age groups and returns index positions.

    Parameters:
        age_group_dict (Dict[str, Tuple[int, int]]): Mapping of age groups to (min_age, max_age).
        config (ACSConfig, optional): ACS configuration containing `labels`.
        kwargs: Additional arguments if `config` is not provided.

    Returns:
        Dict[str, List[int]]: Dictionary with age group names as keys and index positions as values.
    """
    if not config:
        config = ACSConfig(**kwargs)

    def is_within_range(
        test_range: Tuple[int, int], group_range: Tuple[int, int]
    ) -> bool:
        """Returns True if `test_range` fully fits within `group_range`."""
        return group_range[0] <= test_range[0] and group_range[1] >= test_range[1]

    # Extract age labels and their index positions
    parsed_labels: List[Tuple[Tuple[int, int], int]] = [
        (extract_age_range(label), config.labels.index(label))
        for label in config.labels
        if "years" in label
    ]

    # Organize indices by age group
    index_by_age_group: Dict[str, List[int]] = defaultdict(list)
    for age_range, index in parsed_labels:
        matched_group: Optional[str] = next(
            (
                group
                for group, bounds in age_group_dict.items()
                if is_within_range(age_range, bounds)
            ),
            None,
        )
        if matched_group:
            index_by_age_group[matched_group].append(index)

    return dict(index_by_age_group)


# Create pre-configured functions for standard age groups
find_large_age_groups = partial(
    find_index_for_age_group, age_group_dict=LARGE_AGE_GROUPS
)
find_ten_year_age_groups = partial(
    find_index_for_age_group, age_group_dict=TEN_YEAR_AGE_GROUPS
)


# -------------------------------------
# 📌 ACS API HANDLING FUNCTIONS
# -------------------------------------


def batchify_variables(config: ACSConfig) -> List[List[str]]:
    """
    Splits ACS variable lists into batches of 49 (API limitation).

    Parameters:
        config (ACSConfig): Configuration containing variable list.

    Returns:
        List[List[str]]: List of variable sub-lists.
    """
    batch_size: int = 49
    table: List[str] = config.variables
    if len(table) > batch_size:
        num_full: int = len(table) // batch_size
        return [
            table[i * batch_size : (i + 1) * batch_size] for i in range(num_full)
        ] + [table[num_full * batch_size :]]
    return [table]


async def download_for_batch(
    config, table: str, key: str, session: ClientSession
) -> List[str]:
    logger.info(f"Downloading {table} for {config.state_fips}")
    if config.acs_type == "":
        source = "acs/acs5"
    else:
        source = f"acs/acs5/{config.acs_type}"

    #     table = ','.join(batchify_variables(config)[0])
    if isinstance(config.state_fips, str) or isinstance(config.state_fips, int):
        if config.query_level == "state":
            acs_url = f"https://api.census.gov/data/{config.year}/{source}?get={table}&for=state:{config.state_fips}&key={key}"
        elif config.query_level == "county":
            acs_url = f"https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=county:*&in=state:{config.state_fips}&key={key}"
        elif config.query_level == "county subdivision":
            acs_url = f"https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=county%20subdivision:*&in=state:{config.state_fips}&in=county:*&key={key}"
        elif config.query_level == "tract":
            acs_url = f"https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=tract:*&in=state:{config.state_fips}&in=county:*&key={key}"
        elif config.query_level == "block":
            acs_url = f"https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=block%20group:*&in=state:{config.state_fips}&in=county:*&in=tract:*&key={key}"
        elif config.query_level == "zip":
            acs_url = f"https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=zip%20code%20tabulation%20area:*&in=state:{config.state_fips}&key={key}"
        elif config.query_level == "puma":
            acs_url = f"https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=public%20use%20microdata%20area:*&in=state:{config.state_fips}&key={key}"
        else:
            raise ValueError(
                "The region level is not found in the system; select among state, county, county subdivision, tract, block, zip and puma"
            )
    elif isinstance(config.state_fips, list):
        config.state_fips = [str(x) for x in config.state_fips]
        states = ",".join(config.state_fips)
        if config.query_level == "state":
            acs_url = f"https://api.census.gov/data/{config.year}/{source}?get={table}&for=state:{states}&key={key}"
        elif config.query_level == "county":
            acs_url = f"https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=county:*&in=state:{states}&key={key}"
        elif config.query_level == "county subdivision":
            acs_url = f"https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=county%20subdivision:*&in=state:{states}&in=county:*&key={key}"
        elif config.query_level == "tract":
            acs_url = f"https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=tract:*&in=state:{states}&in=county:*&key={key}"
        elif config.query_level == "block":
            acs_url = f"https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=block%20group:*&in=state:{states}&in=county:*&in=tract:*&key={key}"
        elif config.query_level == "zip":
            acs_url = f"https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=zip%20code%20tabulation%20area:*&in=state:{states}&key={key}"
        elif config.query_level == "puma":
            acs_url = f"https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=public%20use%20microdata%20area:*&in=state:{states}&key={key}"
        else:
            raise ValueError(
                "The region level is not found in the system; select among state, county, county subdivision, tract, block, zip and puma"
            )
    async with session.get(acs_url) as resp:
        resp.raise_for_status()
        json_raw = await resp.json()
    return json_raw


async def download_all(config: ACSConfig, key: str) -> List[List[str]]:
    """
    Downloads all ACS data asynchronously using batch requests.

    Parameters:
        config (ACSConfig): ACS configuration.
        key (str): Census API key.

    Returns:
        List[List[str]]: List of JSON responses.
    """
    tables: List[List[str]] = batchify_variables(config)
    connector: TCPConnector = TCPConnector(limit=10)  # Limit concurrent requests
    async with ClientSession(connector=connector) as session:
        tasks = [
            download_for_batch(config, ",".join(table), key, session)
            for table in tables
        ]
        return await asyncio.gather(*tasks)


def acs_data(key: str, config: Optional[ACSConfig] = None, **kwargs) -> pd.DataFrame:
    """
    Retrieves and processes ACS data into a DataFrame.

    Parameters:
        key (str): Census API key.
        config (ACSConfig, optional): ACS configuration.
        kwargs: Additional config parameters if not provided.

    Returns:
        pd.DataFrame: Processed ACS data.
    """
    if not config:
        config = ACSConfig(**kwargs)

    # Run the download function asynchronously
    result: List[List[str]] = asyncio.run(download_all(config, key))

    # Convert JSON response into a DataFrame
    df_list: List[pd.DataFrame] = [
        pd.DataFrame(res[1:], columns=res[0]) for res in result if res
    ]
    if not df_list:
        return pd.DataFrame()

    # Merge all DataFrames on non-numeric columns
    merge_columns: List[str] = (
        df_list[0].columns[df_list[0].columns.str.isalpha()].tolist()
    )
    df: pd.DataFrame = functools.reduce(
        lambda left, right: left.merge(right, how="inner", on=merge_columns), df_list
    )

    # Rename and sort for consistency
    if config.query_level == "puma":
        df.rename(columns={"public use microdata area": "PUMA"}, inplace=True)

    return df


def custom_acs_data(key, config=None, **kwargs):
    def decorator_transform(func, key=key, config=config):
        if config:
            pass
        else:
            config = ACSConfig(**kwargs)

        @functools.wraps(func)
        def wrapper(**kwargs):
            df = acs_data(key, config)
            df = func(df)
            return df

        return wrapper

    return decorator_transform
