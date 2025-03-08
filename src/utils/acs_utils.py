import asyncio
import pandas as pd
from aiohttp import ClientSession

async def fetch_acs_groups(year: int, acs_type: str, session: ClientSession) -> list:
    """
    Fetch ACS group names, descriptions, and ACS type from the Census API.
    
    Args:
        year (int): The year for the ACS data.
        acs_type (str): The type of ACS data ('', 'profile', or 'subject').
        session (ClientSession): An active aiohttp session.
    
    Returns:
        list: A list of dictionaries containing ACS group information.
    """
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    url = f"{base_url}/{acs_type}/groups" if acs_type else f"{base_url}/groups"
    
    async with session.get(url) as response:
        response.raise_for_status()
        json_data = await response.json()
    
    return [
        {"name": group["name"], "description": group["description"], "acs_type": acs_type}
        for group in json_data["groups"]
    ]

async def fetch_all_acs_groups(year: int) -> list:
    """
    Fetch ACS groups for all types ('', 'profile', 'subject') concurrently.
    
    Args:
        year (int): The year for the ACS data.
    
    Returns:
        list: A combined list of dictionaries with group names, descriptions, and ACS type.
    """
    async with ClientSession() as session:
        tasks = [fetch_acs_groups(year, acs_class, session) for acs_class in ["", "profile", "subject"]]
        results = await asyncio.gather(*tasks)
    
    return [group for result in results for group in result]  # Flatten the results list

def gen_group_names_acs(config):
    """
    Retrieve available ACS group names, their descriptions, and ACS type based on a given ACSConfig instance.
    
    Args:
        config (ACSConfig): A configuration object containing the ACS year and group(s).
    
    Returns:
        tuple: (
            list of group names,
            list of metadata [['name', 'description'], ['B01001', 'Population by Age'], ...],
            acs_type (str)
        )
    """
    year = config.year
    groups_data = asyncio.run(fetch_all_acs_groups(year))
    
    # Create a lookup dictionary for quick access
    group_lookup = {group["name"]: group for group in groups_data}
    
    # Convert single group string to a list for uniform handling
    if isinstance(config.acs_group, str):
        config.acs_group = [config.acs_group]
    
    # Identify any missing groups
    missing_groups = [g for g in config.acs_group if g not in group_lookup]
    if missing_groups:
        raise AttributeError(f"Invalid ACS group ID(s): {missing_groups}")
    
    # Ensure all groups belong to the same ACS type
    acs_types = {group_lookup[g]["acs_type"] for g in config.acs_group}
    if len(acs_types) > 1:
        raise AttributeError("All groups must belong to the same ACS type.")
    
    acs_type = acs_types.pop()  # Extract the single ACS type
    
    # Build output lists
    group_names = config.acs_group
    group_metadata = [["name", "description"]] + [
        [group_lookup[g]["name"], group_lookup[g]["description"]] for g in config.acs_group
    ]
    
    return group_names, group_metadata, acs_type


# testing
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add parent directory to path
    # add src to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # add src to path


    from config.acs_config import ACSConfig
    config = ACSConfig(year=2021, acs_group=["B01001", "B01002"], state_fips="06", query_level="county")
    group_names, metadata, acs_type = gen_group_names_acs(config)
    print(group_names)
    print(metadata)
    print(acs_type)