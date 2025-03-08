#%% setup dev env
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add parent directory to path
# add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # add src to path


#%%
from functools import cached_property
import pandas as pd
from dataclasses import dataclass
from typing import Union, List
from utils.acs_utils import gen_group_names_acs # dev env
# from ciftools.utils.acs_utils import gen_group_names_acs
from dotenv import load_dotenv, find_dotenv
import os

# Load environment variables
load_dotenv("/root/CIFTools/.env")
#%%
@dataclass
class ACSConfig:
    """Configuration for pulling ACS data from the Census API."""
    year: Union[str, int]
    state_fips: Union[str, int, List[str], List[int]]
    query_level: str        
    acs_group: Union[str, List[str]]
    acs_type: str = None  

    def __post_init__(self):
        """Validate `acs_group` and fetch all required metadata in a single request."""
        if isinstance(self.acs_group, str):
            self.acs_group = [self.acs_group]  # Convert to list for consistent handling
        
        if not isinstance(self.acs_group, list) or not all(isinstance(g, str) for g in self.acs_group):
            raise ValueError("acs_group must be a string or a list of strings representing ACS group IDs.")
        
        # ✅ Fetch ACS metadata in a single request
        self._fetch_group_metadata()

    def _fetch_group_metadata(self):
        """Fetch ACS variable names, descriptions, and ACS type in a single request."""
        res = gen_group_names_acs(self)
        if not res or len(res) < 3:
            raise ValueError("Error fetching ACS group metadata. Response format is unexpected.")

        # ✅ Assign all fetched values at once
        self.acs_type = res[2]  # ACS Type ('', 'profile', or 'subject')
        self._variables = res[0]  # List of variable names
        self._var_desc = pd.DataFrame(res[1][1:], columns=res[1][0])  # Metadata DataFrame

        # ✅ Validate that all requested groups exist in the fetched data
        existing_groups = set(self._var_desc["name"].tolist())
        missing_groups = [g for g in self.acs_group if g not in existing_groups]
        if missing_groups:
            raise AttributeError(f"Invalid ACS group ID(s): {missing_groups}")

    @cached_property
    def api_key(self) -> str:
        """Fetch the Census API key dynamically from environment variables."""
        return os.getenv("CENSUS_API_KEY", None)

    @cached_property
    def variables(self) -> List[str]:
        """Return pre-fetched list of variable names."""
        return self._variables

    @cached_property
    def var_desc(self) -> pd.DataFrame:
        """Return pre-fetched variable descriptions."""
        return self._var_desc

    @cached_property
    def labels(self) -> List[str]:
        """Generate human-readable labels for the selected variables."""
        if self.var_desc.empty:
            return []

        if "description" not in self._var_desc.columns:
            raise KeyError("Expected 'description' column is missing from variable metadata.")
        
        return self._var_desc["description"].tolist()

# %% test

if __name__ == "__main__":
    config  = ACSConfig(
        year=2021,
        state_fips="06",
        query_level="tract",
        acs_group=["B01001", "B01002"],
    )
    print(config.var_desc)
