#%% import packages
import pandas as pd
from typing import Union, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # add src to path


from utils.ciftools_logger import logger
from utils.states import stateDf

#%%




class svi_data:
    def __init__(self, state_fips: Union[str, List[str]]):
        self.state_fips = state_fips
        logger.info(f"Initialized svi_data with state_fips: {state_fips}")
    
    @property
    def state(self):
        if hasattr(self, '_state'):
            logger.debug("Using cached state data.")
        else:
            logger.info("Fetching state data based on FIPS codes.")
            if isinstance(self.state_fips, str):
                self._state = stateDf.loc[stateDf.FIPS2.eq(self.state_fips), 'State'].values[0]
                logger.info(f"State for FIPS {self.state_fips}: {self._state}")
            else:
                self._state = stateDf.loc[stateDf.FIPS2.isin(self.state_fips), 'State'].values.tolist()
                logger.info(f"States for FIPS {self.state_fips}: {self._state}")
        return self._state
    
    @property
    def svi_data(self):
        if hasattr(self, '_svi_data'):
            logger.debug("Using cached SVI data.")
        else:
            logger.info("Fetching SVI data for county and tract levels.")
            self._svi_data = {'county': self.svi_county(), 'tract': self.svi_tract()}
        return self._svi_data
    
    def svi_county(self):
        url = "https://svi.cdc.gov/Documents/Data/2022/csv/states_counties/SVI_2022_US_county.csv"
        logger.info(f"Downloading SVI county data from {url}")
        try:
            df = pd.read_csv(url, dtype={'ST': str, 'STCNTY': str})
            logger.info("SVI county data downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download SVI county data: {e}")
            raise
        
        logger.info("Processing SVI county data.")
        df = df[['STCNTY', 'COUNTY', 'STATE', 'RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3', 'RPL_THEME4', 'RPL_THEMES']]
        df.rename(columns={
            "STCNTY": "FIPS", "COUNTY": "County", "STATE": "State",
            "RPL_THEME1": "SVI_SES", "RPL_THEME2": "SVI_Household",
            "RPL_THEME3": "SVI_Minority", "RPL_THEME4": "SVI_Housing",
            "RPL_THEMES": "SVI_Overall"
        }, inplace=True)
        
        if isinstance(self.state, str):
            df = df.loc[df.State.eq(self.state), ]
            logger.info(f"Filtered SVI county data for state: {self.state}")
        else:
            df = df.loc[df.State.isin(self.state), ]
            logger.info(f"Filtered SVI county data for states: {self.state}")
        
        logger.debug(f"Processed SVI county data: {df.head()}")
        return df
    
    def svi_tract(self):
        url = "https://svi.cdc.gov/Documents/Data/2022/csv/states/SVI_2022_US.csv"
        logger.info(f"Downloading SVI tract data from {url}")
        try:
            df = pd.read_csv(url, dtype={'ST': str, 'FIPS': str})
            logger.info("SVI tract data downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download SVI tract data: {e}")
            raise
        
        logger.info("Processing SVI tract data.")
        df[['Tract', 'County', 'StateName']] = df.LOCATION.str.split(';', expand=True)
        df = df[['FIPS', 'Tract', 'County', 'STATE', 'RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3', 'RPL_THEME4', 'RPL_THEMES']]
        df.rename(columns={
            "STATE": "State",
            "RPL_THEME1": "SVI_SES", "RPL_THEME2": "SVI_Household",
            "RPL_THEME3": "SVI_Minority", "RPL_THEME4": "SVI_Housing",
            "RPL_THEMES": "SVI_Overall"
        }, inplace=True)
        
        if isinstance(self.state, str):
            df = df.loc[df.State.eq(self.state), ]
            logger.info(f"Filtered SVI tract data for state: {self.state}")
        else:
            df = df.loc[df.State.isin(self.state), ]
            logger.info(f"Filtered SVI tract data for states: {self.state}")
        
        logger.debug(f"Processed SVI tract data: {df.head()}")
        return df

# %% testing
if __name__ == "__main__":
    # Example usage
    state_fips = ['12', '13']  # Example state FIPS codes for Florida and Georgia
    svi = svi_data(state_fips)
    
    print("State: ", svi.state)  # Should print the state names corresponding to the FIPS codes
    print("SVI County Data:\n", svi.svi_data['county'].head())  # Print the first few rows of county data
    print("SVI Tract Data:\n", svi.svi_data['tract'].head())  # Print the first few rows of tract data
# %%
