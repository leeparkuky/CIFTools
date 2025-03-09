import logging
from dataclasses import dataclass, field
from typing import Union, Callable
import re
import functools
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # add src to path

from utils.states import stateDf
from utils.acs_utils import *
from config.acs_config import ACSConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


@dataclass
class acs_sdoh:
    year: int
    state_fips: Union[str, int]
    query_level: str
    key: str

    def cancer_infocus_download(self):
        """
        Runs all functions that generate cancer-related ACS tables and downloads all data,
        with a progress bar to track execution.

        Returns:
            dict: Dictionary containing cleaned ACS tables.
        """
        logging.info("Starting cancer_infocus_download process.")
        cancer_infocus_funcs = [self.__getattribute__(x) for x in self.__dir__() if re.match("gen_\\w.+_table", x)]

        # Progress bar for function execution
        for func in tqdm(cancer_infocus_funcs, desc="Generating ACS tables"):
            logging.info(f"Running function: {func.__name__}")
            func()

        output = self.download_all()
        logging.info("Finished downloading all tables.")
        return output

    def download_all(self):
        logging.info("Starting download_all function.")
        res = Parallel(n_jobs=-1)(
            delayed(self.cleanup_geo_col)(fun(), self.query_level) for fun in self.functions.values()
        )
        logging.info("Completed parallel execution of all functions.")
        return {key: val for key, val in zip(self.functions.keys(), res)}

    def clean_functions(self):
        logging.info("Cleaning function dictionary.")
        if hasattr(self, "functions"):
            self.functions = {}
        else:
            logging.warning("No function dictionary found to clean.")

    @staticmethod
    def cleanup_geo_col(df, level):
        logging.info(f"Cleaning up geographic columns for level: {level}")

        def cap_mc(s):
            if s[:2].lower() == "mc":
                return s[:2].title() + s[2:].title()
            else:
                return s.title()

        if level in ["county subdivision", "tract", "block", "county"]:
            name_series = df.NAME.str.split("[,|;] ")
            county_name = [cap_mc(x[-2]).replace("'S", "'s") for x in name_series]
            state_name = [x[-1] for x in name_series]
            FIPS = df[df.columns[df.columns.isin(["county subdivision", "tract", "block", "county", "state"])]].apply(
                lambda x: "".join(x), axis=1
            )
            if level != "county":
                subgroup_name = [x[0] for x in name_series]
                columns = pd.DataFrame(
                    zip(FIPS, subgroup_name, county_name, state_name),
                    columns=["FIPS", level.title(), "County", "State"],
                )
            else:
                columns = pd.DataFrame(zip(FIPS, county_name, state_name), columns=["FIPS", "County", "State"])

        elif level == "zip":
            logging.info("Processing ZIP-level data.")
            zip_name = [x[-5:] for x in df.NAME]
            states = {x: stateDf.loc[stateDf.FIPS2.eq(str(x)), "State"].values[0] for x in df.state.unique()}
            state_name = df.state.apply(lambda x: states[x])
            columns = pd.DataFrame(zip(zip_name, state_name), columns=["ZCTA5", "State"])

        elif level == "puma":
            logging.info("Processing PUMA-level data.")
            states = {x: stateDf.loc[stateDf.FIPS2.eq(str(x)), "State"].values[0] for x in df.state.unique()}
            state_name = df.state.apply(lambda x: states[x])
            puma_name = df.NAME.apply(lambda x: x.split(",")[0])
            puma_id = df.PUMA.tolist()
            columns = pd.DataFrame(zip(puma_id, puma_name, state_name), columns=["PUMA_ID", "PUMA_NAME", "State"])

        geo_name = ["county subdivision", "tract", "block", "county", "zip", "state", "PUMA"]
        df = df.drop(df.columns[df.columns.isin(geo_name)].tolist() + ["NAME"], axis=1)
        df = pd.concat([columns, df], axis=1)

        sorting_key = {"county": "FIPS", "zip": "ZCTA5", "puma": ["State", "PUMA_ID"]}.get(level, "FIPS")
        df = df.sort_values(sorting_key).reset_index(drop=True)

        logging.info(f"Completed cleanup for level: {level}")
        return df

    def add_function(self, func, name):
        logging.info(f"Adding function {name} to function dictionary.")
        if not hasattr(self, "functions"):
            self.functions = {}
        self.functions[name] = func

    def gen_acs_config(self, **kwargs):
        logging.info(f"Generating ACS config with arguments: {kwargs}")
        arguments = {
            "year": self.year,
            "state_fips": self.state_fips,
            "query_level": self.query_level,
            "acs_group": kwargs["acs_group"],
            "acs_type": kwargs["acs_type"],
        }
        self.config = ACSConfig(**arguments)
        return self.config

    def add_custom_table(self, group_id, acs_type, name):
        logging.info(f"Adding custom table: {name}")
        config = self.gen_acs_config(acs_group=group_id, acs_type=acs_type)

        def transform_data(func):
            @functools.wraps(func)
            def wrapper(**kwargs):
                logging.info(f"Fetching ACS data for {name}.")
                df = acs_data(self.key, config)
                df = func(df, **kwargs)
                return df

            self.add_function(wrapper, name)
            return wrapper

        return transform_data

    def gen_insurance_table(self, return_table=False):
        logging.info("Generating insurance table.")
        config = self.gen_acs_config(acs_group=["B27001", "C27007"], acs_type="")

        @custom_acs_data(self.key, config)
        def transform_df(df):
            logging.info("Transforming insurance data.")
            insurance_col = [config.variables[config.labels.index(x)] for x in config.labels if re.match(".+With health insurance coverage", x)]
            medicaid_col = [config.variables[config.labels.index(x)] for x in config.labels if re.match(".+With Medicaid/means-tested public coverage", x)]
            df["health_insurance_coverage_rate"] = df.loc[:, insurance_col].astype(int).sum(axis=1) / df.B27001_001E.astype(int)
            df["medicaid"] = df.loc[:, medicaid_col].astype(int).sum(axis=1) / df.C27007_001E.astype(int)
            df.drop(config.variables, axis=1, inplace=True)
            return df

        self.add_function(transform_df, "insurance")
        if return_table:
            return transform_df()
        

    def gen_insurance_disparity_table(self, return_table = False):
        config = self.gen_acs_config(**{'acs_group': ['B27010', 'C27001B', 'C27001I'], 'acs_type': ''})
        @custom_acs_data(self.key, config)
        def transform_df(df):
            df['uninsured_child'] = df.B27010_017E.astype(int)/df.B27010_002E.astype(int)
            df['uninsured_black'] = (df.C27001B_004E.astype(int) + df.C27001B_007E.astype(int) 
                                     + df.C27001B_010E.astype(int))/df.C27001B_001E.astype(int)
            df['uninsured_hispanic'] = (df.C27001I_004E.astype(int) + df.C27001I_007E.astype(int) 
                                     + df.C27001I_010E.astype(int))/df.C27001I_001E.astype(int)
            df.drop(config.variables, axis = 1, inplace = True)
            return df
        self.add_function(transform_df, 'insurance_disparity')
        if return_table:
            return transform_df()
    
    def gen_eng_prof_table(self, return_table = False):
        config = self.gen_acs_config(**{'acs_group': 'B16005', 'acs_type': ''})
        @custom_acs_data(self.key, config)
        def transform_df(df):
            notwell_col = [config.variables[config.labels.index(x)] for x in config.labels if re.match('.+not well', x)]
            none_col  = [config.variables[config.labels.index(x)] for x in config.labels if re.match('.+not at all', x)] + notwell_col
            df['Lack_English_Prof'] = df.loc[:, none_col].astype(int).sum(axis = 1)/df.B16005_001E.astype(int)
            df.drop(config.variables, axis = 1, inplace = True)
            return df
        self.add_function(transform_df, 'eng_prof')
        if return_table:
            return transform_df()


    def gen_vacancy_table(self, return_table = False):
        config = self.gen_acs_config(**{'acs_group': 'B25002', 'acs_type': ''})
        @custom_acs_data(self.key, config)
        def transform_df(df):
            df['vacancy_rate'] = df.B25002_003E.astype(int)/df.B25002_001E.astype(int)
            df.drop(df.columns[df.columns.str.contains(config.acs_group)], axis = 1, inplace = True)
            return df
        self.add_function(transform_df, 'vacancy')
        if return_table:
            return transform_df()

    def gen_poverty_table(self, return_table = False):
        config = self.gen_acs_config(**{'acs_group': 'B17026', 'acs_type': ''})
        @custom_acs_data(self.key, config)
        def transform_df(df):
            df['below_poverty_x.5'] = df.B17026_002E.astype(int)/df.B17026_001E.astype(int)
            df['below_poverty'] = df.loc[:, df.columns.str.match(re.compile('B17026_00[2-4]E'))].astype(int).sum(axis = 1).astype(int)/df.B17026_001E.astype(int)
            df['below_poverty_x2'] = (df.B17026_010E.astype(int) + df.loc[:, df.columns.str.match(re.compile('B17026_00[2-9]E'))].astype(int).sum(axis = 1)).astype(int)/df.B17026_001E.astype(int)
            df.drop(df.columns[df.columns.str.contains(config.acs_group)], axis = 1, inplace = True)
            return df
        self.add_function(transform_df, 'poverty')
        if return_table:
            return transform_df()
        
    def gen_poverty_disparity_table(self, return_table = False):
        config = self.gen_acs_config(**{'acs_group': ['B17010B', 'B17010I'], 'acs_type': ''})
        @custom_acs_data(self.key, config)
        def transform_df(df):
            df['below_poverty_black'] = df.B17010B_002E.astype(int)/df.B17010B_001E.astype(int)
            df['below_poverty_hispanic'] = df.B17010I_002E.astype(int)/df.B17010I_001E.astype(int)
            df.drop(config.variables, axis = 1, inplace = True)
            return df
        self.add_function(transform_df, 'poverty_disparity')
        if return_table:
            return transform_df()
        
    def gen_transportation_table(self, return_table = False):
        config = self.gen_acs_config(**{'acs_group': 'B08141', 'acs_type': ''})
        @custom_acs_data(self.key, config)
        def transform_df(df):
            df['no_vehicle'] = df.B08141_002E.astype(int)/df.B08141_001E.astype(int)
            df['two_or_more_vehicle']  = (df.B08141_004E.astype(int) + df.B08141_005E.astype(int))/df.B08141_001E.astype(int)
            df['three_or_more_vehicle'] = df.B08141_005E.astype(int)/df.B08141_001E.astype(int)
            df.drop(df.columns[df.columns.str.contains(config.acs_group)], axis = 1, inplace = True)
            return df
        self.add_function(transform_df, 'transportation')
        if return_table:
            return transform_df()

    def gen_econ_seg_table(self, return_table = False):
        config = self.gen_acs_config(**{'acs_group': ['B19001', 'B19001A', 'B19001B',], 'acs_type': ''})
        @custom_acs_data(self.key, config)
        def transform_df(df):
            df['economic_seg'] = ((df.B19001_016E.astype(int) + df.B19001_017E.astype(int)) - 
                                  (df.B19001_002E.astype(int) + df.B19001_003E.astype(int) + 
                                  df.B19001_004E.astype(int) + df.B19001_005E.astype(int) + 
                                  df.B19001_006E.astype(int)))/df.B19001_001E.astype(int)
            df['racial_economic_seg'] = ((df.B19001A_016E.astype(int) + df.B19001A_017E.astype(int)) - 
                                         (df.B19001B_002E.astype(int) + df.B19001B_003E.astype(int) + 
                                          df.B19001B_004E.astype(int) + df.B19001B_005E.astype(int) + 
                                          df.B19001B_006E.astype(int)))/df.B19001_001E.astype(int)
            df.drop(config.variables, axis = 1, inplace = True)
            return df
        self.add_function(transform_df, 'econ_seg')
        if return_table:
            return transform_df()
        
    def gen_race_seg_table(self, return_table = False):
        config = self.gen_acs_config(**{'acs_group': 'B03002', 'acs_type': ''})
        @custom_acs_data(self.key, config)
        def transform_df(df):
            df['racial_seg'] = (df.B03002_003E.astype(int) - df.B03002_004E.astype(int))/df.B03002_001E.astype(int)
            df.drop(df.columns[df.columns.str.contains(config.acs_group)], axis = 1, inplace = True)
            return df
        self.add_function(transform_df, 'race_seg')
        if return_table:
            return transform_df()
    
    def gen_gender_gap_table(self, return_table = False):
        config = self.gen_acs_config(**{'acs_group': 'B24022', 'acs_type': ''})
        @custom_acs_data(self.key, config)
        def transform_df(df):
            df['gender_pay_gap'] = np.where(df.B24022_038E.astype(int).ge(0) & df.B24022_002E.astype(int).ge(0), 
                                            1 - (df.B24022_038E.fillna(0).astype(int)/df.B24022_002E.astype(int)),
                                            0)
            df.drop(df.columns[df.columns.str.contains(config.acs_group)], axis = 1, inplace = True)
            return df
        self.add_function(transform_df, 'gender_gap')
        if return_table:
            return transform_df()
    
    def gen_employment_table(self, return_table = False):
        config = self.gen_acs_config(**{'acs_group': 'B23025', 'acs_type': ''})
        @custom_acs_data(self.key, config)
        def transform_df(df):
            df['Labor Force Participation Rate'] = df.B23025_003E.astype(int)/(df.B23025_003E.astype(int) + df.B23025_007E.astype(int))
            df['Unemployment Rate'] = df.B23025_005E.astype(int)/df.B23025_003E.astype(int)
            df.drop(df.columns[df.columns.str.contains(config.acs_group)], axis = 1, inplace = True)
            return df
        self.add_function(transform_df, 'employment')
        if return_table:
            return transform_df()
        
        
    def gen_gini_index_table(self, return_table = False):
        config = self.gen_acs_config(**{'acs_group': 'B19083', 'acs_type': ''})
        @custom_acs_data(self.key, config)
        def transform_df(df):
            df = df.rename(columns = {'B19083_001E': 'Gini Index'})
            df['Gini Index'] = df['Gini Index'].astype(float).apply(lambda x: x if x>=0 else np.nan)
            df.drop(df.columns[df.columns.str.contains(config.acs_group)], axis = 1, inplace = True)
            return df
        self.add_function(transform_df, 'gini_index')
        if return_table:
            return transform_df()

    def gen_rent_to_income_table(self, return_table = False):
        config = self.gen_acs_config(**{'acs_group': 'B25070', 'acs_type': ''})
        @custom_acs_data(self.key, config)
        def transform_df(df):
            df['rent_over_40'] = (df.B25070_009E.astype(int) + df.B25070_010E.astype(int))/df.B25070_001E.astype(int)
            df.drop(df.columns[df.columns.str.contains(config.acs_group)], axis = 1, inplace = True)
            return df
        self.add_function(transform_df, 'rent_to_income')
        if return_table:
            return transform_df()
        
    def gen_single_parent_table(self, return_table = False):
        config = self.gen_acs_config(**{'acs_group': 'B11012', 'acs_type': ''})
        @custom_acs_data(self.key, config)
        def transform_df(df):
            df['single_parent_house'] = (df.B11012_010E.astype(int) + df.B11012_015E.astype(int))/df.B11012_001E.astype(int)
            df.drop(df.columns[df.columns.str.contains(config.acs_group)], axis = 1, inplace = True)
            return df
        self.add_function(transform_df, 'single_parent')
        if return_table:
            return transform_df()
        
    def gen_housing_table(self, return_table = False):
        config = self.gen_acs_config(**{'acs_group': 'DP04', 'acs_type': 'profile'})
        @custom_acs_data(self.key, config)
        def transform_df(df):
            df['multi_unit_house'] = (df.DP04_0012E.astype(int) + df.DP04_0013E.astype(int))/df.DP04_0001E.astype(int)
            df['mobile_home'] = df.DP04_0014E.astype(int)/df.DP04_0001E.astype(int)
            df['owner_occupied'] = df.DP04_0046E.astype(int)/df.DP04_0045E.astype(int)
            df['crowding'] = (df.DP04_0078E.astype(int) + df.DP04_0079E.astype(int))/df.DP04_0076E.astype(int)
            df['lack_plumbing'] = df.DP04_0073E.astype(int)/df.DP04_0072E.astype(int)
            df['median_value'] = df.DP04_0089E.astype(float)
            df.loc[df['median_value'].le(0), 'median_value'] = 0
            df['median_mortgage'] = df.DP04_0101E.astype(float)
            df.loc[df['median_mortgage'].le(0), 'median_mortgage'] = 0
            df['median_rent'] = df.DP04_0134E.astype(float)
            df.loc[df['median_rent'].le(0), 'median_rent'] = 0
            df.drop(df.columns[df.columns.str.contains(config.acs_group)], axis = 1, inplace = True)
            return df
        self.add_function(transform_df, 'housing')
        if return_table:
            return transform_df()
        
    def gen_computer_table(self, return_table = False):
        config = self.gen_acs_config(**{'acs_group': 'B28005', 'acs_type': ''})
        @custom_acs_data(self.key, config)
        def transform_df(df):
            df['no_broadband'] = 1 - ((df.B28005_005E.astype(int) + df.B28005_011E.astype(int) + df.B28005_017E.astype(int))/df.B28005_001E.astype(int))
            df.drop(df.columns[df.columns.str.contains(config.acs_group)], axis = 1, inplace = True)
            return df
        self.add_function(transform_df, 'internet')
        if return_table:
            return transform_df()
        
    def gen_old_house_table(self, return_table = False):
        config = self.gen_acs_config(**{'acs_group': 'B25034', 'acs_type': ''})
        @custom_acs_data(self.key, config)
        def transform_df(df):
            df['houses_before_1960'] = df[['B25034_009E','B25034_010E','B25034_011E']].astype(int).sum(axis = 1)/df.B25034_001E.astype(int) ##### DOUBLE CHECK
            df.drop(df.columns[df.columns.str.contains(config.acs_group)], axis = 1, inplace = True)
            return df
        self.add_function(transform_df, 'houses_before_1960')
        if return_table:
            return transform_df()

        
    def gen_public_assistance_table(self, return_table = False):
        config = self.gen_acs_config(**{'acs_group': 'B19058', 'acs_type': ''})
        @custom_acs_data(self.key, config)
        def transform_df(df):
            df['public_assistance_received'] = df.B19058_002E.astype(int)/df.B19058_001E.astype(int)
            df.drop(df.columns[df.columns.str.contains(config.acs_group)], axis = 1, inplace = True)
            return df
        self.add_function(transform_df, 'public_assistance')
        if return_table:
            return transform_df()

               
    def gen_education_table(self, return_table = False):
        config = self.gen_acs_config(**{'acs_group': 'B15003', 'acs_type': ''})

        @custom_acs_data(self.key, config)
        def transform_df(df):
            # col1 is for below 9th grade
            col1_label = ['Nursery school', 'No schooling completed', '5th grade', '3rd grade', '4th grade', 
                          '2nd grade', '1st grade', 'Kindergarten','8th grade', '7th grade', '6th grade']
            col1       = [config.variables[config.labels.index(x)] for x in col1_label]
            # col4 is for advanced degree
            col4_label = ['Doctorate degree','Professional school degree', "Master's degree"]
            col4       = [config.variables[config.labels.index(x)] for x in col4_label]

            # col3 is for 4 years college and above: (it changes at the end, but for now, it includes any college to define col2, which is high school and above)
            col3_label = ["Bachelor's degree", "Associate's degree", "Some college, 1 or more years, no degree",'Some college, less than 1 year'] + col4_label
            col3       = [config.variables[config.labels.index(x)] for x in col3_label]
            # col2 is high school and above
            col2_label = ['Regular high school diploma'] + col3_label 
            col2       = [config.variables[config.labels.index(x)] for x in col2_label]
            # col5 is for completed college
            col5_label = ["Bachelor's degree"] + col4_label
            col5       = [config.variables[config.labels.index(x)] for x in col5_label]
            
            df['Total'] = df.B15003_001E.astype(int)
            df['Below 9th grade'] = df.loc[:, col1].astype(int).sum(axis = 1)/df.Total
            df['High School'] = df.loc[:, col2].astype(int).sum(axis = 1)/df.Total
            df['College'] = df.loc[:, col5].astype(int).sum(axis = 1)/df.Total
            df['Advanced Degree'] = df.loc[:, col4].astype(int).sum(axis = 1)/df.Total
            
            df.drop('Total', axis = 1, inplace = True)
            df.drop(df.columns[df.columns.str.contains(config.acs_group)], axis = 1, inplace = True)
            return df

        self.add_function(transform_df, 'education')
        if return_table:
            return transform_df()
        
    def gen_income_table(self, group_id = "B19013", race = 'all', return_table = False):
        config = self.gen_acs_config(**{'acs_group': group_id, 'acs_type': ''})
        @custom_acs_data(self.key, config)
        def transform_df(df):
            df[f'median_income_{race}'] = df[f'{config.acs_group}_001E'].astype(float)
            df.loc[df[f'median_income_{race}'].le(0), f'median_income_{race}'] = np.nan
            df.drop(df.columns[df.columns.str.contains(config.acs_group)], axis = 1, inplace = True)
            return df
        if race == 'all':
            self.add_function(transform_df, "income")
        else:
            self.add_function(transform_df, f"income_{race}")
        if return_table:
            return transform_df()

        
    def gen_age_demographic_table(self, group_id = "B01001", age_groups: Union[str, dict] = '18-64', return_table = False):
        config = self.gen_acs_config(**{'acs_group': group_id, 'acs_type': ''})
        
        @custom_acs_data(self.key, config)
        def transform_df(df):
            if isinstance(age_groups, str):
                if age_groups in ['ten years', '18-64']:
                    if age_groups == 'ten years':
                        age_group_dict = find_ten_year_age_groups(config = config) # you can chage the age group defnition here
                    else:
                        age_group_dict = find_large_age_groups(config = config) # you can chage the age group defnition here
                else:
                    raise ValueError("you should choose between 'ten years' or '18-64'; otherwise, provide a custom age_groups in a dictionary format")
            elif isinstance(age_groups, dict):
                try:
                    age_group_dict = find_index_for_age_group(age_groups, config = config)
                except:
                    raise ValueError("Please follow the guideline for the custom age_groups")
            
            df['Total'] = df[group_id + "_001E"].astype(int)
            for key, val in age_group_dict.items():
                col = [x for x in config.variables if config.variables.index(x) in val]
                df[key] = df.loc[:, col].astype(int).apply(np.sum, axis = 1)/df['Total']
            df.drop(df.columns[df.columns.str.contains(config.acs_group)], axis = 1, inplace = True)
            return df
        self.add_function(transform_df, 'demographic_age')
        if return_table:
            return transform_df()
    
  
    def gen_race_demographic_table(self, return_table = False):
        config = self.gen_acs_config(**{'acs_group': 'B03002', 'acs_type': ''})
        
        @custom_acs_data(self.key, config)
        def transform_df(df):
            def gen_race_series(race_variable, df = df):
                newdf = df.copy()
                race_series = newdf[race_variable].astype(int)/newdf.B03002_001E.astype(int)
                del newdf
                return race_series
            variables = ['B03002_' + x for x in ['003E','004E','005E','006E','007E','012E']]
            race_names = ['White','Black','AIAN','Asian','NHOPI','Hispanic']
            for v, n in zip(variables, race_names):
                df[n] = gen_race_series(v)
            df['Other_Races'] = (df.B03002_002E.astype(int) - 
                                 df.loc[:,['B03002_003E','B03002_004E','B03002_005E','B03002_006E','B03002_007E']].astype(int).sum(1))/df.B03002_001E.astype(int)

            df.drop(df.columns[df.columns.str.contains(config.acs_group)], axis = 1, inplace = True)
            return df
        
        self.add_function(transform_df, 'demographic_race')
        if return_table:
            return transform_df()


# Testing acs_sdoh
if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    api_key = os.getenv("CENSUS_API_KEY")

    logging.info("Initializing acs_sdoh object.")
    sdoh = acs_sdoh(year=2021, state_fips="21", query_level="county", key=api_key)

    output = sdoh.cancer_infocus_download()
    logging.info("Download complete.")
    print(output)
