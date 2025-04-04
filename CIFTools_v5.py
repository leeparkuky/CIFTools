import logging
from dataclasses import dataclass
from typing import Union, List
import requests
import urllib
import certifi
import asyncio
import re
import functools
import json
from functools import partial
from io import BytesIO
from zipfile import ZipFile
from glob import glob
from os import getcwd, remove
import os
from csv import DictReader
from itertools import product
# for ascync requests
from aiohttp import ClientSession
# cfg
from CIF_Config import ACSConfig, SocrataConfig
# pandas / numpy
import pandas as pd
import numpy as np
# multi-processing
from joblib import Parallel, delayed
# import stateDF from utils
from utils import stateDf
# import beautifulSoup
from bs4 import BeautifulSoup


# Hell World from VIM


def batchify_variables(config: ACSConfig):
    if config.acs_type == '':
        source = 'acs/acs5'
    else:
        source = f'acs/acs5/{config.acs_type}'
    table = config.variables
    batch_size = 49
    if len(table) > batch_size:
        num_full = len(table)//batch_size
        table_split = [table[k*batch_size:(k+1)*batch_size] for k in range(num_full)]
        table_split.append(table[num_full*batch_size:])
        return table_split
    else:
        return [table]


async def donwload_for_batch(config, table: str, key: str, session: ClientSession) -> List[str]:
    if config.acs_type == '':
        source = 'acs/acs5'
    else:
        source = f'acs/acs5/{config.acs_type}'
                
                
#     table = ','.join(batchify_variables(config)[0])
    if isinstance(config.state_fips, str) or isinstance(config.state_fips, int):
        if config.query_level == 'state':
            acs_url = f'https://api.census.gov/data/{config.year}/{source}?get={table}&for=state:{config.state_fips}&key={key}'
        elif config.query_level == 'county':
            acs_url = f'https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=county:*&in=state:{config.state_fips}&key={key}'
        elif config.query_level == 'county subdivision':
            acs_url = f'https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=county%20subdivision:*&in=state:{config.state_fips}&in=county:*&key={key}'
        elif config.query_level == 'tract':
            acs_url = f'https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=tract:*&in=state:{config.state_fips}&in=county:*&key={key}'
        elif config.query_level == 'block':
            acs_url = f'https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=block%20group:*&in=state:{config.state_fips}&in=county:*&in=tract:*&key={key}'
        elif config.query_level == 'zip':
            acs_url = f'https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=zip%20code%20tabulation%20area:*&in=state:{config.state_fips}&key={key}'
        elif config.query_level == 'puma':
            acs_url = f'https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=public%20use%20microdata%20area:*&in=state:{config.state_fips}&key={key}'
        else:
            raise ValueError('The region level is not found in the system; select among state, county, county subdivision, tract, block, zip and puma')
    elif isinstance(config.state_fips, list):
        config.state_fips = [str(x) for x in config.state_fips]
        states = ','.join(config.state_fips)
        if config.query_level == 'state':
            acs_url = f'https://api.census.gov/data/{config.year}/{source}?get={table}&for=state:{states}&key={key}'
        elif config.query_level == 'county':
            acs_url = f'https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=county:*&in=state:{states}&key={key}'
        elif config.query_level == 'county subdivision':
            acs_url = f'https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=county%20subdivision:*&in=state:{states}&in=county:*&key={key}'
        elif config.query_level == 'tract':
            acs_url = f'https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=tract:*&in=state:{states}&in=county:*&key={key}'
        elif config.query_level == 'block':
            acs_url = f'https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=block%20group:*&in=state:{states}&in=county:*&in=tract:*&key={key}'
        elif config.query_level == 'zip':
            acs_url = f'https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=zip%20code%20tabulation%20area:*&in=state:{states}&key={key}'
        elif config.query_level == 'puma':
            acs_url = f'https://api.census.gov/data/{config.year}/{source}?get=NAME,{table}&for=public%20use%20microdata%20area:*&in=state:{states}&key={key}'
        else:
            raise ValueError('The region level is not found in the system; select among state, county, county subdivision, tract, block, zip and puma')
    resp = await session.request(method="GET", url=acs_url)
    resp.raise_for_status()    
    json_raw =  await resp.json()
    return json_raw



async def download_all(config, key):
    tables = batchify_variables(config)
    async with ClientSession() as session:
        tasks = [donwload_for_batch(config, f"{','.join(table)}", key, session) for table in tables]
        return await asyncio.gather(*tasks)
    
    
    
def acs_data(key, config = None, **kwargs):
    import sys
    if config:
        pass
    else:
        config = ACSConfig(**kwargs)
    if sys.platform in ['win32','cygwin']:
#         asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy()) # not working 
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        result = asyncio.get_event_loop().run_until_complete(download_all(config, key))
    else:                                                     
        result = asyncio.run(download_all(config, key))
    if len(result) == 1:
        df = pd.DataFrame(result[0][1:], columns = result[0][0])
        if isinstance(config.acs_group, list):
            data_columns = []
            for group in config.acs_group:
                data_columns += df.columns[df.columns.str.contains(group)].tolist()
            index_columns = df.columns[~df.columns.isin(data_columns)].tolist()
        elif isinstance(config.acs_group, str):
            data_columns = df.columns[df.columns.str.contains(config.acs_group)].tolist()
            index_columns = df.columns[~df.columns.str.contains(config.acs_group)].tolist()
        else:
            raise AttributeError("config.acs_group should be either string or a list")
        df = df.loc[:, index_columns + data_columns]
    else:
        for i, res in enumerate(result):
            if i == 0:
                df = pd.DataFrame(res[1:], columns = res[0])
            else:
                df_1 = pd.DataFrame(res[1:], columns = res[0])
                merge_columns = df.columns[df.columns.str.isalpha()].tolist()
                if config.query_level == 'puma':
                    merge_columns += ['public use microdata area']
                df = df.merge(df_1, how = 'inner', on = merge_columns)
    df = pd.concat([df.loc[:, df.columns.str.isalpha()], df.loc[:, ~df.columns.str.isalpha()]], axis = 1)
    if config.query_level == 'puma':
        df.rename(columns = {'public use microdata area': "PUMA"}, inplace = True)
    return df


def custom_acs_data(key, config = None, **kwargs):    
    def decorator_transform(func, key = key, config = config):
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

###################################################################################################################
####################### census sdoh variables for CIFTools ########################################################
###################################################################################################################

@dataclass
class acs_sdoh:
    year: int
    state_fips: Union[str, int]
    query_level: str        
    key: str
    
    
    
    def cancer_infocus_download(self):
        cancer_infocus_funcs = [self.__getattribute__(x) for x in self.__dir__() if re.match('gen_\w.+_table', x)]
        for func in cancer_infocus_funcs:
            func()
        output = self.download_all()
        return output 

    def download_all(self):
        res = Parallel(n_jobs=-1)(delayed(fun)() for fun in self.functions.values())
        res = Parallel(n_jobs=-1)(delayed(self.cleanup_geo_col)(df, self.query_level) for df in res)
        return {key:val for key, val in zip(self.functions.keys(), res)}
    
    
    def clean_functions(self):
        if hasattr(self, 'functions'):
            self.functions = {}
        else:
            pass
    
    @staticmethod
    def cleanup_geo_col(df, level):
        def cap_mc(s):
            if s[0:2].lower() == 'mc':
                return s[0:2].title() + s[2:].title()
            else:
                return s.title()
        
        if level in ['county subdivision','tract','block', 'county']:
            name_series = df.NAME.str.split('[,|;] ')
            county_name = [cap_mc(x[-2]).replace("'S", "'s") for x in name_series]
            state_name  = [x[-1] for x in name_series]
            FIPS        = df[df.columns[df.columns.isin(
                ['county subdivision','tract','block', 'county', 'state'])]].apply(lambda x: ''.join(x), axis = 1)
            if level != "county":
                subgroup_name = [x[0] for x in name_series]
                columns = pd.DataFrame(zip(FIPS, subgroup_name, county_name, state_name), 
                                       columns = ['FIPS', level.title(), 'County','State'])
            else:
                columns = pd.DataFrame(zip(FIPS, county_name, state_name), columns = ['FIPS','County','State'])
                

        elif level == 'zip':
            zip_name = [x[-5:] for x in df.NAME]
            states = {x:stateDf.loc[stateDf.FIPS2.eq(str(x)),'State'].values[0] for x in df.state.unique()}
            state_name = df.state.apply(lambda x: states[x])
            columns = pd.DataFrame(zip(zip_name, state_name), columns = ['ZCTA5','State'])
            
        elif level == 'puma':
            states = {x:stateDf.loc[stateDf.FIPS2.eq(str(x)),'State'].values[0] for x in df.state.unique()}
            state_name = df.state.apply(lambda x: states[x])
            puma_name = df.NAME.apply(lambda x: x.split(',')[0])
            puma_id = df.PUMA.tolist()
            columns = pd.DataFrame(zip(puma_id, puma_name, state_name), columns = ['PUMA_ID','PUMA_NAME','State'])
            
        geo_name = ['county subdivision','tract','block', 'county', 'zip', 'state', 'PUMA']
        df = df.drop(df.columns[df.columns.isin(geo_name)].tolist() + ['NAME'], axis = 1)
        df = pd.concat([columns, df], axis = 1)
        if level in ['county subdivision','tract','block', 'county']:
            df = df.sort_values('FIPS').reset_index(drop = True)
        elif level == 'zip':
            df = df.sort_values('ZCTA5').reset_index(drop = True)
        elif level == 'puma':
            df = df.sort_values(['State','PUMA_ID']).reset_index(drop = True)
        return df
        
            
            
    def add_function(self, func, name):
        if hasattr(self, "functions"):
            pass
        else:
            self.functions = {}
        self.functions[name] = func

    
    def gen_acs_config(self, **kwargs):
        arguements = {'year': self.year,
                    'state_fips': self.state_fips,
                    'query_level': self.query_level,
                    'acs_group': kwargs['acs_group'],
                    'acs_type': kwargs['acs_type']}
        self.config = ACSConfig(**arguements)
        return self.config
    
    
#####################################################################
# Check this out!!!!!!!!!!!!!
#####################################################################
    def add_custom_table(self, group_id, acs_type, name):
        config = self.gen_acs_config(**{'acs_group': group_id, 'acs_type': acs_type})
        def transform_data(func):
            @functools.wraps(func)
            def wrapper(**kwargs):
                df = acs_data(self.key, config)
                df = func(df, **kwargs)
                return df
            self.add_function(wrapper, name)
            return wrapper
        
        return transform_data

    def gen_insurance_table(self, return_table = False):
        config = self.gen_acs_config(**{'acs_group': ["B27001", "C27007",], 'acs_type': ''})
        @custom_acs_data(self.key, config)
        def transform_df(df):
            insurance_col = [config.variables[config.labels.index(x)] for x in config.labels if re.match('.+With health insurance coverage', x)]
            medicaid_col  = [config.variables[config.labels.index(x)] for x in config.labels if re.match('.+With Medicaid/means-tested public coverage', x)]
            df['health_insurance_coverage_rate'] = df.loc[:, insurance_col].astype(int).sum(axis = 1)/df.B27001_001E.astype(int)
            df['medicaid'] = df.loc[:,medicaid_col].astype(int).sum(axis = 1)/df.C27007_001E.astype(int)
            df.drop(config.variables, axis = 1, inplace = True)
            return df
        self.add_function(transform_df, 'insurance')
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
                        age_group_dict = ten_year_age_groups(config = config) # you can chage the age group defnition here
                    else:
                        age_group_dict = large_age_groups(config = config) # you can chage the age group defnition here
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
            def gen_race_series(race_variable, df = df, config = config):
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
        

        
        
        
        
#########################
# utils for Demographics
#########################

ten_years = dict(zip(['Under 5 years', '5 to 14 years','15 to 24 years',
                      '25 to 34 years','35 to 44 years','45 to 54 years',
                      '55 to 64 years','65 to 74 years','75 to 84 years',
                      '85 years and over'],
                     [(0, 4), (5, 14), (15, 24), (25, 34), (35, 44), (45, 54), (55, 64),
                      (65, 74), (75, 84), (85, 100)]))

total_years = dict(zip(['Under 18', '18 to 64', 'Over 64'],
                       [(0, 17), (18, 64), (65, 100)]))


def find_index_for_age_group(age_group_dict, config = None, **kwargs):
    if config:
        pass
    else:
        config = ACSConfig(**kwargs)
        
    def extract_age_range(text):
        """from the labels for B01001, this extracts age interval in a tuple:
        \d year old    -> (\d, \d)
        Under \d years -> (0, \d)
        \d_1 to \d_2   -> (\d_1, \d_2)
        \d and over    -> (\d, 100)
        """
        def check_integer(s):
            try:
                int(s)
                return True
            except:
                return False
        numbers = [int(x) for x in text.split(' ') if check_integer(x)]
        if len(numbers) == 1:
            numbers = numbers + numbers
        return tuple(numbers)
    
    one = [[x.replace('Under', '0 to').replace('over','100'), config.labels.index(x)] for x in config.labels if re.match('.*years.*', x)]
    two = [(extract_age_range(x[0]), x[1]) for x  in one]

    def check_in_between(t1, t2):
        if t1[0] >= t2[0] and t1[1] <= t2[1]:
            return True
        else:
            return False

    def find_age_group(test):
        for k, v in age_group_dict.items():
            if check_in_between(test[0], v):
                return k
        
# def find_index_for_age_group(new_def, two):
    index_by_age_group = {k: [] for k in age_group_dict.keys()}
    for t in two:
        index_by_age_group[find_age_group(t)].append(t[1])
    return index_by_age_group


large_age_groups = partial(find_index_for_age_group, age_group_dict = total_years)

ten_year_age_groups = partial(find_index_for_age_group, age_group_dict = ten_years)



###################################################################
###################################################################
# facilities    ###################################################
###################################################################
###################################################################


def gen_facility_data(location:Union[List[str], str], taxonomy:List[str] = ['Gastroenterology','colon','obstetrics']):
    data_dict = {}
    data_dict['nppes'] = nppes(location)
    functions = [mammography, hpsa, fqhc, lung_cancer_screening, toxRel_data] #, superfund]
    dataset_names = ['mammography', 'hpsa','fqhc','lung_cancer_screening', 'tri_facility'] #, 'superfund_site']
    datasets = Parallel(n_jobs=-1)(delayed(f)(location) for f in functions)
    for name, df in zip(dataset_names, datasets):
        data_dict[name] = df
    return data_dict

    
    
###################################################################
## toxRel_data
###################################################################
    
def toxRel_data(location:Union[str, List[str]]):
    from tqdm import tqdm
    import datetime
    import os
    today = datetime.date.today(); year = today.year
    flag = True
    while flag: # this while statement will look for the most recent tri data
        # resp = requests.get(f'https://data.epa.gov/efservice/downloads/tri/mv_tri_basic_download/{year}_US/csv', stream=True)
        resp = requests.get(f'https://data.epa.gov/efservice/downloads/tri/mv_tri_basic_download/2022_US/csv', stream=True)
        try:
            resp.raise_for_status()
            flag = False
        except:
            year -= 1
    total = int(resp.headers.get('content-length', 0))
    fname = os.path.join(getcwd(), 'toxRel.csv')
    chunk_size = int(1024*1024/2)
    with open(fname, 'wb') as file, tqdm(
        desc="downloading toxRel data file",
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
        leave = True
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

    with open(fname, newline='') as csvfile:
        reader = DictReader(csvfile)
        ############
        colnames = ['FRS ID', 'FACILITY NAME',    #   'STREET ADDRESS','CITY','ST','ZIP',
                    'LATITUDE', 'LONGITUDE', 'COUNTY', 'CHEMICAL', 'ST']
        csv_keys = [field for col, field in product(colnames, reader.fieldnames) if \
                       re.match("\d+\.\s" + col + "$", field, flags = re.I) ]

        temp_col = ['STREET ADDRESS','CITY','ST', 'ZIP', 'CARCINOGEN']
        temp_field = [field for col, field in product(temp_col, reader.fieldnames) if \
                       re.match("\d+\.\s" + col + "$", field, flags = re.I) ]

        temp = dict(zip(temp_col, temp_field))


    #         csv_keys = ['3. FRS ID', '4. FACILITY NAME', #. '5. STREET ADDRESS', '6. CITY', '8. ST', '9. ZIP',
    #                    '12. LATITUDE', '13. LONGITUDE', '7. COUNTY', '34. CHEMICAL', '8. ST']
        data_dict = dict(zip(colnames, [[] for _ in range(len(colnames))]))
        data_dict['Address'] = []
        if isinstance(location, str):
            assert len(location) == 2
            for row in reader:
                if (row[temp['ST']] == location.upper()) & (
                    row[temp['CARCINOGEN']] == 'YES'):
                    address = row[temp['STREET ADDRESS']].title() + ', ' + row[temp['CITY']].title() + ', ' + row[temp['ST']].upper() + ' ' + str(row[temp['ZIP']])
                    data_dict['Address'].append(address)
                    for dict_key, row_key in zip(colnames, csv_keys):
                        data_dict[dict_key].append(row[row_key]) 
        else:
            for loc in location:
                assert len(loc) == 2
            for row in reader:
                if (row[temp['ST']] in [x.upper() for x in location]) & (
                row[temp['CARCINOGEN']] == 'YES'):
                    address = row[temp['STREET ADDRESS']].title() + ', ' + row[temp['CITY']].title() + ', ' + row[temp['ST']].upper() + ' ' + str(row[temp['ZIP']])
                    data_dict['Address'].append(address)
                    for dict_key, row_key in zip(colnames, csv_keys):
                        data_dict[dict_key].append(row[row_key])  
                    
    df = pd.DataFrame(data_dict)
    remove(fname)
    del data_dict
    df = df.groupby(["FRS ID", 'FACILITY NAME', 
                                 'Address', 'LATITUDE', 
                                 'LONGITUDE', 'COUNTY', 'ST'])['CHEMICAL'].agg(lambda col: ', '.join(col)).reset_index()
    df['Notes'] = 'Chemicals relased: ' + df['CHEMICAL']
    df = df[['FACILITY NAME', 'Address', 'LATITUDE', 'LONGITUDE', 'COUNTY', 'ST', 'Notes']]
    df = df.rename(columns = {'FACILITY NAME': 'Name', 'LATITUDE': 'latitude', 'LONGITUDE': 'longitude',
                             'ST':'State'})
    df['Type'] = 'Toxic Release Inventory Facility'
    df['Phone_number'] = None
    return df[['Type', 'Name', 'Address', 'State', 'Phone_number', 'Notes', 'latitude', 'longitude']] # You can add FIPS5 if you find COUNTY from a county-> fipscode map

    
    
    
    
###################################################################
## superfund
###################################################################

    
def gen_single_superfund(location: str):
    assert len(location) == 2; assert location.isalpha()
    url = f'https://data.epa.gov/efservice/ENVIROFACTS_SITE/FK_REF_STATE_CODE/{location}/JSON'
    sf = pd.read_json(url)
    sf.fips_code = sf.fips_code.apply(lambda x: str(x)[:5])
    sf.zip_code = sf.zip_code.apply(lambda x: str(x)[:5])
    sf2 = sf.loc[sf.npl_status_name.isin(['Currently on the Final NPL', 'Deleted from the Final NPL',
                                          'Site is Part of NPL Site'])]
    sf3 = sf2[['name', 'street_addr_txt', 'city_name', 'fk_ref_state_code', 'zip_code',
             'fips_code', 'npl_status_name', 'primary_latitude_decimal_val', 'primary_longitude_decimal_val']]
    sf3 = sf3.assign(Address = sf3['street_addr_txt'] + ', ' + sf3['city_name'] + ', ' + sf3['fk_ref_state_code'] + ' ' + sf3['zip_code'].astype(str))
    sf3 = sf3.rename(columns = {'name':'Name', 'fips_code':'FIPS', 'npl_status_name':'Notes',
                              'primary_latitude_decimal_val':'latitude', 'primary_longitude_decimal_val':'longitude',
                               'fk_ref_state_code':'State'})
    sf3.drop(['street_addr_txt', 'city_name', 'zip_code'], axis=1, inplace=True)
    sf3['Type'] = 'Superfund Site'
    sf3['Phone_number'] = None
    
    del sf, sf2
    return sf3[['Type', 'Name', 'Address', 'State', 'Phone_number', 'Notes', 'latitude', 'longitude', 'FIPS']]


def superfund(location: Union[str, List[str]]):
    if isinstance(location, str):
        if location.isnumeric():
            location = stateDf.loc[stateDf.FIPS2.eq(location),'StateAbbrev'].values[0]
        try:
            df = gen_single_superfund(location)
            df = df.reset_index(drop = True)
        except:
            print(f'\nsuperfund data for {location} is not available\n')
            df = None
    else:
        if any([x.isnumeric() for x in location]):
            for i, loc in enumerate(location):
                if loc.isnumeric():
                    location[i] = stateDf.loc[stateDf.FIPS2.eq(loc),'StateAbbrev'].values[0]
        datasets = []
        for loc in location:
            try:
                datasets.append(gen_single_superfund(loc))
            except:
                print(f"superfund data for {loc} is not available")
                pass
        df = pd.concat(datasets, axis = 0).reset_index(drop = True)
    return df
    
    

###################################################################
## mammography
###################################################################


def mammography(location: Union[str, List[str]]):

    url = urllib.request.urlopen("http://www.accessdata.fda.gov/premarket/ftparea/public.zip")
    
    with ZipFile(BytesIO(url.read())) as my_zip_file:
        df = pd.DataFrame(my_zip_file.open(my_zip_file.namelist()[0]).readlines(), columns= ['main'])
        df = df.main.astype(str).str.split('|', expand = True)
        df.columns = ['Name','Street','Street2','Street3','City','State','Zip_code','Phone_number', 'Fax']
        # state = state.upper()
        if isinstance(location, str):
            df = df.loc[df.State.eq(location)].reset_index(drop = True)
        else:
            df = df.loc[df.State.isin(location)].reset_index(drop = True)
        df.Name = df.Name.str.extract(re.compile('[bB].(.*)'))
        df['Address'] = df['Street'] + ', ' + df['City'] + ', ' +  df['State'] + ' ' + df['Zip_code']
        df['Type'] = 'Mammography'
        df['Notes'] = ''
        def convert_phone_number(match_obj):
            first =  '(' + match_obj.group(1)[:3] + ") " + match_obj.group(1)[3:]
            if len(match_obj.group(2)) > 4:
                second = match_obj.group(2)[:4] + " ext. " + match_obj.group(2)[4:]
            else:
                second = match_obj.group(2)
            return first + '-' + second
        df['Phone_number'] = df.Phone_number.str.replace("(\d+)-(\d+)", convert_phone_number, regex =True)
    return df.loc[:,['Type','Name','Address','State', 'Phone_number', 'Notes']] #try to add FIPS and State


###################################################################
## hpsa
###################################################################

def download_hpsa_data():
    from tqdm import tqdm
    import os
    resp = requests.get('https://data.hrsa.gov/DataDownload/DD_Files/BCD_HPSA_FCT_DET_PC.xlsx', stream=True)
    total = int(resp.headers.get('content-length', 0))
    fname = os.path.join(getcwd(), 'hpsa.xlsx')
    chunk_size = 1024*10
    with open(fname, 'wb') as file, tqdm(
        desc="downloading hpsa data file",
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
        leave = True
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)
    output = pd.read_excel(fname, engine = 'openpyxl')
    remove(fname)
    return output




def hpsa(location: Union[str, List[str]]):
    df= download_hpsa_data()
    df.columns = df.columns.str.replace(' ','_')
    if isinstance(location, str):
        df = df.loc[df.Primary_State_Abbreviation.eq(location)&
                    df.HPSA_Status.eq('Designated')&
                    df.Designation_Type.ne('Federally Qualified Health Center')].reset_index(drop = True)
    else:
        df = df.loc[df.Primary_State_Abbreviation.isin(location)&
                    df.HPSA_Status.eq('Designated')&
                    df.Designation_Type.ne('Federally Qualified Health Center')].reset_index(drop = True)
    df = df[['HPSA_Name','HPSA_ID','Designation_Type','HPSA_Score','HPSA_Address',
             'HPSA_City', 'State_Abbreviation', 'Common_State_County_FIPS_Code',
             'HPSA_Postal_Code','Longitude','Latitude']]
    pattern = re.compile('(\d+)-\d+')
    df['HPSA_Postal_Code']  = df.HPSA_Postal_Code.str.extract(pattern)
    df['HPSA_Street'] = df['HPSA_Address'] + ', ' + df['HPSA_City'] + \
        ', ' + df['State_Abbreviation'] + ' ' + df['HPSA_Postal_Code']
    df = df.drop_duplicates()
    df['Type'] = 'HPSA '+df.Designation_Type
    df = df.rename(columns = {'HPSA_Name' : 'Name', 'HPSA_Street':'Address', 
                              'Common_State_County_FIPS_Code': 'FIPS', 'State_Abbreviation': 'State',
                              'Longitude': 'longitude', 'Latitude': 'latitude'})
    df = df[['Type','Name','HPSA_ID','Designation_Type','HPSA_Score','Address',
             'FIPS', 'State', 'longitude','latitude']]
    df = df.loc[df.longitude.notnull()|df.Address.notnull()].reset_index(drop = True)
    df['Phone_number'] = None
    df['Notes'] = ''
    return df[['Type','Name', 'Address', 'State', 'Phone_number', 'Notes', 'latitude', 'longitude']] #try to add FIPS 


###################################################################
## fqhc
###################################################################


def download_fqhc_data(location: Union[str, List[str]]):
    from tqdm import tqdm
    import os
    resp = requests.get('https://data.hrsa.gov//DataDownload/DD_Files/Health_Center_Service_Delivery_and_LookAlike_Sites.csv', stream = True)    
    total = int(resp.headers.get('content-length', 0))
    fname = os.path.join(getcwd(), 'fqhc.csv')
    chunk_size = 1024*10
    with open(fname, 'wb') as file, tqdm(
        desc="downloading fqhc data file",
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
        leave = True
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)
            
    with open(fname, newline='', encoding='utf8') as csvfile:
        reader = DictReader(csvfile)
        colnames = ['Health_Center_Type', 'Site_Name','Site_Address','Site_City','Site_State_Abbreviation',
                 'Site_Postal_Code','Site_Telephone_Number', 
                 'Health_Center_Service_Delivery_Site_Location_Setting_Description',
                 'Geocoding_Artifact_Address_Primary_X_Coordinate',
                 'Geocoding_Artifact_Address_Primary_Y_Coordinate']
        csv_keys = [x.replace('_', ' ') for x in colnames]
        data_dict = dict(zip(colnames, [[] for _ in range(len(colnames))]))
        for row in reader:
            if isinstance(location, str):
                if (row['Site State Abbreviation'] == location.upper()) & (
                    row['Health Center Type'] == 'Federally Qualified Health Center (FQHC)') & (
                    row['Site Status Description'] == 'Active'):
                    for dict_key, row_key in zip(colnames, csv_keys):
                        data_dict[dict_key].append(row[row_key])   
            else:
                if (row['Site State Abbreviation'] in [x.upper() for x in location]) & (
                    row['Health Center Type'] == 'Federally Qualified Health Center (FQHC)') & (
                    row['Site Status Description'] == 'Active'):
                    for dict_key, row_key in zip(colnames, csv_keys):
                        data_dict[dict_key].append(row[row_key])   
    output = pd.DataFrame(data_dict)
    del data_dict
    remove(fname)
    return output



def fqhc(location: Union[str, List[str]]):
    df= download_fqhc_data(location)
    df['Type'] = 'FQHC'
    df['Address'] = df['Site_Address'] + ', ' + df['Site_City'] + ', ' + \
                    df['Site_State_Abbreviation'] + ' ' + df['Site_Postal_Code']
    df = df.rename(columns = {'Site_Name':'Name', 'Site_Telephone_Number': 'Phone_number', 
                              'Site_State_Abbreviation': 'State',
                              'Health_Center_Service_Delivery_Site_Location_Setting_Description': 'Notes',
                              'Geocoding_Artifact_Address_Primary_X_Coordinate': 'longitude',
                              'Geocoding_Artifact_Address_Primary_Y_Coordinate': 'latitude'})
    df = df.loc[df.Address.notnull()].reset_index(drop = True)
    return df[['Type', 'Name', 'Address', 'State','Phone_number', 'Notes', 'latitude', 'longitude']]


###################################################################
## nppes
###################################################################


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


taxonomy = ['Gastroenterology','colon','obstetrics', 'Hematology%20&%20Oncology', 'Medical%20Oncology', 'Gynecologic%20Oncology',
            'Pediatric%20Hematology-Oncology', 'Radiation%20Oncology', 'Surgical%20Oncology']

taxonomy_names = dict(zip(taxonomy, ['Gastroenterology','Colon & Rectal Surgeon','Obstetrics & Gynecology', 'Hematology & Oncology',
                                     'Medical Oncology', 'Gynecologic Oncology', 'Pediatric Hematology-Oncology', 'Radiation Oncology',
                                     'Surgical Oncology']))

def gen_nppes_by_taxonomy(taxonomy: str, location: str):
    count = 0
    result_count = 200
    skip = 0
    datasets = []
    while result_count == 200:
        count += 1
        url = f'https://npiregistry.cms.hhs.gov/api/?version=2.1&address_purpose=LOCATION&number=&state={location}&taxonomy_description={taxonomy}&skip={200*(count -1)}&limit=200'
        resp = requests.get(url)
        output = resp.json()
        if 'result_count' in output.keys():
            result_count = output['result_count']
            if result_count:
                df = pd.DataFrame(output['results'])
                df['Name'] = df.basic.apply(parse_basic)
                df['Phone_number'] = df.addresses.apply(lambda x: parse_address(x)[1])
                df['Address'] = df.addresses.apply(lambda x: parse_address(x)[0])
                df['State'] = df.addresses.apply(lambda x: parse_address(x)[2])
                df = df.loc[df.State.eq(location), :].reset_index(drop = True)
                if taxonomy in taxonomy_names.keys():
                    df['Type']    = taxonomy_names[taxonomy]
                else:
                    df['Type']    = taxonomy
                df['Notes']   = ['' if x[-5:].isnumeric() else 'missing zip code' for x in df.Address] # MA has one missing zip code
            if result_count == 200: # if result_count is 200, it is very likely to have more data
                df = df[['Type','Name','Address','State', 'Phone_number', 'Notes']]
                if count % 7 == 0 : # sometimes, it returns the same datasets over and over
                    if (datasets[-1].shape[0]==200) & (df.shape[0] < 200): 
                        result = pd.concat(datasets, axis = 0).reset_index(drop = True)
                        result = result.drop_duplicates()
                        return result
                    elif datasets[-1].shape[0] == df.shape[0]:
                        result = pd.concat(datasets, axis = 0).reset_index(drop = True)
                        result = result.drop_duplicates()
                        return result
                else:
                    datasets.append(df[['Type','Name','Address','State', 'Phone_number', 'Notes']])
            elif count == 1:
                if result_count:
                    df[['Type','Name','Address','State', 'Phone_number', 'Notes']]
                    return df[['Type','Name','Address','State', 'Phone_number', 'Notes']]
                else:
                    df = pd.DataFrame(columns = ['Type','Name','Address','State', 'Phone_number', 'Notes'])
                    return df
            else:
                if result_count:
                    datasets.append(df[['Type','Name','Address','State', 'Phone_number', 'Notes']])
                    result = pd.concat(datasets, axis = 0).reset_index(drop = True)
                    return result
                else:
                    result = pd.concat(datasets, axis = 0).reset_index(drop = True)
                    return result
        else:
            if len(datasets):
                result = pd.concat(datasets, axis = 0).reset_index(drop = True)
                result_count = 0
                return result
            else:
                break

        
def nppes(location:Union[str, List[str]], taxonomy:List[str] = ['Gastroenterology','colon','obstetrics', 'Hematology%20&%20Oncology', 'Medical%20Oncology', 'Gynecologic%20Oncology',
            'Pediatric%20Hematology-Oncology', 'Radiation%20Oncology', 'Surgical%20Oncology']) -> pd.DataFrame:
    if isinstance(location, str):
        res = Parallel(n_jobs=-1)(delayed(gen_nppes_by_taxonomy)(t, location) for t in taxonomy)
    else:
        from itertools import product
        res = Parallel(n_jobs=-1)(delayed(gen_nppes_by_taxonomy)(t, loc) for t, loc in product(taxonomy, location))
        print('Process is complete')
    return pd.concat(res, axis = 0)

###################################################################
## lung_cancer_screening ########################################## -> multiprocessing with multiple states
###################################################################
    
    
        
def setup_chrome_driver():
    import sys
    import os
    if sys.platform in ['win32','cygwin']: # if the platfor is windows
        glob_result = glob(os.path.join(os.getcwd(), '*', 'chromedriver.exe'))
    else: # if the platform is either linux or mac os
        glob_result = glob(os.path.join(os.getcwd(), '*', 'chromedriver'))
    if len(glob_result) == 0: # if chromedriver is not found
        import chromedriver_autoinstaller 
        fp = chromedriver_autoinstaller.install('.') # install chromedriver_autoinstaller
    else:
        fp = glob_result[0]
    return fp
    
def lung_cancer_screening_file_download(chrome_driver_path = None):
    try: # first try requests
        resp = requests.get('https://report.acr.org/t/PUBLIC/views/NRDRLCSLocator/ACRLCSDownload.csv', stream = True)
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        chunk_size = 1024
        from tqdm import tqdm
        with open('./ACRLCSDownload.csv', 'wb') as f, tqdm(
            desc="downloading lcs data file",
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            leave = True
        ) as bar:
            for data in resp.iter_content(chunk_size = chunk_size):
                size = f.write(data)
                bar.update(size)
            f.close()
        return 0
#         print('LCSR data ready')

        
    except:  # if it is not working, I will use the selenium
        from selenium import webdriver
        from selenium.webdriver import ChromeOptions
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        import os
        import time
        chromeOptions =ChromeOptions()
        prefs = {"download.default_directory" : getcwd()}
        chromeOptions.add_experimental_option("prefs",prefs)
        chromeOptions.add_argument(f"download.default_directory={getcwd()}")
        if chrome_driver_path == None:
            if os.getenv("COLAB_RELEASE_TAG"):
                import subprocess
                subprocess.run(['apt','install','chromium-chromedriver'])
                chromeOptions.add_argument('--headless')
                chromeOptions.add_argument('--no-sandbox')
                chromeOptions.add_argument('--disable-dev-shm-usage')
                chromeOptions.add_argument("window-size=1200x600")
                chrome_driver_path = '/usr/lib/chromium-browser/chromedriver'
                print('google chrome driver is ready')
            else:
                chrome_driver_path = setup_chrome_driver()
        driver = webdriver.Chrome(service=Service(chrome_driver_path), options=chromeOptions)
    #     old code
    #     driver = webdriver.Chrome(executable_path=chrome_driver_path, options=chromeOptions)
        url = 'https://report.acr.org/t/PUBLIC/views/NRDRLCSLocator/LCSLocator?:embed=y&:showVizHome=no&:host_url=https%3A%2F%2Freport.acr.org%2F&:embed_code_version=3&:tabs=no&:toolbar=no&:showAppBanner=no&:display_spinner=no&:loadOrderID=0'
        driver.get(url);  time.sleep(10)
        state = driver.find_elements(By.CLASS_NAME, 'tabComboBoxButtonHolder')[2]; state.click(); time.sleep(10)
        state2 = driver.find_elements(By.CLASS_NAME, 'tabMenuItemNameArea')[1]; state2.click(); time.sleep(10)
        download = driver.find_element(By.ID, 'tabZoneId422'); download.click()
        t = 0
        while t == 0:
            time.sleep(5)
            t = len(glob('./ACRLCSDownload*.csv'))
            print('Waiting on LCSR data...')
        else:
            print('LCSR data ready')
        driver.close()
        return 1

    
def process_lcs_data(file_path, location: Union[str, List[str]]):
    
    input_file = DictReader(open(file_path))

    new_names = ['Name','Street','City','State','Zip code','Phone']
    Address = []; Phone = [];Name = []; State = []

    name_dict = {k: v for k, v in product(new_names,input_file.fieldnames) if re.match(".*" + k + '.*', v, flags = re.I)}


    def return_lcs_info(row):
        address = row[name_dict['Street']].title() + ', ' + \
        row[name_dict['City']].title() + ', ' + row[name_dict['State']] \
        + ' ' + row[name_dict['Zip code']][:5]

        phone = row[name_dict['Phone']]

        name = row[name_dict['Name']]
        
        state = row[name_dict['State']]
        return address, phone, name, state



    for row in input_file:
        if isinstance(location, str):
            if row[name_dict['State']] == location:
                address, phone, name, state  = return_lcs_info(row)
                Address.append(address); Phone.append(phone); Name.append(name); State.append(state)
        else:
            if row[name_dict['State']] in location:
                address, phone, name, state = return_lcs_info(row)
                Address.append(address); Phone.append(phone); Name.append(name); State.append(state)
    df = pd.DataFrame(zip(Name, Address, State, Phone), columns = ['Name','Address', 'State', 'Phone_number'])
    df['Type'] = 'Lung Cancer Screening'
    df['Notes'] = ''
    df = df[['Type','Name', 'Address', 'State', 'Phone_number', 'Notes']]

    return df
#     df = pd.read_csv(file_path)
#     df.columns = ['Name','Street','City','State','Zip_code','Phone','Designation', 'Site ID', 'Facility ID', 'Registry Participant']
#     df['Address'] = df['Street'].str.title() + ', ' + df['City'].str.title() + ', ' +  df['State'].str.upper() + ' ' + df['Zip_code'].apply(lambda x: x[:5])
#     df['Type'] = 'Lung Cancer Screening'
#     df['Phone_number'] = df['Phone']
#     df['Notes'] = ''
#     if isinstance(location, str):
#         df = df.loc[df.State.eq(location)]
#     else:
#         df = df.loc[df.State.isin(location)]
#     df = df[['Type','Name', 'Address', 'Phone_number', 'Notes']]
#     return df



def remove_chromedriver(chrome_driver_path):
    import shutil
    import os
    import sys
    directory_path = os.path.dirname(chrome_driver_path)
    try:
        shutil.rmtree(directory_path)
    except:
        print(f"please remove chrome driver located in {directory_path} manually \n when the process is finished \n\n")

    
def lung_cancer_screening(location: Union[str, List[str]]):
    selenium = lung_cancer_screening_file_download()
    downloads = glob('./ACRLCSDownload*.csv')
    df = process_lcs_data(downloads[0], location)
    if selenium:
        chrome_driver_path = setup_chrome_driver()
        remove_chromedriver(chrome_driver_path)
    df = df.reset_index(drop = True)
    for file in downloads:
        remove(file)
    return df

def uscs_incidence():
    file_path = glob('./catchment_area/uscs/uscs_cancer_incidence_county_2017-2021.csv')
    df = pd.read_csv(file_path[0])
    return df
    
    
    

############################################################################
## BLS      ################################################################
############################################################################

@dataclass
class BLS:
    state_fips: Union[str, List[str]]
    most_recent: bool = True
    
    @property
    def bls_data(self):
        user_agent = {'User-agent': 'ciodata@uky.edu'}
        if hasattr(self, '_bls_data'):
            pass
        else:
            state = self.state_fips
            response = requests.get('https://www.bls.gov/web/metro/laucntycur14.txt', headers = user_agent)
            df = pd.DataFrame([x.strip().split('|') for x in response.text.split('\n')[6:-7]],
                              columns = ['LAUS Area Code','State','County','Area',
                                         'Period','Civilian Labor Force','Employed',
                                         'Unemployed','Unemployment Rate'] )
            df['State'] = df.State.str.strip().astype(str)
            if isinstance(state, str):
                assert len(state) == 2
                df = df.loc[df.State.eq(state), :]
            else:
                for s in state:
                    assert len(s) == 2
                df = df.loc[df.State.isin(state), :]
            df['County'] = df.County.str.strip().astype(str)
            df['County'] = ['0'+x if len(x) < 3 else x for x in df.County]
            df['County'] = ['0'+x if len(x) < 3 else x for x in df.County]
            df['Employed'] = pd.to_numeric(df.Employed.str.strip().str.replace(',',''), errors = 'coerce')
            df['Unemployed'] = pd.to_numeric(df.Unemployed.str.strip().str.replace(',',''), errors = 'coerce')
            df['Unemployment Rate'] = pd.to_numeric(df['Unemployment Rate'].str.strip().str.replace(',',''), errors = 'coerce')
            df['FIPS'] = df['State']+df['County']
            df['Period'] = df.Period.str.strip()
            if self.most_recent:
                # df = df.loc[df.Period.str.match(re.compile('.*p\)$'))]
                p = df.iloc[1,4]
                df = df.loc[df.Period == p]
                df['Period'] = [x[:-3] for x in df.Period]
            df = df.loc[:,['FIPS','Unemployment Rate', 'Period']].sort_values('FIPS').reset_index(drop = True)
            df[f'Monthly Unemployment Rate ({df.Period.unique()[0]})'] = df['Unemployment Rate']*0.01
            df = df.drop(columns=['Unemployment Rate', 'Period'])
            self._bls_data = df
        return self._bls_data
        
    @property
    def bls_data_timeseries(self):
        user_agent = {'User-agent': 'ciodata@uky.edu'}
        if hasattr(self, '_bls_data_timeseries'):
            pass
        else:
            state = self.state_fips
            response = requests.get('https://www.bls.gov/web/metro/laucntycur14.txt', headers = user_agent)
            df = pd.DataFrame([x.strip().split('|') for x in response.text.split('\n')[6:-7]],
                              columns = ['LAUS Area Code','State','County','Area',
                                         'Period','Civilian Labor Force','Employed',
                                         'Unemployed','Unemployment Rate'] )
            df['State'] = df.State.str.strip().astype(str)
            if isinstance(state, str):
                assert len(state) == 2
                df = df.loc[df.State.eq(state), :]
            else:
                for s in state:
                    assert len(s) == 2
                df = df.loc[df.State.isin(state), :]
            df['County'] = df.County.str.strip().astype(str)
            df['County'] = ['0'+x if len(x) < 3 else x for x in df.County]
            df['County'] = ['0'+x if len(x) < 3 else x for x in df.County]
            df['Employed'] = pd.to_numeric(df.Employed.str.strip().str.replace(',',''), errors = 'coerce')
            df['Unemployed'] = pd.to_numeric(df.Unemployed.str.strip().str.replace(',',''), errors = 'coerce')
            df['Unemployment Rate'] = pd.to_numeric(df['Unemployment Rate'].str.strip().str.replace(',',''), errors = 'coerce')
            df['Civilian Labor Force'] = pd.to_numeric(df['Civilian Labor Force'].str.strip().str.replace(',',''), errors = 'coerce')

            df['FIPS'] = df['State']+df['County']
            df['Period'] = df.Period.str.strip()
            df['Period'] = df.Period.str.replace('\(p\)','', regex = True)
            df.Period = df.Period.apply(lambda x:  x[:-2] + '20' + x[-2:])
            df['period_for_ordering'] = [pd.Period(x) for x in df.Period]
            df = df.sort_values(['FIPS', 'period_for_ordering']).loc[:,['FIPS','Civilian Labor Force', 'Unemployment Rate', 'Period']].reset_index(drop = True)
            df = df.rename(columns = {'Unemployment Rate':'Monthly Unemployment Rate'})
            df['Monthly Unemployment Rate'] = df['Monthly Unemployment Rate'] * .01
            self._bls_data_timeseries = df
        return self._bls_data_timeseries

        
# ############################################################################
# ## Water Violation      ####################################################  -> multiprocessing if with multiple states
# ############################################################################
        
# @dataclass
# class water_violation:
#     state_fips: Union[str, List[str]]
#     start_year: int = 2016
#     end_year  : int = None
        
        
#     @property
#     def location(self):
#         if hasattr(self, '_location'):
#             pass
#         else:
#             if isinstance(self.state_fips, str):
#                 self._location = stateDf.loc[stateDf.FIPS2.eq(self.state_fips),'StateAbbrev'].values[0]
#             else:
#                 self._location = stateDf.loc[stateDf.FIPS2.isin(self.state_fips),'StateAbbrev'].values.tolist()
#         return self._location
        
#     @property
#     def water_violation_data(self):
#         if hasattr(self, '_water_violation_data'):
#             pass
#         else:
#             self._water_violation_data = pd.concat(self.water_violation_data_dictionary.values(), axis = 0).reset_index(drop = True)
#         return self._water_violation_data
            
        
#     @property
#     def water_violation_data_dictionary(self):
#         if hasattr(self, '_water_violation_data_dictionary'):
#             pass
#         else:
#             if isinstance(self.location, str):
#                 df = self.gen_water_violation(self.location)
#                 data_dict = {self.location: df}
#             else:
#                 datasets = Parallel(n_jobs=-1)(delayed(self.gen_water_violation)(loc) for loc in self.location)
#                 data_dict = dict(zip(self.location, datasets))
#             self._water_violation_data_dictionary = data_dict
#         return self._water_violation_data_dictionary
        
        
#     def gen_water_violation(self, state:str):
#         assert len(state) == 2
#         violation = self.gen_violation(state, self.start_year, self.end_year)
#         profile   = self.gen_profile(state)
#         violation_by_pws = violation[['pwsid','violation_id','indicator']].groupby(['pwsid','violation_id'], as_index = False).max().loc[:,['pwsid','indicator']].groupby('pwsid', as_index = False).sum() # summed the number of violations
#         violation_by_pws.columns = ['pwsid','counts']
#         df = profile.merge(violation_by_pws, on = 'pwsid', how='left')
#         df = df[['county_served', 'primacy_agency_code', 'counts']].groupby('county_served', as_index = False).max() 
#         self.testing = df
#         df['County'] = df.county_served.astype(str) + ' County'
#         df['County']= df.County.str.replace('Parish County','Parish')
#         df['County']= df.County.str.replace('City County','City')
#         df.drop(['county_served', 'primacy_agency_code'], axis = 1, inplace  = True)
#         df.loc[df.counts.isnull(),'counts'] = 0
#         df['State'] = stateDf.loc[stateDf.StateAbbrev.eq(state), "State"].values[0]
        
#         del profile, violation
#         if self.end_year:
#             if self.start_year == self.end_year:
#                 new_name = f"PWS_Violations_in_{self.start_year}"
#             else:
#                 new_name = f"PWS_Violations_Since_{self.start_year}_Until_{self.end_year}"
#         else:
#             new_name = f"PWS_Violations_Since_{self.start_year}"
#         df.rename(columns = {'counts':new_name}, inplace = True)
        
#         return df[['County','State',new_name]]
        
#     @staticmethod
#     def gen_violation(state:str, start_year: int, end_year:int = None):
#         url_violation = f'https://data.epa.gov/efservice/VIOLATION/IS_HEALTH_BASED_IND/Y/PRIMACY_AGENCY_CODE/{state}/COMPL_PER_BEGIN_DATE/%3E/2015-12-31/CSV'
#         violation = pd.read_csv(url_violation)
#         violation.columns = violation.columns.str.replace(re.compile('.*\.'),"", regex = True)
#         violation = violation.loc[violation.compl_per_begin_date.notnull() ,:]
#         violation['date'] = pd.to_datetime(violation.compl_per_begin_date,  format = "%Y-%m-%d")
#         if end_year:
#             violation = violation.loc[(
#                 violation.date.dt.year >= start_year) & (
#                 violation.date.dt.year <= end_year), :].reset_index(drop = True)
#         else:
#             violation = violation.loc[violation.date.dt.year >= start_year, :].reset_index(drop = True)
#         violation['indicator'] = 1
#         return violation

#     @staticmethod
#     def gen_profile(state:str):
#         url_systems = f'https://data.epa.gov/efservice/GEOGRAPHIC_AREA/PWSID/BEGINNING/{state}/CSV'
#         profile = pd.read_csv(url_systems)
#         profile.columns = map(str.lower, profile.columns)
#         if len(profile.index) == 10001:
#             url_systems2 = f'https://data.epa.gov/efservice/GEOGRAPHIC_AREA/PWSID/BEGINNING/{state}/rows/10001:20000/CSV'
#             profile2 = pd.read_csv(url_systems2)
#             profile2.columns = map(str.upper, profile2.columns)
#             profile = pd.concat([profile,profile2]).reset_index(drop=True)
#             del profile2
#         profile.columns = profile.columns.str.replace(re.compile('.*\.'),"", regex = True)
#         profile = profile.loc[profile['pws_type_code'] == 'CWS']
#         profile = profile.assign(county_served = profile.county_served.str.split(',')).explode('county_served')
#         return profile

#         if isinstance(self.location, str):
#             assert len(self.location) == 2
#             self.profile = gen_profile(location)
#             self.violation = gen_violation(location)
#         else:
#             self.profile = {}
#             self.violation = {}
#             for loc in self.location:
#                 assert len(loc) == 2
#                 self.profile[loc] = gen_profile(loc)
#                 self.violation[loc] = gen_violation(loc)

               
############################################################################
## Food Desert      ########################################################
############################################################################

class food_desert:  
    def __init__(self, state_fips: Union[ str,  List[str]], var_name:str = 'LILATracts_Vehicle'):
        self.var_name = var_name
        response = requests.get('https://www.ers.usda.gov/data-products/food-access-research-atlas/download-the-data/')
        soup = BeautifulSoup(response.content, "html.parser")
        hrefs = soup.find_all('a', href = True)
        url_path_series = pd.Series([x['href'] for x in hrefs])
        url = url_path_series[url_path_series.str.match(re.compile('.*FoodAccessResearchAtlasData.*', flags = re.I))].values[0]
        self.url = url
        # self.path = f'https://www.ers.usda.gov{url}'
        self.state_fips = state_fips
        
    @property
    def food_desert_data(self):
        if hasattr(self, '_food_desert_data'):
            pass
        else:
            self._food_desert_data = self.download_data(self.state_fips, var_name = self.var_name)
        return self._food_desert_data
    
    def download_data(self, state, var_name):
        from tqdm import tqdm
        import os
        resp = requests.get(self.url, stream=True)
        total = int(resp.headers.get('content-length', 0))
        fname = os.path.join(getcwd(), 'food_desert.xlsx')
        chunk_size = int(1024*1024/2)
        with open(fname, 'wb') as file, tqdm(
            desc="downloading food desert data file",
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            leave = True
        ) as bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)

        df = pd.read_excel(fname, engine = 'openpyxl', sheet_name = 2, dtype = {"CensusTract":str}) # attention
        df['CensusTract'] = df.CensusTract.str.zfill(11) # census tract fips is 11 digits code
        df['State'] = df.CensusTract.apply(lambda x: str(x)[:2])
        if isinstance(state, str):
            assert len(state) == 2
            df = df.loc[df.State.eq(state)].reset_index(drop = True)
        else:
            for s in state:
                assert len(s) == 2
            df = df.loc[df.State.isin(state),:].reset_index(drop = True)
        df = df[['CensusTract', var_name, 'OHU2010']]
        df2 = df.copy()
        data_dictionary = {}
        # Tract
        df.rename(columns = {'CensusTract':'FIPS'}, inplace = True)
        df['FIPS'] = df.FIPS.astype(str)
        df.drop('OHU2010', axis = 1, inplace = True)
        data_dictionary['Tract'] = df
        # County
        df2['FIPS'] = [str(x)[:5] for x in df2.CensusTract]
        df2 = df2.loc[df2.OHU2010.gt(0),['FIPS',var_name,'OHU2010']]
        df2 = df2[['FIPS',var_name,'OHU2010']].groupby('FIPS', as_index = False).apply(lambda x: pd.Series(np.average(x[var_name], weights=x['OHU2010'])))
        df2.columns = ['FIPS',var_name]
        df2['FIPS'] = df2.FIPS.astype(str)
        data_dictionary['County'] = df2
        remove(fname)
        
        return data_dictionary
    
                   
############################################################################
## EJScreen         ########################################################
############################################################################

class ejscreen:  
    def __init__(self, state_fips: Union[ str,  List[str]]):
        self.name = 'https://gaftp.epa.gov/EJScreen/2024/2.32_August_UseMe/EJScreen_2024_Tract_with_AS_CNMI_GU_VI.csv'
        self.path = self.name + '.zip'
        self.state_fips = state_fips
        
    @property
    def ejscreen_data(self):
        if hasattr(self, '_ejscreen_data'):
            pass
        else:
            self._ejscreen_data = self.download_data(self.state_fips)
        return self._ejscreen_data
    
    def download_data(self, state):
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        url = urllib.request.urlopen(self.path)
        
        with ZipFile(BytesIO(url.read()), 'r') as zip:
            df = pd.read_csv(zip.open(zip.namelist()[0]), dtype={'ID':str})
        
        df = df[['ID', 'PM25', 'OZONE', 'DSLPM', 'RSEI_AIR', 'PTRAF', 
                 'PRE1960PCT', 'PNPL', 'PRMP', 'PTSDF', 'UST', 'PWDIS', 'NO2', 'DWATER']]
        
        df.rename(columns = {"ID" : "CensusTract", "PRE1960PCT" : 'Lead Paint', "DSLPM" : "Diesel PM", 
                              "PTRAF" : "Traffic Proximity", "PWDIS" : "Water Discharge", 
                              "PNPL" : "Superfund Proximity", "PRMP" : "RMP Proximity",
                              "PTSDF" : "Hazardous Waste Proximity", "OZONE" : "Ozone", 
                              "UST" : "Underground Storage Tanks", "RSEI_AIR": "Toxics Release to Air",
                              "NO2" : "Nitrogen Dioxide", "DWATER" : "Drinking Water Noncompliance"}, inplace=True)
        
        df['CensusTract'] = df.CensusTract.str.zfill(11) # census tract fips is 11 digits code
        df['State'] = df.CensusTract.apply(lambda x: str(x)[:2])
        if isinstance(state, str):
            assert len(state) == 2
            df = df.loc[df.State.eq(state)].reset_index(drop = True)
        else:
            for s in state:
                assert len(s) == 2
            df = df.loc[df.State.isin(state),:].reset_index(drop = True)
        data_dictionary = {}
        # Tract
        df.rename(columns = {'CensusTract':'FIPS'}, inplace = True)
        df['FIPS'] = df.FIPS.astype(str)
        data_dictionary['Tract'] = df
        
        return data_dictionary 
   
############################################################################
## FCC Broadband Availability         ######################################
############################################################################ 
   
class fcc_avail:
    def __init__(self, state_fips: Union[ str,  List[str]]):
        self.state_fips = state_fips
        self.s = requests.Session()
        self.s.headers.clear()
        self.s.headers.update({'username': 'ciodata@uky.edu'})
        self.s.headers.update({'hash_value': '5rCA9+H/ZXpgmYdZ5DC32JWlOg997zTP9D6izroMSzE='})
        self.s.headers.update({'user-agent': 'play/0.0.0'})
    
    @property
    def fcc_avail_data(self):
        if hasattr(self, '_fcc_avail_data'):
            pass
        else:
            self._fcc_avail_data = self.download_data(self.state_fips, self.s)
        return self._fcc_avail_data
    
    def download_data(self, state, req):
        stateDict = stateDf.set_index('StateAbbrev')['State'].to_dict()
        
        url = "https://broadbandmap.fcc.gov/api/public/map/listAsOfDates"
        r = req.get(url, timeout=5)
        parsed = json.loads(r.text)
        dates = pd.DataFrame(parsed['data'])
        dates['date'] = dates.as_of_date.str[:10]
        as_of_date = max(dates.date[dates.data_type == 'availability'])

        # as_of_date = '2023-06-30'
        url2 = f"https://broadbandmap.fcc.gov/api/public/map/downloads/listAvailabilityData/{as_of_date}"
        # url2 = f"https://broadbandmap.fcc.gov/api/public/map/downloads/listAvailabilityData/2023-06-30"
        r2 = req.get(url2, timeout=5)
        parsed2 = json.loads(r2.text)
        filingData = pd.DataFrame(parsed2['data'])

        ### select summary by geography
        baseUrl = ("https://broadbandmap.fcc.gov/api/public/map/downloads//downloadFile/availability/", "/1")
        
        fbAvailability = filingData[(filingData.category == 'Summary') & 
                                       (filingData.subcategory == 'Summary by Geography Type - Other Geographies') &
                                       (filingData.technology_type == 'Fixed Broadband')].reset_index()
        
        mbAvailability = filingData[(filingData.category == 'Summary') & 
                                       (filingData.subcategory == 'Summary by Geography Type - Other Geographies') &
                                       (filingData.technology_type == 'Mobile Broadband')].reset_index()

        columnHints={
            'geography_id': str
        }
        
        print(fbAvailability['file_name'][0])
        # print(url)
        url_fb = f"{baseUrl[0]}{fbAvailability['file_id'][0]}{baseUrl[1]}"
        r3 = req.get(url_fb)
        data_fb = r3.content
        zip = ZipFile(BytesIO(data_fb))
        fb = pd.read_csv(zip.open(zip.filelist[0].filename), dtype=columnHints) 
        subsetFb = fb[(fb.geography_type == 'County') & (fb.area_data_type == 'Total') & 
                      (fb.biz_res == 'R') & (fb.technology == 'Any Technology')].reset_index()
        subsetFb2 = subsetFb.loc[:, ['geography_id', 'geography_desc_full', 'speed_100_20', 'speed_1000_100']]
        del fb, subsetFb
        
        print(mbAvailability['file_name'][0])
        # print(url)
        url_mb = f"{baseUrl[0]}{mbAvailability['file_id'][0]}{baseUrl[1]}"
        r4 = req.get(url_mb)
        data_mb = r4.content
        zip = ZipFile(BytesIO(data_mb))
        mb = pd.read_csv(zip.open(zip.filelist[0].filename), dtype=columnHints) 
        subsetMb = mb[(mb.geography_type == 'County') & (mb.area_data_type == 'Total')].reset_index()
        subsetMb2 = subsetMb.loc[:, ['geography_id', 
                                     'geography_desc', 
                                     'mobilebb_5g_spd1_area_st_pct', 
                                     'mobilebb_5g_spd2_area_st_pct']].rename(columns = {'geography_desc': 'geography_desc_full'})
        del mb, subsetMb
        
        df = pd.merge(subsetFb2, subsetMb2, on=['geography_id', 'geography_desc_full'])
        df[['County', 'State']] = df.geography_desc_full.str.split(', ', expand=True)
        df['State'] = df['State'].replace(stateDict)
        df = df.rename(columns = {'geography_id': 'FIPS',
                                              'speed_100_20': 'pctBB_100_20',
                                              'speed_1000_100': 'pctBB_1000_10',
                                              'mobilebb_5g_spd1_area_st_pct': 'pct5G_7_1',
                                              'mobilebb_5g_spd2_area_st_pct': 'pct5G_35_3'})
        df = df[['FIPS', 'County', 'State', 'pctBB_1000_10', 'pctBB_100_20', 'pct5G_7_1', 'pct5G_35_3']]
        df['FIPS2'] = df.FIPS.apply(lambda x: str(x)[:2])
        if isinstance(state, str):
            assert len(state) == 2
            df2 = df.loc[df.FIPS2.eq(state)].reset_index(drop = True)
        else:
            for s in state:
                assert len(s) == 2
            df2 = df.loc[df.FIPS2.isin(state),:].reset_index(drop = True)
        df2.drop('FIPS2', axis = 1, inplace = True)
        
        return df2
    
############################################################################
## scp_cancer_data      ####################################################   -> multiprocessing
############################################################################

# reference (incidence) : https://www.statecancerprofiles.cancer.gov/incidencerates/index.php
# reference (mortality) : https://www.statecancerprofiles.cancer.gov/deathrates/index.php
    
@dataclass
class scp_cancer_data:
    state_fips : Union[str, List[str]] #
    folder_name : str = 'cancer_data'
        
    @property
    def cancer_data(self):
        if hasattr(self, '_cancer_data'):
            pass
        else:
            import shutil
            import os
            data_dict = {}
            # data_dict['incidence'] = self.scp_cancer_inc()
            data_dict['incidence'] = self.uscs_incidence()
            data_dict['mortality'] = self.scp_cancer_mor()
            data_dict['incidence']['AAR'] = self.convert_dtype(data_dict['incidence'].AAR)
            data_dict['mortality']['AAR'] = self.convert_dtype(data_dict['mortality'].AAR)
            self._cancer_data = data_dict
            shutil.rmtree(os.path.join(os.getcwd(), self.folder_name))
        return self._cancer_data
        
    @staticmethod
    def convert_dtype(series):
        series = series.copy()
        series = series.astype(str).str.strip()
        series[~series.str.contains('\d+\.?\d*')] = np.nan    
        series = series.astype(float)
        return series        
        
    def uscs_incidence(self):
        file_path = glob('./catchment_area/uscs/uscs_cancer_incidence_county_2017-2021.csv')
        df = pd.read_csv(file_path[0])
        df.FIPS = df.FIPS.astype(str).str.zfill(5)
        return df
        
    def scp_cancer_inc(self):
        sites = {'001': 'All Site', '071': 'Bladder', '076': 'Brain & ONS', '020': 'Colon & Rectum', '017': 'Esophagus', 
                 '072': 'Kidney & Renal Pelvis', '090': 'Leukemia', '035': 'Liver & IBD', '047': 'Lung & Bronchus',
                 '053': 'Melanoma of the Skin', '086': 'Non-Hodgkin Lymphoma', '003': 'Oral Cavity & Pharynx', '040': 'Pancreas',
                 '018': 'Stomach', '080': 'Thyroid'}

        sitesf = {'001': 'All Site', '071': 'Bladder', '076': 'Brain & ONS', '020': 'Colon & Rectum', '017': 'Esophagus', 
                 '072': 'Kidney & Renal Pelvis', '090': 'Leukemia', '035': 'Liver & IBD', '047': 'Lung & Bronchus',
                 '053': 'Melanoma of the Skin', '086': 'Non-Hodgkin Lymphoma', '003': 'Oral Cavity & Pharynx', '040': 'Pancreas',
                 '018': 'Stomach', '080': 'Thyroid', '055': 'Female Breast', '057': 'Cervix', '061': 'Ovary', '058': 'Corpus Uteri & Uterus, NOS'}
        
        sitesm = {'001': 'All Site', '071': 'Bladder', '076': 'Brain & ONS', '020': 'Colon & Rectum', '017': 'Esophagus', 
                 '072': 'Kidney & Renal Pelvis', '090': 'Leukemia', '035': 'Liver & IBD', '047': 'Lung & Bronchus',
                 '053': 'Melanoma of the Skin', '086': 'Non-Hodgkin Lymphoma', '003': 'Oral Cavity & Pharynx', '040': 'Pancreas',
                 '018': 'Stomach', '080': 'Thyroid', '066': 'Prostate'}
        
        # re = {'00': 'All', '07': 'White NH', '28': 'Black NH', '05': 'Hispanic'} #, '38': 'AIAN', '48': 'API'
        re = {'00': 'All', '07': 'White NH', '28': 'Black NH', '05': 'Hispanic', '38': 'American Indian/Alaska Native NH', 
              '48': 'Asian/Pacific Islander NH'}
        
        gen_single_cancer_inc_all = partial(self.gen_single_cancer_inc, sex = '0', folder_name = self.folder_name)
        gen_single_cancer_inc_male = partial(self.gen_single_cancer_inc, sex = '1', folder_name = self.folder_name)
        gen_single_cancer_inc_female = partial(self.gen_single_cancer_inc, sex = '2', folder_name = self.folder_name)

        incidence_all = Parallel(n_jobs=-1)(
            delayed(gen_single_cancer_inc_all)(
                site[0], site[1], re[0], re[1]) for site, re in product(sites.items(), re.items()))
        incidence_female = Parallel(n_jobs=-1)(
            delayed(gen_single_cancer_inc_female)(
                site[0], site[1], re[0], re[1]) for site, re in product(sitesf.items(), re.items()))
        incidence_male = Parallel(n_jobs=-1)(
            delayed(gen_single_cancer_inc_male)(
                site[0], site[1], re[0], re[1]) for site, re in product(sitesm.items(), re.items()))
            
        df = pd.concat(incidence_all + incidence_female + incidence_male, axis = 0).sort_values(['FIPS','Site']).reset_index(drop = True)
        if df.FIPS.eq('51917').sum(): # if we find 51917 in FIPS
            vaFix = {'51917': '51019', 'Bedford City and County' : 'Bedford County'}
            df = df.replace(vaFix)
        return df

        
        
    @staticmethod
    def gen_single_cancer_inc(cancer_site_id:str, cancer_site:str, re_id:str, re_g:str, sex:int, folder_name:str):
        from requests.adapters import HTTPAdapter
        from requests.packages.urllib3.util.retry import Retry
        
        retry_strategy = Retry(
          total=3,
          backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)
        http.mount("http://", adapter)
        
        assert sex in list('012')
        assert len(cancer_site_id) == 3
        assert len(re_id) == 2
        sex_n = {'0': 'All', '1': 'Male', '2': 'Female'}
        # API get
        # headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'}
        path = f'https://www.statecancerprofiles.cancer.gov/incidencerates/index.php?stateFIPS=00&areatype=county&cancer={cancer_site_id}&race={re_id}&sex={sex}&age=001&stage=999&year=0&type=incd&sortVariableName=rate&sortOrder=desc&output=1'
        resp = http.get(path)        
        resp.raise_for_status()
        
        folder_dir = os.path.join(os.getcwd(), folder_name)                                                     
        # first we will create "cancer_data" directory and download the csv file
        if len(glob(folder_dir)) == 0: # if we don't yet have 'cancer_data' directory
            os.mkdir(folder_dir)
        # We then will select row that are relevant
        flag = False
        fname = os.path.join(folder_dir, f'incidence_us_{cancer_site_id}_{sex}_{re_id}.csv')
                                                             # file name will be unique for each query
        with open(fname, 'w') as f:
            for row in resp.iter_lines(decode_unicode = True): # go through response
                if row[:6] in ['County', 'Area, ']:
                    flag = True
                    row = row.replace('Area', 'County').replace(', ',',').replace(' ,','').replace(' ','')
                elif flag & (row== ''):
                    flag = False
                if flag:
                    f.write(row)
                    f.write('\n')
        # read the file with csv.DictReader
        reader = DictReader(open(fname, 'r'))
        if reader.fieldnames:
            # find relevant field name (AAR and AAC)
            fieldnames = pd.Series(reader.fieldnames)
            AAR_field_name = fieldnames[fieldnames.str.contains('^Age-Adjusted', flags = re.I)].values[0]
            AAC_field_name = fieldnames[fieldnames.str.contains('Count$', flags = re.I)].values[0]
            # Go through reader
            stateDict = stateDf.set_index('FIPS2')['State'].to_dict()

            FIPS = []
            County = []
            State = []
            RE = []
            Sex = []
            AAR  = []
            AAC  = []
            colname = ['FIPS','County','State','Type','Site', 'RE', 'Sex', 'AAR','AAC']
            for row in reader:
                if row['FIPS'][:2] in stateDf.FIPS2.tolist():
                    FIPS.append(row['FIPS'])
                    # County.append(row['County'].rstrip('\(0123456789\)'))
                    County.append(row['County'].split(", ")[0])
                    State.append(stateDict[row['FIPS'][:2]])
                    try:
                        row[AAR_field_name] = float(row[AAR_field_name])
                    except:
                        row[AAR_field_name] = None
                    AAR.append(row[AAR_field_name])
                    try:
                        row[AAC_field_name] = int(row[AAC_field_name])
                    except:
                        row[AAC_field_name] = None
                    AAC.append(row[AAC_field_name])
            Type = ['Incidence' for _ in range(len(FIPS))]
            Site = [cancer_site for _ in range(len(FIPS))]
            RE = [re_g for _ in range(len(FIPS))]
            Sex = [sex_n[sex] for _ in range(len(FIPS))]
            df = pd.DataFrame(zip(FIPS, County, State, Type, Site, RE, Sex, AAR, AAC), columns = colname)
            df = df.sort_values('FIPS').reset_index(drop = True)
            del FIPS, County, State, Type, Site, RE, Sex, AAR, AAC, reader, resp
            remove(fname)
            return df
        else:
            return pd.DataFrame(None, columns = ['FIPS','County','State','Type','Site', 'RE', 'Sex', 'AAR','AAC'])
            
#         df = pd.read_csv(path, skiprows=11, header=None, usecols=[0,1,2,8],  names=['County', 'FIPS', 'AAR', 'AAC'],
#                          dtype={'County':str, 'FIPS':str}).dropna()
#         df['County'] = df['County'].map(lambda x: x.rstrip('\(0123456789\)'))
#         df['Site'] = cancer_site
#         df['Type'] = 'Incidence'
#         df['State'] = stateDf.loc[stateDf.FIPS2.eq(state), 'State'].values[0]
#         df['AAR'] = df.AAR.replace('* ', np.nan).astype(float)

#         df = df[['FIPS', 'County', 'State', 'Type', 'Site', 'AAR', 'AAC']].sort_values('FIPS')
#         return df
        

    def scp_cancer_mor(self):
        sites = {'001': 'All Site', '071': 'Bladder', '076': 'Brain & ONS', '020': 'Colon & Rectum', '017': 'Esophagus', 
                 '072': 'Kidney & Renal Pelvis', '090': 'Leukemia', '035': 'Liver & IBD', '047': 'Lung & Bronchus',
                 '053': 'Melanoma of the Skin', '086': 'Non-Hodgkin Lymphoma', '003': 'Oral Cavity & Pharynx', '040': 'Pancreas',
                 '018': 'Stomach', '080': 'Thyroid'}

        sitesf = {'001': 'All Site', '071': 'Bladder', '076': 'Brain & ONS', '020': 'Colon & Rectum', '017': 'Esophagus', 
                 '072': 'Kidney & Renal Pelvis', '090': 'Leukemia', '035': 'Liver & IBD', '047': 'Lung & Bronchus',
                 '053': 'Melanoma of the Skin', '086': 'Non-Hodgkin Lymphoma', '003': 'Oral Cavity & Pharynx', '040': 'Pancreas',
                 '018': 'Stomach', '080': 'Thyroid', '055': 'Female Breast', '057': 'Cervix', '061': 'Ovary', 
                 '058': 'Corpus Uteri & Uterus, NOS'}
        
        sitesm = {'001': 'All Site', '071': 'Bladder', '076': 'Brain & ONS', '020': 'Colon & Rectum', '017': 'Esophagus', 
                 '072': 'Kidney & Renal Pelvis', '090': 'Leukemia', '035': 'Liver & IBD', '047': 'Lung & Bronchus',
                 '053': 'Melanoma of the Skin', '086': 'Non-Hodgkin Lymphoma', '003': 'Oral Cavity & Pharynx', '040': 'Pancreas',
                 '018': 'Stomach', '080': 'Thyroid','066': 'Prostate'}
        
        # re = {'00': 'All', '07': 'White NH', '28': 'Black NH', '05': 'Hispanic'} #, '38': 'AIAN', '48': 'API'
        re = {'00': 'All', '07': 'White NH', '28': 'Black NH', '05': 'Hispanic', '38': 'American Indian/Alaska Native NH', 
              '48': 'Asian/Pacific Islander NH'}
        
        gen_single_cancer_mor_all = partial(self.gen_single_cancer_mor, sex = '0', folder_name = self.folder_name)
        gen_single_cancer_mor_male = partial(self.gen_single_cancer_mor, sex = '1', folder_name = self.folder_name)
        gen_single_cancer_mor_female = partial(self.gen_single_cancer_mor, sex = '2', folder_name = self.folder_name)

        mortality_all = Parallel(n_jobs=-1)(
            delayed(gen_single_cancer_mor_all)(
                site[0], site[1], re[0], re[1]) for site, re in product(sites.items(), re.items()))
        mortality_female = Parallel(n_jobs=-1)(
            delayed(gen_single_cancer_mor_female)(
                site[0], site[1], re[0], re[1]) for site, re in product(sitesf.items(), re.items()))
        mortality_male = Parallel(n_jobs=-1)(
            delayed(gen_single_cancer_mor_male)(
                site[0], site[1], re[0], re[1]) for site, re in product(sitesm.items(), re.items()))
            
        df = pd.concat(mortality_all + mortality_female + mortality_male, axis = 0).sort_values(['FIPS','Site']).reset_index(drop = True)
        if df.FIPS.eq('51917').sum(): # if we find 51917 in FIPS
            vaFix = {'51917': '51019', 'Bedford City and County' : 'Bedford County'}
            df = df.replace(vaFix)
        return df

        
    @staticmethod
    def gen_single_cancer_mor(cancer_site_id:str, cancer_site:str, re_id:str, re_g:str, sex:int, folder_name:str):
        from requests.adapters import HTTPAdapter
        from requests.packages.urllib3.util.retry import Retry
        
        retry_strategy = Retry(
          total=3,
          backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)
        http.mount("http://", adapter)
        
        assert sex in list('012')
        assert len(cancer_site_id) == 3
        assert len(re_id) == 2
        sex_n = {'0': 'All', '1': 'Male', '2': 'Female'}
        path = f'https://www.statecancerprofiles.cancer.gov/deathrates/index.php?stateFIPS=00&areatype=county&cancer={cancer_site_id}&race={re_id}&sex={sex}&age=001&year=0&type=death&sortVariableName=rate&sortOrder=desc&output=1'
        resp = http.get(path)        
        resp.raise_for_status()
        
        # first we will create "cancer_data" directory and download the csv file
                                                             
        folder_dir = os.path.join(os.getcwd(), folder_name)                                                     
        # first we will create "cancer_data" directory and download the csv file
        if len(glob(folder_dir)) == 0: # if we don't yet have 'cancer_data' directory
            os.mkdir(folder_dir)
        # We then will select row that are relevant
        flag = False
        fname = os.path.join(folder_dir, f'mortality_us_{cancer_site_id}_{sex}_{re_id}.csv')
                                                             # file name will be unique for each query
        with open(fname, 'w') as f:
            for row in resp.iter_lines(decode_unicode = True): # go through response
                if row[:6] in ['County', 'Area, ']:
                    flag = True
                    row = row.replace('Area', 'County').replace(', ',',').replace(' ,','').replace(' ','')
                elif flag & (row== ''):
                    flag = False
                if flag:
                    f.write(row)
                    f.write('\n')
        # read the file with csv.DictReader
        reader = DictReader(open(fname, 'r'))
        if reader.fieldnames:
            # find relevant field name (AAR and AAC)
            fieldnames = pd.Series(reader.fieldnames)
            AAR_field_name = fieldnames[fieldnames.str.contains('^Age-Adjusted', flags = re.I)].values[0]
            AAC_field_name = fieldnames[fieldnames.str.contains('Count$', flags = re.I)].values[0]
            # Go through reader
            stateDict = stateDf.set_index('FIPS2')['State'].to_dict()

            FIPS = []
            County = []
            State = []
            RE = []
            Sex = []
            AAR  = []
            AAC  = []
            colname = ['FIPS','County','State','Type','Site', 'RE', 'Sex', 'AAR','AAC']
            for row in reader:
                if row['FIPS'][:2] in stateDf.FIPS2.tolist():
                    FIPS.append(row['FIPS'])
                    # County.append(row['County'].rstrip('\(0123456789\)'))
                    County.append(row['County'].split(", ")[0])
                    State.append(stateDict[row['FIPS'][:2]])
                    try:
                        row[AAR_field_name] = float(row[AAR_field_name])
                    except:
                        row[AAR_field_name] = None
                    AAR.append(row[AAR_field_name])
                    try:
                        row[AAC_field_name] = int(row[AAC_field_name])
                    except:
                        row[AAC_field_name] = None
                    AAC.append(row[AAC_field_name])
            Type = ['Mortality' for _ in range(len(FIPS))]
            Site = [cancer_site for _ in range(len(FIPS))]
            RE = [re_g for _ in range(len(FIPS))]
            Sex = [sex_n[sex] for _ in range(len(FIPS))]
            df = pd.DataFrame(zip(FIPS, County, State, Type, Site, RE, Sex, AAR, AAC), columns = colname)
            df = df.sort_values('FIPS').reset_index(drop = True)
            del FIPS, County, State, Type, Site, RE, Sex, AAR, AAC, reader, resp
            remove(fname)
            return df

        else:
            return pd.DataFrame(None, columns = ['FIPS','County','State','Type','Site', 'RE', 'Sex', 'AAR','AAC'])
        
#         df = pd.read_csv(path, skiprows=11, header=None, usecols=[0,1,3,9],  names=['County', 'FIPS', 'AAR', 'AAC'],
#                          dtype={'County':str, 'FIPS':str}).dropna()
#         df['County'] = df['County'].map(lambda x: x.rstrip('\(0123456789\)'))
#         df['Site'] = cancer_site
#         df['Type'] = 'Incidence'
#         df['State'] = stateDf.loc[stateDf.FIPS2.eq(state), 'State'].values[0]
#         df = df[['FIPS', 'County', 'State', 'Type', 'Site', 'AAR', 'AAC']].sort_values('FIPS')
#         return df

##################################################################
## CDC PLACES (county/tract level risk factors and screening data)
##################################################################


@dataclass
class places_data:
    state_fips: Union[str, List[str]] 
    config: SocrataConfig
    
    
    @property
    def state(self):
        if hasattr(self, '_state'):
            pass
        else:
            if isinstance(self.state_fips, str):
                self._state = stateDf.loc[stateDf.FIPS2.eq(self.state_fips),'StateAbbrev'].values[0]
            else:
                self._state = stateDf.loc[stateDf.FIPS2.isin(self.state_fips),'StateAbbrev'].values.tolist()
        return self._state
    
    @property
    def places_data(self):
        if hasattr(self, '_places_data'):
            pass
        else:
            self._places_data = {'county': self.places_county(), 'tract':self.places_tract()}
        return self._places_data
        
        
    def places_county(self):
        if isinstance(self.state, str):
            results = self.config.client.get("i46a-9kgh", where=f'stateabbr="{self.state}"',  limit = 100_000)
        else:
            state = '("' + '","'.join(self.state) + '")'
            results = self.config.client.get("i46a-9kgh", where=f'stateabbr in {state}',  limit = 100_000)
        results_df = pd.DataFrame.from_records(results)
        results_df2 = results_df.loc[:, results_df.columns.isin(['countyfips', 'countyname', 'stateabbr', 'cancer_crudeprev', 
                                                                 'colon_screen_crudeprev', 'csmoking_crudeprev', 
                                                                 'mammouse_crudeprev', 'obesity_crudeprev', 'binge_crudeprev', 
                                                                 'bphigh_crudeprev', 'bpmed_crudeprev', 'casthma_crudeprev', 
                                                                 'chd_crudeprev', 'checkup_crudeprev', 'copd_crudeprev', 
                                                                 'dental_crudeprev', 'depression_crudeprev', 'diabetes_crudeprev', 
                                                                 'ghlth_crudeprev', 'highchol_crudeprev', 'lpa_crudeprev', 
                                                                 'mhlth_crudeprev', 'phlth_crudeprev', 'sleep_crudeprev', 
                                                                 'stroke_crudeprev', 'teethlost_crudeprev',
                                                                 'hearing_crudeprev', 'vision_crudeprev', 'cognition_crudeprev',
                                                                 'mobility_crudeprev', 'selfcare_crudeprev', 'indeplive_crudeprev',
                                                                 'isolation_crudeprev', 'foodinsecu_crudeprev', 'housinsecu_crudeprev',
                                                                 'lacktrpt_crudeprev', 'emotionspt_crudeprev'])]
        if isinstance(self.state, str):
            state_abbr_to_name = {x: stateDf.loc[stateDf.StateAbbrev.eq(x), 'State'].values[0] for x in [self.state]}
        else:
            state_abbr_to_name = {x: stateDf.loc[stateDf.StateAbbrev.eq(x), 'State'].values[0] for x in self.state}
        states_name = results_df2.stateabbr.copy().apply(lambda x: state_abbr_to_name[x]).tolist()
        results_df3 = results_df2.rename(columns={'countyfips': 'FIPS', 'countyname': 'County',  
                                                  'cancer_crudeprev': 'Cancer_Prevalence',
                                                  'colon_screen_crudeprev': 'Met_Colon_Screen', 
                                                  'mammouse_crudeprev': 'Met_Breast_Screen', 
                                                  'csmoking_crudeprev': 'Currently_Smoke',
                                                  'obesity_crudeprev': 'BMI_Obese', 
                                                  'binge_crudeprev' : 'Binge_Drinking',
                                                  'bphigh_crudeprev' : 'High_BP',
                                                  'bpmed_crudeprev' : 'BP_Medicine',
                                                  'casthma_crudeprev' : 'Asthma',
                                                  'chd_crudeprev' : 'CHD',
                                                  'checkup_crudeprev' : 'Recent_Checkup',
                                                  'copd_crudeprev' : 'COPD',
                                                  'dental_crudeprev' : 'Recent_Dentist',
                                                  'depression_crudeprev' : 'Depression',
                                                  'diabetes_crudeprev' : 'Diabetes_DX',
                                                  'ghlth_crudeprev' : 'Bad_Health',
                                                  'highchol_crudeprev' : 'High_Cholesterol', 
                                                  'lpa_crudeprev' : 'Physically_Inactive',
                                                  'mhlth_crudeprev' : 'Poor_Mental',
                                                  'phlth_crudeprev' : 'Poor_Physical',
                                                  'sleep_crudeprev' : 'Sleep_Debt',
                                                  'stroke_crudeprev' : 'Had_Stroke',
                                                  'teethlost_crudeprev' : 'No_Teeth',
                                                  'hearing_crudeprev' : 'Hearing_Disability', 
                                                  'vision_crudeprev' : 'Vision_Disability', 
                                                  'cognition_crudeprev' : 'Cognitive_Disability',
                                                  'mobility_crudeprev' : 'Mobility_Disability', 
                                                  'selfcare_crudeprev' : 'Selfcare_Disability', 
                                                  'indeplive_crudeprev' : 'Independent_Living_Disability',
                                                  'isolation_crudeprev' : 'Socially_Isolated', 
                                                  'foodinsecu_crudeprev' : 'Food_Insecure', 
                                                  'housinsecu_crudeprev' : 'Housing_Insecure',
                                                  'lacktrpt_crudeprev' : 'Lacked_Reliable_Transportation', 
                                                  'emotionspt_crudeprev' : 'Lacked_Social_Emotional_Support'})
        results_df3['State'] = states_name
        results_df3 = results_df3[['FIPS','County','State','Met_Breast_Screen',
                                  'Met_Colon_Screen', 'Currently_Smoke', 
                                  'BMI_Obese', 'Physically_Inactive', 
                                  'Binge_Drinking', 'Sleep_Debt', 'Cancer_Prevalence',  
                                  'Bad_Health', 'Poor_Physical', 
                                  'Poor_Mental', 'Depression',
                                  'Diabetes_DX', 'High_BP', 'High_Cholesterol', 'BP_Medicine', 
                                  'CHD', 'Had_Stroke', 
                                  'Asthma', 'COPD', 'No_Teeth', 
                                  'Recent_Checkup', 'Recent_Dentist',
                                  'Hearing_Disability', 'Vision_Disability', 
                                  'Cognitive_Disability', 'Mobility_Disability', 
                                  'Selfcare_Disability', 'Independent_Living_Disability',
                                  'Socially_Isolated', 'Food_Insecure', 
                                  'Housing_Insecure', 'Lacked_Reliable_Transportation', 
                                  'Lacked_Social_Emotional_Support']]
        results_df3[['Met_Breast_Screen',
                                  'Met_Colon_Screen', 'Currently_Smoke', 
                                  'BMI_Obese', 'Physically_Inactive', 
                                  'Binge_Drinking', 'Sleep_Debt', 'Cancer_Prevalence',  
                                  'Bad_Health', 'Poor_Physical', 
                                  'Poor_Mental', 'Depression',
                                  'Diabetes_DX', 'High_BP', 'BP_Medicine', 
                                  'CHD', 'Had_Stroke', 
                                  'Asthma', 'COPD', 'No_Teeth', 
                                  'Recent_Checkup', 'Recent_Dentist',
                                  'Hearing_Disability', 'Vision_Disability', 
                                  'Cognitive_Disability', 'Mobility_Disability', 
                                  'Selfcare_Disability', 'Independent_Living_Disability',
                                  'Socially_Isolated', 'Food_Insecure', 
                                  'Housing_Insecure', 'Lacked_Reliable_Transportation', 
                                  'Lacked_Social_Emotional_Support']] = results_df3[['Met_Breast_Screen',
                                                            'Met_Colon_Screen', 'Currently_Smoke', 
                                                            'BMI_Obese', 'Physically_Inactive', 
                                                            'Binge_Drinking', 'Sleep_Debt', 'Cancer_Prevalence',  
                                                            'Bad_Health', 'Poor_Physical', 
                                                            'Poor_Mental', 'Depression',
                                                            'Diabetes_DX', 'High_BP', 'BP_Medicine', 
                                                            'CHD', 'Had_Stroke',
                                                            'Asthma', 'COPD', 'No_Teeth', 
                                                            'Recent_Checkup', 'Recent_Dentist',
                                                            'Hearing_Disability', 'Vision_Disability', 
                                                            'Cognitive_Disability', 'Mobility_Disability', 
                                                            'Selfcare_Disability', 'Independent_Living_Disability',
                                                            'Socially_Isolated', 'Food_Insecure', 
                                                            'Housing_Insecure', 'Lacked_Reliable_Transportation', 
                                                            'Lacked_Social_Emotional_Support']].astype(float)
        del results_df, results_df2
        return results_df3
        
    def places_tract(self):
        if isinstance(self.state, str):
            results = self.config.client.get("yjkw-uj5s", where=f'stateabbr="{self.state}"', limit = 100_000)
        else:
            state = '("' + '","'.join(self.state) + '")'
#             while 
            results = self.config.client.get("yjkw-uj5s", where=f'stateabbr in {state}', offset = 0, limit = 100_000)

        results_df = pd.DataFrame.from_records(results)
        results_df2 = results_df.loc[:, results_df.columns.isin(['tractfips', 'countyfips', 'countyname', 
                                                                 'stateabbr', 'cancer_crudeprev', 'mammouse_crudeprev', 
                                                                 'colon_screen_crudeprev', 'csmoking_crudeprev',
                                                                 'obesity_crudeprev', 'binge_crudeprev', 'bphigh_crudeprev', 
                                                                 'bpmed_crudeprev', 'casthma_crudeprev', 'chd_crudeprev', 
                                                                 'checkup_crudeprev', 'copd_crudeprev', 'dental_crudeprev', 
                                                                 'depression_crudeprev', 'diabetes_crudeprev', 'ghlth_crudeprev', 
                                                                 'highchol_crudeprev', 'lpa_crudeprev', 'mhlth_crudeprev', 
                                                                 'phlth_crudeprev', 'sleep_crudeprev', 'stroke_crudeprev', 
                                                                 'teethlost_crudeprev', 'hearing_crudeprev', 'vision_crudeprev', 'cognition_crudeprev',
                                                                 'mobility_crudeprev_', 'selfcare_crudeprev', 'indeplive_crudeprev',
                                                                 'isolation_crudeprev', 'foodinsecu_crudeprev', 'housinsecu_crudeprev',
                                                                 'lacktrpt_crudeprev', 'emotionspt_crudeprev'])]
        
        if isinstance(self.state, str):
            state_abbr_to_name = {x: stateDf.loc[stateDf.StateAbbrev.eq(x), 'State'].values[0] for x in [self.state]} #inefficient...
        else:
            state_abbr_to_name = {x: stateDf.loc[stateDf.StateAbbrev.eq(x), 'State'].values[0] for x in self.state}

        states_name = results_df2.stateabbr.copy().apply(lambda x: state_abbr_to_name[x]).tolist()
        results_df3 = results_df2.rename(columns={'tractfips': 'FIPS', 'countyfips': 'FIPS5', 
                                                  'countyname': 'County',   
                                                  'cancer_crudeprev': 'Cancer_Prevalence',
                                                  'colon_screen_crudeprev': 'Met_Colon_Screen', 
                                                  'mammouse_crudeprev': 'Met_Breast_Screen', 
                                                  'csmoking_crudeprev': 'Currently_Smoke',
                                                  'obesity_crudeprev': 'BMI_Obese', 
                                                  'binge_crudeprev' : 'Binge_Drinking',
                                                  'bphigh_crudeprev' : 'High_BP',
                                                  'bpmed_crudeprev' : 'BP_Medicine',
                                                  'casthma_crudeprev' : 'Asthma',
                                                  'chd_crudeprev' : 'CHD',
                                                  'checkup_crudeprev' : 'Recent_Checkup',
                                                  'copd_crudeprev' : 'COPD',
                                                  'dental_crudeprev' : 'Recent_Dentist',
                                                  'depression_crudeprev' : 'Depression',
                                                  'diabetes_crudeprev' : 'Diabetes_DX',
                                                  'ghlth_crudeprev' : 'Bad_Health',
                                                  'highchol_crudeprev' : 'High_Cholesterol', 
                                                  'lpa_crudeprev' : 'Physically_Inactive',
                                                  'mhlth_crudeprev' : 'Poor_Mental',
                                                  'phlth_crudeprev' : 'Poor_Physical',
                                                  'sleep_crudeprev' : 'Sleep_Debt',
                                                  'stroke_crudeprev' : 'Had_Stroke',
                                                  'teethlost_crudeprev' : 'No_Teeth',
                                                  'hearing_crudeprev' : 'Hearing_Disability', 
                                                  'vision_crudeprev' : 'Vision_Disability', 
                                                  'cognition_crudeprev' : 'Cognitive_Disability',
                                                  'mobility_crudeprev_' : 'Mobility_Disability', 
                                                  'selfcare_crudeprev' : 'Selfcare_Disability', 
                                                  'indeplive_crudeprev' : 'Independent_Living_Disability',
                                                  'isolation_crudeprev' : 'Socially_Isolated', 
                                                  'foodinsecu_crudeprev' : 'Food_Insecure', 
                                                  'housinsecu_crudeprev' : 'Housing_Insecure',
                                                  'lacktrpt_crudeprev' : 'Lacked_Reliable_Transportation', 
                                                  'emotionspt_crudeprev' : 'Lacked_Social_Emotional_Support'})
        results_df3['State'] = states_name

        results_df3 = results_df3[['FIPS','FIPS5','County','State', 'Met_Breast_Screen',
                                  'Met_Colon_Screen', 'Currently_Smoke', 
                                  'BMI_Obese', 'Physically_Inactive', 
                                  'Binge_Drinking', 'Sleep_Debt', 'Cancer_Prevalence',  
                                  'Bad_Health', 'Poor_Physical', 
                                  'Poor_Mental', 'Depression',
                                  'Diabetes_DX', 'High_BP', 'High_Cholesterol', 'BP_Medicine', 
                                  'CHD', 'Had_Stroke', 
                                  'Asthma', 'COPD', 'No_Teeth', 
                                  'Recent_Checkup', 'Recent_Dentist',
                                  'Hearing_Disability', 'Vision_Disability', 
                                  'Cognitive_Disability', 'Mobility_Disability', 
                                  'Selfcare_Disability', 'Independent_Living_Disability',
                                  'Socially_Isolated', 'Food_Insecure', 
                                  'Housing_Insecure', 'Lacked_Reliable_Transportation', 
                                  'Lacked_Social_Emotional_Support']]
        results_df3[['Met_Breast_Screen',
                                  'Met_Colon_Screen', 'Currently_Smoke', 
                                  'BMI_Obese', 'Physically_Inactive', 
                                  'Binge_Drinking', 'Sleep_Debt', 'Cancer_Prevalence',  
                                  'Bad_Health', 'Poor_Physical', 
                                  'Poor_Mental', 'Depression',
                                  'Diabetes_DX', 'High_BP', 'High_Cholesterol', 'BP_Medicine', 
                                  'CHD', 'Had_Stroke', 
                                  'Asthma', 'COPD', 'No_Teeth', 
                                  'Recent_Checkup', 'Recent_Dentist',
                                  'Hearing_Disability', 'Vision_Disability', 
                                  'Cognitive_Disability', 'Mobility_Disability', 
                                  'Selfcare_Disability', 'Independent_Living_Disability',
                                  'Socially_Isolated', 'Food_Insecure', 
                                  'Housing_Insecure', 'Lacked_Reliable_Transportation', 
                                  'Lacked_Social_Emotional_Support']] = results_df3[['Met_Breast_Screen',
                                                            'Met_Colon_Screen', 'Currently_Smoke', 
                                                            'BMI_Obese', 'Physically_Inactive', 
                                                            'Binge_Drinking', 'Sleep_Debt', 'Cancer_Prevalence',  
                                                            'Bad_Health', 'Poor_Physical', 
                                                            'Poor_Mental', 'Depression',
                                                            'Diabetes_DX', 'High_BP', 'High_Cholesterol', 'BP_Medicine', 
                                                            'CHD', 'Had_Stroke', 
                                                            'Asthma', 'COPD', 'No_Teeth', 
                                                            'Recent_Checkup', 'Recent_Dentist',
                                                            'Hearing_Disability', 'Vision_Disability', 
                                                            'Cognitive_Disability', 'Mobility_Disability', 
                                                            'Selfcare_Disability', 'Independent_Living_Disability',
                                                            'Socially_Isolated', 'Food_Insecure', 
                                                            'Housing_Insecure', 'Lacked_Reliable_Transportation', 
                                                            'Lacked_Social_Emotional_Support']].astype(float)
        del results_df, results_df2
        return results_df3

    
##############
# Urban_rural
##############

def urban_rural_counties(state_fips: Union[str, List[str]]):
    urban_rural_counties = pd.read_excel('https://www2.census.gov/geo/docs/reference/ua/2020_UA_COUNTY.xlsx',
                                        dtype = {'STATE':str, 'COUNTY':str})
    if isinstance(state_fips, str):
        urban_rural_counties = urban_rural_counties.loc[urban_rural_counties.STATE.eq(state_fips),['STATE','COUNTY','STATE_NAME','COUNTY_NAME','POPPCT_URB']]
    else:
        urban_rural_counties = urban_rural_counties.loc[urban_rural_counties.STATE.isin(state_fips),['STATE','COUNTY','STATE_NAME','COUNTY_NAME','POPPCT_URB']]
    urban_rural_counties['FIPS'] = urban_rural_counties.STATE + urban_rural_counties.COUNTY
    urban_rural_counties = urban_rural_counties[['FIPS','COUNTY_NAME','STATE_NAME','POPPCT_URB']]
    urban_rural_counties.rename(columns = {'COUNTY_NAME': 'County','STATE_NAME':'State','POPPCT_URB':'Urban_Percentage'}, inplace = True)
    urban_rural_counties['Urban_Percentage'] = urban_rural_counties.Urban_Percentage #* 0.01
    urban_rural_counties['County'] = urban_rural_counties.County + ' County'
    urban_rural_counties = urban_rural_counties.sort_values('FIPS').reset_index(drop = True)
    return urban_rural_counties



    
if __name__ == '__main__':
    import argparse
    import pickle
    import os
    from tqdm import tqdm
    from utils import check_ca_file    

    parser = argparse.ArgumentParser()
    # where to save the data
    parser.add_argument('--download_dir', required = False, default = None)
    # arguments for acs config
    parser.add_argument('--ca_file_path', help = 'catchment area csv file name', required = False, default = None) #uky_ca.csv
    parser.add_argument('--state_fips', nargs = '+', type = int, required = False, default = None)
    parser.add_argument('--census_api_key', required = True)
    parser.add_argument('--query_level', nargs = '+', required = True, 
                        choices = ['county subdivision','tract','block', 'county', 'state','zip','puma'])
    parser.add_argument('--year', required = True, type = int)
    # argument for SocrataConfig
    parser.add_argument('--socrata_user_name', required = False, default = None)
    parser.add_argument('--socrata_password', required = False, default = None)

    args = parser.parse_args()
    
    
    if args.ca_file_path:
        ca_file_path = check_ca_file(args.ca_file_path)
        ca = pd.read_csv(ca_file_path, dtype={'FIPS':str})
        ca['FIPS'] = ca.FIPS.str.zfill(5)
        state_FIPS = ca.FIPS.apply(lambda x: x[:2]).unique().tolist()
        if len(state_FIPS) == 1:
            state_fips = state_FIPS[0]
        else:
            state_fips = state_FIPS
    
    elif args.state_fips:
        if len(args.state_fips) == 1:
            state_fips = str(args.state_fips[0])
        else:
            state_fips = [str(x) for x in args.state_fips]
    else:
        raise AttributeError("You must provide either catchment area csv file or list of state fips")
        
        
    ####### query level can be multiple
   
    sdoh_by_query_level = {}

    # Setting tqdm bar
    pbar = tqdm(range(5), desc = "collecting acs data", leave = False)

    
    #### Step 1: ACS data
    
    for query_level in args.query_level:
        sdoh = acs_sdoh(args.year, state_fips, query_level, key = args.census_api_key)
        sdoh_by_query_level[query_level] = sdoh.cancer_infocus_download()

    # update tqdm
    pbar.update(1)
    pbar.set_description("collecting cancer data")

    
    #### Step 2: Cancer Data        
    # cancer data
    for level in args.query_level:
        if level not in ['county','tract']:
            import warnings 
            warnings.warn("cancer data is only avaialbe at the county level")
            break
    cancer = scp_cancer_data(state_fips)
    sdoh_by_query_level['cancer'] = cancer.cancer_data 
        
        
        
    # update tqdm
    pbar.update(1)
    pbar.set_description("collecting facility data")
    
    
    #### Step 3: Facility Data
    # facility data
    from utils import stateDf
    if isinstance(state_fips, str):
        location = stateDf.loc[stateDf.FIPS2.eq(state_fips), 'StateAbbrev'].values[0]
    else:
        location = stateDf.loc[stateDf.FIPS2.isin(state_fips), 'StateAbbrev'].values.tolist()
    
    sdoh_by_query_level['facility'] = gen_facility_data(location)
    sdoh_by_query_level['facility']['all'] = pd.concat(sdoh_by_query_level['facility'].values(), axis = 0).reset_index(drop = True)
    

    
    ##################################
    ## Append other datasets to sdoh_by_query_level
    ##################################
    
    # Note: they are either in county or tract level. So we must make sure sdoh_by_query_level have 
    # county or tract level
    
    if 'county' not in sdoh_by_query_level.keys():
        sdoh_by_query_level['county'] = {}
    if 'tract' not in sdoh_by_query_level.keys():
        sdoh_by_query_level['tract']  = {}
    
    
    
    # risk_and_screening
    if args.socrata_user_name:
        kwargs = {"domain": "data.cdc.gov",
              "app_token": "nx4zQ2205wpLwaaaZeZp9zAOs",
                 "user_name": args.socrata_user_name,
                 "password": args.socrata_password}
    else:
        kwargs = {"domain": "data.cdc.gov",
              "app_token": "nx4zQ2205wpLwaaaZeZp9zAOs"}

    cfg = SocrataConfig(**kwargs)

    def cdc_risk_and_screening(state_fips = state_fips, cfg = cfg):
        cdc = places_data(state_fips, cfg)
        return cdc.places_data
    
    # BLS
    def bls_func(state_fips = state_fips):
        bls = BLS(state_fips)
        return bls.bls_data
    
    # Food Desert
    def food_desert_func(state_fips = state_fips):
        fd = food_desert(state_fips)
        return fd.food_desert_data
    
    # Food Desert
    def ejscreen_func(state_fips = state_fips):
        ej = ejscreen(state_fips)
        return ej.ejscreen_data
    
    # FCC Broadband Availability
    def fcc_func(state_fips = state_fips):
        fcc = fcc_avail(state_fips)
        return fcc.fcc_avail_data  
    
    # # Water Violation (multiprocessing if state_fips is a list)
    # def water_violation_func(state_fips = state_fips):
    #     wv = water_violation(state_fips)
    #     return wv.water_violation_data
            
    # Urban Rural
    def urban_rural_func(state_fips = state_fips):
        return urban_rural_counties(state_fips)
    
    # superfund
    def superfund_func(state_fips = state_fips):
        return superfund(state_fips)
    
    ### Step 4: Other Data    
    # Using joblibs, retrieve and allocation datasets concurrently
    # water violation is the only function that runs concurrently if state_fips is a list of more than one state_fips code
    # update tqdm
    pbar.update(1)
    pbar.set_description("collecting bls, food desert, urban-rural-counties, and risk-and-screening data")

    if isinstance(state_fips, str):
        functions = [food_desert_func, ejscreen_func, cdc_risk_and_screening, bls_func, urban_rural_func, 
                     superfund_func, fcc_func]
        dataset_name = ['food_desert', 'ejscreen', 'cdc','bls','urban_rural','superfund', 'fcc']
        res = Parallel(n_jobs = -1)(delayed(f)() for f in functions)
        other_data = {k: v for k,v in zip(dataset_name, res)}
    else:
        functions = [food_desert_func, ejscreen_func, cdc_risk_and_screening, bls_func, 
                     urban_rural_func, superfund_func, fcc_func]
        dataset_name = ['food_desert', 'ejscreen', 'cdc','bls', 'urban_rural', 'superfund', 'fcc']
        res = Parallel(n_jobs = -1)(delayed(f)() for f in functions)
        other_data = {k: v for k,v in zip(dataset_name, res)}
        
    
    # appending cdc
    sdoh_by_query_level['county']['risk_and_screening'] = other_data['cdc']['county']
    sdoh_by_query_level['tract']['risk_and_screening']  = other_data['cdc']['tract']
    
    # appending bls
    sdoh_by_query_level['county']['bls_unemployment']   = other_data['bls']
    
    # appending food_desert
    sdoh_by_query_level['county']['food_desert'] = other_data['food_desert']['County']
    sdoh_by_query_level['tract']['food_desert']  = other_data['food_desert']['Tract']
    
    # appending ejscreen
    sdoh_by_query_level['tract']['ejscreen']  = other_data['ejscreen']['Tract']
    
    # appending fcc
    sdoh_by_query_level['county']['fcc']  = other_data['fcc']

    # appending urban_rural
    sdoh_by_query_level['county']['urban_rural'] = other_data['urban_rural']
    
    # appending superfund
    sdoh_by_query_level['facility']['superfund'] = other_data['superfund']
    
    pbar.update(1)
    pbar.set_description("data collection is complete")

    
    
    
    
    if args.download_dir:
        file_path = os.path.join(args.download_dir, 'cif_raw_data.pickle')
    else:
        file_path = os.path.join(os.getcwd(), 'cif_raw_data.pickle')
                        
    with open(file_path, 'wb') as dataset:
        pickle.dump(sdoh_by_query_level, dataset, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'dataset is stored at {file_path}')