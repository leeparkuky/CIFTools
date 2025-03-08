import asyncio
from aiohttp import ClientSession
import pandas as pd
import re
from dataclasses import dataclass
from typing import Union, List
import requests
import dotenv
dotenv.load_dotenv()


def gen_variable_names(year: Union[str, int], 
                       acs_type: Union[str, List[str]], 
                       group_id:Union[str, List[str]] = None) -> List[str]:
    """
    This function retunrs a list of ACS groups IDs available for each acs types: acs, profile, and subject.
    """
    if isinstance(acs_type, str):
        if acs_type in ['','profile','subject']:
            pass
        else:
            raise ValueError
        if acs_type != '':
            url = f'https://api.census.gov/data/{year}/acs/acs5/{acs_type}/variables'
        else:
            url = f'https://api.census.gov/data/{year}/acs/acs5/variables'
        resp = requests.get(url)
        json_raw =  resp.json()

    elif isinstance(acs_type, list):
        urls = []
        for at in acs_type:
            if at in ['','profile','subject']:
                if at != '':
                    url = f'https://api.census.gov/data/{year}/acs/acs5/{at}/variables'
                    urls.append(url)
                else:
                    url = f'https://api.census.gov/data/{year}/acs/acs5/variables'
                    urls.append(url)
            else:
                raise ValueError
        urls = pd.Series(urls).unique().tolist()
        json_raw = []
        for url in urls:
            resp = requests.get(url)
            json_raw += resp.json()

    if group_id:
        if isinstance(group_id, str):
            variables = [x[0] for x in json_raw if re.match(f"{group_id}_\d+E",x[0])]
        else:
            variables = []
            for gid in group_id:
                variables += [x[0] for x in json_raw if re.match(f"{gid}_\d+E",x[0])]
    else:
        variables = [x[0] for x in json_raw if re.match(".+_\d+E",x[0])]
    variables.sort()
    
    return variables, json_raw


async def gen_group_names(year: Union[str, int], acs_type: str, session: ClientSession) -> List[str]:
    """
    This function retunrs a list of ACS groups IDs available for each acs types: acs, profile, and subject.
    """
    if acs_type in ['','profile','subject']:
        pass
    else:
        raise ValueError
    if acs_type != '':
        url = f'https://api.census.gov/data/{year}/acs/acs5/{acs_type}/groups'
    else:
        url = f'https://api.census.gov/data/{year}/acs/acs5/groups'
    
    resp = await session.request(method="GET", url=url)
    resp.raise_for_status()    
    json_raw =  await resp.json()
    groups = [x['name'] for x in json_raw['groups']]
    return groups

async def groups(year: Union[str, int]):
    async with ClientSession() as session:
        tasks = [gen_group_names(year, acs_class, session) for acs_class in ['','profile','subject']]
        return await asyncio.gather(*tasks)
    
def gen_group_names_acs(config):
    year = config.year
    result = asyncio.run(groups(year))
    output = []
    for r in result:
        output += r
    del result
    return output

def check_acs_type(config):
    year = config.year
    result = asyncio.run(groups(year))
    acs_class = ['','profile','subject']
    output = None
    if isinstance(config.acs_group, str):
        for acs, r in zip(acs_class, result):
            if config.acs_group in r:
                output = acs
                break
        if output is None:
            raise AttributeError("Check the ACS group id")
        else:
            return output
    elif isinstance(config.acs_group, list):
        for acs_group in config.acs_group:
            for acs, r in zip(acs_class, result):
                if acs_group in r:
                    if output == None:
                        output = [acs]
                    else:
                        output.append(acs)
        if len(output) != len(config.acs_group):
            raise AttributeError("Check the ACS group id")
        else:
            if pd.Series(output).unique().shape[0] == 1:
                return output[0]
            else:
                raise AttributeError("All the groups must be in the same acs_type")



@dataclass
class ACSConfig:
    year: Union[str, int]
    state_fips: Union[str, int, List[str], List[int]]
    query_level: str        
    acs_group: Union[str, List[str]]
    acs_type: str = None
    
    
    def reset_attributes(self):
        for attr in ['_labels', '_variables', '_var_desc']:
            if hasattr(self, attr):
                delattr(self, attr)
        
    
    def raise_for_status(self, groups:List[str] = None)-> bool:
        if groups is None:
            if hasattr(self, '_all_groups'):
                groups = self._all_groups
            else:
                groups = gen_group_names_acs(self)
            
        if self.acs_group not in groups:
            raise AttributeError("Check your ACSConfig attributes")
        else:
            pass
        
    def find_acs_type(self):
        self.acs_type = check_acs_type(self)
    
    @property
    def labels(self):
        if hasattr(self, "_labels"):
            pass
        else:
            if hasattr(self, '_variables'):
                pass
            else:
                self.variables
            label_col = self.var_desc.loc[self.var_desc.name.isin(self.variables),:].sort_values('name').label.reset_index(drop = True)
            
            def join_labels(seq, num_labels):
                text = seq[-num_labels].replace(":",'').replace("Estimate!!","") 
                for i in range(num_labels-1):
                    text += ' - '
                    text += seq[-num_labels + i + 1].replace(":",'').replace("Estimate!!","")
                return text
            
            labels  = []

            for x in label_col.str.split(":!!"):
                if len(x) <= 2:
                    labels.append(x[-1].replace(":",'').replace("Estimate!!",""))
                else:
                    labels.append(join_labels(x, num_labels = len(x)))

            self._labels = labels

        return self._labels
    
    @property
    def var_desc(self):
        if hasattr(self, '_var_desc'):
            pass
        else:
            self.varaibels
        return self._var_desc
    

    @property
    def variables(self):
        if hasattr(self, "_variables"):
            pass
        else:
            if self.acs_type == None:
                self.find_acs_type()
            res = gen_variable_names(self.year, self.acs_type, self.acs_group)
            self._variables = res[0]
            self._var_desc = pd.DataFrame(res[1][1:], columns = res[1][0])
            self._var_desc = self.var_desc.loc[self.var_desc.name.isin(self._variables),:].reset_index(drop = True)
                
        return self._variables


#%% testing all functionality

if __name__ == "__main__":
    config = ACSConfig(year=2021, acs_group=["B01001"], state_fips="21", query_level="tract")
    print(config.variables)
    print(config.labels)
    print(config.var_desc)
    config.reset_attributes()
    print(config.variables)
    print(config.labels)
    print(config.var_desc)  
