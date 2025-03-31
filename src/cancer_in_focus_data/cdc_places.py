# import packages
#%%
import pandas as pd
from typing import Union, List
from dataclasses import dataclass
import sys
import shutil
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # add src to path

from utils.ciftools_logger import logger
from dotenv import load_dotenv
from utils.states import stateDf
load_dotenv()
from config.socrata_config import SocrataConfig

socrata_cdc_county = os.environ.get("SOCRATA_CDC_COUNTY", None)
socrata_cdc_tract = os.environ.get("SOCRATA_CDC_TRACT", None)

#%%


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
            results = self.config.client.get(socrata_cdc_county, where=f'stateabbr="{self.state}"',  limit = 100_000)
        else:
            state = '("' + '","'.join(self.state) + '")'
            results = self.config.client.get(socrata_cdc_county, where=f'stateabbr in {state}',  limit = 100_000)
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
            results = self.config.client.get(socrata_cdc_tract, where=f'stateabbr="{self.state}"', limit = 100_000)
        else:
            state = '("' + '","'.join(self.state) + '")'
#             while 
            results = self.config.client.get(socrata_cdc_tract, where=f'stateabbr in {state}', offset = 0, limit = 100_000)

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


#%% testing
if __name__ == "__main__":
    # Example usage
    state_fips = ['01', '02']  # Alabama and Alaska for example
    config = SocrataConfig(domain = "data.cdc.gov")  # Initialize the SocrataConfig

    places_data_instance = places_data(state_fips=state_fips, config=config)
    
    # Fetch and print county data
    county_data = places_data_instance.places_data['county']
    print("County Data:")
    print(county_data.head())

    # Fetch and print tract data
    tract_data = places_data_instance.places_data['tract']
    print("\nTract Data:")
    print(tract_data.head())