
import argparse
from tqdm import tqdm
from glob import glob
import time
import os
from utils import write_bash_script
import subprocess
import pandas as pd
import geopandas as gpd
import pickle
# multi-processing
from joblib import Parallel, delayed
from datetime import datetime as dt
import shutil
from functools import partial
from typing import Union, List


multi_processing = False



today = dt.today().strftime('%m-%d-%Y')


def gen_spatial_zip_file(cancer_center_name_abb, catchment_area_df, data_dictionary):
    # make sure you have a folder "data" before calling this function
    directory_path = os.path.join(os.getcwd(), cancer_center_name_abb) # folder that will contain csv files
    ca_name = os.path.join(os.getcwd(), cancer_center_name_abb, cancer_center_name_abb) # will be used for file_name
    os.mkdir(cancer_center_name_abb) #create the folder
    cancer_center_fips = catchment_area_df.loc[catchment_area_df.name_short.eq(cancer_center_name_abb),:].FIPS.tolist()
    cancer_center_state = catchment_area_df.loc[catchment_area_df.name_short.eq(cancer_center_name_abb),:].State.unique().tolist()
    for table_name, df in data_dictionary.items():
        if 'tract' in table_name:
            df.FIPS = df.FIPS.str.zfill(11)
            df = df.loc[df.FIPS.str[:5].isin(cancer_center_fips),:].reset_index(drop = True)
            gdf = gpd.GeoDataFrame(df)
            gdf = gdf.set_geometry('Shape')
            gdf.to_file(ca_name + table_name + '_' + today)
            del gdf
        else: # county
            df.FIPS = df.FIPS.str.zfill(5)
            df = df.loc[df.FIPS.isin(cancer_center_fips),:].reset_index(drop = True)
            gdf = gpd.GeoDataFrame(df)
            gdf = gdf.set_geometry('Shape')
            gdf.to_file(ca_name + table_name + '_' + today)
            del gdf
    
    shutil.make_archive(os.path.join(os.getcwd(), 'data', cancer_center_name_abb + '_shape'), 'zip', directory_path)
    shutil.rmtree(directory_path)
    return 1

def gen_zip_file(cancer_center_name_abb, catchment_area_df, data_dictionary):
    # make sure you have a folder "data" before calling this function
    directory_path = os.path.join(os.getcwd(), cancer_center_name_abb) # folder that will contain csv files
    ca_name = os.path.join(os.getcwd(), cancer_center_name_abb, cancer_center_name_abb) # will be used for file_name
    os.mkdir(cancer_center_name_abb) #create the folder
    catchment_area_df = catchment_area_df.loc[catchment_area_df.name_short.eq(cancer_center_name_abb),:]
    cancer_center_fips = catchment_area_df.FIPS.unique().tolist()
    cancer_center_state = catchment_area_df.State.unique().tolist()
    cdata = {}
    for table_name, df in data_dictionary.items():
        if 'facilities' in table_name:
            cdata[table_name] = df.loc[df.State.isin(cancer_center_state),:]
        elif 'tract' in table_name:
            cdata[table_name] = df.loc[df.FIPS.str[:5].isin(cancer_center_fips),:]
        elif 'puma' in table_name: 
            cdata[table_name] = df
        elif 'shape' in table_name: # shape tables
            county_shape = df['county_shape']; tract_shape = df['tract_shape']
            # county_shape = county_shape.loc[county_shape.FIPS.isin(cancer_center_fips),:].sort_values('FIPS').reset_index(drop = True)
            # tract_shape = tract_shape.loc[tract_shape.FIPS.str[:5].isin(cancer_center_fips),:].sort_values('FIPS').reset_index(drop = True)
            # county_shape = gpd.GeoDataFrame(county_shape); county_shape = county_shape.set_geometry('Shape')
            # tract_shape =  gpd.GeoDataFrame(tract_shape);  tract_shape  = tract_shape.set_geometry('Shape')
            # cdata['county_shape'] = county_shape
            # cdata['tract_shape'] = tract_shape
        else:
            cdata[table_name] = df.loc[df.FIPS.str[:5].isin(cancer_center_fips),:]
    # cdata['county_shape'].to_file(ca_name + '_county_shape')
    # cdata['tract_shape'].to_file(ca_name + '_tract_shape')
    # md.to_csv(ca_name + '_measure_dictionary_' + today + '.csv', encoding='utf-8', index=False)
    md.to_csv(directory_path + '\measure_dictionary_v5.csv', encoding='utf-8', index=False)
    cdata['cancer_incidence'].to_csv(ca_name + '_cancer_incidence_county_' + today + '.csv', encoding='utf-8', index=False)
    cdata['cancer_mortality'].to_csv(ca_name + '_cancer_mortality_county_' + today + '.csv', encoding='utf-8', index=False)
    cdata['cancer_incidence_long'].to_csv(ca_name + '_cancer_incidence_county_long_' + today + '.csv', encoding='utf-8', index=False)
    cdata['cancer_mortality_long'].to_csv(ca_name + '_cancer_mortality_county_long_' + today + '.csv', encoding='utf-8', index=False)
    cdata['economy_county'].to_csv(ca_name + '_economy_county_' + today + '.csv', encoding='utf-8', index=False)
    cdata['economy_county_long'].to_csv(ca_name + '_economy_county_long_' + today + '.csv', encoding='utf-8', index=False)
    cdata['economy_tract'].to_csv(ca_name + '_economy_tract_' + today + '.csv', encoding='utf-8', index=False)
    cdata['economy_tract_long'].to_csv(ca_name + '_economy_tract_long_' + today + '.csv', encoding='utf-8', index=False)
    cdata['environment_county'].to_csv(ca_name + '_environment_county_' + today + '.csv', encoding='utf-8', index=False)
    cdata['environment_county_long'].to_csv(ca_name + '_environment_county_long_' + today + '.csv', encoding='utf-8', index=False)
    cdata['environment_tract'].to_csv(ca_name + '_environment_tract_' + today + '.csv', encoding='utf-8', index=False)
    cdata['environment_tract_long'].to_csv(ca_name + '_environment_tract_long_' + today + '.csv', encoding='utf-8', index=False)
    cdata['food_desert_tract'].to_csv(ca_name + '_food_desert_tract_' + today + '.csv', encoding='utf-8', index=False)
    cdata['food_desert_tract_long'].to_csv(ca_name + '_food_desert_tract_long_' + today + '.csv', encoding='utf-8', index=False)
    cdata['ht_county'].to_csv(ca_name + '_housing_trans_county_' + today + '.csv', encoding='utf-8', index=False)
    cdata['ht_county_long'].to_csv(ca_name + '_housing_trans_county_long_' + today + '.csv', encoding='utf-8', index=False)
    cdata['ht_tract'].to_csv(ca_name + '_housing_trans_tract_' + today + '.csv', encoding='utf-8', index=False)
    cdata['ht_tract_long'].to_csv(ca_name + '_housing_trans_tract_long_' + today + '.csv', encoding='utf-8', index=False)
    cdata['disp_county'].to_csv(ca_name + '_disparity_county_' + today + '.csv', encoding='utf-8', index=False)
    cdata['disp_county_long'].to_csv(ca_name + '_disparity_county_long_' + today + '.csv', encoding='utf-8', index=False)
    cdata['disp_tract'].to_csv(ca_name + '_disparity_tract_' + today + '.csv', encoding='utf-8', index=False)
    cdata['disp_tract_long'].to_csv(ca_name + '_disparity_tract_long_' + today + '.csv', encoding='utf-8', index=False)
    cdata['rf_and_screening_county'].to_csv(ca_name + '_rf_and_screening_county_' + today + '.csv', encoding='utf-8', index=False)
    cdata['rf_and_screening_tract'].to_csv(ca_name + '_rf_and_screening_tract_' + today + '.csv', encoding='utf-8', index=False)
    cdata['rf_and_screening_county_long'].to_csv(ca_name + '_rf_and_screening_county_long_' + today + '.csv', encoding='utf-8', index=False)
    cdata['rf_and_screening_tract_long'].to_csv(ca_name + '_rf_and_screening_tract_long_' + today + '.csv', encoding='utf-8', index=False)
    cdata['sociodemographics_county'].to_csv(ca_name + '_sociodemographics_county_' + today + '.csv', encoding='utf-8', index=False)
    cdata['sd_county_long'].to_csv(ca_name + '_sociodemographics_county_long_' + today + '.csv', encoding='utf-8', index=False)
    cdata['sociodemographics_tract'].to_csv(ca_name + '_sociodemographics_tract_' + today + '.csv', encoding='utf-8', index=False)
    cdata['sd_tract_long'].to_csv(ca_name + '_sociodemographics_tract_long_' + today + '.csv', encoding='utf-8', index=False)
    # cdata['broadband_speeds'].to_csv(ca_name + '_broadband_speeds_' + today + '.csv', encoding='utf-8', index=False)
    cdata['facilities_and_providers'].to_csv(ca_name + '_facilities_and_providers_' + today + '.csv', encoding='utf-8', index=False)
    
    # cdata['economy_puma'].to_csv(ca_name + '_economy_puma_' + today + '.csv', encoding='utf-8', index=False)
    # cdata['economy_puma_long'].to_csv(ca_name + '_economy_puma_long_' + today + '.csv', encoding='utf-8', index=False)
    # cdata['ht_puma'].to_csv(ca_name + '_housing_trans_puma_' + today + '.csv', encoding='utf-8', index=False)
    # cdata['ht_puma_long'].to_csv(ca_name + '_housing_trans_puma_long_' + today + '.csv', encoding='utf-8', index=False)
    # cdata['sociodemographics_puma'].to_csv(ca_name + '_sociodemographics_puma_' + today + '.csv', encoding='utf-8', index=False)
    # cdata['sd_puma_long'].to_csv(ca_name + '_sociodemographics_puma_long_' + today + '.csv', encoding='utf-8', index=False)
    
    shutil.make_archive(os.path.join(os.getcwd(), 'data', cancer_center_name_abb), 'zip', directory_path)
    shutil.rmtree(directory_path)
    return 1
    
    
    
def gen_unique_states(fips):
    states = fips.str[:2].unique().tolist()
    return states

def gen_shapes(fips):
    states = gen_unique_states(fips)
    





if __name__ == '__main__':
    
    bash_script_kwargs = {
    "bash_file_name" : 'all_catchment_areas.sh', #the name of a bash file to run
    "catchment_area_name": "all", # the name of the catchment area name
    "ca_file_path": "all_counties.csv",
    "query_level" : ['county','tract'],
    "add_puma_level": True,
    "acs_year"    : 2023,
    "download_file_type": ['pickle'],
    "census_api_key": 'f1a4c4de1f35fe90fc1ceb60fd97b39c9a96e436',
    "generate_zip_file" : False,
    "install_packages" : False, # We already installed required packages above,
    "socrata_user_name": "ciodata@uky.edu",
    "socrata_password" : "MarkeyCancer123!"
}
        
    write_bash_script(**bash_script_kwargs)
    
    
    
    subprocess.run(["bash", "all_catchment_areas.sh"], shell=True)

    ca_path = glob('*/all_counties.csv')[0]
    md_path = glob('*/measure_dictionary_v5.csv')[0]
    dd_path = glob('*/cif_variable_codebook_v5.xlsx')[0]
    ca = pd.read_csv(ca_path, dtype = {"FIPS":str})
    md = pd.read_csv(md_path)
    dd = pd.read_excel(dd_path, index_col=0, engine='openpyxl')
    ca['FIPS'] = ca.FIPS.str.zfill(5)
    pickle_path = [x for x in glob('*/all_catchment_data*.pickle') if 'spatial' not in x][0]
    with open(pickle_path, 'rb') as f:
        data_dictionary = pickle.load(f)
        
        
    if os.path.exists(os.path.join(os.getcwd(), 'data')):
        pass
    else:
        os.mkdir('data')
    
    gen_zip_file_partial = partial(gen_zip_file, catchment_area_df = ca, data_dictionary = data_dictionary)
    if multi_processing:
        Parallel(n_jobs = -1)(delayed(gen_zip_file_partial)(abb) for abb in tqdm(ca.name_short.unique().tolist()))
    else:
        for abb in tqdm(ca.name_short.unique().tolist()):
            gen_zip_file_partial(abb)
    os.remove('cif_raw_data.pickle');
    os.remove('all_catchment_areas.sh')
    
