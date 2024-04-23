# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# INSTALL AND IMPORTS                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import gc
import os
import sys
import time
import json
import boto3
import pygrib
import swifter
import requests
import itertools

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from functools import partial
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict

from multiprocessing import Pool, cpu_count
from multiprocessing import set_start_method, get_context

from sklearn.metrics import RocCurveDisplay
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (brier_score_loss, f1_score, log_loss,
                                precision_score, recall_score, roc_auc_score)

import warnings
warnings.filterwarnings('ignore')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# COLAB MARKDOWN AND USER CONFIGS                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Scripted Inputs #

season = sys.argv[1]
start_date = sys.argv[2]# YYYY-mm-dd
end_date = sys.argv[3]# YYYY-mm-dd
element = sys.argv[4].lower()# str
lead_days_selection = int(sys.argv[5])# int
region_selection = sys.argv[6].upper()# str
cwa_selection = sys.argv[7].upper()# str
user_token = sys.argv[8].lower() #str

network_selection = 'NWS+RAWS+HADS' if 'qpf' in element.lower() else 'NWS+RAWS'

# --------------- #

# @markdown <FONT SIZE=5>**1. Please Provide Your Synoptic API Token...**
# user_token = "ecd8cc8856884bcc8f02f374f8eb87fc" # @param {type:"string"}

# @markdown <FONT SIZE=5>**2. Select Start and End Dates**
# start_date = "2023-12-01" # @param {type:"date"}
# end_date = "2023-12-31" # @param {type:"date"}

# @markdown <FONT SIZE=5>**3. For Which Element?**
# element = "qpf24" # @param ["maxt", "mint", "qpf24", "qpf12", "qpf06"]

# Split element/interval
interval_selection = int(element[-2:]) if "qpf" in element else False
element = element[:3] if "qpf" in element else element

#6/12/24/48/72, if element==temp then False
# interval_selection = "24" #@param ["24", "12", "6"]
# interval_selection = interval_selection if element == "qpf" else False

#temperature_threshold = -60 #@param {type:"slider", min:-60, max:140, step:10}
#qpf_threshold = 0.31 #@param {type:"slider", min:0.01, max:5.00, step:0.01}

#if element in ["maxt","mint"]:
#    threshold = temperature_threshold
#elif element in ["qpf"]:
#    threshold = qpf_threshold

# @markdown <FONT SIZE=5>**4. For Which Lead Time (in days)?**
# lead_days_selection = 1 #@param {type:"slider", min:1, max:8, step:1}

# @markdown <FONT SIZE=5>**5. For Which Region?**
# region_selection = "CWA" #@param ["WR", "SR", "CR", "ER", "CONUS","CWA"]

#@markdown If CWA selected, which one? (i.e. "SLC" for Salt Lake City)
# cwa_selection = "SEW" #@param {type:"string"}

# @markdown For Which Networks?
# network_selection = 'NWS+RAWS+HADS'#@param ["NWS+RAWS", "NWS+RAWS+HADS", "NWS", "RAWS", "ALL"]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# GLOBAL VARIABLES AND GENERAL CONFIG                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Multiprocess settings
process_pool_size = 20 #cpu_count()*16
print(f'Process Pool Size: {process_pool_size}')

# Backend APIs
metadata_api = "https://api.synopticdata.com/v2/stations/metadata?"
qc_api = "https://api.synopticdata.com/v2/stations/qcsegments?"

# Data Query APIs
timeseries_api = "https://api.synopticdata.com/v2/stations/timeseries?"
statistics_api = "https://api.synopticlabs.org/v2/stations/statistics?"
precipitation_api = "https://api.synopticdata.com/v2/stations/precipitation?"

# Assign API to element name
synoptic_apis = {
    'qpf':precipitation_api,
    'maxt':statistics_api,
    'mint':statistics_api}

synoptic_networks = {"NWS+RAWS+HADS":"1,2,106",
                     "NWS+RAWS":"1,2",
                     "NWS":"1",
                     "RAWS": "2",
                     "ALL":None}
                    #  "CUSTOM": "&network="+network_input,
                    #  "LIST": "&stid="+network_input}

# Assign synoptic variable to element name
synoptic_vars = {
    'qpf':None,
    'maxt':'air_temp',
    'mint':'air_temp'}

synoptic_vars_out = {
    'qpf':'OBSERVATIONS.precipitation',
    'maxt':'STATISTICS.air_temp_set_1.maximum',
    'mint':'STATISTICS.air_temp_set_1.minimum',}

# Assign stat type to element name
stat_type = {
    'qpf':'interval',
    'maxt':'maximum',
    'mint':'minimum'}

ob_hours = {
    'qpf':[['0000', '0000'], ['1200', '1200']],
    'maxt':[['1200', '0600']],
    'mint':[['0000', '1800']]}

# NBM Globals
aws_bucket_nbm = 'noaa-nbm-grib2-pds'
aws_bucket_urma = 'noaa-urma-pds'

# Where to place the grib files (subdirs can be added in local) (not used)
output_dir = '/nas/stid/data/nbm-verification/'

# Which grib variables do each element correlate with
nbm_vars = {'qpf':'APCP',
                  'maxt':'TMP',
                  'mint':'TMP'}

# Which grib levels do each element correlate with
nbm_levs = {'qpf':'surface',
               'maxt':'2 m above ground',
               'mint':'2 m above ground'}

# If a grib message contains any of these, exclude
excludes = ['ens std dev', '% lev']

# Fix MDL's kelvin thresholds...
tk_fix = {233.0:233.15, 244.0:244.261, 249.0:249.817, 255.0:255.372,
    260:260.928, 270.0:270.928, 273.0:273.15, 299.0:299.817,
    305.0:305.372, 310.0:310.928, 316.0:316.483, 322.0:322.039}

# Convert user input to datetime objects
start_date, end_date = [datetime.strptime(date+' 0000', '%Y-%m-%d %H%M')
    for date in [start_date, end_date]]

# Build synoptic arg dict
synoptic_api_args = {
    'ob_stat':stat_type[element],
    'api':synoptic_apis[element],
    'element':element,
    'interval':interval_selection if element == 'qpf' else False,
    'region':region_selection,
    'network_query':synoptic_networks[network_selection], # add config feature later
    'vars_query':None if element == 'qpf'
        else f'{synoptic_vars[element]}',
    'days_offset':1 if element != 'mint' else 0}

# Build nbm/urma arg dict
nbm_request_args = {
    'interval':interval_selection if element == 'qpf' else False,
    'lead_time_days':lead_days_selection,
    'nbm_area':'co',
    'element':element,
    'var':nbm_vars[element],
    'level':nbm_levs[element]}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FUNCTIONS AND METHODS (GENERAL)                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def mkdir_p(check_dir):
    from pathlib import Path
    check_dir = output_dir + check_dir
    Path(check_dir).mkdir(parents=True, exist_ok=True)
    return check_dir

def cwa_list(input_region):

    input_region = input_region.upper()

    region_dict ={
        "WR":["BYZ", "BOI", "LKN", "EKA", "FGZ", "GGW", "TFX", "VEF", "LOX", "MFR",
            "MSO", "PDT", "PSR", "PIH", "PQR", "REV", "STO", "SLC", "SGX", "MTR",
            "HNX", "SEW", "OTX", "TWC"],

        "CR":["ABR", "BIS", "CYS", "LOT", "DVN", "BOU", "DMX", "DTX", "DDC", "DLH",
            "FGF", "GLD", "GJT", "GRR", "GRB", "GID", "IND", "JKL", "EAX", "ARX",
            "ILX", "LMK", "MQT", "MKX", "MPX", "LBF", "APX", "IWX", "OAX", "PAH",
            "PUB", "UNR", "RIW", "FSD", "SGF", "LSX", "TOP", "ICT"],

        "ER":["ALY", "LWX", "BGM", "BOX", "BUF", "BTV", "CAR", "CTP", "RLX", "CHS",
            "ILN", "CLE", "CAE", "GSP", "MHX", "OKX", "PHI", "PBZ", "GYX", "RAH",
            "RNK", "AKQ", "ILM"],

        "SR":["ABQ", "AMA", "FFC", "EWX", "BMX", "BRO", "CRP", "EPZ", "FWD", "HGX",
            "HUN", "JAN", "JAX", "KEY", "MRX", "LCH", "LZK", "LUB", "MLB", "MEG",
            "MAF", "MFL", "MOB", "MRX", "OHX", "LIX", "OUN", "SJT", "SHV", "TAE",
            "TBW", "TSA"]}

    if input_region == "CONUS":
        return np.hstack([region_dict[region] for region in region_dict.keys()])
    else:
        return region_dict[input_region]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FUNCTIONS AND METHODS (SYNOPTIC API)                                        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def fetch_obs_from_API(valid_datetime, cwa='', output_type='csv',
                       use_saved=True, **req):

    if req["element"] == 'qpf':
        start_adjusted = (datetime.strptime(valid_datetime, '%Y%m%d%H%M')
                          - timedelta(hours=req["interval"]))
        end_adjusted = datetime.strptime(valid_datetime, '%Y%m%d%H%M')

    elif ((req["element"] == 'maxt') or (req["element"] == 'mint')):
        start_adjusted = (datetime.strptime(valid_datetime, '%Y%m%d%H%M')
                          - timedelta(hours=18))
        end_adjusted = datetime.strptime(valid_datetime, '%Y%m%d%H%M')

    valid = True
    cwa_filename = (region_selection if region_selection != 'CWA' 
                    else cwa_selection)

    element_label = req['element'] if req['element'] != 'qpf' else \
                        'qpe' + f'{req["interval"]:02d}'


    output_file = mkdir_p(f'obs_{output_type}/') +\
        f'obs.{element_label}.{req["ob_stat"]}' +\
        f'.{valid_datetime}.{cwa_filename}.{output_type}'

    if os.path.isfile(output_file) & use_saved:
        # print(f'Output file exists for:{iter_item}')
        return output_file

    else:
        json_file = mkdir_p('obs_json/') +\
            f'obs.{element_label}.{req["ob_stat"]}' +\
            f'.{valid_datetime}.{cwa_filename}.json'

        if os.path.isfile(json_file) & use_saved:
            # print(f'Polling archived JSON for: {iter_item}')

            with open(json_file, 'rb+') as rfp:
                response_dataframe = pd.json_normalize(json.load(rfp)['STATION'])

        else:
            api_query_args = {
                'api_token':f'&token={user_token}',
                'station_query':f'&cwa={cwa}',
                'network_query':(f'&network={req["network_query"]}'
                                 if req["network_query"] is not None else ''),

                'start_date_query':f'&start={start_adjusted.strftime("%Y%m%d%H%M")}',
                'end_date_query':f'&end={end_adjusted.strftime("%Y%m%d%H%M")}',

                'vars_query':(f'&pmode=intervals&interval={req["interval"]}'
                              if req["element"] == 'qpf'
                                else f'&vars={req["vars_query"]}'),
                'stats_query':f'&type={req["ob_stat"]}',
                'timezone_query':'&obtimezone=utc',
                'api_extras':'&units=temp|f&complete=True'}
                    #'&fields=name,status,latitude,longitude,elevation'

            api_query = req['api'] + ''.join(
                [api_query_args[k] for k in api_query_args.keys()])

            print(f'Polling API for: {iter_item}\n{api_query}')

            status_code, response_count = None, 0
            while (status_code != 200) & (response_count <= 10):
                print(f'{iter_item}, HTTP:{status_code}, #:{response_count}')

                # Don't sleep first try, sleep increasing amount for each retry
                time.sleep(2*response_count)

                response = requests.get(api_query)
                # response.raise_for_status()

                status_code = response.status_code
                response_count += 1

            try:
                response_dataframe = pd.json_normalize(
                    response.json()['STATION'])
            except:
                valid = False
            else:
                with open(json_file, 'wb+') as wfp:
                    wfp.write(response.content)

        if valid:
            # Check ACTIVE flag (Can disable in config above if desired)
            response_dataframe = response_dataframe[
                response_dataframe['STATUS'] == "ACTIVE"]

            # Un-nest the QPF totals
            if req['element'] == 'qpf':
                response_dataframe['TOTAL'] = [i[0]['total']
                    for i in response_dataframe['OBSERVATIONS.precipitation']]

            if output_type == 'pickle':
            # Save out df as pickle
                response_dataframe.to_pickle(output_file)

            elif output_type == 'csv':
            # Save out df as csv
                response_dataframe.to_csv(output_file)

            return None

        else:
            return iter_item

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FUNCTIONS AND METHODS (NBM)                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def ll_to_index(loclat, loclon, datalats, datalons):
    # index, loclat, loclon = loclatlon
    abslat = np.abs(datalats-loclat)
    abslon = np.abs(datalons-loclon)
    c = np.maximum(abslon, abslat)
    latlon_idx_flat = np.argmin(c)
    latlon_idx = np.unravel_index(latlon_idx_flat, datalons.shape)
    return latlon_idx

def fetch_NBMgrib_from_AWS(iter_item, save_dir='nbm_grib2/', **req):

    from botocore import UNSIGNED
    from botocore.client import Config

    nbm_sets = ['qmd'] #, 'core']

    # As strings
    yyyymmdd = iter_item[:-4]
    hh = iter_item[-4:-2]

    element_label = req['element'] if req['element'] != 'qpf' else \
                    req['element'] + f'{req["interval"]:02d}'

    save_dir = mkdir_p(save_dir)

    output_file = (save_dir +
        f'{yyyymmdd}.t{hh}z.fhr{req["lead_time_days"]*24:03d}.{element_label}.grib2')

    if os.path.isfile(output_file):
        pass
        # return output_file

    else:
        client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

        for nbm_set in nbm_sets:

            bucket_dir = f'blend.{yyyymmdd}/{hh}/{nbm_set}/'

            grib_file = f'{bucket_dir}blend.t{hh}z.'+\
                        f'{nbm_set}.f{req["lead_time_days"]*24:03d}.{req["nbm_area"]}.grib2'

            index_file = f'{grib_file}.idx'

            index_data_raw = client.get_object(
                Bucket=aws_bucket_nbm, Key=index_file)['Body'].read().decode().split('\n')

            cols = ['num', 'byte', 'date', 'var', 'level',
                'forecast', 'fthresh', 'ftype', '']

            n_data_cols = len(index_data_raw[0].split(':'))

        while len(cols) > n_data_cols:
            cols = cols[:-1]

        index_data = pd.DataFrame(
            [item.split(':') for item in index_data_raw],
                        columns=cols)

        # Clean up any ghost indicies, set the indexA
        index_data = index_data[index_data['num'] != '']
        index_data['num'] = index_data['num'].astype(int)
        index_data = index_data.set_index('num')

        # Allow byte ranging to '' (EOF)
        index_data.loc[index_data.shape[0]+1] = ['']*index_data.shape[1]

        # Isolate the correct forecast interval
        if req['element'] == 'qpf':
            index_data = index_data.query('byte != ""')

            forecast_step = []
            for item in index_data['forecast']:
                step, steptype = item.replace(' acc fcst', '').split(' ')
                step = np.array(step.split('-')).astype(int)
                step = f'{(step[-1] - step[0]):02d}'

                if ((steptype == 'day') & (step == '01')):
                    step, steptype = '24', 'hour'

                forecast_step.append(step)

            index_data.insert(4, 'step', forecast_step)

        index_subset = index_data[
            ((index_data['var'] == req['var']) &
            (index_data['level'] == req['level']))]

        if req['element'] == 'qpf':
            index_subset = index_subset[
                index_data['step'] == f"{req['interval']:02d}"]

        # Depreciated, old pandas style
        # byte start >> byte range
        # for i in index_subset.index:
        #     try:
        #         index_data.loc[int(i)+1]
        #     except:
        #         index_subset.loc[i]['byte'] = [
        #             index_data.loc[i, 'byte'], '']
        #     else:
        #         index_subset.loc[i]['byte'] = [
        #             index_data.loc[i, 'byte'],
        #             index_data.loc[i+1, 'byte']]

        # byte start >> byte range
        for i in index_subset.index:
            try:
                index_data.loc[int(i)+1]
            except:
                index_subset['byte'][i] = (
                    index_data.loc[i, 'byte'], '')
            else:
                index_subset['byte'][i] = (
                    index_data.loc[i, 'byte'],
                    index_data.loc[i+1, 'byte'])

        # Filter out excluded vars
        for ex in excludes:
            mask = np.column_stack([index_subset[col].str.contains(ex, na=False)
                                    for col in index_subset])

            index_subset = index_subset.loc[~mask.any(axis=1)]

        # Fetch the data by byte range, write from stream
        for index, item in index_subset.iterrows():
            byte_range = f"bytes={item['byte'][0]}-{item['byte'][1]}"

            output_bytes = client.get_object(
                Bucket=aws_bucket_nbm, Key=grib_file, Range=byte_range)

            with open(output_file, 'ab') as wfp:
                for chunk in output_bytes['Body'].iter_chunks(chunk_size=4096):
                    wfp.write(chunk)

        client.close()

def fetch_URMAgrib_from_AWS(iter_item, save_dir='urma_grib2/', **req):
    from botocore import UNSIGNED
    from botocore.client import Config

    save_dir = mkdir_p(save_dir)
    yyyymmdd = iter_item

    if req["element"] == 'maxt':
        hh_set = [8]

    elif req["element"] == 'mint':
        hh_set = [20]

    elif req["element"] == 'qpf':
        if req["interval"] == 6:
            # Change this if aggregating all 4 runs/only one run
            hh_set = [0, 12]

        elif req["interval"] == 12:
            # Change this if aggregating all 4 runs/only one run
            hh_set = [0, 6, 12, 18]

        else:
            hh_set = [0, 6, 12, 18]

    client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    element_label = element if element != 'qpf' else 'qpe06'

    for hh in hh_set:

        output_file = (save_dir +
            f'urma2p5.{yyyymmdd}.t{hh:02d}z.{element_label}.grib2')

        if os.path.isfile(output_file):
            pass

        else:

            # Buffering the dates can be done outside the loop when building iterable
            bucket_dir = f'urma2p5.{yyyymmdd}/'

            if element == 'qpf':
                grib_file = f'{bucket_dir}urma2p5.{yyyymmdd}{hh:02d}.pcp_06h.wexp.grb2'
            else:
                grib_file = f'{bucket_dir}urma2p5.t{hh:02d}z.2dvaranl_ndfd.grb2_wexp'

            try:
                output_bytes = client.get_object(Bucket=aws_bucket_urma, Key=grib_file)
            except:
                pass
            else:
                with open(output_file, 'ab') as wfp:
                    for chunk in output_bytes['Body'].iter_chunks(chunk_size=4096):
                        wfp.write(chunk)

    client.close()

def extract_nbm_value(grib_index, nbm_data):
    return nbm_data[grib_index]

def brier_skill_score(_y_test, _y_prob):

    _y_ref = _y_test.sum()/y_test.size

    bss = 1 - (brier_score_loss(_y_test, _y_prob) /
                brier_score_loss(_y_test, np.full(_y_test.shape, _y_ref)))

    return bss

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# INPUT-BASED GLOBAL VARIABLES AND CONFIG                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":

    # Build an iterable date list from range
    iter_date = start_date
    valid_date_iterable = []
    valid_datetime_iterable = []
    forecast_datetime_iterable = []

    while iter_date <= end_date:

        valid_date_iterable.append(iter_date.strftime('%Y%m%d'))

        for hour_range in ob_hours[element]:
            end_hour = hour_range[-1]

            valid_datetime_iterable.append(iter_date.strftime('%Y%m%d') + end_hour)

            forecast_datetime_iterable.append(
                    (iter_date-timedelta(days=lead_days_selection)
                ).strftime('%Y%m%d') + end_hour)

        iter_date += timedelta(days=1)

    # Assign the fixed kwargs to the function
    cwa_query = ','.join(cwa_list(region_selection)
                        ) if region_selection != 'CWA' else cwa_selection

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # DATA ACQUISITION                                                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    multiprocess_function = partial(fetch_obs_from_API,
                                    cwa=cwa_query,
                                    **synoptic_api_args)

    # Multithreaded requests currently not supported by the Synoptic API
    for iter_item in valid_datetime_iterable:
        multiprocess_function(iter_item)

    # with Pool(process_pool_size) as pool:
    #     print(f'Spooling up process pool for {len(valid_datetime_iterable)} tasks '
    #           f'across {process_pool_size} workers')

    #     retry = pool.map(multiprocess_function, valid_datetime_iterable)
    #     pool.terminate()

    #     print('Multiprocessing Complete')

    # Glob together csv files
    # Need to filter by variable/region in case of region change or re-run!
    synoptic_varname = synoptic_vars_out[element]

    csv_element = element if element != 'qpf' else 'qpe'

    searchstring = (f'*{csv_element}*{region_selection}*.csv'
        if region_selection != 'CWA' else f'*{csv_element}*{cwa_selection}*.csv')

    filelist = np.array(glob(os.path.join(output_dir + 'obs_csv/', searchstring)))

    datecheck = np.array(
        [datetime.strptime(f.split('.')[-3], "%Y%m%d%H%M") for f in filelist])

    datecheck_mask = np.where(
        (datecheck >= start_date.replace(hour=0, minute=0)) 
        & (datecheck <= end_date.replace(hour=23, minute=59)))
    
    filelist = filelist[datecheck_mask]

    df = pd.concat(map(pd.read_csv, filelist),
                ignore_index=True)

    if element == 'qpf':
        # Un-nest precipitation observations
        # df_qpf = pd.concat([pd.DataFrame(json.loads(row.replace("'", '"')))
        #         for row in df[synoptic_varname]], ignore_index=True)

        # df = df.drop(columns=synoptic_varname).join(df_qpf)

        # # Rename the variable since we've changed the column name
        print('Un-nesting precipitation observations')

        qpf_df = []
        for row in df.iterrows():
            row = row[1]

            _qpf_df = pd.DataFrame(eval(row[synoptic_varname]))

            if 'CWA' in df.columns:
                _qpf_df.insert(0, 'STATE', row['STATE'])
                _qpf_df.insert(0, 'CWA', row['CWA'])

            _qpf_df.insert(0, 'ELEVATION', row['ELEVATION'])
            _qpf_df.insert(0, 'LONGITUDE', row['LONGITUDE'])
            _qpf_df.insert(0, 'LATITUDE', row['LATITUDE'])
            _qpf_df.insert(0, 'STID', row['STID'])

            qpf_df.append(_qpf_df)

        # Rename the variable since we've changed the column name
        synoptic_varname = 'total'

        print('Concatenating DataFrame')
        df = pd.concat(qpf_df).reset_index()
        df['last_report'] = pd.to_datetime(df['last_report']).round('6H')

    # Identify the timestamp column (changes with variable)
    for k in df.keys():
        if (('date_time' in k) or ('last_report' in k)):
            time_col = k

    df.rename(columns={time_col:'timestamp'}, inplace=True)
    time_col = 'timestamp'

    # Convert read strings to datetime object
    df[time_col] = pd.to_datetime(df['timestamp']).round('60min')

    # DATETIME OFFSET FIX FOR TEMPS
    if element == 'maxt':
        # Attribute to the day prior if UTC < 06Z otherwise attribute as stamped
        df['timestamp'] = df['timestamp'].where(df['timestamp'].dt.hour < 8,
                        df['timestamp'] - pd.Timedelta(1, unit='D'))

        # Falls outside of URMA and NBM timeframe for MaxT
        df['MAXT'] = df['timestamp'].where(
            ((df['timestamp'].dt.hour >= 8)&(df['timestamp'].dt.hour < 12)), np.nan)

        df['timestamp'] = df['timestamp'].dt.date
        df['timestamp'] = pd.to_datetime(df['timestamp']) + timedelta(hours=8)

    elif element == 'mint':
        df['timestamp'] = df['timestamp'].dt.date
        df['timestamp'] = pd.to_datetime(df['timestamp']) + timedelta(hours=20)

    # Drop any NaNs and sort by date with station as secondary index
    df.set_index(['timestamp'], inplace=True)
    df = df[df.index.notnull()].reset_index().set_index(['timestamp', 'STID'])
    df.sort_index(inplace=True)

    if 'CWA' in df.columns:
        df = df[['CWA', 'STATE', 'LATITUDE', 'LONGITUDE', 'ELEVATION', synoptic_varname]]
    else:
        df = df[['LATITUDE', 'LONGITUDE', 'ELEVATION', synoptic_varname]]

    # df = df.rename(columns={synoptic_varname:element.upper()})
    df = df.rename(columns={synoptic_varname:'OBS'})

    # Build an iterable date list from range
    iter_date = start_date
    date_selection_iterable = []
    while iter_date <= (end_date + timedelta(days=2)):
        date_selection_iterable.append(iter_date.strftime('%Y%m%d'))
        iter_date += timedelta(days=1)

    # Assign the fixed kwargs to the function
    multiprocess_function = partial(fetch_NBMgrib_from_AWS, **nbm_request_args)

    # Set up this way for later additions (e.g. a 2D iterable)
    # multiprocess_iterable = [item for item in itertools.product(
    #     other_iterable, date_selection_iterable)]

    multiprocess_iterable = forecast_datetime_iterable

    # for iter_item in multiprocess_iterable:
    #     multiprocess_function(iter_item)

    # with get_context('fork').Pool(process_pool_size) as pool:
    with Pool(process_pool_size) as pool:
        print(f'Spooling up process pool for {len(multiprocess_iterable)} NBM tasks '
            f'across {process_pool_size} workers')
        NBMgrib_output_files = pool.map(multiprocess_function, multiprocess_iterable)
        pool.terminate()
        print('Multiprocessing Complete')

    # Gridded URMA pull for verification using NBM pull framework (AWS)
    # Assign the fixed kwargs to the function
    multiprocess_function = partial(fetch_URMAgrib_from_AWS, **nbm_request_args)
    multiprocess_iterable = date_selection_iterable

    # with get_context('fork').Pool(process_pool_size) as pool:
    with Pool(process_pool_size) as pool:
        print(f'Spooling up process pool for {len(multiprocess_iterable)} URMA tasks '
            f'across {process_pool_size} workers')
        URMAgrib_output_files = pool.map(multiprocess_function, multiprocess_iterable)
        pool.terminate()

        print('Multiprocessing Complete')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # EXTRACT DATA AND CALCULATE STATISTICS                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Loop over dates in the DataFrame, open one NBM file at a time
    for valid_date in df.index.get_level_values(0).unique():

        # We are looping over the VALID DATE... the filenames are stamped
        # with the INIT DATE. We need to offset the valid dates to work!
        init_date = valid_date - pd.Timedelta(
            nbm_request_args['lead_time_days'], 'day')

        if ((element == 'maxt') or (element == 'mint')):
            init_hour = 6 if element == 'maxt' else 18

        else:
            init_hour = init_date.hour

        datestr = datetime.strftime(init_date, '%Y%m%d')

        # # # # # # # # # # #
        # File Arrangement

        element_name = f'{element}{interval_selection:02d}' if element == 'qpf' \
                            else f'{element}'

        nbm_file = output_dir + f'nbm_grib2/{datestr}.t{init_hour:02d}z' +\
                f'.fhr{nbm_request_args["lead_time_days"]*24:03d}' +\
                f'.{element_name}.grib2'

        if element == 'qpf':
            urma_name = 'QPE_URMA'

            if interval_selection == 6:
                valid_set = [valid_date]

            if interval_selection == 12:
                valid_set = [valid_date - timedelta(hours=offset)
                    for offset in [6, 0]]

            elif interval_selection == 24:
                valid_set = [valid_date - timedelta(hours=offset)
                    for offset in [18, 12, 6, 0]]

        elif element == 'maxt':
            urma_name = 'MAXT_URMA'
            valid_set = [valid_date]

        elif element == 'mint':
            urma_name = 'MINT_URMA'
            valid_set = [valid_date]

        # # # # # # # # # # #
        # Data Extraction (NBM)

        # print(nbm_file)

        if os.path.isfile(nbm_file):
            nbm = pygrib.open(nbm_file)

            print(f'\nextracting i:{init_date}, nbf:{nbm_file}')

            # If not yet indexed, go ahead and build the indexer
            if 'grib_index' not in df.columns:

                nbmlats, nbmlons = nbm.message(1).latlons()

                df_indexed = df.reset_index()[
                    ['STID', 'LATITUDE', 'LONGITUDE', 'ELEVATION']].drop_duplicates()

                ll_to_index_mapped = partial(ll_to_index,
                                            datalats=nbmlats, datalons=nbmlons)

                print('\nFirst pass: creating y/x grib indicies from lat/lon\n')

                df_indexed['grib_index'] = df_indexed.swifter.apply(
                    lambda x: ll_to_index_mapped(x.LATITUDE, x.LONGITUDE), axis=1)

                # Extract the grid latlon
                extract_nbm_lats_mapped = partial(extract_nbm_value,
                                    nbm_data=nbmlats)

                extract_nbm_lons_mapped = partial(extract_nbm_value,
                                    nbm_data=nbmlons)

                df_indexed['grib_lat'] = df_indexed['grib_index'].apply(
                    extract_nbm_lats_mapped)

                df_indexed['grib_lon'] = df_indexed['grib_index'].apply(
                    extract_nbm_lons_mapped)

                df_indexed.set_index('STID', inplace=True)

                df = df.join(
                    df_indexed[['grib_index', 'grib_lat', 'grib_lon']]).sort_index()

            # Extract the data for that date and re-insert into DataFrame
            # Loop over each variable in the NBM file and store to DataFrame
            # May need a placeholder column of NaNs in df for each var to make this work...
            # Use .swifter.apply() as needed if this will speed up the process
            # Alternatively, can use multiprocess pool to thread out the work over each date
            # First pass this seems fast enough as it is...
            for msg in nbm:
                if (('Probability' in str(msg)) & (('temperature' in str(msg)) or
                    ((msg.lengthOfTimeRange == interval_selection)))):

                    # Deal with column names
                    if (('Precipitation' in str(msg)) &
                    (msg.lengthOfTimeRange == interval_selection)):

                        threshold_in = round(msg['upperLimit']*0.0393701, 2)

                        name = f"tp_ge_{str(threshold_in).replace('.','p')}"

                    elif 'temperature' in str(msg):
                        gtlt = 'le' if 'below' in str(msg) else 'ge'
                        tk = (msg['lowerLimit'] if 'below'
                                in str(msg) else msg['upperLimit'])
                        tk = tk_fix[tk]
                        tc = tk-273
                        tf = (((tc)*(9/5))+32)
                        name = f"temp_{gtlt}_{tf:.0f}".replace('-', 'm')

                    if name not in df.columns:
                        df[name] = np.nan

                    extract_nbm_value_mapped = partial(extract_nbm_value,
                                                    nbm_data=msg.values)

                    df.loc[valid_date, name] = df.loc[valid_date]['grib_index'].apply(
                        extract_nbm_value_mapped).values

                elif 'temperature at 2 metres' in str(msg): # OR precipitation clause
                    name = 'FXMAXT' if element == 'maxt' else 'FXMINT'
                    if name not in df.columns:
                        df[name] = np.nan

                    extract_nbm_value_mapped = partial(extract_nbm_value,
                                                    nbm_data=msg.values)

                    # Convert to F from K
                    df.loc[valid_date, name] = (((df.loc[valid_date]['grib_index'].apply(
                        extract_nbm_value_mapped).values - 273.15)*(9/5))+32)

                elif (('Precipitation' in str(msg)) &
                    (msg.lengthOfTimeRange == interval_selection)):

                    name = 'FXQPF'
                    if name not in df.columns:
                        df[name] = np.nan

                    extract_nbm_value_mapped = partial(extract_nbm_value,
                                                    nbm_data=msg.values)

                    df.loc[valid_date, name] = df.loc[valid_date]['grib_index'].apply(
                                        extract_nbm_value_mapped).values

            nbm.close()

            # # # # # # # # # # #
            # Data Extraction (URMA)

            # lat shape lon shape are 2d and interchangable
            msg = np.zeros(nbmlats.shape) if element == 'qpf' else None

            urma_element = element if element != 'qpf' else 'qpe06'

            baddata = False
            for urma_datetime in valid_set:

                urma_file = output_dir + f'urma_grib2/urma2p5.' +\
                    f'{(urma_datetime).strftime("%Y%m%d")}.'+\
                    f't{urma_datetime.hour:02d}z.{urma_element}.grib2'

                print(f'extracting v:{valid_date}, urf:{urma_file}')

                if os.path.isfile(urma_file):
                    urma = pygrib.open(urma_file)

                    if element == 'qpf':
                        # Sum onto the initalized zero array
                        urma = pygrib.open(urma_file)
                        msg += urma.select(shortName='tp')[0].values

                    elif element == 'maxt':
                        msg = ((urma.select(shortName='tmax')[0].values - 273.15) * (9/5)) + 32

                    elif element == 'mint':
                        msg = ((urma.select(shortName='tmin')[0].values - 273.15) * (9/5)) + 32

                    urma.close()

                else:
                    print(f'{urma_file} not found, skipping and backfill nan\n')
                    baddata = True

            if urma_name not in df.columns:
                df[urma_name] = np.nan

            msg = msg if baddata == False else np.full(nbmlats.shape, np.nan)

            extract_urma_value_mapped = partial(extract_nbm_value,
                                    nbm_data=(msg))

            df.loc[valid_date, urma_name] = df.loc[valid_date]['grib_index'].apply(
                                extract_urma_value_mapped).values

        # print()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # GENERATE FIGURES AND TABLES                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    C1, C2 = 'Lime', 'Magenta'

    df = df.dropna(how='any')
    df.index.get_level_values(0).unique()

    varname = element.upper()
    urma_varname = varname if element != 'qpf' else 'QPE'

    keylist = [k for k in df.columns if (('ge' in k) or ('le' in k))]

    # threshlist = [float(
    #     k.split('_')[-1].replace('p', '.').replace('m', '-')) for k in keylist]

    # if element == 'qpf':
    #     threshlist = [t*25.4 for t in threshlist]

    # for i, t in enumerate(zip(keylist, threshlist)):
    for k in sorted(keylist):

        fig = plt.figure(constrained_layout=True, figsize=(6, 8))
        gs = fig.add_gridspec(nrows=4, ncols=3, left=0.05, right=0.5, 
                              hspace=-0.05, wspace=0.05)

        ax1 = fig.add_subplot(gs[:-1, :])
        ax2 = fig.add_subplot(gs[-1, :-1])
        ax3 = fig.add_subplot(gs[-1, -1])

        # thresh_text, thresh = t
        thresh_text = k
        thresh = float(k.split('_')[-1].replace('p', '.').replace('m', '-'))
        thresh = thresh*25.4 if element == 'qpf' else thresh

        if 'ge' in thresh_text:
            y_test_ob = np.where(df['OBS'] >=  thresh, 1, 0) #ge vs gt
            y_test = np.where(df[f'{urma_varname}_URMA'] >= thresh, 1, 0)

        elif 'le' in thresh_text:
            y_test_ob = np.where(df['OBS'] <=  thresh, 1, 0) #le vs lt
            y_test = np.where(df[f'{urma_varname}_URMA'] <= thresh, 1, 0)

        y_prob = df[thresh_text]/100
        y_pred = np.where(df['FX'+varname] >= thresh, 1, 0) #ge vs gt

        x_ref = y_prob.sum()/y_prob.size
        y_ref = y_test.sum()/y_test.size
        y_ref_ob = y_test_ob.sum()/y_test_ob.size

        noskill_x, noskill_y = np.array([(bin, (bin + y_ref)/2)
                                        for bin in np.arange(0, 1.1, .1)]).T

        _, noskill_y_ob = np.array([(bin, (bin + y_ref_ob)/2)
                                    for bin in np.arange(0, 1.1, .1)]).T

        # Calibration Curves/Reliability Diagrams
        CalibrationDisplay.from_predictions(y_test, y_prob, n_bins=10, ax=ax1, name='URMA',
                                            ref_line=False, marker='o', markersize=10,
                                            markerfacecolor='none', linewidth=2, color=C1,
                                            zorder=10)

        CalibrationDisplay.from_predictions(y_test_ob, y_prob, n_bins=10, ax=ax1, name='OBS',
                                            ref_line=False, marker='^', markersize=10,
                                            markerfacecolor='none', linewidth=2, color=C2,
                                            zorder=9)

        # ax1.fill_between(np.linspace(x_ref, 1, noskill_y[noskill_y.round(3) >= x_ref.round(3)].size),
        #                  noskill_y[noskill_y.round(3) >= x_ref.round(3)], 1,
        #                  zorder=-10, color='gray', alpha=0.4)

        # ax1.fill_between(np.linspace(x_ref, 1, noskill_y[noskill_y.round(3) >= x_ref.round(3)].size),
        #                 y_ref, noskill_y[noskill_y.round(3) >= x_ref.round(3)],
        #                 zorder=-10, color='gray', alpha=0.25)

        # ax1.fill_between(np.linspace(0, x_ref, noskill_y[noskill_y.round(3) <= x_ref.round(3)].size), 0,
        #                  noskill_y[noskill_y.round(3) <= x_ref.round(3)],
        #                  zorder=-10, color='gray', alpha=0.4)

        # ax1.fill_between(np.linspace(0, x_ref, noskill_y[noskill_y.round(3) <= x_ref.round(3)].size),
        #                 noskill_y[noskill_y.round(3) <= x_ref.round(3)], y_ref,
        #                 zorder=-10, color='gray', alpha=0.25)

        ax1.plot(noskill_x, noskill_y, linestyle='--', linewidth=1.5, color='black', alpha=0.5, zorder=8)
        # ax1.plot(noskill_x, noskill_y_ob, linestyle='--', color=C2)

        ax1.axhline(y_ref, linestyle='--', linewidth=1.5, color='black', alpha=0.5, zorder=6)
        # ax1.axhline(y_ref_ob, linestyle='--', linewidth=1.5, color=C2, alpha=0.5, zorder=-1)

        ax1.axvline(x_ref, linestyle='--', linewidth=1.5, color='black', alpha=0.5, zorder=6)
        ax1.plot([0, 1], [0, 1], '-', linewidth=1, color='black', alpha=1, zorder=7)

        ax1.set_xticks(np.arange(0, 1.1, 0.1))
        ax1.set_yticks(np.arange(0, 1.1, 0.1))

        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])

        ax1.set_xlabel('Forecast Probability')
        ax1.set_ylabel('Observed Frequency')
        ax1.grid(True)

        # Sharpness Diagram
        ax2.hist(y_prob, bins=10, density=True, rwidth=0.8, log=True, color='black')
        ax2.set_xlabel('Forecast Probability')
        ax2.set_ylabel('Relative Frequency')
        ax2.grid(True)

        # ROC-AUC Curves
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax3, name='URMA', color=C1)
        RocCurveDisplay.from_predictions(y_test_ob, y_prob, ax=ax3, name='OBS', color=C2)

        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        
        ax3.yaxis.set_label_position("right")
        ax3.yaxis.tick_right()

        ax3.grid(True)
        ax3.get_legend().remove()

        # Skill Scores
        scores = defaultdict(list)
        for name in ['OBS', 'URMA']:

            _y_test = y_test if name == 'URMA' else y_test_ob

            scores["Classifier"].append(name)

            for metric in [brier_score_loss, brier_skill_score, roc_auc_score]: #log_loss
                try:
                    score_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
                    scores[score_name].append(metric(_y_test, y_prob))
                except:
                    scores[score_name].append(np.nan)

            for metric in [recall_score, precision_score]: #f1_score
                score_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
                scores[score_name].append(metric(_y_test, y_pred))

        score_df = pd.DataFrame(scores).set_index("Classifier")
        score_df.round(decimals=3)

        score_df.rename(columns={'Brier  loss':'Brier Score',
                                'Brier skill ':'Brier Skill',
                                'Log loss':'Log Loss',
                                'Roc auc ':'ROC AUC',
                                'Recall ':'Recall\n(POD)',
                                'Precision ':'Precision\n(1-FAR)'}, inplace=True)

        ax1.table(cellText=score_df.values.round(3),
            colWidths=[0.25]*len(score_df.columns),
            rowLabels=score_df.index,
            colLabels=score_df.columns,
            cellLoc='center', rowLoc='center',
            loc='bottom', bbox=[0., -1.075, 1., 0.25])

        # Title/Labels
        n_urma = y_test.sum()
        n_ob = y_test_ob.sum()
        n_sites = df.index.get_level_values(1).unique().size

        locname = cwa_selection if region_selection == 'CWA' else region_selection

        all_dates = df.index.get_level_values(0).unique()
        label_start = all_dates[0].strftime('%Y%m%d%H')
        label_end = all_dates[-1].strftime('%Y%m%d%H')

        suptitle = (f'{locname} n_stations ({network_selection}): {n_sites}\n'+\
                    f'valid: {label_start} - {label_end} F{lead_days_selection*24:-03d}')

        reformat = {'tp':'Total Precipitation', 'MAXT':'Max Temp', 'tmin':'Min Temp',
                'ge':'≥', 'le':'≤', 'p':'.', 'm':'-', '_':' ', 'temp':''}

        thresh_text = thresh_text.split('_')
        for k in reformat.keys():
            thresh_text[2] = thresh_text[2].replace(k, reformat[k])

        thresh_text = ' '.join([
            thresh_text[0].replace(thresh_text[0], reformat[thresh_text[0]]),
            thresh_text[1].replace(thresh_text[1], reformat[thresh_text[1]]),
            thresh_text[2]])

        if element == 'qpf':
            ax1.set_title(suptitle + '\n'*2 +\
                    f'{varname}{interval_selection:02d} {thresh_text}"' +\
                        f'\nn_events (URMA/OBS): {n_urma}/{n_ob}')

            plot_filename = f'{locname}_{season}_f{lead_days_selection*24:03d}' +\
                f'_{varname}{interval_selection:02d}_{thresh_text}.jpg'

        elif ((element == 'maxt') or (element == 'mint')):
            # Add elif to only assign F to temp if adding future vars
            ax1.set_title(suptitle + '\n'*2 +\
                    f'{varname} {thresh_text}F\nn_events (URMA/OBS): {n_urma}/{n_ob}')

        # plt.show()

        plot_dir = f"{element}{interval_selection if interval_selection != False else ''}/" +\
            f"{region_selection if region_selection != 'CWA' else cwa_selection}/" +\
            f"{season}/f{lead_days_selection*24:03d}/"

        plot_fname = f"reliability_{thresh_text.replace(' ', '_').replace('.','p')}.jpg"
        output_plotfile = mkdir_p(plot_dir.upper()) + plot_fname.lower()

        print(f'Writing: {output_plotfile}')

        plt.savefig(output_plotfile, bbox_inches='tight', dpi=fig.dpi)
        
        plt.close()


