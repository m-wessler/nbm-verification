!pip install boto3
!pip install pygrib

import os, boto3, pygrib

import numpy as np
import pandas as pd

from botocore import UNSIGNED
from botocore.client import Config

# Define Globals
aws_bucket = 'noaa-nbm-grib2-pds'

# Where to place the grib file (subdirs can be added in local) (not used)
# output_dir = './'

# Which grib variables do each element correlate with
element_var = {'qpf':'APCP',
                  'maxt':'TMP',
                  'mint':'TMP'}

# Which grib levels do each element correlate with 
element_lev = {'qpf':'surface',
               'maxt':'2 m above ground',
               'mint':'2 m above ground'}

# If a grib message contains any of these, exclude
excludes = ['ens std dev', '% lev']

def fetch_grib_from_AWS(**req):

    output_file = f'{req["yyyymmdd"]}.t{req["hh"]:02d}z.fhr{req["fhr"]:03d}.{req["var"]}.grib2'

    if os.path.isfile(output_file):
        return output_file

    else:
        for nbm_set in req['nbm_set']:

            bucket_dir = f'blend.{req["yyyymmdd"]}/{req["hh"]:02d}/{nbm_set}/'

            grib_file = f'{bucket_dir}blend.t{req["hh"]:02d}z.'+\
                        f'{nbm_set}.f{req["fhr"]:03d}.{req["nbm_area"]}.grib2'

            index_file = f'{grib_file}.idx'

            client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

            index_data_raw = client.get_object(
                Bucket=aws_bucket, Key=index_file)['Body'].read().decode().split('\n')

            cols = ['num', 'byte', 'date', 'var', 'level', 
                'forecast', 'fthresh', 'ftype', '']

            index_data = pd.DataFrame([item.split(':') for item in index_data_raw],
                            columns=cols if nbm_set == 'core' else cols[:-1])

            # Clean up any ghost indicies, set the index
            index_data = index_data[index_data['num'] != '']
            index_data['num'] = index_data['num'].astype(int)
            index_data = index_data.set_index('num')

            # Allow byte ranging to '' (EOF)
            index_data.loc[index_data.shape[0]+1] = ['']*index_data.shape[1]

            index_subset = index_data[
                ((index_data['var'] == req['var']) &
                (index_data['level'] == req['level']))]

            # byte start >> byte range
            for i in index_subset.index:
                index_subset.loc[i]['byte'] = (
                    index_data.loc[i, 'byte'],
                    index_data.loc[int(i)+1, 'byte'])
                                
            # Filter out excluded vars
            for ex in excludes:
                mask = np.column_stack([index_subset[col].str.contains(ex, na=False) 
                                        for col in index_subset])
                
                index_subset = index_subset.loc[~mask.any(axis=1)]

            # Fetch the data by byte range, write from stream
            for index, item in index_subset.iterrows():
                byte_range = f"bytes={item['byte'][0]}-{item['byte'][1]}"

                output_bytes = client.get_object(
                    Bucket=aws_bucket, Key=grib_file, Range=byte_range)
                                
                with open(output_file, 'ab') as wfp:
                    for chunk in output_bytes['Body'].iter_chunks(chunk_size=4096):
                        wfp.write(chunk)
        
    client.close()
    return output_file

if __name__ == __main__:

    element = input('Desired element? (QPF/MaxT/MinT)').lower()

    request_args = {
        'yyyymmdd':input('Desired init date (YYYYMMDD)? '),
        'hh':int(input('Desired init hour int(HH)? ')),
        'fhr':int(input('Desired forecast hour/lead time int(HHH)?')),
        'nbm_set':['core', 'qmd'] if element == 'qpf' else ['core'],
        'nbm_area':'co',
        'var':element_var[element],
        'level':element_lev[element]}

    grib_output_file = fetch_grib_from_AWS(**request_args)

    for item in pygrib.open(grib_output_file).read():
        print(item)

    os.remove(grib_output_file)