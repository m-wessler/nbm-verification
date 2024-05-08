import subprocess

pythonpath = '/home/michael.wessler/anaconda3/envs/py310/bin/python'
logpath = '/nas/stid/projects/michael.wessler/nbm-verification'

# pythonpath = r"C:\Users\Michael.wessler\AppData\Local\anaconda3" +\
#                 r"\envs\py310\python.exe"
            
season_dates = {
    # 'POR':['2023-01-18', '2024-04-16'],
    # 'YEAR':['2023-04-15', '2024-04-16'],
    'DJF':['2023-12-01', '2024-03-01'],
    # 'JJA':['2023-06-01', '2023-09-01'],
    # 'MAM':['2023-03-01', '2024-06-01'],
    # 'SON':['2023-09-01', '2023-12-01'],
    # 'WPC':['2023-10-02', '2024-02-05'],
    }

api_tokens = [
   # 'a2386b75ecbc4c2784db1270695dde73',
    'ecd8cc8856884bcc8f02f374f8eb87fc',
    '97985ec5837949cf807aa36544a7ca57',
    '1445303bab134661bdae8e1155482ff0',
    '265e9a7f586d45219a63a7afbe256b33',
    '2eb808b92c9841adb31e2b2608e9afc3']

elements = ['qpf24']#'qpf24', 'qpf12', 'qpf06']#, 'maxt', 'mint']

regions = {
    'WR':['SEW']}#'STO', 'SEW', 'SLC', 'BOI', 'MTR', 'LOX', 'PSR']}


        #"BYZ", "BOI", "LKN", "EKA", "FGZ", "GGW", "TFX", "VEF", "LOX", "MFR",
        #"MSO", "PDT", "PSR", "PIH", "PQR", "REV", "STO", "SLC", "SGX", "MTR",
        #"HNX", "SEW", "OTX", "TWC"]}

arg_set = []

for lead_time_days in [1]:#, 2, 3, 5, 7, 9]:

    for season in season_dates.keys():

        start_date, end_date = season_dates[season]

        # for region in regions.keys():
        region = 'CWA'
        for cwa in regions['WR']:

            processes, logfile = [], {}
            for token, element in zip(api_tokens[:len(elements)], elements):

                arg = f"{season} {start_date} {end_date} {element} " +\
                        f"{lead_time_days} {region} {cwa} {token}".lower()
                        # f"{lead_time_days} {region} {None} {token}".lower()
                        
                cmd_string = f"{pythonpath} ./nbm_4_1_reliability.py {arg}"

                logfile_path = f"{logpath}/{element}.log"

                logfile[element] = open(logfile_path, 'a+')


                print(cmd_string)
                p = subprocess.Popen(cmd_string,
                        stdout=logfile[element], stderr=logfile[element], 
                        shell=True) 
                
                processes.append(p)

            for p in processes:
                p.communicate()