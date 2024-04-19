import subprocess

pythonpath = r"C:\Users\Michael.wessler\AppData\Local\anaconda3" +\
                r"\envs\py310\python.exe"
            
season_dates = {
    'DJF':['2023-12-01', '2024-03-01'],
    'MAM':['2023-03-01', '2024-06-01'],
    'JJA':['2023-06-01', '2023-09-01'],
    'SON':['2023-09-01', '2023-12-01'],
    'YEAR':['2023-04-15', '2024-04-16'],
    'POR':['2023-01-18', '2024-04-16']}

api_tokens = [
    'a2386b75ecbc4c2784db1270695dde73',
    'ecd8cc8856884bcc8f02f374f8eb87fc',
    '97985ec5837949cf807aa36544a7ca57',
    '1445303bab134661bdae8e1155482ff0',
    '265e9a7f586d45219a63a7afbe256b33',
    '2eb808b92c9841adb31e2b2608e9afc3']

elements = ['qpf06', 'qpf12', 'qpf24', 'maxt', 'mint']

regions = {
    'WR':["BYZ", "BOI", "LKN", "EKA", "FGZ", "GGW", "TFX", "VEF", "LOX", "MFR",
        "MSO", "PDT", "PSR", "PIH", "PQR", "REV", "STO", "SLC", "SGX", "MTR",
        "HNX", "SEW", "OTX", "TWC"]}

arg_set = []
for season in season_dates.keys():

    start_date, end_date = season_dates[season]

    for element in elements:

        for lead_time_days in [1, 2, 3, 5, 7]:

            for region in regions.keys():

                arg = f"{season} {start_date} {end_date} {element} " +\
                        f"{lead_time_days} {region} {None}".lower()
                
                arg_set.append(arg)

                for cwa in regions[region]:

                    arg = f"{season} {start_date} {end_date} {element} " +\
                        f"{lead_time_days} cwa {cwa}".lower()
                    
                    arg_set.append(arg)

processes = []
task_count = 0

while(task_count < len(arg_set)):

    # for api_token in api_tokens:
    # if task_count < len(arg_set):

    arg_string = f"{arg_set[task_count]} {api_tokens[0]}"
    cmd_string = f"{pythonpath} ./nbm_4_1_reliability.py {arg_string}"
    # print(cmd_string)

    p = subprocess.Popen(cmd_string)
    # processes.append(p)
    
    # for p in processes:
        # p.communicate()
    
    p.wait()
    task_count += 1