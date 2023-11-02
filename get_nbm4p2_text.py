#!/usr/local/anaconda2/envs/py3_pygrib/bin/python

"""-------------------------------------------------------------
    Script Name: 	get_nbm4p2_text.py
    Description: 	Downloads NBM 4.2 text products
    Author: 		Chad Kahler (WRH STID), Michael Wessler (WRH STID)
    Date:			11/2/2023
-------------------------------------------------------------"""

#---------------------------------------------------------------
# Import python packages
#---------------------------------------------------------------
import os, sys, datetime, time, shutil, traceback, pycurl, json
 
#-------------------------------------------------------
# Global configuration options
#-------------------------------------------------------
lookback = 6
num_hrs = 48
download_dir = "/nas/stid/data/nbm/v4p2_text"
download_again = False
files = []

#-------------------------------------------------------
# Convert file size to useful units
#------------------------------------------------------
def downloadFiles(dt_valid):
    """
    this function will download NBM text files
    """

    print("#----------------------------------------------------")
    print("# Looking for NBM 4.2 text product valid: " +\
          "%sZ" % dt_valid.strftime("%b %d, %Y %H"))
    print("#----------------------------------------------------")

    date_dir = dt_valid.strftime("%Y%m%d")

    # make download directories if they don't exist
    if not os.path.exists(download_dir + "/" + date_dir):
        os.makedirs(download_dir + "/" + date_dir)
        os.system("/usr/bin/chmod 775 " + download_dir + "/" + date_dir)

    # NBH	Hourly	1-Hourly	Hours 1-25
    # NBS	Short	3-Hourly	Hours 6-72*
    # NBE	Extended	12-Hourly	Hours 24-192*
    # NBX	Super-Extended	12-Hourly	Hours 204-264* (continuation of NBE)
    # NBP	Probabilistic (Extended period)	12-Hourly	Hours 24-228*

    for text_product in ["nbs", "nbe", "nbx", "nbp"]:

        text_file = f"blend_{text_product}tx.t%sz.txt" % dt_valid.strftime("%H")

        download_path = download_dir + "/" + date_dir + "/" + text_file

        remote_path = f"https://blend.mdl.nws.noaa.gov/nbm/txtdev/{text_file}"

        print("Remote: " + remote_path)
        print("Local: " + download_path)
                
        if not os.path.exists(download_path) or download_again:
            print("Attempting to download: " + remote_path)

            try:
                c = pycurl.Curl()
                f = open(download_path,"wb")
                c.setopt(c.URL, remote_path)
                c.setopt(c.WRITEDATA, f)

                c.perform()

                # Elapsed time for the transfer.
                print('Download time: %f' % c.getinfo(c.TOTAL_TIME))
                
                c.close()
                f.close()

                if os.path.getsize(download_path) < 1000000:  # check if < 1 MB
                    print("Download file is too small...removing: ", 
                        "%s" % download_path)
                    os.remove(download_path)
            
            except Exception as err:
                print(traceback.format_exc())
                if os.path.getsize(download_path) < 1000000:  # check if < 1 MB
                    print("Download file is too small...removing: ", 
                        "%s" % download_path)
                    os.remove(download_path)
                pass

            if os.path.exists(download_path):
                print("Successfully downloaded NBM 4.2 text product valid: ",
                    dt_valid.strftime("%a %b %d, %Y %HZ"))
                print("Filesize: " + fileSize(download_path) + "\n")

                os.system("/usr/bin/chmod 775 %s" % download_path)
            else:
                print("Unable to download NBM 4.2 text product valid: ",
                        dt_valid.strftime("%a %b %d, %Y %HZ") + "\n")
        else:
            print(download_path + " already exists\ndownload_again = False\n")

#-------------------------------------------------------
# Convert file size to useful units
#------------------------------------------------------
def convertBytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0

#-------------------------------------------------------
# Check file size and return in useful units
#------------------------------------------------------
def fileSize(file_path):
    """
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convertBytes(file_info.st_size)

def main():

    start = datetime.datetime.utcnow()
    print("\nScript executed at " + start.strftime("%a %b %d, %Y %H:%M:%S Z\n"))

    dt = datetime.datetime.utcnow() - datetime.timedelta(hours=3)

    for hr in range(0, 49):

        dt_chk = dt - datetime.timedelta(hours=hr)

        if dt_chk.hour == 1 or dt_chk.hour == 13:

            try:
                downloadFiles(dt_chk)		
                
            except Exception as err:
                print(traceback.format_exc())

    end = datetime.datetime.utcnow()
    print("\nScript completed at " + end.strftime("%a %b %d, %Y %H:%M:%S Z"))
    diff_minute = (end-start).total_seconds()/60
    print("Script execution: %.2f" % diff_minute + " minutes\n")

if __name__ == "__main__":
    main()
