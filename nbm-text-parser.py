import re
import os
import sys
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import logging

# Supported file types
supported_file_types = ['nbp', 'nbe', 'nbs', 'nbx']

# Hardcoded parent directory of text files
parent_directory = '/nas/stid/data/nbm/v5p0_text/' #'/data/nbm_text/v4p2_text/'

# Hardcoded output directory
output_directory = '/data/nbm_text/csv/'

# Hardcoded logging directory
logging_directory = '/data/nbm_text/'

# Debugging option
debug = False  # Toggle between single-threaded (True) and multi-threaded (False) processing

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Command-line argument for file type
file_type = sys.argv[1] if len(sys.argv) > 1 else None

if file_type not in supported_file_types:
    raise ValueError(f"Unsupported file type: {file_type}. Supported types are: {', '.join(supported_file_types)}")

# Optional hardcoded init hours (can be set to None for all hours)
init_hours_input = None

if init_hours_input:
    try:
        init_hours = [f"{int(hour):02d}z" for hour in init_hours_input.split(",")]  # Format as '01z', '13z'
    except ValueError:
        raise ValueError("Invalid init hours. Please enter integers separated by commas (e.g., '1,13').")
else:
    init_hours = None  # Process all init hours if no specific input

# Set up logging configuration
log_file_name = f"parse_{file_type}_t{init_hours[0] if init_hours else 'all'}z_log.log"
log_file = os.path.join(logging_directory, log_file_name)
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Log the start of the script
logging.info("Script started.")

# Build the search pattern based on selected init hours
matching_files = []
if init_hours:
    for hour in init_hours:
        hour_pattern = f".t{hour}.txt"  # e.g., ".t01z.txt"
        search_pattern = f"{parent_directory}/**/blend_*{file_type}*{hour_pattern}"
        matching_files.extend(glob.glob(search_pattern, recursive=True))
else:
    # Search all init hours if no specific hour is selected
    search_pattern = f"{parent_directory}/**/blend_*{file_type}*.txt"
    matching_files = glob.glob(search_pattern, recursive=True)

if not matching_files:
    logging.info(f"No files found matching the file type '{file_type}' or init hours in the directory: {parent_directory}")
    exit()

logging.info(f"Found {len(matching_files)} file(s) matching the file type '{file_type}' and init hours in the directory: {parent_directory}")

# Sort the files for consistent processing
matching_files.sort()

# Filter out files that already have corresponding .csv files
files_to_process = []
for input_file in matching_files:
    # Extract the directory name (e.g., '20231118') and input file name
    directory_name = os.path.basename(os.path.dirname(input_file))
    input_file_name = os.path.basename(input_file)

    # Extract the hour (e.g., 't01z' -> '01') from the input file name
    hour = input_file_name.split('.')[-2][1:3]  # Extract the hour part (e.g., '01' from 't01z')

    output_file_name = f"blend_{file_type}tx_{directory_name}{hour}.csv"
    output_file_path = os.path.join(output_directory, output_file_name)

    # Add the file to the processing list if the output file doesn't already exist
    if not os.path.exists(output_file_path):
        files_to_process.append(input_file)

if not files_to_process:
    logging.info("All files have already been processed. No work to do.")
    exit()

logging.info(f"Processing {len(files_to_process)} file(s) that do not have existing .csv outputs.")

def process_file_with_logging(input_file_path, total_files, file_index):
    """
    Process a single file and log progress.

    Args:
        input_file_path (str): Path to the input file to process.
        total_files (int): Total number of files to process.
        file_index (int): Index of the current file (1-based).
    """
    try:
        # Extract the base name of the input file
        input_file_name = os.path.basename(input_file_path)
        logging.info(f"({file_index}/{total_files}) Processing file: {input_file_name}")

        # Read the file
        with open(input_file_path, "r") as file:
            lines = file.read().splitlines()

        # Split the content into blocks based on blank rows
        blocks = []
        current_block = []
        for line in lines:
            if not line.strip():  # Blank line indicates end of a block
                if current_block:
                    blocks.append(current_block)
                    current_block = []
            else:
                current_block.append(line)

        # Add the last block if it wasn't added
        if current_block:
            blocks.append(current_block)

        # Process metadata from the first row of each block, discarding invalid blocks
        first = True
        valid_blocks = []
        output_init_time = None  # Will store the init_time for the output filename
        for block in blocks:
            if block:  # Ensure the block isn't empty
                metadata_line = block[0]  # First row contains metadata
                parts = re.split(r"\s{2,}", metadata_line.strip())  # Split by 2+ spaces

                # Validate metadata: ensure we have a proper init time (last two parts of the metadata line)
                if len(parts) >= 2:
                    site_id = parts[0].split()[0]  # Extract SITE_ID (first part)
                    init_time = " ".join(parts[-2:])  # Combine date and time (last two parts)

                    # Check if the init_time is in the correct format
                    if re.match(r"^\d{1,2}/\d{1,2}/\d{4} \d{4} UTC$", init_time):
                        # Convert init_time to datetime
                        init_time_dt = datetime.strptime(init_time, "%m/%d/%Y %H%M %Z")
                        directory_date = os.path.basename(os.path.dirname(input_file_path))

                        # Debugging: Print init_time and directory_date
                        if debug:
                            print(f"File: {input_file_name}")
                            print(f"  Parsed init_time: {init_time_dt.strftime('%Y-%m-%d %H:%M:%S')} (from metadata)")
                            print(f"  Directory date: {directory_date} (from folder structure)")

                        # Check if init_time matches the directory date
                        if init_time_dt.strftime('%Y%m%d') != directory_date:
                            if debug:
                                print(f"  Skipping block: init_time ({init_time_dt.strftime('%Y%m%d')}) "
                                      f"does not match directory_date ({directory_date}).")
                            if first: 
                                first = False  # Only log the first occurrence
                                logging.warning(f"({file_index}/{total_files}) init_time ({init_time_dt.strftime('%Y%m%d')}) does not match directory date ({directory_date}) for file: {input_file_path}")
                            continue  # Skip this block

                        valid_blocks.append((site_id, init_time, block))  # Store valid blocks
                        if output_init_time is None:
                            output_init_time = init_time_dt.strftime("%Y%m%d%H")
                    else:
                        if debug:
                            print(f"  Skipping block: Invalid init_time format in metadata: {init_time}")
                        logging.warning(f"Invalid init_time format in metadata: {init_time} for file: {input_file_name}")
                        continue

        # Skip the file if no valid blocks were found
        if not valid_blocks:
            logging.warning(f"No valid blocks found in file: {input_file_name}")
            if debug:
                print(f"No valid blocks found in {input_file_name}. File skipped.")
            return

        # Prepare the DataFrame
        all_dataframes = []

        # Start parsing data rows for each valid block
        for site_id, init_time, block in valid_blocks:
            # Convert init_time to a datetime object
            init_time_dt = datetime.strptime(init_time, "%m/%d/%Y %H%M %Z")

            # Skip the first three rows (metadata + headers)
            data_rows = block[3:]

            # Initialize a dictionary to store the data for the current block
            block_data = {}

            # Define bad data codes
            bad_data_codes = ["-459", "-99"]

            # Parsing logic for each file type
            if file_type == 'nbp':
                for row in data_rows:
                    variable_name = row[1:6].strip()
                    if any(bad_code in row for bad_code in bad_data_codes):
                        parsed_data = [np.nan] * 17
                    else:
                        parsed_data = [row[i:i + 3].strip() for i in range(7, 74, 3)]
                        parsed_data = [np.nan if val == "" else val for val in parsed_data]
                    block_data[variable_name] = parsed_data

            elif file_type == 'nbs':
                fhr_processed = False
                for row in data_rows:
                    variable_name = row[1:4].strip()
                    if any(bad_code in row for bad_code in bad_data_codes):
                        parsed_data = [np.nan] * 23
                    else:
                        if not fhr_processed and variable_name == "FHR":
                            parsed_data = [row[i:i + 3].strip() for i in range(5, 74, 3)]
                            fhr_processed = True
                        else:
                            parsed_data = [row[i:i + 3].strip() for i in range(5, len(row), 3)]
                        parsed_data = [np.nan if val == "" else val for val in parsed_data]
                    block_data[variable_name] = parsed_data

            elif file_type == 'nbe':
                for row in data_rows:
                    variable_name = row[1:4].strip()
                    if any(bad_code in row for bad_code in bad_data_codes):
                        parsed_data = [np.nan] * 23
                    else:
                        parsed_data = [row[i:i + 3].strip() for i in range(7, len(row) - 7, 4)]
                        parsed_data = [np.nan if val == "" else val for val in parsed_data]
                    block_data[variable_name] = parsed_data

            elif file_type == 'nbx':
                for row in data_rows:
                    variable_name = row[1:4].strip()
                    if any(bad_code in row for bad_code in bad_data_codes):
                        parsed_data = [np.nan] * 23
                    else:
                        parsed_data = [row[i:i + 3].strip() for i in range(5, len(row) - 6, 4)]
                        parsed_data = [np.nan if val == "" else val for val in parsed_data]
                    block_data[variable_name] = parsed_data

            # Convert block_data to a DataFrame and align rows/columns
            block_df = pd.DataFrame({k: pd.Series(v) for k, v in block_data.items()})

            # Add site_id and init_time columns
            block_df["site_id"] = site_id
            block_df["init_time"] = init_time_dt

            # Store the block DataFrame
            all_dataframes.append(block_df)

        # Combine all block DataFrames into a single DataFrame, aligning variables
        df = pd.concat(all_dataframes, ignore_index=True)

        # Set MultiIndex [init_time, site_id]
        df.set_index(["init_time", "site_id"], inplace=True)

        # Generate the output filename
        directory_date = os.path.basename(os.path.dirname(input_file_path))
        output_file_name = f"blend_{file_type}tx_{directory_date}{hour}.csv"
        output_file_path = os.path.join(output_directory, output_file_name)

        # Save the DataFrame to a CSV file
        df.to_csv(output_file_path)

        logging.info(f"({file_index}/{total_files}) File saved: {output_file_name}")
    except Exception as e:
        logging.error(f"Error processing file {input_file_path}: {e}")
        logging.error(f"Error processing file {input_file_path}: {e}")

if debug:
    # Single-threaded processing for debugging
    logging.info("Debug mode enabled: Processing files in single-threaded mode.")
    for idx, file in enumerate(files_to_process):
        process_file_with_logging(file, len(files_to_process), idx + 1)
else:
    # Multi-threaded processing
    num_workers = cpu_count() * 1
    logging.info(f"Processing files in multi-threaded mode with {num_workers} workers.")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_file_with_logging, file, len(files_to_process), idx + 1): file
            for idx, file in enumerate(files_to_process)
        }
        for future in futures:
            future.result()

logging.info("Script completed.")