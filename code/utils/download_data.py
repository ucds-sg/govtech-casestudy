import json
import requests
import time
import yaml
import pandas as pd
from pathlib import Path

# Leverage the function available in Google Collab on data.gov.sg
def download_file(DATASET_ID):
  # initiate download
  initiate_download_response = s.get(
      f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}/initiate-download",
      headers={"Content-Type":"application/json"},
      json={}
  )
  print(initiate_download_response.json()['data']['message'])

  # poll download
  MAX_POLLS = 5
  for i in range(MAX_POLLS):
    poll_download_response = s.get(
        f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}/poll-download",
        headers={"Content-Type":"application/json"},
        json={}
    )
    print("Poll download response:", poll_download_response.json())
    if "url" in poll_download_response.json()['data']:
      print(poll_download_response.json()['data']['url'])
      DOWNLOAD_URL = poll_download_response.json()['data']['url']
      df = pd.read_csv(DOWNLOAD_URL)

      print(df.head())
      print("\nDataframe loaded!")
      return df
    if i == MAX_POLLS - 1:
      print(f"{i+1}/{MAX_POLLS}: No result found, possible error with dataset, please try again or let us know at https://go.gov.sg/datagov-supportform\n")
    else:
      print(f"{i+1}/{MAX_POLLS}: No result yet, continuing to poll\n")
    time.sleep(3)

# Dynamically identify path of script
script_dir = Path(__file__).parent.parent

# Navigate to config file in parent directory
config_path = script_dir / 'config.yaml'
print(config_path) ## Ensure path is correct

# Open the config file for all paths and ids
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Loop through every project and datasets necessary for the project
for question in config:
  data_folder = config[question]['data_loc']
  data = config[question]['data']
  # download all files based on config 
  for DATASET_ID, name in zip(data['DATASET_ID'], data['name']):
    file_loc = data_folder + name
    df = download_file(DATASET_ID)
    df.to_csv(file_loc = file_loc, index = False)

