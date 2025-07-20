import requests
import os
import time
import yaml
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Access environment variables
SECRET_ONEMAP_KEY = os.getenv("ACCESS_TOKEN")

# Load config file
# Dynamically identify path of script
script_dir = Path(__file__).parent.parent.parent

# Navigate to config file in parent directory
config_path = script_dir / 'config.yaml'

# Open the config file for all paths and ids
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Config for this question
config = config['s2_q1']
image_loc = config['image']
data_loc = config['data_loc']

# Specify additional file locations
raw_data_loc = 'raw_data'

# Preprocess center data
center_filename = 'ListingofCentres.csv'
preschool_center_data_loc = script_dir.parent / data_loc / raw_data_loc / center_filename

# Read data
preschool_center_df = pd.read_csv(preschool_center_data_loc)

# Assume 'postal_code' column exists
postal_codes = preschool_center_df['postal_code'].unique()

# Dictionary to cache responses
cache = {}
results = []

# Step 1: Get latitude and longitude coordinates from Postal codes of the centers
for idx, postal_code in enumerate(postal_codes):
    if postal_code in cache:
        lat, lon = cache[postal_code]
    else:
        url = f'https://www.onemap.gov.sg/api/common/elastic/search?searchVal={postal_code}&returnGeom=Y&getAddrDetails=Y&pageNum=1'
        try:
            response = requests.get(url, verify=False)
            if response.status_code == 200:
                data = response.json()
                if data['results']:
                    lat, lon = data['results'][0]['LATITUDE'], data['results'][0]['LONGITUDE']
                    try:
                        lat, lon = float(lat), float(lon)
                    except (TypeError, ValueError):
                        lat, lon = None, None
                else:
                    lat, lon = None, None
            else:
                print(f"No coordinates found for: {postal_code}")
                lat, lon = None, None
        except Exception as e:
            print(f"Error fetching {postal_code}: {e}")
            lat, lon = None, None
        cache[postal_code] = (lat, lon)
        # Respect API rate limits
        if idx % 50 == 0:
            print(f"Processed {idx} postal codes, pausing for a bit...")
            time.sleep(2)

    results.append({'postal_code': postal_code, 'latitude': lat, 'longitude': lon})

# Convert results to DataFrame
coords_df = pd.DataFrame(results)
# Merge with original DataFrame
df = preschool_center_df.merge(coords_df, on='postal_code', how='left')

# Step 2: Get the planning area from the latitide and longitude coordinates

api_token = SECRET_ONEMAP_KEY
headers = {"Authorization": api_token}
year = 2019

def fetch_planning_area(lat, lon, year, headers):
    url = f"https://www.onemap.gov.sg/api/public/popapi/getPlanningarea?latitude={lat}&longitude={lon}&year={year}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Check the structure of the returned JSON
            if "pln_area_nm" in data:
                return data["pln_area_nm"]
            elif "pln_area" in data:  # Sometimes the field name varies
                return data["pln_area"]
            else:
                print(f"No planning zone found in API response: {data}")
                return None
        else:
            print(f"API call failed: ({lat},{lon}) status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error for ({lat}, {lon}): {e}")
        return None

planning_zones = []

for idx, row in df.iterrows():
    lat, lon = row["latitude"], row["longitude"]
    if pd.notnull(lat) and pd.notnull(lon):
        zone = fetch_planning_area(lat, lon, year, headers)
    else:
        zone = None
    planning_zones.append(zone)
    # For rate-limiting
    if idx % 50 == 0 and idx != 0:
        print(f"Processed {idx} centers, pausing for rate limit...")
        time.sleep(2)

# We finally have planning zone which we can use in our dataset
preschool_center_df["planning_zone"] = planning_zones