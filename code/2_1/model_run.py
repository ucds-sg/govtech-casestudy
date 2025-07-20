import pandas as pd
import yaml

from pathlib import Path

from preprocess_utils.preprocess import population_preprocess, hdb_merge_with_fertility, model_train_df
from model.model import model_build_and_forecast

# Load config file
# Dynamically identify path of script
script_dir = Path(__file__).parent.parent

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
forecast_data_loc = 'forecast'

# Preprocess Stage

# Preprocess poulation data
## NOTE: Since the population data is only available till 2020, we assume that ECDA wants to plan from 2021-2025
## Note: Some sub-zones are newly created in 2019 and will not be considered for forecast
population_file_name = 'respopagesex2000to2020e.xlsx'
population_data_loc = script_dir.parent / data_loc / raw_data_loc / population_file_name
population_df = population_preprocess(file_loc = population_data_loc)

# Preprocess BTO and fertility data to arrive at drivers for time-series models
fertility_file_name = 'BirthsAndFertilityRatesAnnual.csv'
fertility_data_loc = script_dir.parent / data_loc / raw_data_loc / fertility_file_name

bto_file_name = 'btomapping.csv'
bto_data_loc = script_dir.parent / data_loc / raw_data_loc / bto_file_name

hdb_driver_data = hdb_merge_with_fertility(fertility_data_loc = fertility_data_loc, hdb_data_loc=bto_data_loc)

# Arrive at training data
Y_train, X_pred = model_train_df(population_data = population_df, hdb_data = hdb_driver_data)
print("Training data:\n", Y_train.head())
print("Future driver data:\n", X_pred.head())

# Model Training, Validation and forecast

subzone_forecasts = model_build_and_forecast(Y_train = Y_train, X_pred = X_pred)
forecast_name = 'subzone_forecast.csv'
subzone_forecasts.to_csv(script_dir.parent / data_loc / forecast_data_loc / forecast_name)