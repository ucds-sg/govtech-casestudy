# Load libraries
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from plots import market_plot_1, market_plot_2, ratio_plot

# Load config file
# Dynamically identify path of script
script_dir = Path(__file__).parent.parent

# Navigate to config file in parent directory
config_path = script_dir / 'config.yaml'

# Open the config file for all paths and ids
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Config for this question
config = config['s1_q1']
image_loc = config['image']
data_loc = config['data_loc']
file = config['data']['name']

# Load necessary datasets 
file_path = script_dir.parent / data_loc
agent_property_txn_record = pd.read_csv(file_path / file[0])
print(agent_property_txn_record.describe())
print(agent_property_txn_record.dtypes) # Convert transaction_date to "date" type

resale_prices = pd.read_csv(file_path / file[1])
print(resale_prices.describe())
print(resale_prices.dtypes) # Convert transaction_date to "date" type

# Data Cleaning and pre-processing
## Convert columns to correct data-type
agent_property_txn_record['transaction_date'] = pd.to_datetime(agent_property_txn_record['transaction_date'], format = '%b-%Y')

# The question is specifically focused on HDB resale flats
## Filter agent transaction data to only resale HDB
agent_property_txn_record_filtered = agent_property_txn_record[(agent_property_txn_record['property_type'].isin(['HDB' ,'EXECUTIVE_CONDOMINIUM'])) &\
                                                               (agent_property_txn_record['transaction_type'] == 'RESALE')]
print(agent_property_txn_record.shape)
print(agent_property_txn_record_filtered.shape)

# Clean up date column for resale data
resale_prices['month'] = pd.to_datetime(resale_prices['month'], format = '%Y-%m')
print("Checking to make sure data-correction has taken place:\n", resale_prices.dtypes) # Convert transaction_date to "date" type

# 1st Analysis: # of property agents vs. total resale properties vs. total market of resale
num_property_agents = agent_property_txn_record_filtered.groupby('transaction_date')['salesperson_reg_num'].nunique().reset_index()
monthly_agent_txns = agent_property_txn_record_filtered.groupby('transaction_date').agg(
    total_num_txn_by_agent=('property_type', 'size'),
    num_unique_agents=('salesperson_reg_num', 'nunique')
).reset_index()

# NOTE: Assumption - every house that has a buyer agent in this dataset also has a seller agent ateast, so 
monthly_agent_txns['num_txn_by_agent_per_house'] = monthly_agent_txns['total_num_txn_by_agent']//2
print(monthly_agent_txns.head())

resale_txns = resale_prices.groupby('month').agg(
    num_txn=('town', 'size'),
    sum_txn=('resale_price', 'sum')
).reset_index()
print(resale_txns.head())

# Merge the datasets on date
merged_data = monthly_agent_txns.merge(resale_txns, left_on='transaction_date', right_on='month')
print(merged_data.head())

# Graph 1
market_plot_1(df = merged_data)

# Graph 2
market_plot_2(df = merged_data)

# 2nd Analysis: Impact on buyer vs. seller agents
num_buyer_seller_agents = agent_property_txn_record_filtered.groupby(['transaction_date','represented']).agg(
    total_num_txn_by_representation=('property_type', 'size')).reset_index()

# Pivot 'represented' to columns
num_buyer_seller_agents_pivot = num_buyer_seller_agents.pivot_table(
    index='transaction_date',
    columns='represented',
    values='total_num_txn_by_representation',
    fill_value=0
).reset_index()

merged_data = merged_data.merge(num_buyer_seller_agents_pivot, on = 'transaction_date')
print(merged_data.head())

# Calculate ratios
merged_data['agents_per_txn'] = merged_data['total_num_txn_by_agent'] / merged_data['num_txn']
merged_data['buyer_agents_per_txn'] = merged_data['BUYER'] / merged_data['num_txn']
merged_data['seller_agents_per_txn'] = merged_data['SELLER'] / merged_data['num_txn']

# Graph 3
ratio_plot(df = merged_data, title = 'Buyer and Seller Agents per transaction')

# 3rd Analysis: Demand by Town
monthly_agent_txns_by_town = agent_property_txn_record_filtered.groupby(['transaction_date','town']).agg(
    total_num_txn_by_agent=('property_type', 'size'),
    num_unique_agents=('salesperson_reg_num', 'nunique')
).reset_index()

num_buyer_seller_agents_by_town = agent_property_txn_record_filtered.groupby(['transaction_date','represented','town']).agg(
    total_num_txn_by_representation=('property_type', 'size')).reset_index()
# Pivot 'represented' to columns
num_buyer_seller_agents_pivot_by_town = num_buyer_seller_agents_by_town.pivot_table(
    index=['transaction_date','town'],
    columns='represented',
    values='total_num_txn_by_representation',
    fill_value=0
).reset_index()

merged_data_by_town = monthly_agent_txns_by_town.merge(num_buyer_seller_agents_pivot_by_town, on = ['transaction_date','town'])

resale_txns_by_town = resale_prices.groupby(['month', 'town']).agg(
    num_txn=('town', 'size'),
    sum_txn=('resale_price', 'sum')
).reset_index()

merged_data_by_town = merged_data_by_town.merge(resale_txns_by_town, left_on=['transaction_date','town'], right_on=['month', 'town'])

# Calculate ratios
merged_data_by_town['agents_per_txn'] = merged_data_by_town['total_num_txn_by_agent'] / merged_data_by_town['num_txn']
merged_data_by_town['buyer_agents_per_txn'] = merged_data_by_town['BUYER'] / merged_data_by_town['num_txn']
merged_data_by_town['seller_agents_per_txn'] = merged_data_by_town['SELLER'] / merged_data_by_town['num_txn']
print(merged_data_by_town.head())

# Graph(s) 4
for town in merged_data_by_town['town'].unique():
    subset_df = merged_data_by_town[merged_data_by_town['town'] == town]
    
    if town == 'KALLANG/WHAMPOA':
        town = 'KALLANG_WHAMPOA'
    ratio_plot(df = subset_df, title = 'Buyer and Seller Agents per transaction - ' + town)
