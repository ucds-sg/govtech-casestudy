import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

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

# Graph 1
def market_plot_1(df):
    """
    Plots two different values on primary and secondary axis

    Args:
    df: dataframe with relevant columns for the plots

    Returns:
    None
    """
    # Plot setup
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid")

    # Create the primary axis
    fig, ax1 = plt.subplots()

    # Plot 'num_unique_agents' on primary axis
    ax1.plot(df['transaction_date'], df['num_unique_agents'], label='Unique Agents', color='deepskyblue', linewidth=1.5)

    # Style primary y-axis
    ax1.set_xlabel('Transaction Date', fontsize=12)
    ax1.set_ylabel('Number of Unique Agents', fontsize=12, color='deepskyblue')
    ax1.tick_params(axis='y', labelcolor='deepskyblue')
    #ax1.yaxis.set_major_locator(MaxNLocator(nbins=8))

    # Create secondary axis
    ax2 = ax1.twinx()

    # Plot 'sum_txn' on secondary axis
    ax2.plot(df['transaction_date'], df['sum_txn'], label='Total Value of Resale Txns', color='tomato', linewidth=1.5)
    ax2.set_ylabel('Total Resale Value', fontsize=12, color='tomato')
    ax2.tick_params(axis='y', labelcolor='tomato')

    # For Y-axes, choose min/max from data and generate common ticks
    y1_min,y1_max = 0, df['num_unique_agents'].max()
    y1_ticks = np.linspace(y1_min,y1_max, 8)
    ax1.set_yticks(y1_ticks)
    ax1.set_ylim(y1_min,y1_max )

    y2_min,y2_max = 0, df['sum_txn'].max()
    y2_ticks = np.linspace(y2_min,y2_max, 8)
    ax2.set_yticks(y2_ticks)
    ax2.set_ylim(y2_min,y2_max )

    # Title and layout
    plt.title("Unique # of Agents and Resale Value Transacted", fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Add legends (handle both axes)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=10)

    image_file = script_dir.parent / image_loc / 'No. of agents vs. Transaction Value.png'
    plt.savefig(image_file,dpi = 300)
    plt.close()


def market_plot_2(df):
    """
    Plots two different values on primary and secondary axis

    Args:
    df: dataframe with relevant columns for the plots

    Returns:
    None
    """
    # Plot setup
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid")

    # Create the primary axis
    fig, ax1 = plt.subplots()

    # Plot 'num_unique_agents' on primary axis
    ax1.plot(df['transaction_date'], df['num_unique_agents'], label='Unique Agents', color='deepskyblue', linewidth=1.5)

    # Style primary y-axis
    ax1.set_xlabel('Transaction Date', fontsize=12)
    ax1.set_ylabel('Number of Unique Agents', fontsize=12, color='deepskyblue')
    ax1.tick_params(axis='y', labelcolor='deepskyblue')
    #ax1.yaxis.set_major_locator(MaxNLocator(nbins=8))

    # Create secondary axis
    ax2 = ax1.twinx()

    # Plot 'sum_txn' on secondary axis
    ax2.plot(df['transaction_date'], df['num_txn'], label='Total Number of Resale Txns', color='red', linewidth=1.5)
    ax2.set_ylabel('Number of Transactions', fontsize=12, color='tomato')
    ax2.tick_params(axis='y', labelcolor='tomato')

    # For Y-axes, choose min/max from data and generate common ticks
    y1_min,y1_max = 0, df['num_unique_agents'].max()
    y1_ticks = np.linspace(y1_min,y1_max, 8)
    ax1.set_yticks(y1_ticks)
    ax1.set_ylim(y1_min,y1_max )

    y2_min,y2_max = 0, df['num_txn'].max()
    y2_ticks = np.linspace(y2_min,y2_max, 8)
    ax2.set_yticks(y2_ticks)
    ax2.set_ylim(y2_min,y2_max )

    # Title and layout
    plt.title('Unique # of Agents vs Number of Resale Transactions', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Add legends (handle both axes)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=10)

    image_file = script_dir.parent / image_loc / 'No. of agents vs. No. of transactions.png'
    plt.savefig(image_file,dpi = 300)
    plt.close()

def ratio_plot(df, title):
    """
    Plots multiple values on same y-axis

    Args:
    df: dataframe with relevant columns for the plots
    title: title of the chart

    Returns:
    None
    """
    # Plot df
    plt.figure(figsize=(14, 8))
    sns.set(style='whitegrid')

    # Plot all three ratios on the same axis
    plt.plot(df['transaction_date'], df['agents_per_txn'], label='# of Agents per Resale', color='royalblue')
    plt.plot(df['transaction_date'], df['buyer_agents_per_txn'], label='Buyer Agents per Resale', color='lightsteelblue', linestyle = '--')
    plt.plot(df['transaction_date'], df['seller_agents_per_txn'], label='Seller Agents per Resale' , color='lightskyblue', linestyle = '--')

    # Labels and title
    plt.xlabel('Transaction Date', fontsize=12)
    plt.ylabel('Ratios', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')

    # Set y-limit axis
    ax = plt.gca()
    ax.set_ylim(0,5)
    ax.set_yticks([1,2,3,4,5])

    # Improve tick layout
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)

    title = title + '.png'
    image_file = script_dir.parent / image_loc / title
    plt.savefig(image_file,dpi = 300)
    plt.close()
