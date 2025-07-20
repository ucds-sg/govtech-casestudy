import pandas as pd
import yaml
from pathlib import Path

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

# Since the population data is only available till 2020, we assume that ECDA wants to plan from 2020-2025
file_name = 'btomapping.csv'
population_data_loc = script_dir.parent / data_loc / raw_data_loc / file_name

# Function to preprocess population data
def population_preprocess(file_loc):

    """
    Returns preprocessed population data from provided location
    """

    # Population data pre-processing
    ## Read sheet, skipping first 2 header/descriptive rows
    sheets = pd.read_excel(file_loc, sheet_name=None, skiprows=2)
    all_data = []

    for sheet_name, df in sheets.items():

        # Assuming header is the first row, otherwise specify header row
        df = df.copy()

        # Convert all column headers to strings
        df.columns = df.columns.astype(str)

        # Identify year columns dynamically
        year_cols = [col for col in df.columns if col.isdigit()]

        # Ensure that all necessary columns are present in input data
        fixed_cols = ['Planning Area', 'Subzone', 'Age', 'Sex']
        for col in fixed_cols:
            if col not in df.columns:
                raise Exception(f"Error: '{col}' not found in sheet '{sheet_name}'")
        
        # Pivot the data: melt year columns into "Year" and "Population"
        melted = df.melt(
            id_vars=fixed_cols,
            value_vars=year_cols,
            var_name='Year',
            value_name='Population'
        )

        # Convert 'Year' to int type for filtering
        melted['Year'] = melted['Year'].astype(int)

        # Convert 'Age' to numeric, dropping all other dtypes
        melted['Age'] = pd.to_numeric(melted['Age'], errors='coerce')
        melted = melted.dropna(subset=['Age'])
        melted['Age'] = melted['Age'].astype(int)

        # Filter based on pre-school criteria and focusing on correct sub-zones only:
        filtered = melted[
            (melted['Age'] >= 1) & (melted['Age'] <= 6) &
            (melted['Subzone'] != 'Total') &
            (melted['Sex'] == 'Total') &
            (melted['Planning Area'] != 'Total')
        ]

        # Filter out Notes at the end of every sheet
        notes_keywords = [
            'Planning areas refer to areas demarcated',
            '"-" Nil or negligible.',
            'Data has been rounded',
            'The data may not add up'
        ]
        for keyword in notes_keywords:
            melted = melted[~melted.apply(lambda row: row.astype(str).str.contains(keyword, case=False)).any(axis=1)]


        all_data.append(filtered)

    # Concatenate all sheets into one DataFrame
    final_df = pd.concat(all_data, ignore_index=True)

    # Convert 'Population' to numeric, dropping all other dtypes
    final_df['Population'] = pd.to_numeric(final_df['Population'], errors='coerce')
    final_df = final_df.dropna(subset=['Population'])
    final_df['Population'] = final_df['Population'].astype(int)
    # Aggregate pre-school population by Year, Planning Area and Subzone
    final_df = final_df.groupby(['Planning Area', 'Subzone', 'Year'])['Population'].sum().reset_index()

    return final_df

def fertility_rate_preprocess(file_loc):
    """
    Returns preprocessed fertility data from provided location
    """
        
    # Read the file
    df = pd.read_csv(file_loc)

    # Remove white space from column
    df['DataSeries'] = df['DataSeries'].str.strip()

    # Filter for the ethnicities of interest
    ethnicities_of_interest = ['Chinese', 'Malays', 'Indians']
    df_filtered = df[df['DataSeries'].isin(ethnicities_of_interest)]

    # First, set 'DataSeries' as index
    df_filtered = df_filtered.set_index('DataSeries')

    # Select only relevant years
    years = ['2018', '2019', '2020']
    
    # Keep only relevant columns (assuming columns exist)
    df_filtered = df_filtered[years]

    # Transpose so years and ethnicities are flipped
    pivot_df = df_filtered.T

    # Reset index to make years a column
    pivot_df = pivot_df.reset_index()

    # Rename 'index' to 'Year'
    pivot_df = pivot_df.rename(columns={'index': 'Year'})

    return pivot_df

def hdb_merge_with_fertility(hdb_data_loc, fertility_data_loc):
    """
    Preprocessed HDB data and merges it with fertility data to arrive at drivers
    """

    # Preprocess fertilitity data first
    fertility_df = fertility_rate_preprocess(file_loc = fertility_data_loc)

    # Read bto data
    bto_data = pd.read_csv(hdb_data_loc)
    agg_bto_data = bto_data.groupby(['Planning area','Subzone', 'Estimated completion year'])['Total number of units'].sum().reset_index()
    agg_bto_data = agg_bto_data.rename(columns={'Total number of units': 'num_units'})

    # For future, we are using the fertility rates from 2020, so we create a new column to join the data
    agg_bto_data['year_join_col'] = agg_bto_data['Estimated completion year'].clip(upper = 2020)
    
    # Change dtype to merge
    agg_bto_data['Estimated completion year'] = agg_bto_data['Estimated completion year'].astype(int)
    fertility_df['Year'] = fertility_df['Year'].astype(int)

    # Map to fertility data
    merged_df = agg_bto_data.merge(fertility_df, left_on = 'year_join_col', right_on = 'Year', how = 'left')

    # Create new population metrics
    ## NOTE: Assume that distribution of Chineese:Malay:Indian population is 75:15:10
    merged_df['chineese_preschool_population'] = merged_df['num_units'] * 0.75 * merged_df['Chinese']
    merged_df['malay_preschool_population'] = merged_df['num_units'] * 0.15 * merged_df['Malays']
    merged_df['indian_preschool_population'] = merged_df['num_units'] * 0.10 * merged_df['Indians']

    train_data_columns = ['Planning area', 'Subzone', 'Estimated completion year', 'num_units', 'chineese_preschool_population', 
                          'malay_preschool_population', 'indian_preschool_population']
    
    train_hdb_data = merged_df[train_data_columns]
    train_hdb_data = train_hdb_data[train_hdb_data['Estimated completion year'] <= 2025]

    return train_hdb_data

def model_train_df(population_data, hdb_data):
    """Create model data by merging population data and merging it with drivers
    """
    model_data = population_data.merge(hdb_data, left_on = ['Planning Area', 'Subzone', 'Year'], 
                                       right_on = ['Planning area', 'Subzone', 'Estimated completion year'], how = 'left')
    
    model_data = model_data.fillna(0)

    # We rename the columns in the format that Statsforecast requires
    model_data['unique_id'] = model_data['Planning Area'] + '_' + model_data['Subzone']
    model_data = model_data.rename(columns={'Year': 'ds','Population':'y'})
    Y_train = model_data[['unique_id', 'ds', 'y', 'num_units', 
                          'chineese_preschool_population', 'malay_preschool_population', 'indian_preschool_population']]
    
    # Remove any series which appears less than 2 times as its too new
    counts = Y_train['unique_id'].value_counts()
    valid_ids = counts[counts > 2].index
    Y_train = Y_train[Y_train['unique_id'].isin(valid_ids)]

    # Create future regressors from HDB data
    X_test = hdb_data[(hdb_data['Estimated completion year'] > 2020) & (hdb_data['Estimated completion year'] <= 2025)].reset_index(drop = True)
    X_test['unique_id'] = X_test['Planning area'] + '_' + X_test['Subzone']
    
    # Cross join with all possible combinations of unique id and all years in forecast window
    unique_ids = Y_train['unique_id'].unique()
    years = list(range(2021, 2026,1))

    combos = list(pd.MultiIndex.from_product([unique_ids, years]))
    grid = pd.DataFrame(combos, columns = ['unique_id', 'ds'])
    X_test_complete = grid.merge(X_test, how = 'left', left_on=['unique_id', 'ds'], right_on = ['unique_id', 'Estimated completion year'])

    # Fill blanks and NAs with 0
    X_test_complete = X_test_complete.fillna(0)
    X_test_complete = X_test_complete[['unique_id', 'ds', 'num_units', 
                                        'chineese_preschool_population', 'malay_preschool_population', 'indian_preschool_population']]

    return Y_train, X_test_complete