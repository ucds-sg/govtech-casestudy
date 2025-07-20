# Load libraries
import pandas as pd
import numpy as np

def preprocess_COE_data(df_loc):
    """
    This function preprocesses COE data to create multiple regressors for our regression and tree-based models
    """

    df = pd.read_csv(df_loc)
    print(df.describe())
    print(df.dtypes)

    # Convert "month" to datetime
    df['date'] = pd.to_datetime(df['month'], format='%Y-%m')

    # Sort data
    df = df.sort_values(['date', 'vehicle_class', 'bidding_no'])

    # Calculate 10 year cumulative sum of quota as COE is valid for 10 years
    # Aggregate by month and vehicle_class
    monthly_agg = df.groupby(['date', 'vehicle_class'], as_index=False).agg({
        'quota': 'sum'
    })

    monthly_agg['quota_10yr_cum'] = (monthly_agg.groupby('vehicle_class').apply(lambda x : 
                                                                                x.sort_values('date').
                                                                                rolling(window = '3650D', on = 'date', min_periods = 1)['quota'].sum()).
                                                                                reset_index(level = 0, drop = True))

    # Merge cumulative data back with original data
    df = df.merge(monthly_agg[['date', 'vehicle_class','quota_10yr_cum']],
                on = ['date', 'vehicle_class'],
                how = 'left')
    # Also create a log transformed cumulative quota column
    df['log_quota_10yr_cum'] = np.log(df['quota_10yr_cum'])

    # Create 6 month and 12 month lags of COE premiums
    ## NOTE: One of our assumptions is that LTA's expected forcast timeline is atleast 6 months in the future
    df = df.sort_values(['date','bidding_no','vehicle_class'])
    grouped = df.groupby(['bidding_no', 'vehicle_class'])
    df['premium_6_month_lag'] = grouped['premium'].shift(6)
    df['premium_12_month_lag'] = grouped['premium'].shift(12)

    # Create bids received/success ratio for a moving 3-month window with a 6 month lag

    df['bids_received'] = df['bids_received'].str.replace(',','').astype(int)
    df['bids_success'] = df['bids_success'].str.replace(',','').astype(int)
    monthly_data = df.groupby(['date', 'vehicle_class'], as_index = False).agg({
        'bids_received': 'sum',
        'bids_success': 'sum'
    }).reset_index(drop = True)
    monthly_data = monthly_data.sort_values(['vehicle_class', 'date'])

    results = []
    for vehicle, group in monthly_data.groupby('vehicle_class'):
        group = group.copy()
        group = group.sort_values('date')

        # Calculate 3-month rolling sum on 'bids_received' and 'bids_success'
        group['bids_received_3m'] = group.rolling(window = '92D', on = 'date', min_periods=1)['bids_received'].sum()
        group['bids_success_3m'] = group.rolling(window = '92D', on = 'date', min_periods=1)['bids_success'].sum()

        # Shift these sums back by 6 months to align with forecast window for the model
        group['bids_received_6m_prior'] = group['bids_received_3m'].shift(6)
        group['bids_success_6m_prior'] = group['bids_success_3m'].shift(6)

        # Calculate bid ratio
        group['bid_ratio'] = (
            group['bids_received_6m_prior'] / group['bids_success_6m_prior']
        ).replace([float('inf'), -float('inf')], None)

        results.append(group)

    # Concatenate all groups
    final_df = pd.concat(results)
    final_df = final_df[['date', 'vehicle_class', 'bid_ratio']]

    # Merge with main dataframe
    df = df.merge(final_df, how = 'left', on = ['date', 'vehicle_class'])

    # Extracting time-based column
    #df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month.astype('category')

    df = df[['bidding_no','vehicle_class','quota','premium','quota_10yr_cum', 'log_quota_10yr_cum' ,'bid_ratio', 'month', 'premium_6_month_lag' ,'premium_12_month_lag']]
    df.dropna(inplace = True)

    print("View of final model input:")
    print(df.head(20))
    print(df.tail(20))

    return df



