# Load libraries
import yaml
import pandas as pd
import numpy as np
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
config = config['s1_q2']
image_loc = config['image']
data_loc = config['data_loc']
file = config['data']['name']

# Load necessary datasets 
file_path = script_dir.parent / data_loc

# Understand the data
COEdata = pd.read_csv(file_path / file[1])
print(COEdata.describe())
print(COEdata.dtypes)
df = COEdata

# Preprocess to pass onto the function
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
df = df.merge(final_df, how = 'left', on = ['date', 'vehicle_class'])

print(df.head(20))
print(df.tail(20))

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

def regression_pipeline(df, target_column, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    num_cols = X.select_dtypes(include=['number']).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    print("One-hot encoding categorical variables:")
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy = 'median')),
            ('scaler', StandardScaler())
        ]),num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy = 'most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]),cat_cols)
    ])

    models = {
        'Linear Regression (L1)': (
            Lasso(random_state = random_state, max_iter = 50000), 
            {'model__alpha': [0.01,0.1,1,10]}
        ),
        'Random Forest': (
            RandomForestRegressor(random_state=random_state),
            {'model__n_estimators': [50, 100], 'model__max_depth': [3, 5, None]}
        ),
        'Gradient Boosting': (
            GradientBoostingRegressor(random_state=random_state),
            {'model__n_estimators': [50, 100], 'model__learning_rate': [0.05, 0.1]}
        ),
        'Hist GB Regressor': (
            HistGradientBoostingRegressor(random_state=random_state),
            {'model__max_iter': [100, 200], 'model__learning_rate': [0.05, 0.1]}
        ),
        'Support Vector Regressor': (
            SVR(),
            {'model__kernel': ['rbf'], 'model__C': [1, 10]}
        ),
        'XGBoost': (
            XGBRegressor(random_state=random_state, verbosity=0),
            {'model__n_estimators': [50, 100], 'model__learning_rate': [0.05, 0.1], 'model__max_depth': [3, 5]}
        ),
    }

    best_score = -float('inf')
    best_model = None
    best_name = None
    best_grid = None

    print("Performing 5-fold Grid Search for best hyperparameters:")
    for name, (model, params) in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        grid = GridSearchCV(pipe, param_grid=params, cv=5, scoring='r2')
        grid.fit(X_train, y_train)
        score = grid.best_score_
        print(f'{name} CV R^2 score: {score:.4f}')
        if score > best_score:
            best_score = score
            best_model = grid.best_estimator_
            best_name = name
            best_grid = grid

    y_pred = best_model.predict(X_test)
    test_score = r2_score(y_test, y_pred)
    print(f"\nBest Model: {best_name} (CV R^2: {best_score:.4f}, Test R^2: {test_score:.4f})\n")
    return X_train, X_test, y_train, y_test, best_model, best_grid

X_train, X_test, y_train, y_test, best_model, best_grid = regression_pipeline(df = df, target_column = 'premium')
print(best_model)

import matplotlib.pyplot as plt
import seaborn as sns

def plot_regression_results(model, X_train, y_train, X_test, y_test, title_prefix=""):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    plt.figure(figsize=(14, 6))
    
    # Scatter plot: Train
    plt.subplot(1,2,1)
    sns.scatterplot(x=y_train, y=y_train_pred, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')  # Diagonal
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{title_prefix} Train: Actual vs. Predicted")
    
    # Scatter plot: Test
    plt.subplot(1,2,2)
    sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{title_prefix} Test: Actual vs. Predicted")
    
    plt.tight_layout()
    plt.show()

# Example usage:
# best_model, best_grid = regression_pipeline(...)
plot_regression_results(best_model, X_train, y_train, X_test, y_test, title_prefix="Best Model")




