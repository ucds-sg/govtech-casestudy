# Load libraries
import yaml
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from preprocess_utils.data_preprocess import preprocess_COE_data
from model.model_training import regression_pipeline, plot_regression_results
from model.feature_importance import extract_model_importance

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
model_loc = config['model']
file = config['data']['name']

# Load necessary datasets 
df_loc = script_dir.parent / data_loc / file[1]
image_save_loc = script_dir.parent / image_loc

# Preprocess data
preprocessed_df = preprocess_COE_data(df_loc = df_loc)

# Test and validate for best model
X_train, X_test, y_train, y_test, best_model, best_grid = regression_pipeline(df = preprocessed_df, target_column='premium')

# Snippet of performance for best model
plot_regression_results(best_model, X_train, y_train, X_test, y_test, image_save_loc=image_save_loc)

# Train best model on ALL of available data
best_model.fit(preprocessed_df.drop(columns=['premium']), preprocessed_df['premium'])

# Save the model, features and parameters for deployment
model_name = 'best_regressor.pkl'
model_final_loc = script_dir.parent / model_loc / model_name
joblib.dump(best_model, model_final_loc)
print(f"Model retrained on full data and saved for deployment!")

# We use feature importance/model coefficients to understand relationship between permium and quota
feature_names, importances = extract_model_importance(best_model=best_model, X=preprocessed_df.drop(columns=['premium']))