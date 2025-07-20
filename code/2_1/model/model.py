# We use StatsForecast package for modelling, which,
# offers a collection of popular univariate and multivariate time series forecasting models optimized for high performance and scalability.

import yaml
from statsforecast import StatsForecast
from statsforecast.models import (
    HoltWinters,
    HistoricAverage,
    AutoETS,
    AutoARIMA,
    AutoMFLES, 
    Naive
)
from pathlib import Path
from utilsforecast.losses import mse

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

# AutoARIMA, AutoMFLES are models being used for exogenous features
season_length = 1 #Frequency for annual data from: https://robjhyndman.com/hyndsight/seasonal-periods/
cores = 1 #Adjust based on machine

# We pick models which can capture different patterns such as seasonal, ARIMA, tree-based
models = [
    HoltWinters(),
    HistoricAverage(),
    AutoETS(season_length=season_length, model = 'ZZZ'), 
    AutoARIMA(season_length=season_length),
    AutoMFLES(season_length=season_length, test_size = 2)
]

def model_build_and_forecast(Y_train, X_pred, models = models, cores = cores):
    """
    Function to train, validate and forecast model based on input data and regressors
    """

    # Instantiate StatsForecast class as sf
    sf = StatsForecast( 
        models=models,
        freq=1, 
        fallback_model = Naive(),
        n_jobs=cores
    )

    print("Starting cross-validation:")
    # Cross validate for 2 years across multiple models
    cv_df = sf.cross_validation(
    df=Y_train,
    h=1,
    step_size=1,
    n_windows=2
    )

    # Check cross-validation results
    print("Sample of cross-validation results for our models:")
    print(cv_df.head())

    # Automatically evaluate the best model based on Mean Squared Error (mse) in validation window
    models = cv_df.columns.drop(['unique_id', 'ds', 'y', 'cutoff']).tolist()
    evals = mse(cv_df, models=models)
    evals['best_model'] = evals[models].idxmin(axis=1)
    
    print("Let's check how evaluation of best model has performed:")
    print(evals.head())

    # Finally, let's check which models got selected
    print("Frequency of models used: ")
    print(evals['best_model'].value_counts().to_frame().reset_index())

    print("Building forecasts for 5 years:")
    # Product forecasts and select the best model based on CV
    forecasts_df = sf.forecast(df=Y_train, X_df=X_pred, h=5, level=[90])
    #forecasts_df.head()
    plot_save_file_name = "model_result_plot.png"
    plot_path = script_dir.parent / image_loc / plot_save_file_name
    sf.plot(Y_train,forecasts_df).savefig(plot_path)

    with_best = forecasts_df.merge(evals[['unique_id', 'best_model']])
    res = with_best[['unique_id', 'ds']].copy()
    for suffix in ('', '-lo-90', '-hi-90'):
        res[f'best_model{suffix}'] = with_best.apply(lambda row: row[row['best_model'] + suffix], axis=1)
    
    return res

