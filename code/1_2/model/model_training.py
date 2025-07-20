# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
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

def plot_regression_results(model, X_train, y_train, X_test, y_test, image_save_loc, title_prefix=""):
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

    file_name = 'best_model_plot.png'
    save_file_param = image_save_loc / file_name
    plt.savefig(save_file_param,dpi = 300)
    plt.close()

    return "Best Model Test and Training image generated!"
