import numpy as np

def extract_model_importance(best_model, X):
    """
    Prints and returns feature coefficients (for Lasso/linear) or feature importances (for tree-based models).
    
    Parameters:
    - best_model: Fitted Pipeline (with preprocessor and final estimator)
    - X: Raw DataFrame of predictors (before preprocessing)
    
    Returns:
    - feature_names: Names after transformation
    - importances: Coefficients or importances
    """
    preprocessor = best_model.named_steps['preprocessor']
    model = best_model.named_steps['model']

    # Get feature names — scikit-learn 1.0+ syntax
    num_features = preprocessor.named_transformers_['num'].named_steps['scaler'].get_feature_names_out(
        X.select_dtypes(include=np.number).columns)
    cat_features = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(
        X.select_dtypes(exclude=np.number).columns)
    feature_names = np.concatenate([num_features, cat_features])

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        label = "Feature Importance"
    elif hasattr(model, "coef_"):
        importances = model.coef_
        label = "Coefficient"
    else:
        raise ValueError("Model does not provide feature_importances_ or coef_.")

    print(f"\n{label} (top 15):")
    sorted_idx = np.argsort(np.abs(importances))[::-1]
    for i in sorted_idx[:15]:
        print(f"{feature_names[i]}: {importances[i]:.4f}")

    return feature_names, importances
