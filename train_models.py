import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score, balanced_accuracy_score
import optuna
import joblib
import os

def load_data():
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    # Identify feature columns (exclude metadata)
    meta_cols = ['canonical_smiles', 'pIC50', 'active']
    X_train = train_df.drop(columns=meta_cols)
    y_train_reg = train_df['pIC50']
    y_train_clf = train_df['active']
    
    X_test = test_df.drop(columns=meta_cols)
    y_test_reg = test_df['pIC50']
    y_test_clf = test_df['active']
    
    return X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf

def tune_xgboost_reg(X_train, y_train):
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': 42
        }
        model = xgb.XGBRegressor(**param)
        # Simple 3-fold CV for speed
        from sklearn.model_selection import cross_val_score
        score = cross_val_score(model, X_train, y_train, cv=3, scoring='r2').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    return study.best_params

def main():
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = load_data()
    print(f"Data loaded. Features: {X_train.shape[1]}")
    
    # Feature selection: remove low variance
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)
    X_train_sel = selector.fit_transform(X_train)
    X_test_sel = selector.transform(X_test)
    selected_features = X_train.columns[selector.get_support()]
    print(f"Features after variance threshold: {X_train_sel.shape[1]}")
    
    # Hyperparameter tuning
    print("Tuning XGBoost Regressor...")
    best_params = tune_xgboost_reg(X_train_sel, y_train_reg)
    print(f"Best params: {best_params}")
    
    # Train final regression model
    print("Training final XGBoost Regressor...")
    reg_model = xgb.XGBRegressor(**best_params, random_state=42)
    reg_model.fit(X_train_sel, y_train_reg)
    
    # Evaluate regression
    y_pred_reg = reg_model.predict(X_test_sel)
    r2 = r2_score(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    
    print("\nRegression Results (Test Set):")
    print(f"R2: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Train classification model (using same features for simplicity)
    print("\nTraining XGBoost Classifier...")
    clf_model = xgb.XGBClassifier(n_estimators=500, random_state=42)
    clf_model.fit(X_train_sel, y_train_clf)
    
    # Evaluate classification
    y_pred_clf = clf_model.predict(X_test_sel)
    y_prob_clf = clf_model.predict_proba(X_test_sel)[:, 1]
    auc = roc_auc_score(y_test_clf, y_prob_clf)
    acc = balanced_accuracy_score(y_test_clf, y_pred_clf)
    
    print("\nClassification Results (Test Set):")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Balanced Accuracy: {acc:.4f}")
    
    # Save models and metadata
    os.makedirs('models', exist_ok=True)
    joblib.dump(reg_model, 'models/xgboost_regressor.joblib')
    joblib.dump(clf_model, 'models/xgboost_classifier.joblib')
    joblib.dump(selected_features, 'models/selected_features.joblib')
    joblib.dump(selector, 'models/variance_selector.joblib')
    
    # Save results to a text file
    with open('models/results.txt', 'w') as f:
        f.write(f"Regression R2: {r2:.4f}\n")
        f.write(f"Regression RMSE: {rmse:.4f}\n")
        f.write(f"Regression MAE: {mae:.4f}\n")
        f.write(f"Classification ROC-AUC: {auc:.4f}\n")
        f.write(f"Classification Balanced Accuracy: {acc:.4f}\n")

if __name__ == "__main__":
    main()
