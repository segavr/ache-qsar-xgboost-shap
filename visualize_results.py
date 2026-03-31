import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_data():
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    meta_cols = ['canonical_smiles', 'pIC50', 'active']
    X_train = train_df.drop(columns=meta_cols)
    y_train = train_df['pIC50']
    X_test = test_df.drop(columns=meta_cols)
    y_test = test_df['pIC50']
    return X_train, X_test, y_train, y_test, train_df, test_df

def main():
    os.makedirs('reports/figures', exist_ok=True)
    
    # Load models and data
    reg_model = joblib.load('models/xgboost_regressor.joblib')
    selected_features = joblib.load('models/selected_features.joblib')
    selector = joblib.load('models/variance_selector.joblib')
    X_train, X_test, y_train, y_test, train_df, test_df = load_data()
    
    # Apply selection
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]
    
    # 1. Actual vs Predicted Plot
    print("Generating Actual vs Predicted plot...")
    y_pred = reg_model.predict(X_test_sel)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual pIC50')
    plt.ylabel('Predicted pIC50')
    plt.title('Actual vs Predicted pIC50 (Test Set)')
    plt.tight_layout()
    plt.savefig('reports/figures/actual_vs_predicted.png')
    plt.close()
    
    # 2. Residual Plot
    print("Generating Residual plot...")
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted pIC50')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig('reports/figures/residuals.png')
    plt.close()
    
    # 3. SHAP Analysis
    print("Generating SHAP summary plot...")
    explainer = shap.TreeExplainer(reg_model)
    shap_values = explainer.shap_values(X_test_sel)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sel, show=False)
    plt.tight_layout()
    plt.savefig('reports/figures/shap_summary.png')
    plt.close()
    
    # 4. Chemical Space Visualization (PCA)
    print("Generating Chemical Space (PCA) plot...")
    pca = PCA(n_components=2)
    X_all = pd.concat([X_train_sel, X_test_sel])
    y_all = pd.concat([y_train, y_test])
    X_pca = pca.fit_transform(X_all)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_all, cmap='viridis', alpha=0.6)
    plt.colorbar(label='pIC50')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('Chemical Space Visualization (PCA)')
    plt.tight_layout()
    plt.savefig('reports/figures/chemical_space_pca.png')
    plt.close()

    print("Visualizations saved to reports/figures/")

if __name__ == "__main__":
    main()
