import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import os
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

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
    X_train, X_test, y_train, y_test, train_df, test_df = load_data()
    
    # Apply selection
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]
    
    # SHAP Analysis
    print("Analyzing SHAP values...")
    explainer = shap.TreeExplainer(reg_model)
    shap_values = explainer.shap_values(X_test_sel)
    
    # Get top features
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X_test_sel.columns, vals)), columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    top_features = feature_importance.head(5)['col_name'].tolist()
    print(f"Top 5 SHAP features: {top_features}")

    # 1. SHAP Dependence Plots for top 2 features
    for feat in top_features[:2]:
        print(f"Generating SHAP dependence plot for {feat}...")
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feat, shap_values, X_test_sel, show=False)
        plt.title(f'SHAP Dependence Plot for {feat}')
        plt.tight_layout()
        plt.savefig(f'reports/figures/shap_dependence_{feat}.png')
        plt.close()

    # 2. Example Molecules with Highlighted Substructures (if any Morgan bits are in top features)
    print("Generating molecule visualizations...")
    # Find a highly active molecule
    top_mol_idx = y_test.idxmax()
    smiles = test_df.loc[top_mol_idx, 'canonical_smiles']
    mol = Chem.MolFromSmiles(smiles)
    
    if mol:
        img = Draw.MolToImage(mol, size=(400, 400))
        img.save('reports/figures/example_active_molecule.png')
        print(f"Saved example active molecule: {smiles}")

    # 3. Correlation Heatmap of top features
    print("Generating correlation heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(X_test_sel[top_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Top SHAP Features')
    plt.tight_layout()
    plt.savefig('reports/figures/top_features_correlation.png')
    plt.close()

    print("New visualizations saved to reports/figures/")

if __name__ == "__main__":
    main()
