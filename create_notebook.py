import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title and Intro
cells.append(nbf.v4.new_markdown_cell("# Exploratory Analysis: QSAR for Acetylcholinesterase Inhibitors\n\nThis notebook provides an overview of the dataset, feature engineering, and model interpretability for the AChE QSAR project. It is part of a human-AI collaboration experiment between **Semen Gavrilov** and **Manus AI**.\n\n**Author:** Semen Gavrilov\n**Date:** 2026"))

# Imports
cells.append(nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom rdkit import Chem\nfrom rdkit.Chem import Draw, Descriptors\nimport joblib\nimport shap\nimport os\n\n%matplotlib inline\nsns.set_theme(style='whitegrid')"))

# Data Overview
cells.append(nbf.v4.new_markdown_cell("## 1. Data Overview\nWe load the curated dataset from ChEMBL for Human Acetylcholinesterase (CHEMBL220)."))
cells.append(nbf.v4.new_code_cell("train_df = pd.read_csv('data/processed/train.csv')\ntest_df = pd.read_csv('data/processed/test.csv')\nprint(f'Training set: {train_df.shape[0]} compounds')\nprint(f'Test set: {test_df.shape[0]} compounds')\ntrain_df.head()"))

# Target Distribution
cells.append(nbf.v4.new_markdown_cell("## 2. Bioactivity Distribution\nThe target variable is **pIC50** ($-\\log_{10}[\\text{IC50 M}]$). A higher pIC50 indicates higher potency."))
cells.append(nbf.v4.new_code_cell("plt.figure(figsize=(8, 5))\nsns.histplot(train_df['pIC50'], kde=True, color='skyblue')\nplt.title('Distribution of pIC50 (Training Set)')\nplt.xlabel('pIC50')\nplt.ylabel('Frequency')\nplt.show()"))

# Chemical Space PCA
cells.append(nbf.v4.new_markdown_cell("## 3. Chemical Space Visualization\nUsing Principal Component Analysis (PCA) on the molecular descriptors to visualize the distribution of compounds."))
cells.append(nbf.v4.new_code_cell("from sklearn.decomposition import PCA\n\nmeta_cols = ['canonical_smiles', 'pIC50', 'active']\nX_train = train_df.drop(columns=meta_cols)\ny_train = train_df['pIC50']\n\npca = PCA(n_components=2)\nX_pca = pca.fit_transform(X_train)\n\nplt.figure(figsize=(10, 8))\nplt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis', alpha=0.6)\nplt.colorbar(label='pIC50')\nplt.xlabel('PCA 1')\nplt.ylabel('PCA 2')\nplt.title('Chemical Space Visualization (PCA)')\nplt.show()"))

# SHAP Summary
cells.append(nbf.v4.new_markdown_cell("## 4. Model Interpretability (SHAP)\nWe use SHAP to understand which features drive the XGBoost model's predictions."))
cells.append(nbf.v4.new_code_cell("reg_model = joblib.load('models/xgboost_regressor.joblib')\nselected_features = joblib.load('models/selected_features.joblib')\nX_test = test_df[selected_features]\n\nexplainer = shap.TreeExplainer(reg_model)\nshap_values = explainer.shap_values(X_test)\n\nplt.figure(figsize=(10, 8))\nshap.summary_plot(shap_values, X_test, plot_type='bar', show=False)\nplt.title('Top Features by SHAP Importance')\nplt.show()"))

# Sample Molecules
cells.append(nbf.v4.new_markdown_cell("## 5. Sample Active Molecules\nVisualizing some of the most potent inhibitors in the training set."))
cells.append(nbf.v4.new_code_cell("top_mols = train_df.nlargest(8, 'pIC50')\nmols = [Chem.MolFromSmiles(s) for s in top_mols['canonical_smiles']]\nDraw.MolsToGridImage(mols, legends=[f'pIC50: {p:.2f}' for p in top_mols['pIC50']], molsPerRow=4)"))

nb['cells'] = cells

with open('exploratory_analysis.ipynb', 'w') as f:
    nbf.write(nb, f)

print("exploratory_analysis.ipynb created.")
