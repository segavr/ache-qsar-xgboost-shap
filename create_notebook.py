import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("# Exploratory Analysis: QSAR for Acetylcholinesterase Inhibitors\n\nThis notebook provides an overview of the dataset, feature engineering, and model interpretability for the AChE QSAR project.\n\n**Author:** Semen Gavrilov"))

# Imports
cells.append(nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom rdkit import Chem\nfrom rdkit.Chem import Draw, Descriptors\nimport joblib\nimport shap\nimport os\n\n%matplotlib inline\nsns.set_theme(style='whitegrid')"))

# Data Loading
cells.append(nbf.v4.new_markdown_cell("## 1. Load Data\nWe'll load the processed training and test sets."))
cells.append(nbf.v4.new_code_cell("train_df = pd.read_csv('data/processed/train.csv')\ntest_df = pd.read_csv('data/processed/test.csv')\nprint(f'Train set size: {len(train_df)}')\nprint(f'Test set size: {len(test_df)}')\ntrain_df.head()"))

# Target Distribution
cells.append(nbf.v4.new_markdown_cell("## 2. Target Distribution (pIC50)\nLet's visualize the distribution of biological activity."))
cells.append(nbf.v4.new_code_cell("plt.figure(figsize=(8, 5))\nsns.histplot(train_df['pIC50'], kde=True, color='skyblue')\nplt.title('Distribution of pIC50 in Training Set')\nplt.xlabel('pIC50')\nplt.ylabel('Frequency')\nplt.show()"))

# Molecule Visualization
cells.append(nbf.v4.new_markdown_cell("## 3. Visualize Sample Molecules\nLet's look at some highly active compounds."))
cells.append(nbf.v4.new_code_cell("top_5 = train_df.nlargest(5, 'pIC50')\nmols = [Chem.MolFromSmiles(s) for s in top_5['canonical_smiles']]\nDraw.MolsToGridImage(mols, legends=[f'pIC50: {p:.2f}' for p in top_5['pIC50']], molsPerRow=5)"))

# Model Interpretability
cells.append(nbf.v4.new_markdown_cell("## 4. Model Interpretability (SHAP)\nWe'll load the trained XGBoost model and explain its predictions."))
cells.append(nbf.v4.new_code_cell("reg_model = joblib.load('models/xgboost_regressor.joblib')\nselected_features = joblib.load('models/selected_features.joblib')\n\nX_test = test_df[selected_features]\nexplainer = shap.TreeExplainer(reg_model)\nshap_values = explainer.shap_values(X_test)\n\n# Summary Plot\nshap.summary_plot(shap_values, X_test)"))

nb['cells'] = cells

with open('exploratory_analysis.ipynb', 'w') as f:
    nbf.write(nb, f)

print("exploratory_analysis.ipynb created.")
