import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import joblib
import shap
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="AChE QSAR Predictor",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 AChE QSAR Predictor")
st.markdown("""
Predict the bioactivity (pIC50) of small molecules against Human Acetylcholinesterase (AChE) 
using machine learning and explainable AI (SHAP).
""")

# Load models
@st.cache_resource
def load_models():
    reg_model = joblib.load('models/xgboost_regressor.joblib')
    clf_model = joblib.load('models/xgboost_classifier.joblib')
    selected_features = joblib.load('models/selected_features.joblib')
    selector = joblib.load('models/variance_selector.joblib')
    return reg_model, clf_model, selected_features, selector

reg_model, clf_model, selected_features, selector = load_models()

def calculate_features(smiles):
    """Calculate molecular features for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Calculate descriptors
    desc_list = Descriptors._descList
    desc_names = [d[0] for d in desc_list]
    desc_values = []
    for name, func in desc_list:
        try:
            desc_values.append(func(mol))
        except:
            desc_values.append(np.nan)
    
    desc_dict = dict(zip(desc_names, desc_values))
    
    # Calculate Morgan fingerprints
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_dict = {f'Bit_{i}': int(fp[i]) for i in range(2048)}
    
    # Combine
    all_features = {**desc_dict, **fp_dict}
    return pd.DataFrame([all_features])

# Input section
col1, col2 = st.columns([2, 1])

with col1:
    smiles_input = st.text_input(
        "Enter SMILES string:",
        placeholder="e.g., CC(C)Cc1ccc(cc1)C(C)C(O)=O",
        help="Input a SMILES string for a small molecule"
    )

with col2:
    predict_button = st.button("🔮 Predict", use_container_width=True)

# Prediction section
if predict_button and smiles_input:
    # Validate SMILES
    mol = Chem.MolFromSmiles(smiles_input)
    if mol is None:
        st.error("❌ Invalid SMILES string. Please check your input.")
    else:
        # Calculate features
        features_df = calculate_features(smiles_input)
        
        if features_df is None or features_df.isnull().any().any():
            st.error("❌ Could not calculate features for this molecule.")
        else:
            # Select features
            features_sel = features_df[selected_features]
            
            # Predict
            pic50_pred = reg_model.predict(features_sel)[0]
            ic50_pred = 10 ** (9 - pic50_pred)  # Convert back to nM
            active_prob = clf_model.predict_proba(features_sel)[0, 1]
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted pIC50", f"{pic50_pred:.2f}")
            with col2:
                st.metric("Predicted IC50 (nM)", f"{ic50_pred:.2e}")
            with col3:
                st.metric("Active Probability", f"{active_prob*100:.1f}%")
            
            # Activity classification
            if pic50_pred >= 6.0:
                st.info(f"🟢 **Predicted as ACTIVE** (pIC50 ≥ 6.0)")
            else:
                st.warning(f"🟡 **Predicted as INACTIVE** (pIC50 < 6.0)")
            
            # SHAP explanation
            st.subheader("📊 Feature Importance (SHAP)")
            explainer = shap.TreeExplainer(reg_model)
            shap_values = explainer.shap_values(features_sel)
            
            # Create SHAP force plot
            fig, ax = plt.subplots(figsize=(12, 4))
            shap.force_plot(explainer.expected_value, shap_values[0], features_sel.iloc[0], 
                           matplotlib=True, show=False)
            st.pyplot(fig)
            
            # Molecular properties
            st.subheader("🧬 Molecular Properties")
            props = {
                "Molecular Weight": f"{Descriptors.MolWt(mol):.2f}",
                "LogP": f"{Descriptors.MolLogP(mol):.2f}",
                "TPSA": f"{Descriptors.TPSA(mol):.2f}",
                "H-Bond Donors": f"{Descriptors.NumHDonors(mol)}",
                "H-Bond Acceptors": f"{Descriptors.NumHAcceptors(mol)}",
                "Rotatable Bonds": f"{Descriptors.NumRotatableBonds(mol)}",
            }
            props_df = pd.DataFrame(list(props.items()), columns=["Property", "Value"])
            st.table(props_df)

# Information section
st.markdown("---")
st.subheader("ℹ️ About This Tool")
st.markdown("""
This predictor is based on a machine learning model trained on **5,940 compounds** from ChEMBL 
with experimental bioactivity data against human Acetylcholinesterase (AChE).

**Model Performance:**
- Regression R²: 0.517
- Classification ROC-AUC: 0.855
- Balanced Accuracy: 0.777

**Limitations:**
- Predictions are based on 2D molecular descriptors and fingerprints only
- No 3D conformational information is considered
- Activity cliffs and applicability domain issues may affect predictions
- Use predictions as a starting point for experimental validation

**Reference:**
- Target: Human Acetylcholinesterase (ChEMBL ID: CHEMBL220)
- Data Source: ChEMBL Database
""")
