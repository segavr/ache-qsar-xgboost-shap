import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.SaltRemover import SaltRemover
import os

def curate_data(target_id='CHEMBL220'):
    print(f"Fetching data for target {target_id}...")
    activity = new_client.activity
    res = activity.filter(target_chembl_id=target_id).filter(standard_type="IC50")
    df = pd.DataFrame.from_dict(res)
    
    print(f"Initial records: {len(df)}")
    
    # Filter criteria
    df = df[df['standard_units'] == 'nM']
    df = df[df['standard_relation'] == '=']
    df = df[df['target_organism'] == 'Homo sapiens']
    # Some versions of chembl client might have different column names, let's check and be robust
    # df = df[df['target_type'] == 'SINGLE PROTEIN'] # This might be in target table, not activity
    
    print(f"After basic filtering: {len(df)}")
    
    # Drop missing values
    df = df.dropna(subset=['standard_value', 'canonical_smiles'])
    df['standard_value'] = pd.to_numeric(df['standard_value'])
    
    print(f"After dropping NaNs: {len(df)}")
    
    # Structure standardization
    remover = SaltRemover()
    
    def standardize(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            # Remove salts
            mol = remover.StripMol(mol)
            # Keep largest fragment
            frags = Chem.GetMolFrags(mol, asMols=True)
            if len(frags) > 1:
                mol = max(frags, key=lambda m: m.GetNumAtoms())
            else:
                mol = frags[0]
            # Canonicalize
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return None

    print("Standardizing structures...")
    df['canonical_smiles'] = df['canonical_smiles'].apply(standardize)
    df = df.dropna(subset=['canonical_smiles'])
    
    print(f"After standardization: {len(df)}")
    
    # Handle duplicates: average pIC50
    # First convert to pIC50 to average in log space
    df['pIC50'] = -np.log10(df['standard_value'] * 1e-9)
    
    # Group by SMILES and take mean pIC50
    df_unique = df.groupby('canonical_smiles')['pIC50'].mean().reset_index()
    
    print(f"Final unique compounds: {len(df_unique)}")
    
    # Add binary label
    df_unique['active'] = (df_unique['pIC50'] >= 6.0).astype(int)
    
    # Save
    os.makedirs('data', exist_ok=True)
    df_unique.to_csv('data/curated_data.csv', index=False)
    print("Data saved to data/curated_data.csv")
    
    # Summary stats
    print("\nSummary Statistics:")
    print(df_unique['pIC50'].describe())
    print(f"\nActive compounds (pIC50 >= 6): {df_unique['active'].sum()}")
    print(f"Inactive compounds (pIC50 < 6): {len(df_unique) - df_unique['active'].sum()}")

if __name__ == "__main__":
    curate_data()
