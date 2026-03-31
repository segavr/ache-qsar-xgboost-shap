import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.SaltRemover import SaltRemover
import os

def process_data():
    if not os.path.exists('data/raw_data.csv'):
        print("Raw data not found!")
        return
    
    df = pd.read_csv('data/raw_data.csv')
    print(f"Initial records: {len(df)}")
    
    # Filter criteria
    print("Applying filters...")
    df = df[df['standard_units'] == 'nM']
    df = df[df['standard_relation'] == '=']
    df = df[df['target_organism'] == 'Homo sapiens']
    
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
            mol = remover.StripMol(mol)
            frags = Chem.GetMolFrags(mol, asMols=True)
            if len(frags) > 1:
                mol = max(frags, key=lambda m: m.GetNumAtoms())
            elif len(frags) == 1:
                mol = frags[0]
            else:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return None

    print("Standardizing structures (this may take a while)...")
    df['canonical_smiles'] = df['canonical_smiles'].apply(standardize)
    df = df.dropna(subset=['canonical_smiles'])
    
    print(f"After standardization: {len(df)}")
    
    # Handle duplicates: average pIC50
    # pIC50 = -log10(IC50_in_molar) = -log10(IC50_nM * 1e-9) = 9 - log10(IC50_nM)
    df['pIC50'] = 9 - np.log10(df['standard_value'])
    
    # Group by SMILES and take mean pIC50
    df_unique = df.groupby('canonical_smiles')['pIC50'].mean().reset_index()
    
    print(f"Final unique compounds: {len(df_unique)}")
    
    # Add binary label
    df_unique['active'] = (df_unique['pIC50'] >= 6.0).astype(int)
    
    # Save
    df_unique.to_csv('data/curated_data.csv', index=False)
    print("Data saved to data/curated_data.csv")
    
    # Summary stats
    print("\nSummary Statistics:")
    print(df_unique['pIC50'].describe())
    print(f"\nActive compounds (pIC50 >= 6): {df_unique['active'].sum()}")
    print(f"Inactive compounds (pIC50 < 6): {len(df_unique) - df_unique['active'].sum()}")

if __name__ == "__main__":
    process_data()
