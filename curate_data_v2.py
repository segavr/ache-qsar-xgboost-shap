import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.SaltRemover import SaltRemover
import os
import sys

def curate_data(target_id='CHEMBL220'):
    print(f"Fetching data for target {target_id}...")
    activity = new_client.activity
    # Filter for target and IC50
    res = activity.filter(target_chembl_id=target_id).filter(standard_type="IC50")
    
    # Fetch in chunks to show progress
    print("Downloading data from ChEMBL...")
    all_data = []
    count = 0
    for item in res:
        all_data.append(item)
        count += 1
        if count % 500 == 0:
            print(f"Downloaded {count} records...")
            # Limit for testing if needed, but here we want all
    
    df = pd.DataFrame(all_data)
    print(f"Total records downloaded: {len(df)}")
    
    if len(df) == 0:
        print("No data found!")
        return

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
