import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
import os

def generate_features(df):
    print("Generating RDKit 2D descriptors...")
    # Get all descriptor functions
    desc_list = Descriptors._descList
    
    def calc_descriptors(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return [None] * len(desc_list)
        results = []
        for name, func in desc_list:
            try:
                results.append(func(mol))
            except:
                results.append(None)
        return results
    
    desc_names = [d[0] for d in desc_list]
    desc_data = df['canonical_smiles'].apply(calc_descriptors).tolist()
    desc_df = pd.DataFrame(desc_data, columns=desc_names)
    
    print("Generating Morgan Fingerprints (ECFP4)...")
    def calc_morgan(smiles, radius=2, nBits=2048):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return [0] * nBits
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return list(fp)
    
    fp_data = df['canonical_smiles'].apply(calc_morgan).tolist()
    fp_df = pd.DataFrame(fp_data, columns=[f'Bit_{i}' for i in range(2048)])
    
    return pd.concat([df.reset_index(drop=True), desc_df, fp_df], axis=1)

def scaffold_split(df, train_size=0.8):
    print("Performing scaffold split...")
    scaffolds = {}
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['canonical_smiles'])
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
        else:
            scaffold_smiles = ''
            
        if scaffold_smiles not in scaffolds:
            scaffolds[scaffold_smiles] = []
        scaffolds[scaffold_smiles].append(idx)
    
    # Sort scaffolds by size
    sorted_scaffolds = sorted(scaffolds.values(), key=len, reverse=True)
    
    train_indices = []
    test_indices = []
    n_train = int(len(df) * train_size)
    
    for scaffold_set in sorted_scaffolds:
        if len(train_indices) + len(scaffold_set) <= n_train:
            train_indices.extend(scaffold_set)
        else:
            test_indices.extend(scaffold_set)
            
    return df.iloc[train_indices], df.iloc[test_indices]

def main():
    if not os.path.exists('data/curated_data.csv'):
        print("Curated data not found!")
        return
    
    df = pd.read_csv('data/curated_data.csv')
    print(f"Loaded {len(df)} compounds.")
    
    # Generate features
    df_features = generate_features(df)
    
    # Drop columns with NaNs (if any)
    initial_cols = len(df_features.columns)
    df_features = df_features.dropna(axis=1)
    print(f"Features generated. Dropped {initial_cols - len(df_features.columns)} columns with NaNs.")
    print(f"Total columns: {len(df_features.columns)}")
    
    # Split data
    train_df, test_df = scaffold_split(df_features)
    print(f"Train set: {len(train_df)}, Test set: {len(test_df)}")
    
    # Save
    os.makedirs('data/processed', exist_ok=True)
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    print("Data saved to data/processed/")

if __name__ == "__main__":
    main()
