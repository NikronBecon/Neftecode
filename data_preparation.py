import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Fragments
import sascorer

def clean_and_filter_smiles_dataset(df):
    # Канонизация SMILES
    def canonicalize_smiles(smiles_str):
        try:
            mol = Chem.MolFromSmiles(smiles_str, sanitize=True)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        except:
            return None

    df['SMILES'] = df['SMILES'].apply(canonicalize_smiles)
    df = df[df['SMILES'].notna()].drop_duplicates(subset="SMILES")

    # Фильтрация по химическим критериям
    allowed_atoms = {'C', 'H', 'O', 'N', 'P', 'S'}

    def is_valid_molecule(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        if Chem.GetFormalCharge(mol) != 0:
            return False
        if any(atom.GetNumRadicalElectrons() != 0 for atom in mol.GetAtoms()):
            return False
        if Descriptors.MolWt(mol) > 1000:
            return False
        if not {atom.GetSymbol() for atom in mol.GetAtoms()}.issubset(allowed_atoms):
            return False
        if Crippen.MolLogP(mol) <= 1:
            return False
        return True

    df = df[df['SMILES'].apply(is_valid_molecule)].drop_duplicates(subset="SMILES")

    # Содержит ли фенол или ароматическую аминогруппу
    def has_target_group(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Fragments.fr_phenol(mol) > 0 or Fragments.fr_aniline(mol) > 0

    df = df[df['SMILES'].apply(has_target_group)].drop_duplicates(subset="SMILES")

    # Фильтрация по SA score
    def calculate_sa_score(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return sascorer.calculateScore(mol) if mol else None

    df['SA_Score'] = df['SMILES'].apply(calculate_sa_score)
    df = df[df['SA_Score'] < 6].drop(columns=['SA_Score']).drop_duplicates(subset="SMILES").reset_index(drop=True)

    return df
