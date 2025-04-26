# Project README

## Overview
This project is designed for processing, filtering, and optimizing molecular SMILES strings using machine learning models. It includes tools for SMILES cleaning, latent space encoding/decoding, and optimization.

## Prerequisites
- Python 3.8+
- requirements.txt

## Installation
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    cd hgraph2graph
    pip install .
    cd ..
    ```
2. Unzip this file to the root directory and rename it to "data". Link - https://drive.google.com/file/d/1R24W4YYKVeVc6G-EdGMnFRFMJWJOeo9e/view?usp=sharing.
It contains csv file with antioxidants and their pseudo labling.

### 1. Regression model
Used form our RL algorith to pseudo label smiles
Example of usage in get_smiles.ipynb

### 2. Optimisation algorithm
**file** - gt_converted.py
Variables inside:
**path_to_vocab** - path to vocabulaty of the model VAE
**path_to_data_to_train_MLP** - path to csv file with pseudo labeled smiles

## File Descriptions
- **`data_preparation.py`**: Cleans and filters SMILES strings based on chemical properties.
- **`gt_converted.py`**: Encodes SMILES into latent vectors, trains an MLP, and optimizes latent vectors.
- **`get_smiles.ipynb`**: Predicts target values for SMILES and extracts top candidates.

## Example Workflow
1. Run gt_converted.py
2. Follow the steps in get_smiles.ipynb

## Notes
- Ensure `rdkit` is properly installed for SMILES processing.
- Adjust hyperparameters in `gt_converted.py` for better optimization results.
- Original repo with VAE - https://github.com/wengong-jin/hgraph2graph/tree/master
 
