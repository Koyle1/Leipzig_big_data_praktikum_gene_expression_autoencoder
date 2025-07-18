# Big Data Praktikum SS25: Cell VAE Repository

This repository contains the code for the lecture **"Big Data Praktikum SS25"**.  
The objective was to train a **variational autoencoder (VAE)** on cellxgene data and find meaningful latent representations of the gene data by using data preparation methods and hyperparameter tuning.

The code is specifically written for the Leipzig SC Cluster. The recommended node configurations are already included in the `.job` files.

---

## Setup Instructions

### 1) Conda

- Make sure conda is installed  
  - Open the terminal  
  - Move to the directory where this file is contained  
  - Run `module load Anaconda3`  
  - Close the terminal & reopen it afterwards for Anaconda to take effect  

- Setup the conda environment for this project  
  - Open the terminal  
  - Move to the directory where this file is contained  
  - Run `conda env create -n cell-vae -f environment.yaml`  
    (This will set up a conda environment named **cell-vae** containing all necessary packages)  
  - Run `conda activate cell-vae` to activate the environment  

---

### 2) Get the Data

- To fetch the data used in the experiment:  
  - Open the terminal  
  - Move to the directory where this file is contained  
  - Run `sbatch jobfiles/fetch_data.job`  
  - The execution of the job might take a while (~20-30 min)  

---

### 3) Train the Model

- To train the VAE:  
  - Open the terminal  
  - Move to the directory where this file is contained  
  - Run `sbatch jobfiles/train_1GPU.job` or `sbatch jobfiles/train_5GPU.job`  
    (for training with either 1 GPU or 5 GPUs)  
  - Logs of the training will be stored in the `logs` folder  
  - After training, model information including weights will be saved as `model.pth`  
  - Model architecture & loss function are in `src/vae.py`  
  - Hyperparameters for dataset & model can be set in `train_vae.py`  

---

### 4) Demo

To present the results, a Jupyter Notebook guides users through the results of the project and compares them to other dimension reduction techniques like PCA or UMAP.  

- Open the Jupyter Notebook `demo.ipynb`  
- Click through the different fields to display the results  

---

### 5) Notes

**Important files:**

- `src/vae.py`: contains the model architecture & loss function  
- `train_vae.py`: contains training logic including hyperparameters for model & dataset  
- `src/dataloader.py`: contains caching and preprocessing logic called by `train_vae.py`  
- `demo.ipynb`: contains the results in an interpretable format  

---

**Authors:** Frederik Dammeier & Felix Coy (18.7.2025)
