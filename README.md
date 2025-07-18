This repository contains the code for the lecutre "Big data praktikum SS25". 
The objective was to train a varational autoencoder on cellxgene data and find meaningful latent representation of the gene data
by using data preparation methods and hyperparameter tuning.

The code is specifically written for the Leipzig Sc Cluster. The recommended node configurations are already included in the .job files.

To set up:

1) Conda
-Make sure conda is installed
    -> open the terminal
    -> move to the directory where this file is contained
    -> run "module load Anaconda3"
    -> close the terminal & reopen it afterwards for Anaconda to take effect 

-Setup the conda enviorment for this project
    -> open the terminal 
    -> move to the directory where this file is contained
    -> run "conda env create -n environment.yaml" (This will set up a conda env named "cell-vae" containing all necessary packages)
    -> run "conda activate cell-vae" to activate the conda enviorment

2) Get the data
-To fetch the data we used in our experiment
    -> open the terminal
    -> move to the directory where this file is contained
    -> run "sbatch job_files/fetch_data.job"
    -> The execution of the job might take a while (~20-30min)

3) train the model
-to train the VAE
    -> open the terminal
    -> move to the directory where this file is contained
    -> run "sbatch job_files/train_1GPU.job" or"sbatch job_files/train_1GPU.job" for training with either 1 GPU or 5 GPUs
    -> the logs of the training will be stored in the "logs" folder
    -> after training modle information including model weights will be saved as "model.pth"
    -> The model architecture & loss function can be found in "src/vae.py"
    -> Hyperparameter for dataset & model can be set in "train_vae.py"

4) Demo
- To present our results, we have created a Jupyter Notebook which guides users trough the results of our project and compares them to other dimension reduction techniques 
  like PCA or UMAP
        -> Open the Jupyter Notebook "demo.ipynb"
        -> Click trough the different fields to display the results

5) Notes
Important files:
    -> src/vae.py: contains the model architecture & loss funciton
    -> train_vae.py: Contains the training logic of the model including hyper parameters for the model & dataset
    -> autocell/data_loader.py: Contains the logic for the caching and preprocessing of the dataset. This file is called by "train_vae.py"
    -> demo.ipynb: Contains the results in a interpretable format
    

Authors: Frederick Dammeier & Felix Coy (18.7.2025)
