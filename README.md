# A Variational Autoencoder for Single Cell Transcriptomics in the CELLxGENE Dataset

This repository contains code and artifacts for the "Big Data Praktikum" course at Leipzig University. Specifically, we implement a variational autoencoder (VAE) for single cell transcriptomics data in the CELLxGENE dataset[^1]. Our VAE is based on the paper and implementation by Wang et al (2018)[^2]

[^1]: https://cellxgene.cziscience.com/datasets
[^2]: https://academic.oup.com/gpb/article/16/5/320/7225045

# Setup

Setup conda env with:
    - (conda init) #if you have never used conda before
    - conda env create -f env.yaml
    - conda activate cell-vae
