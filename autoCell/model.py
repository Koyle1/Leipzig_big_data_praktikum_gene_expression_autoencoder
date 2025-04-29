import torch
import torch.nn as nn
import pytorch_lightning as pl

class GeneAutoencoder(pl.LightningModule):
    def __init__(self):
        super.__init__()
        