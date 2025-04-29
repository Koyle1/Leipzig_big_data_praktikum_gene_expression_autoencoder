from abc import ABC, abstractmethod
import cellxgene_census
import gget
import numpy as np
import pandas as pd
import anndata as ad



class view():

    def __init__(self):
        self.census = cellxgene_census.open_soma()
        gget.setup("cellxgene")


    def fetch_data(self) -> ad.Anndata:
        '''
            Input:
            Output: dataframe
            Description: Function fetches Data from the cellxgene database
        '''
        cellxgene_census.get_anndata()

    def build_pipeline(self) -> None:
        pass

    def execute_pipeline(self) -> None:
        pass


    def load_view() -> view:
        pass

    def save_view() -> None:
        pass