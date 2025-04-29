from abc import ABC, abstractmethod
import cellxgene_census
import gget
import numpy as np
import pandas as pd
import anndata as ad


class View():

    def __init__(self):
        self.census = cellxgene_census.open_soma()
        gget.setup("cellxgene")
        self.data = None


    def fetch_data(self) -> None:
        '''
            Input:
            Output: dataframe
            Description: Function fetches Data from the cellxgene database
        '''
        cellxgene_census.get_anndata()

    def build_pipeline(self, steps: dict = {}) -> None:
        '''
            For future use -> Define Data Transformation with functions from pipeline steps
        '''
        pass

    def execute_pipeline(self) -> None:
        '''
            For future use -> Execute the pipeline build with build_pipeline
        '''
        pass

    
    @classmethod
    def load_view():
        '''
            Load view from saved data
        '''
        pass

    def save_view() -> None:
        '''
            Save view for future use
        '''
        pass