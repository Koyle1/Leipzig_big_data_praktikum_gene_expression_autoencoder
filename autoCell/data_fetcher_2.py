import cellxgene_census
import scanpy.pp as pp

import anndata
import pandas as pd


with cellxgene_census.open_soma() as census:

    obs_df = census["census_data"]["homo_sapiens"].obs.read(
        value_filter = "tissue_general == 'lung' and disease in ['normal','lung adenocarcinoma', 'squamous cell lung carcinoma', 'small cell lung carcinoma', 'non-small cell lung carcinoma', 'pleomorphic carcinoma', 'lung large cell carcinoma'] and is_primary_data == True",
    ).concat().to_pandas() # Used for data exploration

    unique_values = obs_df["disease"].unique()
    for value in unique_values:
        print(value)

    columns = obs_df.columns
    for column in columns:
        print(column)

    df_sample = obs_df.sample(n=50_000, random_state=42)
    ids = df_sample['soma_joinid']
    ids_list = ids.tolist()
    ids_str = "[" + ",".join(str(i) for i in ids_list) + "]"

    print("indices sampled, starting download...")

    adata = cellxgene_census.get_anndata(
        census = census,
        organism = "Homo sapiens",
        obs_value_filter = f"soma_joinid in {ids_str}",
        obs_column_names=["tissue", "disease"],
    )

    print("download finished")
    # pp.filter_genes_dispersion(adata)
    # pp.highly_variable_genes(adata, n_top_genes=50, subset=True)
    
    adata.write("data.h5ad")