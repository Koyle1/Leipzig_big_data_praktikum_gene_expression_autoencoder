import cellxgene_census
import pandas as pd

with cellxgene_census.open_soma() as census:
    human = census["census_data"]["homo_sapiens"]

    # Step 1: Read a small number of cell metadata rows
    obs_df = human["obs"].read(value_filter="tissue_general == 'lung'", column_names=["cell_type"]).concat().to_pandas()
    obs_df = obs_df.head(100)  # Limit to first 100 cells for this example

    # Step 2: Read corresponding expression values
    cell_ids = obs_df.index.to_list()  # These are the cell observation IDs

    # Read some genes (e.g., first 10 genes for brevity)
    var_df = human["var"].read().concat().to_pandas()
    gene_ids = var_df.index[:10].to_list()

    # Now read from the X layer (expression matrix)
    expr_matrix = human["X"]["raw"].read(coords={"obs": cell_ids, "var": gene_ids}).to_pandas()

    print(expr_matrix.head())
