import cellxgene_census
import scanpy as sc
import numpy as np
import gget

import scanpy.pp as pp # for preprocessing

import argparse

gget.setup("cellxgene")

#help(gget.cellxgene)
def main():
    parser = argparse.ArgumentParser(description="Specify your data")

    #Data Options
    parser.add_argument("--species", type=str, default="Homo sapiens", help="Specify the species of the organism")
    parser.add_argument("--cell_type", type=str, default=None, help="Specify the cell type")
    parser.add_argument("--sex", type=str, default=None, help="Choose the sex of the cell donor")
    parser.add_argument("--disease", type=str, default=None, help="Choose the desease of the target")

    # Preprocessing options
    parser.add_argument("--num_genes", type=int, default=None, help="Top N highly variable genes to keep")
    parser.add_argument("--filter_cells_min_genes", type=int, default=None, help="Minimum genes per cell")
    parser.add_argument("--filter_genes_min_cells", type=int, default=None, help="Minimum cells per gene")
    parser.add_argument("--normalize", action="store_true", help="Apply total-count normalization")
    parser.add_argument("--log1p", action="store_true", help="Log1p transform the data")
    parser.add_argument("--sqrt", action="store_true", help="Square-root transform the data (alternative to log1p)")
    parser.add_argument("--scale", action="store_true", help="Scale to zero mean and unit variance")
    parser.add_argument("--regress_out", nargs="+", default=None, help="Regress out confounding variables (e.g. 'total_counts', 'pct_counts_mt')")
    parser.add_argument("--run_pca", action="store_true", help="Run PCA")
    parser.add_argument("--n_pcs", type=int, default=50, help="Number of PCs to compute (used in PCA)")
    parser.add_argument("--run_neighbors", action="store_true", help="Compute neighbors after PCA")
    parser.add_argument("--subsample", type=float, default=None, help="Fraction of cells to subsample (0 < x < 1)")
    parser.add_argument("--downsample_counts", type=int, default=None, help="Downsample total counts per cell to this value")
    parser.add_argument("--combat", action="store_true", help="Apply batch correction using ComBat (requires 'batch' in adata.obs)")
    parser.add_argument("--magic", action="store_true", help="Apply MAGIC imputation (requires magic package)")

    #Save options
    parser.add_argument("--save_name", type=str, default=None, help="Name of the save file")
    
    args = parser.parse_args()

    #Data Extraction
    adata = gget.cellxgene(
        species = args.species,
        cell_type= args.cell_type,
        disease= args.disease,
        sex = args.sex,
    )

    #Preprocessing
    if args.filter_cells_min_genes:
        pp.filter_cells(adata, min_genes=args.filter_cells_min_genes)
    if args.filter_genes_min_cells:
        pp.filter_genes(adata, min_cells=args.filter_genes_min_cells)
    
    if args.num_genes is not None:
        pp.highly_variable_genes(adata, n_top_genes= args.num_genes, subset=True)

    
    #Save the extracted data

    return 1

if __name__ == "__main__":
    main()
