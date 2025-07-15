import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import sparse



# Create output directory
os.makedirs("figures", exist_ok=True)

# Load data
adata = sc.read_h5ad("data.h5ad")

# -------------------------------
# Total counts per cell
# -------------------------------
adata.obs['total_counts'] = (
    adata.X.sum(axis=1).A1 if hasattr(adata.X, 'A1') else adata.X.sum(axis=1)
)

plt.figure(figsize=(6, 4))
plt.hist(adata.obs['total_counts'], bins=50, range=(0, np.percentile(adata.obs['total_counts'], 99)))
plt.xlabel('Total counts per cell')
plt.ylabel('Number of cells')
plt.title('Total Counts per Cell (Clipped at 99th percentile)')
plt.tight_layout()
plt.savefig("figures/cell_count_clipped.png")
plt.close()

# Clean up
del adata.obs['total_counts']

# -------------------------------
# Raw expression distribution
# -------------------------------
if hasattr(adata.X, 'todense'):
    all_expr = np.array(adata.X.todense()).flatten()
else:
    all_expr = adata.X.flatten()

plt.figure(figsize=(6,4))
plt.hist(all_expr, bins=100, range=(0, np.percentile(all_expr, 99)))
plt.xlabel('Gene expression (raw counts)')
plt.ylabel('Frequency')
plt.title('Distribution of Raw Gene Expression Values')
plt.tight_layout()
plt.savefig("figures/expression_raw_counts.png")
plt.close()

# Clean up
del all_expr

# -------------------------------
# Sparsity plots (fixed)
# -------------------------------
if sparse.issparse(adata.X):
    nonzeros_per_gene = np.diff(adata.X.indptr)
    zero_gene_pct = 1 - (adata.X.getnnz(axis=0) / adata.n_obs)
    zero_cell_pct = 1 - (adata.X.getnnz(axis=1) / adata.n_vars)
else:
    zero_gene_pct = (adata.X == 0).sum(axis=0) / adata.n_obs
    zero_cell_pct = (adata.X == 0).sum(axis=1) / adata.n_vars

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.hist(zero_gene_pct, bins=50)
plt.xlabel('Fraction of zeros per gene')
plt.ylabel('Number of genes')
plt.title('Sparsity Across Genes')

plt.subplot(1,2,2)
plt.hist(zero_cell_pct, bins=50)
plt.xlabel('Fraction of zeros per cell')
plt.ylabel('Number of cells')
plt.title('Sparsity Across Cells')

plt.tight_layout()
plt.savefig("figures/zeros_per_expression.png")
plt.close()

# Clean up
del zero_gene_pct, zero_cell_pct

# -------------------------------
# Quality control metrics
# -------------------------------
adata.var['mt'] = adata.var_names.str.upper().str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)

sc.pl.violin(adata, ['pct_counts_mt'], jitter=0.4, multi_panel=True, show=False, save="_pct_counts_mt.png")
sc.pl.violin(adata, ['n_genes_by_counts'], jitter=0.4, multi_panel=True, show=False, save="_n_genes_by_counts.png")

# -------------------------------
# Normalize, log1p, HVG
# -------------------------------
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=2000)
sc.pl.highly_variable_genes(adata, show=False, save="_hvg.png")

# Subset to HVGs
adata = adata[:, adata.var['highly_variable']]

# -------------------------------
# PCA & variance
# -------------------------------
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata)
sc.pl.pca_variance_ratio(adata, log=True, show=False, save="_pca_variance.png")

# -------------------------------
# Neighbors + UMAP
# -------------------------------
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=['pct_counts_mt', 'n_genes_by_counts'], show=False, save="_qc_umap.png")