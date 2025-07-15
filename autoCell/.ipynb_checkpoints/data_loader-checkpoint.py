import torch
from torch.utils.data import Dataset
import anndata as ad
import numpy as np
from typing import Optional, Union, List, Tuple
import pandas as pd
from pathlib import Path
import scanpy as sc


class SingleCellDataset(Dataset):
    def __init__(
        self,
        file_path: Union[str, Path],
        gene_subset: Optional[List[str]] = None,
        cell_subset: Optional[Union[List[int], np.ndarray]] = None,
        obs_keys: Optional[List[str]] = None,
        var_keys: Optional[List[str]] = None,
        select_n_genes: Optional[int] = None,
        transform: Optional[callable] = None,
        remove_outliers: Optional[List[float]] = None,
        log_transform: bool = False,
        normalize: bool = False,
        scale_factor: float = 10000.0,
        return_labels: bool = False,
        label_key: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        use_vae_preprocessing: bool = False,
    ):
        self.file_path = Path(file_path)
        self.gene_subset = gene_subset
        self.cell_subset = cell_subset
        self.obs_keys = obs_keys or []
        self.var_keys = var_keys or []
        self.transform = transform
        self.normalize = normalize
        self.scale_factor = scale_factor
        self.return_labels = return_labels
        self.label_key = label_key
        self.dtype = dtype
        self.log_transform = log_transform
        self.remove_outliers = remove_outliers
        self.select_n_genes = select_n_genes
        self.use_vae_preprocessing = use_vae_preprocessing

        if self.return_labels and self.label_key is None:
            raise ValueError("label_key must be specified when return_labels=True")

        # Load and preprocess data
        self._load_data()

    def _load_data(self):
        self.adata = ad.read_h5ad(self.file_path)

        # Subset cells
        if self.cell_subset is not None:
            if isinstance(self.cell_subset, (list, np.ndarray)):
                # Boolean mask or list of indices
                self.adata = self.adata[self.cell_subset, :]

        # Subset genes
        if self.gene_subset is not None:
            available_genes = set(self.adata.var_names)
            requested_genes = set(self.gene_subset)
            missing_genes = requested_genes - available_genes
            if missing_genes:
                print(f"Warning: {len(missing_genes)} genes not found in dataset: {missing_genes}")

            genes_to_keep = list(requested_genes & available_genes)
            if genes_to_keep:
                self.adata = self.adata[:, genes_to_keep]
            else:
                raise ValueError("None of the specified genes found in the dataset")

        # VAE or custom preprocessing
        if self.use_vae_preprocessing:
            print("Using VAE-optimized preprocessing...")
            # Preprocessing returns processed numpy array and indices of HVGs
            processed_X, hvg_indices = self.preprocess_for_vae(
                self.adata, n_top_genes=self.select_n_genes or 3000, target_sum=self.scale_factor
            )
            # Subset adata to HVGs and replace X
            self.adata = self.adata[:, hvg_indices].copy()
            self.adata.X = processed_X
            self.n_cells, self.n_genes = self.adata.shape
        else:
            if self.select_n_genes is not None or self.normalize or self.log_transform or self.remove_outliers is not None:
                self._preprocess_expression()
            else:
                self.n_cells, self.n_genes = self.adata.shape

        # Extract labels if requested
        if self.return_labels:
            if self.label_key not in self.adata.obs.columns:
                raise ValueError(f"Label key '{self.label_key}' not found in obs")
            self.labels = self.adata.obs[self.label_key]

            if pd.api.types.is_categorical_dtype(self.labels):
                self.label_encoder = {cat: i for i, cat in enumerate(self.labels.cat.categories)}
                self.labels = self.labels.cat.codes.values
            elif self.labels.dtype == object:
                unique_labels = np.unique(self.labels)
                self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
                self.labels = np.array([self.label_encoder[label] for label in self.labels])
            else:
                # Assume numeric labels
                self.labels = self.labels.values

        print(f"Dataset loaded: {self.adata.n_obs} cells × {self.adata.n_vars} genes")

    def _preprocess_expression(self):
        # HVG selection first
        if self.select_n_genes is not None:
            # Filter genes expressed in at least 1% of cells
            sc.pp.filter_genes(self.adata, min_cells=int(0.01 * self.adata.n_obs))
            
            # Highly variable genes selection (before transform)
            sc.pp.highly_variable_genes(
                self.adata,
                n_top_genes=self.select_n_genes,
                min_mean=0.01,
                max_mean=5,
                flavor='seurat'
            )
            self.adata = self.adata[:, self.adata.var.highly_variable]

        X = self.adata.X.copy()

        if hasattr(X, 'todense'):
            X = np.asarray(X.todense())

        # Outlier removal
        if self.remove_outliers is not None:
            assert len(self.remove_outliers) == 2, "remove_outliers must be a list of [low_quantile, high_quantile]"
            low, high = self.remove_outliers
            for i in range(X.shape[1]):
                lower_q = np.quantile(X[:, i], low)
                upper_q = np.quantile(X[:, i], high)
                X[:, i] = np.clip(X[:, i], lower_q, upper_q)

        # Size factor normalization
        counts_per_cell = X.sum(axis=1, keepdims=True)
        counts_per_cell[counts_per_cell == 0] = 1
        X = X / counts_per_cell * self.scale_factor

        # Log transform
        if self.log_transform:
            X = np.log1p(X)

        # Normalization (Z-score gene-wise)
        if self.normalize:
            means = np.mean(X, axis=0, keepdims=True)
            stds = np.std(X, axis=0, keepdims=True)
            stds[stds == 0] = 1
            X = (X - means) / stds

        # Clip extreme values after log transform
        if self.log_transform:
            upper_clip = np.percentile(X, 99)
            X = np.clip(X, 0, upper_clip)

        self.adata.X = X
        self.n_cells, self.n_genes = self.adata.shape

    @staticmethod
    def preprocess_for_vae(raw_counts, n_top_genes=3000, target_sum=10000):
        """
        Complete preprocessing pipeline optimized for VAE training.
        Returns processed matrix and indices of selected HVGs.
        """
        if hasattr(raw_counts, 'X'):
            X = raw_counts.X.copy()
            if hasattr(X, 'todense'):
                X = np.asarray(X.todense())
        else:
            X = np.array(raw_counts)

        # HVG selection based on variance
        if n_top_genes is not None and n_top_genes < X.shape[1]:
            gene_vars = np.var(X, axis=0)
            top_genes = np.argsort(gene_vars)[-n_top_genes:]
            X = X[:, top_genes]
        else:
            top_genes = np.arange(X.shape[1])

        # Size factor normalization
        counts_per_cell = X.sum(axis=1, keepdims=True)
        counts_per_cell[counts_per_cell == 0] = 1
        X = X / counts_per_cell * target_sum

        # Log1p transform
        X = np.log1p(X)

        # Clip at 99th percentile
        upper_clip = np.percentile(X, 99)
        X = np.clip(X, 0, upper_clip)

        return X.astype(np.float32), top_genes

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple]:
        if idx >= self.n_cells:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.n_cells}")

        # Expression data
        if hasattr(self.adata.X, 'todense'):
            expression = np.array(self.adata.X[idx, :].todense()).flatten()
        else:
            expression = self.adata.X[idx, :].copy()

        expression_tensor = torch.tensor(expression, dtype=self.dtype)

        if self.transform:
            expression_tensor = self.transform(expression_tensor)

        return_items = [expression_tensor]

        # Labels
        if self.return_labels:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
            return_items.append(label_tensor)

        # Observation keys metadata
        if self.obs_keys:
            obs_data = []
            for key in self.obs_keys:
                if key in self.adata.obs.columns:
                    value = self.adata.obs.iloc[idx][key]
                    # Numeric obs keys to tensor, else string
                    if pd.api.types.is_numeric_dtype(type(value)) or isinstance(value, (int, float, np.number)):
                        obs_data.append(torch.tensor(float(value), dtype=torch.float32))
                    else:
                        # Could add encoding here if needed, but just append string
                        obs_data.append(value)
            if obs_data:
                return_items.append(obs_data)

        if len(return_items) == 1:
            return return_items[0]
        else:
            return tuple(return_items)

    def get_gene_names(self) -> List[str]:
        return self.adata.var_names.tolist()

    def get_cell_metadata(self) -> pd.DataFrame:
        return self.adata.obs.copy()

    def get_gene_metadata(self) -> pd.DataFrame:
        return self.adata.var.copy()

    def get_expression_stats(self) -> dict:
        X = self.adata.X
        if hasattr(X, 'todense'):
            X = X.todense()
        X = np.array(X)

        stats = {
            'mean': float(np.mean(X)),
            'std': float(np.std(X)),
            'min': float(np.min(X)),
            'max': float(np.max(X)),
            'sparsity': float(np.mean(X == 0)),
            'total_counts_per_cell_mean': float(np.mean(np.sum(X, axis=1))),
            'total_counts_per_cell_std': float(np.std(np.sum(X, axis=1))),
        }
        return stats


def create_dataloader(
    file_path: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True,
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    dataset = SingleCellDataset(file_path, **dataset_kwargs)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


if __name__ == "__main__":
    dataset = SingleCellDataset(
        file_path="data.h5ad",
        log_transform=True,
        normalize=True,
    )

    dataset_with_labels = SingleCellDataset(
        file_path="data.h5ad",
        gene_subset=["CD4", "CD8A", "CD19", "CD14"],
        return_labels=True,
        label_key="cell_type",
        log_transform=True,
        normalize=True,
    )

    dataloader = create_dataloader(
        file_path="data.h5ad",
        batch_size=64,
        shuffle=True,
        return_labels=True,
        label_key="cell_type",
        normalize=True,
        log_transform=True,
    )

    for batch_idx, batch in enumerate(dataloader):
        if dataset_with_labels.return_labels:
            expressions, labels = batch
            print(f"Batch {batch_idx}: expressions {expressions.shape}, labels {labels.shape}")
        else:
            expressions = batch
            print(f"Batch {batch_idx}: expressions {expressions.shape}")

        if batch_idx >= 2:
            break
