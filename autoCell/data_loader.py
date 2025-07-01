import torch
from torch.utils.data import Dataset
import anndata as ad
import numpy as np
from typing import Optional, Union, List, Tuple
import pandas as pd
from pathlib import Path


class SingleCellDataset(Dataset):
    """
    PyTorch Dataset for single cell gene expression data in AnnData HDF5 format.
    
    Args:
        file_path: Path to the .h5ad file
        gene_subset: Optional list of gene names to subset. If None, uses all genes.
        cell_subset: Optional list of cell indices or boolean mask to subset cells.
        obs_keys: Optional list of observation (cell metadata) keys to include
        var_keys: Optional list of variable (gene metadata) keys to include
        transform: Optional transform function to apply to gene expression data
        log_transform: Whether to apply log1p transformation to expression data
        normalize: Whether to normalize expression data (divide by total counts * scale_factor)
        scale_factor: Scale factor for normalization (default: 10000)
        return_labels: Whether to return labels along with expression data
        label_key: Key in obs for labels (required if return_labels=True)
        dtype: Data type for expression values (default: torch.float32)
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        gene_subset: Optional[List[str]] = None,
        cell_subset: Optional[Union[List[int], np.ndarray]] = None,
        obs_keys: Optional[List[str]] = None,
        var_keys: Optional[List[str]] = None,
        transform: Optional[callable] = None,
        log_transform: bool = False,
        normalize: bool = False,
        scale_factor: float = 10000.0,
        return_labels: bool = False,
        label_key: Optional[str] = None,
        dtype: torch.dtype = torch.float32
    ):
        self.file_path = Path(file_path)
        self.gene_subset = gene_subset
        self.cell_subset = cell_subset
        self.obs_keys = obs_keys or []
        self.var_keys = var_keys or []
        self.transform = transform
        self.log_transform = log_transform
        self.normalize = normalize
        self.scale_factor = scale_factor
        self.return_labels = return_labels
        self.label_key = label_key
        self.dtype = dtype
        
        if self.return_labels and self.label_key is None:
            raise ValueError("label_key must be specified when return_labels=True")
        
        # Load and preprocess data
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess the AnnData object."""
        # Load the full dataset
        self.adata = ad.read_h5ad(self.file_path)
        
        # Subset cells if specified
        if self.cell_subset is not None:
            if isinstance(self.cell_subset, (list, np.ndarray)):
                if len(self.cell_subset) > 0 and isinstance(self.cell_subset[0], bool):
                    # Boolean mask
                    self.adata = self.adata[self.cell_subset, :]
                else:
                    # Integer indices
                    self.adata = self.adata[self.cell_subset, :]
        
        # Subset genes if specified
        if self.gene_subset is not None:
            # Check if genes exist in the dataset
            available_genes = set(self.adata.var_names)
            requested_genes = set(self.gene_subset)
            missing_genes = requested_genes - available_genes
            
            if missing_genes:
                print(f"Warning: {len(missing_genes)} genes not found in dataset: {missing_genes}")
            
            # Keep only genes that exist
            genes_to_keep = list(requested_genes & available_genes)
            if genes_to_keep:
                self.adata = self.adata[:, genes_to_keep]
            else:
                raise ValueError("None of the specified genes found in the dataset")
        
        # Store dimensions
        self.n_cells, self.n_genes = self.adata.shape
        
        # Preprocess expression data if needed
        if self.normalize or self.log_transform:
            self._preprocess_expression()
            
        # Extract labels if needed
        if self.return_labels:
            if self.label_key not in self.adata.obs.columns:
                raise ValueError(f"Label key '{self.label_key}' not found in obs")
            self.labels = self.adata.obs[self.label_key].values
            
            # Convert categorical labels to integers if needed
            if pd.api.types.is_categorical_dtype(self.labels):
                self.label_encoder = {cat: i for i, cat in enumerate(self.labels.categories)}
                self.labels = self.labels.cat.codes.values
            elif self.labels.dtype == object:
                unique_labels = np.unique(self.labels)
                self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
                self.labels = np.array([self.label_encoder[label] for label in self.labels])
        
        print(f"Dataset loaded: {self.n_cells} cells × {self.n_genes} genes")
        
    def _preprocess_expression(self):
        """Apply normalization and/or log transformation to expression data."""
        X = self.adata.X.copy()
        
        # Convert to dense if sparse
        if hasattr(X, 'todense'):
            X = X.todense()
        
        if self.log_transform:
            X = np.log1p(X + 1)

        if self.normalize:
            # Normalize by max counts per cell
            max_counts = np.array(X.max(axis=1)).flatten()
            # Avoid division by zero -> not needed when adding one in the log transform
            # if not self.log_transform: 
            max_counts[max_counts == 0] = 1
            X = X / max_counts[:, np.newaxis] * self.scale_factor
            
        # Update the AnnData object
        self.adata.X = X
        
    def __len__(self) -> int:
        """Return the number of cells in the dataset."""
        return self.n_cells
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Get a single cell's data.
        
        Args:
            idx: Cell index
            
        Returns:
            If return_labels=False: torch.Tensor of shape (n_genes,)
            If return_labels=True: Tuple of (expression_tensor, label_tensor)
            Additional obs/var data included if specified
        """
        if idx >= self.n_cells:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.n_cells}")
        
        # Get expression data
        if hasattr(self.adata.X, 'todense'):
            # Sparse matrix
            expression = np.array(self.adata.X[idx, :].todense()).flatten()
        else:
            # Dense matrix
            expression = self.adata.X[idx, :].copy()
            
        expression_tensor = torch.tensor(expression, dtype=self.dtype)
        
        # Apply custom transform if provided
        if self.transform:
            expression_tensor = self.transform(expression_tensor)
        
        # Prepare return values
        return_items = [expression_tensor]
        
        # Add labels if requested
        if self.return_labels:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
            return_items.append(label_tensor)
        
        # Add observation metadata if requested
        if self.obs_keys:
            obs_data = []
            for key in self.obs_keys:
                if key in self.adata.obs.columns:
                    value = self.adata.obs.iloc[idx][key]
                    # Convert to tensor based on data type
                    if pd.api.types.is_numeric_dtype(type(value)):
                        obs_data.append(torch.tensor(float(value), dtype=torch.float32))
                    else:
                        # For categorical or string data, you might want to encode them
                        obs_data.append(str(value))
            if obs_data:
                return_items.append(obs_data)
        
        # Return single tensor or tuple
        if len(return_items) == 1:
            return return_items[0]
        else:
            return tuple(return_items)
    
    def get_gene_names(self) -> List[str]:
        """Return list of gene names in the dataset."""
        return self.adata.var_names.tolist()
    
    def get_cell_metadata(self) -> pd.DataFrame:
        """Return cell metadata (obs) as DataFrame."""
        return self.adata.obs.copy()
    
    def get_gene_metadata(self) -> pd.DataFrame:
        """Return gene metadata (var) as DataFrame."""
        return self.adata.var.copy()
    
    def get_expression_stats(self) -> dict:
        """Return basic statistics about the expression data."""
        X = self.adata.X
        if hasattr(X, 'todense'):
            X = X.todense()
            
        stats = {
            'mean': float(np.mean(X)),
            'std': float(np.std(X)),
            'min': float(np.min(X)),
            'max': float(np.max(X)),
            'sparsity': float(np.mean(X == 0)),
            'total_counts_per_cell_mean': float(np.mean(np.sum(X, axis=1))),
            'total_counts_per_cell_std': float(np.std(np.sum(X, axis=1)))
        }
        return stats


# Example usage and utility functions
def create_dataloader(
    file_path: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True,
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    """
    Convenience function to create a DataLoader for single cell data.
    
    Args:
        file_path: Path to .h5ad file
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        **dataset_kwargs: Additional arguments for SingleCellDataset
        
    Returns:
        torch.utils.data.DataLoader
    """
    dataset = SingleCellDataset(file_path, **dataset_kwargs)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # Set to > 0 for multiprocessing
    )


# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage
    dataset = SingleCellDataset(
        file_path="data.h5ad",
        log_transform=True,
        normalize=True
    )
    
    # Example 2: With labels and gene subset
    dataset_with_labels = SingleCellDataset(
        file_path="data.h5ad",
        gene_subset=["CD4", "CD8A", "CD19", "CD14"],
        return_labels=True,
        label_key="cell_type",
        log_transform=True,
        normalize=True
    )
    
    # Example 3: Create DataLoader
    dataloader = create_dataloader(
        file_path="data.h5ad",
        batch_size=64,
        shuffle=True,
        return_labels=True,
        label_key="cell_type",
        normalize=True,
        log_transform=True
    )
    
    # Iterate through batches
    for batch_idx, (expressions, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: {expressions.shape}, {labels.shape}")
        if batch_idx >= 2:  # Just show first few batches
            break