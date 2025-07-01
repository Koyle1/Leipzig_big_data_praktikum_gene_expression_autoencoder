import torch
from torch.utils.data import Dataset
import anndata as ad
import numpy as np
from typing import Optional, Union, List, Tuple
import pandas as pd
from pathlib import Path
import h5py


class SingleCellDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for single cell gene expression data in AnnData HDF5 format.
    Loads data on-demand to handle large datasets that don't fit in memory.
    
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
        cache_size: Number of cells to cache in memory (0 = no caching, -1 = cache all)
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
        dtype: torch.dtype = torch.float32,
        cache_size: int = 0
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
        self.cache_size = cache_size
        
        if self.return_labels and self.label_key is None:
            raise ValueError("label_key must be specified when return_labels=True")
        
        # Initialize metadata and indices
        self._load_metadata()
        
        # Initialize cache
        self._cache = {}
        self._cache_order = []
        
    def _load_metadata(self):
        """Load only metadata (obs, var) and determine indices, not expression data."""
        # Load minimal data to get metadata
        adata_meta = ad.read_h5ad(self.file_path, backed='r')  # Read-only backed mode
        
        # Store original dimensions
        self.original_n_cells, self.original_n_genes = adata_meta.shape
        
        # Determine cell indices
        if self.cell_subset is not None:
            if isinstance(self.cell_subset, (list, np.ndarray)):
                if len(self.cell_subset) > 0 and isinstance(self.cell_subset[0], bool):
                    # Boolean mask
                    self.cell_indices = np.where(self.cell_subset)[0]
                else:
                    # Integer indices
                    self.cell_indices = np.array(self.cell_subset)
            else:
                raise ValueError("cell_subset must be a list or numpy array")
        else:
            self.cell_indices = np.arange(self.original_n_cells)
            
        # Determine gene indices
        if self.gene_subset is not None:
            available_genes = adata_meta.var_names.tolist()
            gene_indices = []
            missing_genes = []
            
            for gene in self.gene_subset:
                try:
                    idx = available_genes.index(gene)
                    gene_indices.append(idx)
                except ValueError:
                    missing_genes.append(gene)
            
            if missing_genes:
                print(f"Warning: {len(missing_genes)} genes not found: {missing_genes}")
            
            if not gene_indices:
                raise ValueError("None of the specified genes found in the dataset")
                
            self.gene_indices = np.array(gene_indices)
            self.gene_names = [available_genes[i] for i in gene_indices]
        else:
            self.gene_indices = np.arange(self.original_n_genes)
            self.gene_names = adata_meta.var_names.tolist()
        
        # Store final dimensions
        self.n_cells = len(self.cell_indices)
        self.n_genes = len(self.gene_indices)
        
        # Load and subset metadata
        self.obs = adata_meta.obs.iloc[self.cell_indices].copy()
        self.var = adata_meta.var.iloc[self.gene_indices].copy()
        
        # Process labels if needed
        if self.return_labels:
            if self.label_key not in self.obs.columns:
                raise ValueError(f"Label key '{self.label_key}' not found in obs")
            self.labels = self.obs[self.label_key].values
            
            # Convert categorical labels to integers if needed
            if pd.api.types.is_categorical_dtype(self.labels):
                self.label_encoder = {cat: i for i, cat in enumerate(self.labels.categories)}
                self.labels = self.labels.cat.codes.values
            elif self.labels.dtype == object:
                unique_labels = np.unique(self.labels)
                self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
                self.labels = np.array([self.label_encoder[label] for label in self.labels])
        
        # Close the backed AnnData object
        adata_meta.file.close()
        
        print(f"Dataset initialized: {self.n_cells} cells × {self.n_genes} genes")
        print(f"Memory usage: Metadata only (~{self._estimate_metadata_size():.1f} MB)")
        
    def _estimate_metadata_size(self):
        """Estimate memory usage of metadata in MB."""
        obs_size = self.obs.memory_usage(deep=True).sum()
        var_size = self.var.memory_usage(deep=True).sum()
        indices_size = (self.cell_indices.nbytes + self.gene_indices.nbytes)
        if hasattr(self, 'labels'):
            indices_size += self.labels.nbytes
        return (obs_size + var_size + indices_size) / (1024 * 1024)
    
    def _load_cell_expression(self, cell_idx: int) -> np.ndarray:
        """Load expression data for a single cell on-demand."""
        # Check cache first
        if self.cache_size > 0 and cell_idx in self._cache:
            # Move to end of cache order (LRU)
            self._cache_order.remove(cell_idx)
            self._cache_order.append(cell_idx)
            return self._cache[cell_idx]
        
        # Load from file
        original_cell_idx = self.cell_indices[cell_idx]
        
        # Use h5py for efficient single-row access
        with h5py.File(self.file_path, 'r') as f:
            # AnnData stores expression data in /X (dense) or /X/data, /X/indices, /X/indptr (sparse)
            if 'X' in f and isinstance(f['X'], h5py.Dataset):
                # Dense matrix
                expression = f['X'][original_cell_idx, self.gene_indices]
            elif 'X' in f and 'data' in f['X']:
                # Sparse CSR matrix - more complex to extract single row
                expression = self._load_sparse_row(f, original_cell_idx)
            else:
                # Fallback: use anndata (slower but more reliable)
                adata = ad.read_h5ad(self.file_path, backed='r')
                if hasattr(adata.X, 'todense'):
                    expression = np.array(adata.X[original_cell_idx, self.gene_indices].todense()).flatten()
                else:
                    expression = adata.X[original_cell_idx, self.gene_indices].copy()
                adata.file.close()
        
        # Apply preprocessing
        expression = self._preprocess_expression(expression)
        
        # Cache if enabled
        if self.cache_size > 0:
            self._update_cache(cell_idx, expression)
            
        return expression
    
    def _load_sparse_row(self, h5_file: h5py.File, row_idx: int) -> np.ndarray:
        """Load a single row from a sparse CSR matrix stored in HDF5."""
        # Read sparse matrix components
        data = h5_file['X/data'][:]
        indices = h5_file['X/indices'][:]
        indptr = h5_file['X/indptr'][:]
        
        # Extract row
        start_idx = indptr[row_idx]
        end_idx = indptr[row_idx + 1]
        
        row_data = data[start_idx:end_idx]
        row_indices = indices[start_idx:end_idx]
        
        # Create dense row
        expression = np.zeros(self.original_n_genes, dtype=np.float32)
        expression[row_indices] = row_data
        
        # Subset to selected genes
        expression = expression[self.gene_indices]
        
        return expression
    
    def _preprocess_expression(self, expression: np.ndarray) -> np.ndarray:
        """Apply preprocessing to expression data."""
        if self.log_transform:
            expression = np.log1p(expression + 1.0)

        if self.normalize:
            max_expression = np.max(expression)
            if max_expression > 0:
                expression = expression / max_expression * self.scale_factor  
            
        return expression
    
    def _update_cache(self, cell_idx: int, expression: np.ndarray):
        """Update cache with LRU eviction."""
        # Remove oldest if cache is full
        if len(self._cache) >= self.cache_size:
            oldest_idx = self._cache_order.pop(0)
            del self._cache[oldest_idx]
        
        # Add new entry
        self._cache[cell_idx] = expression.copy()
        self._cache_order.append(cell_idx)
    
    def __len__(self) -> int:
        """Return the number of cells in the dataset."""
        return self.n_cells
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Get a single cell's data, loading on-demand.
        
        Args:
            idx: Cell index (in the subsetted dataset)
            
        Returns:
            Expression tensor and optionally labels/metadata
        """
        if idx >= self.n_cells:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.n_cells}")
        
        # Load expression data on-demand
        expression = self._load_cell_expression(idx)
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
                if key in self.obs.columns:
                    value = self.obs.iloc[idx][key]
                    # Convert to tensor based on data type
                    if pd.api.types.is_numeric_dtype(type(value)):
                        obs_data.append(torch.tensor(float(value), dtype=torch.float32))
                    else:
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
        return self.gene_names.copy()
    
    def get_cell_metadata(self) -> pd.DataFrame:
        """Return cell metadata (obs) as DataFrame."""
        return self.obs.copy()
    
    def get_gene_metadata(self) -> pd.DataFrame:
        """Return gene metadata (var) as DataFrame."""
        return self.var.copy()
    
    def preload_cache(self, indices: Optional[List[int]] = None):
        """
        Preload specific cells into cache for faster access.
        
        Args:
            indices: List of cell indices to preload. If None, preloads first cache_size cells.
        """
        if self.cache_size <= 0:
            print("Caching is disabled (cache_size <= 0)")
            return
            
        if indices is None:
            indices = list(range(min(self.cache_size, self.n_cells)))
        
        print(f"Preloading {len(indices)} cells into cache...")
        for idx in indices[:self.cache_size]:
            if idx < self.n_cells:
                self._load_cell_expression(idx)
        print("Preloading complete")
    
    def clear_cache(self):
        """Clear the expression data cache."""
        self._cache.clear()
        self._cache_order.clear()
        print("Cache cleared")
    
    def get_memory_usage(self) -> dict:
        """Return current memory usage statistics."""
        metadata_size = self._estimate_metadata_size()
        cache_size_mb = sum(arr.nbytes for arr in self._cache.values()) / (1024 * 1024)
        
        return {
            'metadata_mb': metadata_size,
            'cache_mb': cache_size_mb,
            'total_mb': metadata_size + cache_size_mb,
            'cached_cells': len(self._cache),
            'cache_limit': self.cache_size
        }


class BatchedSingleCellDataset(SingleCellDataset):
    """
    Extended dataset that can load entire batches at once for better I/O efficiency.
    Useful when using large batch sizes.
    """
    
    def __init__(self, *args, batch_size: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self._batch_cache = {}
        
    def _get_batch_id(self, idx: int) -> int:
        """Get batch ID for a given cell index."""
        return idx // self.batch_size
    
    def _load_batch_expression(self, batch_id: int) -> np.ndarray:
        """Load expression data for an entire batch."""
        if batch_id in self._batch_cache:
            return self._batch_cache[batch_id]
        
        start_idx = batch_id * self.batch_size
        end_idx = min((batch_id + 1) * self.batch_size, self.n_cells)
        
        batch_size_actual = end_idx - start_idx
        batch_expressions = np.zeros((batch_size_actual, self.n_genes), dtype=np.float32)
        
        # Load batch of cells
        original_indices = self.cell_indices[start_idx:end_idx]
        
        with h5py.File(self.file_path, 'r') as f:
            if 'X' in f and isinstance(f['X'], h5py.Dataset):
                # Dense matrix - load multiple rows at once
                batch_data = f['X'][original_indices[:, None], self.gene_indices]
                batch_expressions = batch_data
            else:
                # Fallback to loading individual cells
                for i, cell_idx in enumerate(range(start_idx, end_idx)):
                    batch_expressions[i] = super()._load_cell_expression(cell_idx)
        
        # Apply preprocessing to entire batch
        for i in range(batch_size_actual):
            batch_expressions[i] = self._preprocess_expression(batch_expressions[i])
        
        # Cache the batch
        if len(self._batch_cache) >= 10:  # Limit batch cache size
            # Remove oldest batch
            oldest_batch = min(self._batch_cache.keys())
            del self._batch_cache[oldest_batch]
            
        self._batch_cache[batch_id] = batch_expressions
        return batch_expressions
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Get item using batch-based loading."""
        batch_id = self._get_batch_id(idx)
        batch_expressions = self._load_batch_expression(batch_id)
        
        # Get expression for specific cell within batch
        within_batch_idx = idx % self.batch_size
        expression = batch_expressions[within_batch_idx]
        expression_tensor = torch.tensor(expression, dtype=self.dtype)
        
        # Apply custom transform if provided
        if self.transform:
            expression_tensor = self.transform(expression_tensor)
        
        # Handle labels and metadata same as parent class
        return_items = [expression_tensor]
        
        if self.return_labels:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
            return_items.append(label_tensor)
        
        if self.obs_keys:
            obs_data = []
            for key in self.obs_keys:
                if key in self.obs.columns:
                    value = self.obs.iloc[idx][key]
                    if pd.api.types.is_numeric_dtype(type(value)):
                        obs_data.append(torch.tensor(float(value), dtype=torch.float32))
                    else:
                        obs_data.append(str(value))
            if obs_data:
                return_items.append(obs_data)
        
        if len(return_items) == 1:
            return return_items[0]
        else:
            return tuple(return_items)


# Utility functions
def create_memory_efficient_dataloader(
    file_path: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    use_batched_dataset: bool = False,
    cache_size: int = 0,
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    """
    Create a memory-efficient DataLoader for large single cell datasets.
    
    Args:
        file_path: Path to .h5ad file
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes (0 = single process)
        use_batched_dataset: Whether to use BatchedSingleCellDataset for better I/O
        cache_size: Number of cells to cache (0 = no caching)
        **dataset_kwargs: Additional arguments for dataset
        
    Returns:
        torch.utils.data.DataLoader
    """
    if use_batched_dataset:
        dataset = BatchedSingleCellDataset(
            file_path, 
            batch_size=batch_size,
            cache_size=cache_size,
            **dataset_kwargs
        )
    else:
        dataset = SingleCellDataset(
            file_path,
            cache_size=cache_size,
            **dataset_kwargs
        )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


# Example usage
if __name__ == "__main__":
    # Example 1: Memory-efficient dataset for large data
    dataset = SingleCellDataset(
        file_path="large_dataset.h5ad",  # 50k x 60k dataset
        log_transform=True,
        normalize=True,
        cache_size=1000,  # Cache 1000 cells in memory
        return_labels=True,
        label_key="cell_type"
    )
    
    print("Memory usage:", dataset.get_memory_usage())
    
    # Example 2: Batched dataset for better I/O efficiency
    batched_dataset = BatchedSingleCellDataset(
        file_path="large_dataset.h5ad",
        batch_size=64,
        log_transform=True,
        normalize=True
    )
    
    # Example 3: Create memory-efficient DataLoader
    dataloader = create_memory_efficient_dataloader(
        file_path="large_dataset.h5ad",
        batch_size=64,
        shuffle=True,
        num_workers=4,  # Use multiple workers for I/O
        use_batched_dataset=True,
        cache_size=500,
        return_labels=True,
        label_key="cell_type",
        normalize=True,
        log_transform=True
    )
    
    # Training loop example
    print("Starting training...")
    for epoch in range(3):
        for batch_idx, (expressions, labels) in enumerate(dataloader):
            # Your training code here
            print(f"Epoch {epoch}, Batch {batch_idx}: {expressions.shape}")
            
            if batch_idx >= 5:  # Just show first few batches
                break
        
        # Optionally clear cache between epochs
        if hasattr(dataloader.dataset, 'clear_cache'):
            dataloader.dataset.clear_cache()