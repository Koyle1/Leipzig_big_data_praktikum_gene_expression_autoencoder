import torch
from torch.utils.data import Dataset
import anndata as ad
import numpy as np
from typing import Optional, Union, List, Tuple, Dict
import pandas as pd
from pathlib import Path
import scanpy as sc
from scipy import sparse
import warnings


class SingleCellDataset(Dataset):
    """
    PyTorch Dataset for single cell gene expression data with proper preprocessing.
    
    This class implements standard scRNA-seq preprocessing pipeline including:
    - Quality control filtering
    - Normalization strategies
    - Feature selection
    - Outlier detection and removal
    - Proper scaling for deep learning
    
    Args:
        file_path: Path to the .h5ad file
        preprocessing_config: Dictionary with preprocessing parameters
        gene_subset: Optional list of gene names to subset
        cell_subset: Optional list of cell indices or boolean mask
        return_labels: Whether to return labels along with expression data
        label_key: Key in obs for labels (required if return_labels=True)
        dtype: Data type for expression values (default: torch.float32)
        verbose: Whether to print preprocessing steps
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        preprocessing_config: Optional[Dict] = None,
        gene_subset: Optional[List[str]] = None,
        cell_subset: Optional[Union[List[int], np.ndarray]] = None,
        return_labels: bool = False,
        label_key: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        verbose: bool = True
    ):
        self.file_path = Path(file_path)
        self.gene_subset = gene_subset
        self.cell_subset = cell_subset
        self.return_labels = return_labels
        self.label_key = label_key
        self.dtype = dtype
        self.verbose = verbose
        
        # Set default preprocessing configuration
        self.preprocessing_config = self._get_default_preprocessing_config()
        if preprocessing_config:
            self.preprocessing_config.update(preprocessing_config)
        
        if self.return_labels and self.label_key is None:
            raise ValueError("label_key must be specified when return_labels=True")
        
        # Load and preprocess data
        self._load_and_preprocess_data()
        
    def _get_default_preprocessing_config(self) -> Dict:
        """Get default preprocessing configuration."""
        return {
            # Quality control thresholds
            'min_genes_per_cell': 200,
            'max_genes_per_cell': 6000,
            'min_counts_per_cell': 1000,
            'max_counts_per_cell': 30000,
            'min_cells_per_gene': 3,
            'max_mitochondrial_fraction': 0.2,
            'max_ribosomal_fraction': 0.5,
            
            # Outlier detection
            'use_mad_outlier_detection': True,
            'mad_threshold': 3.0,
            
            # Normalization
            'normalization_method': 'cpm',  # 'cpm', 'tpm', 'scran', 'none'
            'target_sum': 1e4,
            'log_transform': True,
            'log_base': 'natural',  # 'natural', '2', '10'
            
            # Feature selection
            'feature_selection': True,
            'n_top_genes': 2000,
            'highly_variable_method': 'seurat_v3',
            
            # Scaling
            'scale_method': 'standard',  # 'standard', 'minmax', 'robust', 'none'
            'clip_values': True,
            'clip_max': 10.0,
            
            # Technical corrections
            'filter_mitochondrial': True,
            'filter_ribosomal': False,
            'remove_doublets': False,  # Requires scrublet
            
            # Advanced options
            'preserve_raw': True,
            'batch_correction': False,
            'batch_key': None,
        }
    
    def _load_and_preprocess_data(self):
        """Load and preprocess the AnnData object."""
        if self.verbose:
            print("Loading data...")
        
        # Load the full dataset
        self.adata = ad.read_h5ad(self.file_path)
        
        if self.verbose:
            print(f"Initial data shape: {self.adata.shape}")
            
        # Store raw data if requested
        if self.preprocessing_config['preserve_raw']:
            self.adata.raw = self.adata.copy()
        
        # Apply preprocessing pipeline
        self._preprocess_pipeline()
        
        # Final subsetting
        self._apply_subsetting()
        
        # Store dimensions
        self.n_cells, self.n_genes = self.adata.shape
        
        # Extract labels if needed
        if self.return_labels:
            self._extract_labels()
            
        if self.verbose:
            print(f"Final data shape: {self.adata.shape}")
            self._print_preprocessing_summary()
    
    def _preprocess_pipeline(self):
        """Apply the full preprocessing pipeline."""
        # 1. Calculate QC metrics
        self._calculate_qc_metrics()
        
        # 2. Filter cells and genes
        self._filter_cells_and_genes()
        
        # 3. Normalize data
        self._normalize_data()
        
        # 4. Feature selection
        if self.preprocessing_config['feature_selection']:
            self._select_highly_variable_genes()
        
        # 5. Scale data
        self._scale_data()
        
        # 6. Final quality checks
        self._final_quality_checks()
    
    def _calculate_qc_metrics(self):
        """Calculate quality control metrics."""
        if self.verbose:
            print("Calculating QC metrics...")
            
        # Basic metrics
        self.adata.var['n_cells'] = np.array((self.adata.X > 0).sum(axis=0)).flatten()
        self.adata.obs['n_genes'] = np.array((self.adata.X > 0).sum(axis=1)).flatten()
        self.adata.obs['total_counts'] = np.array(self.adata.X.sum(axis=1)).flatten()
        
        # Mitochondrial genes
        mt_genes = self.adata.var_names.str.startswith('MT-')
        self.adata.var['mitochondrial'] = mt_genes
        if mt_genes.sum() > 0:
            self.adata.obs['pct_counts_mt'] = (
                np.array(self.adata[:, mt_genes].X.sum(axis=1)).flatten() / 
                self.adata.obs['total_counts'] * 100
            )
        else:
            self.adata.obs['pct_counts_mt'] = 0
            
        # Ribosomal genes
        ribo_genes = self.adata.var_names.str.startswith(('RPS', 'RPL'))
        self.adata.var['ribosomal'] = ribo_genes
        if ribo_genes.sum() > 0:
            self.adata.obs['pct_counts_ribo'] = (
                np.array(self.adata[:, ribo_genes].X.sum(axis=1)).flatten() / 
                self.adata.obs['total_counts'] * 100
            )
        else:
            self.adata.obs['pct_counts_ribo'] = 0
    
    def _filter_cells_and_genes(self):
        """Filter cells and genes based on QC metrics."""
        if self.verbose:
            print("Filtering cells and genes...")
            
        n_cells_before = self.adata.n_obs
        n_genes_before = self.adata.n_vars
        
        # Filter cells
        cell_filter = (
            (self.adata.obs['n_genes'] >= self.preprocessing_config['min_genes_per_cell']) &
            (self.adata.obs['n_genes'] <= self.preprocessing_config['max_genes_per_cell']) &
            (self.adata.obs['total_counts'] >= self.preprocessing_config['min_counts_per_cell']) &
            (self.adata.obs['total_counts'] <= self.preprocessing_config['max_counts_per_cell']) &
            (self.adata.obs['pct_counts_mt'] <= self.preprocessing_config['max_mitochondrial_fraction'] * 100) &
            (self.adata.obs['pct_counts_ribo'] <= self.preprocessing_config['max_ribosomal_fraction'] * 100)
        )
        
        # MAD-based outlier detection
        if self.preprocessing_config['use_mad_outlier_detection']:
            cell_filter = cell_filter & self._detect_outliers_mad()
        
        self.adata = self.adata[cell_filter, :]
        
        # Filter genes
        gene_filter = self.adata.var['n_cells'] >= self.preprocessing_config['min_cells_per_gene']
        
        # Filter mitochondrial genes if requested
        if self.preprocessing_config['filter_mitochondrial']:
            gene_filter = gene_filter & ~self.adata.var['mitochondrial']
            
        # Filter ribosomal genes if requested
        if self.preprocessing_config['filter_ribosomal']:
            gene_filter = gene_filter & ~self.adata.var['ribosomal']
            
        self.adata = self.adata[:, gene_filter]
        
        if self.verbose:
            print(f"Filtered {n_cells_before - self.adata.n_obs} cells "
                  f"({n_cells_before} -> {self.adata.n_obs})")
            print(f"Filtered {n_genes_before - self.adata.n_vars} genes "
                  f"({n_genes_before} -> {self.adata.n_vars})")
    
    def _detect_outliers_mad(self) -> np.ndarray:
        """Detect outliers using Median Absolute Deviation."""
        def is_outlier_mad(data, threshold=3.0):
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            if mad == 0:
                return np.zeros(len(data), dtype=bool)
            modified_z_score = 0.6745 * (data - median) / mad
            return np.abs(modified_z_score) <= threshold
        
        # Check for outliers in total counts and n_genes
        counts_ok = is_outlier_mad(self.adata.obs['total_counts'], 
                                  self.preprocessing_config['mad_threshold'])
        genes_ok = is_outlier_mad(self.adata.obs['n_genes'], 
                                 self.preprocessing_config['mad_threshold'])
        
        return counts_ok & genes_ok
    
    def _normalize_data(self):
        """Normalize the expression data."""
        if self.verbose:
            print(f"Normalizing data using {self.preprocessing_config['normalization_method']}...")
            
        method = self.preprocessing_config['normalization_method']
        
        if method == 'cpm':
            # Counts per million
            sc.pp.normalize_total(self.adata, target_sum=self.preprocessing_config['target_sum'])
        elif method == 'tpm':
            # Transcript per million (requires gene lengths)
            warnings.warn("TPM normalization requires gene lengths. Using CPM instead.")
            sc.pp.normalize_total(self.adata, target_sum=self.preprocessing_config['target_sum'])
        elif method == 'scran':
            # Size factor normalization (requires scran R package)
            warnings.warn("Scran normalization not implemented. Using CPM instead.")
            sc.pp.normalize_total(self.adata, target_sum=self.preprocessing_config['target_sum'])
        elif method == 'none':
            pass
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Log transformation
        if self.preprocessing_config['log_transform']:
            if self.preprocessing_config['log_base'] == 'natural':
                sc.pp.log1p(self.adata)
            elif self.preprocessing_config['log_base'] == '2':
                sc.pp.log1p(self.adata, base=2)
            elif self.preprocessing_config['log_base'] == '10':
                sc.pp.log1p(self.adata, base=10)
    
    def _select_highly_variable_genes(self):
        """Select highly variable genes."""
        if self.verbose:
            print(f"Selecting top {self.preprocessing_config['n_top_genes']} highly variable genes...")
            
        sc.pp.highly_variable_genes(
            self.adata,
            n_top_genes=self.preprocessing_config['n_top_genes'],
            flavor=self.preprocessing_config['highly_variable_method']
        )
        
        # Keep only highly variable genes
        self.adata = self.adata[:, self.adata.var.highly_variable]
    
    def _scale_data(self):
        """Scale the expression data."""
        if self.verbose:
            print(f"Scaling data using {self.preprocessing_config['scale_method']}...")
            
        method = self.preprocessing_config['scale_method']
        
        if method == 'standard':
            sc.pp.scale(self.adata, max_value=self.preprocessing_config['clip_max'])
        elif method == 'minmax':
            # Min-Max scaling to [0, 1]
            X = self.adata.X.copy()
            if sparse.issparse(X):
                X = X.toarray()
            
            X_min = X.min(axis=0)
            X_max = X.max(axis=0)
            X_range = X_max - X_min
            X_range[X_range == 0] = 1  # Avoid division by zero
            
            X = (X - X_min) / X_range
            self.adata.X = X
            
        elif method == 'robust':
            # Robust scaling using median and MAD
            X = self.adata.X.copy()
            if sparse.issparse(X):
                X = X.toarray()
            
            median = np.median(X, axis=0)
            mad = np.median(np.abs(X - median), axis=0)
            mad[mad == 0] = 1  # Avoid division by zero
            
            X = (X - median) / mad
            if self.preprocessing_config['clip_values']:
                X = np.clip(X, -self.preprocessing_config['clip_max'], 
                           self.preprocessing_config['clip_max'])
            self.adata.X = X
            
        elif method == 'none':
            pass
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def _final_quality_checks(self):
        """Perform final quality checks and corrections."""
        # Check for NaN or infinite values
        X = self.adata.X
        if sparse.issparse(X):
            X = X.toarray()
            
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            warnings.warn("Found NaN or infinite values in expression data. Replacing with zeros.")
            X[np.isnan(X) | np.isinf(X)] = 0
            self.adata.X = X
        
        # Final clipping if requested
        if self.preprocessing_config['clip_values'] and self.preprocessing_config['scale_method'] != 'robust':
            if sparse.issparse(self.adata.X):
                self.adata.X = self.adata.X.toarray()
            self.adata.X = np.clip(self.adata.X, -self.preprocessing_config['clip_max'], 
                                  self.preprocessing_config['clip_max'])
    
    def _apply_subsetting(self):
        """Apply cell and gene subsetting."""
        # Subset cells if specified
        if self.cell_subset is not None:
            if isinstance(self.cell_subset, (list, np.ndarray)):
                if len(self.cell_subset) > 0 and isinstance(self.cell_subset[0], bool):
                    self.adata = self.adata[self.cell_subset, :]
                else:
                    self.adata = self.adata[self.cell_subset, :]
        
        # Subset genes if specified
        if self.gene_subset is not None:
            available_genes = set(self.adata.var_names)
            requested_genes = set(self.gene_subset)
            missing_genes = requested_genes - available_genes
            
            if missing_genes and self.verbose:
                print(f"Warning: {len(missing_genes)} genes not found in dataset")
            
            genes_to_keep = list(requested_genes & available_genes)
            if genes_to_keep:
                self.adata = self.adata[:, genes_to_keep]
            else:
                raise ValueError("None of the specified genes found in the dataset")
    
    def _extract_labels(self):
        """Extract and encode labels."""
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
    
    def _print_preprocessing_summary(self):
        """Print summary of preprocessing steps."""
        stats = self.get_expression_stats()
        print("\nPreprocessing Summary:")
        print(f"  Expression stats: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        print(f"  Value range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"  Sparsity: {stats['sparsity']:.3f}")
        print(f"  Cells: {self.n_cells}, Genes: {self.n_genes}")
    
    def __len__(self) -> int:
        """Return the number of cells in the dataset."""
        return self.n_cells
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Get a single cell's data."""
        if idx >= self.n_cells:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.n_cells}")
        
        # Get expression data
        if sparse.issparse(self.adata.X):
            expression = np.array(self.adata.X[idx, :].toarray()).flatten()
        else:
            expression = self.adata.X[idx, :].copy()
            
        expression_tensor = torch.tensor(expression, dtype=self.dtype)
        
        # Return with or without labels
        if self.return_labels:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
            return expression_tensor, label_tensor
        else:
            return expression_tensor
    
    def get_expression_stats(self) -> Dict:
        """Return comprehensive statistics about the expression data."""
        X = self.adata.X
        if sparse.issparse(X):
            X = X.toarray()
            
        stats = {
            'mean': float(np.mean(X)),
            'std': float(np.std(X)),
            'min': float(np.min(X)),
            'max': float(np.max(X)),
            'median': float(np.median(X)),
            'q25': float(np.percentile(X, 25)),
            'q75': float(np.percentile(X, 75)),
            'sparsity': float(np.mean(X == 0)),
            'total_counts_per_cell_mean': float(np.mean(np.sum(X, axis=1))),
            'total_counts_per_cell_std': float(np.std(np.sum(X, axis=1))),
            'total_counts_per_cell_median': float(np.median(np.sum(X, axis=1))),
            'n_cells': self.n_cells,
            'n_genes': self.n_genes
        }
        return stats
    
    def get_gene_names(self) -> List[str]:
        """Return list of gene names in the dataset."""
        return self.adata.var_names.tolist()
    
    def get_cell_metadata(self) -> pd.DataFrame:
        """Return cell metadata (obs) as DataFrame."""
        return self.adata.obs.copy()
    
    def get_gene_metadata(self) -> pd.DataFrame:
        """Return gene metadata (var) as DataFrame."""
        return self.adata.var.copy()
    
    def save_processed_data(self, output_path: Union[str, Path]):
        """Save the processed data to a new .h5ad file."""
        self.adata.write(output_path)
        if self.verbose:
            print(f"Processed data saved to: {output_path}")


def create_dataloader(
    file_path: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True,
    preprocessing_config: Optional[Dict] = None,
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    """Convenience function to create a DataLoader with proper preprocessing."""
    
    # Default preprocessing for VAE training
    if preprocessing_config is None:
        preprocessing_config = {
            'normalization_method': 'cpm',
            'target_sum': 1e4,
            'log_transform': True,
            'feature_selection': True,
            'n_top_genes': 2000,
            'scale_method': 'standard',
            'clip_values': True,
            'clip_max': 10.0,
        }
    
    dataset = SingleCellDataset(
        file_path=file_path,
        preprocessing_config=preprocessing_config,
        **dataset_kwargs
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )


# Example usage for VAE training
if __name__ == "__main__":
    # Configuration specifically for VAE training
    vae_preprocessing_config = {
        # More stringent QC for VAE
        'min_genes_per_cell': 500,
        'max_genes_per_cell': 5000,
        'min_counts_per_cell': 1000,
        'max_counts_per_cell': 25000,
        'min_cells_per_gene': 10,
        'max_mitochondrial_fraction': 0.15,
        
        # Normalization optimized for VAE
        'normalization_method': 'cpm',
        'target_sum': 1e4,
        'log_transform': True,
        'log_base': 'natural',
        
        # Feature selection
        'feature_selection': True,
        'n_top_genes': 2000,
        'highly_variable_method': 'seurat_v3',
        
        # Scaling for neural networks
        'scale_method': 'standard',
        'clip_values': True,
        'clip_max': 10.0,
        
        # Outlier detection
        'use_mad_outlier_detection': True,
        'mad_threshold': 3.0,
    }
    
    # Create dataset
    dataset = SingleCellDataset(
        file_path="lung_data.h5ad",
        preprocessing_config=vae_preprocessing_config,
        return_labels=True,
        label_key="cell_type",
        verbose=True
    )
    
    # Check preprocessing results
    stats = dataset.get_expression_stats()
    print("\nFinal preprocessing statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Create DataLoader
    dataloader = create_dataloader(
        file_path="lung_data.h5ad",
        batch_size=128,
        shuffle=True,
        preprocessing_config=vae_preprocessing_config,
        return_labels=True,
        label_key="cell_type"
    )
    
    # Test the dataloader
    for batch_idx, (expressions, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: expressions {expressions.shape}, labels {labels.shape}")
        print(f"  Expression range: [{expressions.min():.3f}, {expressions.max():.3f}]")
        if batch_idx >= 2:
            break