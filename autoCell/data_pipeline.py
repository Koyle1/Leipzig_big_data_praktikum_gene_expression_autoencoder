from sklearn.model_selection import KFold

import scanpy as sc

adata = sc.read_h5ad("data.h5ad")
X = adata.X  # This could be a NumPy array or sparse matrix

X = X.toarray()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    print('success')