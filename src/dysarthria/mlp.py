"""
PyTorch port of the classifier head in legacy/mlp_head_keras.py
(Dense 128 relu → Dropout 0.3 → Dense 64 relu → Dropout 0.3 → softmax,
Adam, cross-entropy), wrapped as a scikit-learn estimator so it plugs
into the same evaluation code as the SVM. The original called it a CNN;
it has no convolutions.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class MLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden=(128, 64), dropout=0.3, epochs=50, batch_size=32, lr=1e-3, seed=0):
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

    def _net(self, d_in: int, n_out: int):
        from torch import nn

        layers, prev = [], d_in
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        import torch

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.as_tensor(y_idx, dtype=torch.long)
        self.net_ = self._net(X_t.shape[1], len(self.classes_))
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        self.net_.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(X_t))
            for i in range(0, len(order), self.batch_size):
                b = order[i:i + self.batch_size]
                opt.zero_grad()
                loss_fn(self.net_(X_t[b]), y_t[b]).backward()
                opt.step()
        return self

    def predict(self, X):
        import torch

        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.as_tensor(np.asarray(X), dtype=torch.float32))
        return self.classes_[logits.argmax(dim=1).numpy()]
