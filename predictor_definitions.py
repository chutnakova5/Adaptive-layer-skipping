import torch
import torch.nn as nn

DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')




class StandardScaler:
    def __init__(self, device=DEV, eps=1e-6):
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.device = device
        self.eps = eps

    def fit(self, X: torch.Tensor):
        if X.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got shape {X.shape}")

        self.mean_ = X.mean(dim=0).to(self.device)
        self.var_ = X.var(dim=0, unbiased=False).to(self.device)
        self.scale_ = (torch.sqrt(self.var_) + self.eps).to(self.device)
        return self

    def transform(self, X: torch.Tensor):
        if X.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got shape {X.shape}")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: torch.Tensor):
        self.fit(X)
        return self.transform(X)

    def to(self, device):
        """Move scaler stats to device."""
        if self.mean_ is not None:
            self.mean_ = self.mean_.to(device)
            self.var_ = self.var_.to(device)
            self.scale_ = self.scale_.to(device)
        self.device = device
        return self




class MinMaxScaler:
    def __init__(self, device=DEV, feature_range=(0, 1), eps=1e-6):
        self.feature_range = feature_range
        self.eps = eps
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.device = device

    def fit(self, X: torch.Tensor):
        if X.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got shape {X.shape}")

        self.data_min_ = X.min(dim=0)[0].to(self.device)
        self.data_max_ = X.max(dim=0)[0].to(self.device)

        data_range = self.data_max_ - self.data_min_
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / (data_range + self.eps).to(self.device)
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        return self

    def transform(self, X: torch.Tensor):
        if self.scale_ is None or self.min_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        return X * 0 + (X - self.data_min_) * self.scale_ + self.min_  # broadcast safely

    def fit_transform(self, X: torch.Tensor):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled: torch.Tensor):
        if self.scale_ is None or self.min_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        return (X_scaled - self.min_) / self.scale_ + self.data_min_

    def to(self, device):
        if self.min_ is not None:
            self.min_ = self.min_.to(device)
            self.scale_ = self.scale_.to(device)
            self.data_min_ = self.data_min_.to(device)
            self.data_max_ = self.data_max_.to(device)
        self.device = device
        return self




### simple 1-layer models ###

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000, patience=20, device=DEV, min_delta=1e-5):
        self.lr = lr
        self.n_iters = n_iters
        self.patience = patience
        self.min_delta = min_delta
        self.model = None
        self.scaler = self.scaler_y = None
        self.device = device

    def fit(self, X, y, X_val=None, y_val=None):        
        if X_val is None or y_val is None:
            X_val, y_val = X, y

        self.scaler = StandardScaler(device=self.device)
        X = self.scaler.fit_transform(X)
        X_val = self.scaler.transform(X_val)

        self.scaler_y = MinMaxScaler(device=self.device)
        y = self.scaler_y.fit_transform(y.reshape(-1, 1))
        self.scaler_y = self.scaler_y.to(DEV)
        
        n_samples, n_features = X.shape
        self.model = nn.Linear(n_features, 1).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        with torch.no_grad():
            self.model.weight.data.normal_(0.0, 0.01)
            self.model.bias.data.zero_()
        
        # Early stopping variables
        best_loss = float('inf')
        counter = 0
        best_model_state = None
        
        for i in range(self.n_iters):
            # Forward pass and training loss
            y_pred = self.model(X)
            train_loss = criterion(y_pred, y)
            # if i % 10 == 0: print(train_loss.item(), end=' ')
            
            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # Calculate validation loss
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred, y_val)
                # if i % 10 == 0: print(val_loss)

            
            # Check for improvement
            if val_loss < best_loss - self.min_delta:
                best_loss = val_loss
                counter = 0
                # Save the best model
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                counter += 1
            
            # Early stopping check
            if counter >= self.patience:
                break
        
        # Load the best model if we found one
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        y_scaled = self.model(X_scaled)
        return self.scaler_y.inverse_transform(y_scaled)

    def to(self, device):
        if self.model is not None:
            self.model.to(device)
        if self.scaler is not None:
            self.scaler.to(device)
        if self.scaler_y is not None:
            self.scaler_y.to(device)
        self.device = device
        return self




class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000, device=DEV, patience=10, min_delta=1e-4):
        self.lr = lr
        self.n_iters = n_iters
        self.patience = patience
        self.min_delta = min_delta
        self.model = None
        self.scaler = None
        self.device = device

        
    def fit(self, X, y, X_val=None, y_val=None):
        if X_val is None or y_val is None:
            X_val, y_val = X, y
        
        self.scaler = StandardScaler(device=self.device)
        X = self.scaler.fit_transform(X)
        X_val = self.scaler.transform(X_val)
        
        n_samples, n_features = X.shape
        self.model = nn.Sequential(
            nn.Linear(n_features, 1),
            nn.Sigmoid()
        ).to(self.device)
        criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        
        with torch.no_grad():
            # Xavier/Glorot initialization for the linear layer
            nn.init.xavier_uniform_(self.model[0].weight)
            nn.init.zeros_(self.model[0].bias)
        
        # Early stopping variables
        best_loss = float('inf')
        counter = 0
        best_model_state = None
        
        for i in range(self.n_iters):
            # Forward pass and training loss
            y_pred = self.model(X)
            train_loss = criterion(torch.sigmoid(y_pred), y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # Calculate validation loss
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred, y_val).item()
            
            # Check for improvement
            if val_loss < best_loss - self.min_delta:
                best_loss = val_loss
                counter = 0
                # Save the best model
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                counter += 1
            
            # Early stopping check
            if counter >= self.patience:
                break
        
        # Load the best model if we found one
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model(X_scaled)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).int()

    def to(self, device):
        if self.model is not None:
            self.model.to(device)
        if self.scaler is not None:
            self.scaler.to(device)
        self.device = device
        return self




### MLP ###
hidden_dim = 32


class MLPRegressor:
    def __init__(self, lr=0.01, n_iters=1000, patience=10, min_delta=1e-4, device=DEV):
        self.lr = lr
        self.n_iters = n_iters
        self.patience = patience
        self.min_delta = min_delta
        self.model = None
        self.scaler = self.scaler_y = None
        self.device = device
        
    def fit(self, X, y, X_val=None, y_val=None):        
        if X_val is None or y_val is None:
            X_val, y_val = X, y
        
        self.scaler = StandardScaler(device=self.device)
        X = self.scaler.fit_transform(X)
        X_val = self.scaler.transform(X_val)

        self.scaler_y = MinMaxScaler(device=self.device)
        y = self.scaler_y.fit_transform(y.reshape(-1, 1))
        
        n_samples, n_features = X.shape
        self.model = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        with torch.no_grad():
            self.model[0].weight.data.normal_(0.0, 0.001)
            self.model[0].bias.data.zero_()
            self.model[2].weight.data.normal_(0.0, 0.001)
            self.model[2].bias.data.zero_()
        
        # Early stopping variables
        best_loss = float('inf')
        counter = 0
        best_model_state = None
        
        for i in range(self.n_iters):
            # Forward pass and training loss
            y_pred = self.model(X)
            train_loss = criterion(y_pred, y)
            # if i % 10 == 0: print(train_loss.item(), end=' ')
            
            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # Calculate validation loss
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred, y_val).item()
                # if i % 10 == 0: print(val_loss)
            
            # Check for improvement
            if val_loss < best_loss - self.min_delta:
                best_loss = val_loss
                counter = 0
                # Save the best model
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                counter += 1
            
            # Early stopping check
            if counter >= self.patience:
                break
        
        # Load the best model if we found one
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        y_scaled = self.model(X_scaled)
        return self.scaler_y.inverse_transform(y_scaled)

    def to(self, device):
        if self.model is not None:
            self.model.to(device)
        if self.scaler is not None:
            self.scaler.to(device)
        if self.scaler_y is not None:
            self.scaler_y.to(device)
        self.device = device
        return self



class MLPClassifier:
    def __init__(self, lr=0.01, n_iters=1000, patience=20, min_delta=1e-4, device=DEV):
        self.lr = lr
        self.n_iters = n_iters
        self.patience = patience
        self.min_delta = min_delta
        self.model = None
        self.scaler = None
        self.device = device
        
    def fit(self, X, y, X_val=None, y_val=None):        
        if X_val is None or y_val is None:
            X_val, y_val = X, y
            
        self.scaler = StandardScaler(device=self.device)
        X = self.scaler.fit_transform(X)
        X_val = self.scaler.transform(X_val)
        
        n_samples, n_features = X.shape
        self.model = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(DEV)
        criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        
        with torch.no_grad():
            nn.init.xavier_uniform_(self.model[0].weight)
            nn.init.zeros_(self.model[0].bias)
            nn.init.xavier_uniform_(self.model[2].weight)
            nn.init.zeros_(self.model[2].bias)
        
        # Early stopping variables
        best_loss = float('inf')
        counter = 0
        best_model_state = None
        
        for i in range(self.n_iters):
            # Forward pass and training loss
            y_pred = self.model(X)
            train_loss = criterion(torch.sigmoid(y_pred), y)
            # if i % 10 == 0: print(train_loss.item(), end=' ')            
            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # Calculate validation loss
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred, y_val).item()
                # if i % 10 == 0: print(val_loss)
            
            # Check for improvement
            if val_loss < best_loss - self.min_delta:
                best_loss = val_loss
                counter = 0
                # Save the best model
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                counter += 1
            
            # Early stopping check
            if counter >= self.patience:
                break
        
        # Load the best model if we found one
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model(X_scaled)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).int()

    def to(self, device):
        if self.model is not None:
            self.model.to(device)
        if self.scaler is not None:
            self.scaler.to(device)
        self.device = device
        return self