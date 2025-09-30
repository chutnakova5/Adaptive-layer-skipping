import torch
import torch.nn as nn

DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

### simple 1-layer models ###

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000, patience=20, min_delta=1e-5):
        self.lr = lr
        self.n_iters = n_iters
        self.patience = patience
        self.min_delta = min_delta
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):        
        if X_val is None or y_val is None:
            X_val, y_val = X, y
        
        n_samples, n_features = X.shape
        self.model = nn.Linear(n_features, 1).to(DEV)
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
            y_pred = self.model(X.to(DEV))
            train_loss = criterion(y_pred, y.to(DEV))
            # if i % 10 == 0: print(train_loss.item(), end=' ')
            
            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # Calculate validation loss
            with torch.no_grad():
                val_pred = self.model(X_val.to(DEV))
                val_loss = criterion(val_pred, y_val.to(DEV)).item()
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
        return self.model(X.to(DEV)).detach().cpu().numpy()




class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000, patience=10, min_delta=1e-4):
        self.lr = lr
        self.n_iters = n_iters
        self.patience = patience
        self.min_delta = min_delta
        self.model = None
        
    def fit(self, X, y, X_val=None, y_val=None):
        if X_val is None or y_val is None:
            X_val, y_val = X, y
        
        n_samples, n_features = X.shape
        self.model = nn.Sequential(
            nn.Linear(n_features, 1),
            nn.Sigmoid()
        ).to(DEV)
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
            y_pred = self.model(X.to(DEV))
            train_loss = criterion(torch.sigmoid(y_pred), y.to(DEV))
            
            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # Calculate validation loss
            with torch.no_grad():
                val_pred = self.model(X_val.to(DEV))
                val_loss = criterion(val_pred, y_val.to(DEV)).item()
            
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
        return self.model(X.to(DEV)).detach().cpu().numpy()

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)



### MLP ###
hidden_dim = 32


class MLPRegressor:
    def __init__(self, lr=0.01, n_iters=1000, patience=10, min_delta=1e-4):
        self.lr = lr
        self.n_iters = n_iters
        self.patience = patience
        self.min_delta = min_delta
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):        
        if X_val is None or y_val is None:
            X_val, y_val = X, y
        
        n_samples, n_features = X.shape
        self.model = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        ).to(DEV)
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
            y_pred = self.model(X.to(DEV))
            train_loss = criterion(y_pred, y.to(DEV))
            # if i % 10 == 0: print(train_loss.item(), end=' ')
            
            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # Calculate validation loss
            with torch.no_grad():
                val_pred = self.model(X_val.to(DEV))
                val_loss = criterion(val_pred, y_val.to(DEV)).item()
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
        return self.model(X.to(DEV)).detach().cpu().numpy()




class MLPClassifier:
    def __init__(self, lr=0.01, n_iters=1000, patience=20, min_delta=1e-4):
        self.lr = lr
        self.n_iters = n_iters
        self.patience = patience
        self.min_delta = min_delta
        self.model = None
        
    def fit(self, X, y, X_val=None, y_val=None):        
        if X_val is None or y_val is None:
            X_val, y_val = X, y
            
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
            y_pred = self.model(X.to(DEV))
            train_loss = criterion(torch.sigmoid(y_pred), y.to(DEV))
            # if i % 10 == 0: print(train_loss.item(), end=' ')            
            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # Calculate validation loss
            with torch.no_grad():
                val_pred = self.model(X_val.to(DEV))
                val_loss = criterion(val_pred, y_val.to(DEV)).item()
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
        return self.model(X.to(DEV)).detach().cpu().numpy()

    def predict(self, X):
        proba = self.predict_proba(X.to(DEV))
        return (proba >= 0.5).astype(int)