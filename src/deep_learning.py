from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # The error message in fit gives the setup instruction.
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


if nn is not None:

    class _RegressionNetwork(nn.Module):
        def __init__(self, input_size, hidden_layer_sizes, dropout):
            super().__init__()
            layers = []
            previous_size = input_size
            for hidden_size in hidden_layer_sizes:
                layers.extend(
                    [
                        nn.Linear(previous_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                previous_size = hidden_size
            layers.append(nn.Linear(previous_size, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, features):
            return self.network(features).squeeze(1)


class TorchMLPRegressor(RegressorMixin, BaseEstimator):
    """A scikit-learn compatible PyTorch regressor for tabular game data."""

    def __init__(
        self,
        hidden_layer_sizes=(128, 64),
        dropout=0.20,
        learning_rate=0.001,
        weight_decay=0.0001,
        batch_size=128,
        max_epochs=200,
        patience=15,
        validation_fraction=0.15,
        refit_full=True,
        random_state=42,
        device="auto",
        verbose=False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.refit_full = refit_full
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _select_device(self):
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        # CUDA provides a clear benefit in Colab. CPU is more reliable than MPS
        # for this small tabular network and avoids Apple backend startup costs.
        return torch.device("cpu")

    def _new_model(self, input_size):
        return _RegressionNetwork(
            input_size=input_size,
            hidden_layer_sizes=tuple(self.hidden_layer_sizes),
            dropout=self.dropout,
        ).to(self.device_)

    def _train_epochs(self, model, x_train, y_train, epochs, validation=None, early_stop=False):
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_function = nn.HuberLoss(delta=1.0)
        history = []
        best_loss = np.inf
        best_epoch = 1
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            model.train()
            batch_losses = []
            for batch_features, batch_target in loader:
                batch_features = batch_features.to(self.device_)
                batch_target = batch_target.to(self.device_)
                optimizer.zero_grad()
                prediction = model(batch_features)
                loss = loss_function(prediction, batch_target)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))

            validation_loss = np.nan
            if validation is not None:
                model.eval()
                with torch.no_grad():
                    x_validation, y_validation = validation
                    prediction = model(x_validation.to(self.device_))
                    validation_loss = float(
                        torch.mean(torch.abs(prediction - y_validation.to(self.device_))).cpu()
                    )

            history.append(
                {
                    "epoch": epoch,
                    "training_loss": float(np.mean(batch_losses)),
                    "validation_mae_scaled": validation_loss,
                }
            )

            if validation is not None and validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            elif validation is not None:
                epochs_without_improvement += 1

            if self.verbose and (epoch == 1 or epoch % 20 == 0):
                print(
                    f"Epoch {epoch}: training loss={history[-1]['training_loss']:.4f}, "
                    f"validation MAE={validation_loss:.4f}"
                )
            if early_stop and epochs_without_improvement >= self.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model, history, best_epoch

    def fit(self, X, y):
        if torch is None:
            raise ImportError(
                "PyTorch is required for TorchMLPRegressor. Install the packages in requirements.txt."
            )
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float32)
        if len(X) < 20:
            raise ValueError("At least 20 training records are required for the neural network.")

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self.n_features_in_ = X.shape[1]
        self.device_ = self._select_device()
        if self.device_.type == "cpu":
            # Small tabular batches are faster with one BLAS thread because the
            # overhead from a large thread pool is greater than the matrix work.
            torch.set_num_threads(1)
        self.target_mean_ = float(np.mean(y))
        self.target_scale_ = float(np.std(y)) or 1.0
        scaled_y = (y - self.target_mean_) / self.target_scale_

        validation_size = max(1, int(len(X) * self.validation_fraction))
        split = len(X) - validation_size
        x_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(scaled_y, dtype=torch.float32)

        model = self._new_model(self.n_features_in_)
        model, history, best_epoch = self._train_epochs(
            model,
            x_tensor[:split],
            y_tensor[:split],
            self.max_epochs,
            validation=(x_tensor[split:], y_tensor[split:]),
            early_stop=True,
        )

        self.early_stopping_history_ = history
        self.best_epoch_ = best_epoch
        if self.refit_full:
            torch.manual_seed(self.random_state)
            model = self._new_model(self.n_features_in_)
            model, self.training_history_, _ = self._train_epochs(
                model,
                x_tensor,
                y_tensor,
                best_epoch,
                validation=None,
                early_stop=False,
            )
        else:
            self.training_history_ = history

        self.model_ = model.to("cpu")
        self.device_ = torch.device("cpu")
        return self

    def predict(self, X):
        check_is_fitted(self, ["model_", "target_mean_", "target_scale_"])
        if next(self.model_.parameters()).device.type == "cpu":
            torch.set_num_threads(1)
        X = check_array(X, accept_sparse=False, dtype=np.float32)
        self.model_.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                batch = torch.as_tensor(X[start : start + self.batch_size], dtype=torch.float32)
                predictions.append(self.model_(batch).numpy())
        scaled_prediction = np.concatenate(predictions)
        return self.target_mean_ + self.target_scale_ * scaled_prediction
