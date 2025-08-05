"""
PyTorch implementation of a 1D convolutional neural network inspired by the
Keras/TensorFlow `Novel_CNN` architecture.  This module defines the model
class, weight initialization, and a simple training loop that demonstrates
how to fit the network on an EEG dataset.  The architecture closely follows
the one described in the prompt: a sequence of Conv1d layers with ReLU
activation, interleaved with average pooling and dropout, followed by a
flatten layer and a dense (fully‑connected) output.

The network is designed to accept inputs of shape ``(batch_size, channels,
sequence_length)``.  It halves the sequence length six times via average
pooling, so lengths divisible by ``2**6 = 64`` (such as 512 or 1024) work
naturally, though other lengths are also possible.  The final dense layer
produces a fixed dimensionality (1024) regardless of the input length.

Example usage:

>>> import torch
>>> from torch.utils.data import DataLoader, TensorDataset
>>> from novel_cnn_pytorch import NovelCNN, train_model
>>> # Dummy data: 100 examples of length 512 with 1 channel
>>> X = torch.randn(100, 1, 512)
>>> y = torch.randn(100, 1024)  # target features for demonstration
>>> dataset = TensorDataset(X, y)
>>> model = NovelCNN(input_channels=1, output_features=1024)
>>> train_model(model, dataset, epochs=5)

This code is ready to be adapted to real EEG/EOG/EMG datasets.  Replace
the dummy data with your actual tensors for ``X_train_EOG`` and
``y_train_EOG``, ``X_train_EMG``, ``y_train_EMG``, etc., and adjust
hyperparameters as needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Iterable, Tuple, Any


class NovelCNN(nn.Module):
    """A 1D CNN inspired by the Keras/TensorFlow Novel_CNN architecture.

    This network stacks seven convolutional blocks.  Each block consists of
    two 1D convolutions with ReLU activations.  Between blocks, the
    temporal dimension is downsampled using ``AvgPool1d`` with a stride of 2,
    effectively halving the sequence length.  Dropout layers are applied to
    deeper blocks to mitigate overfitting.  Finally, the output is flattened
    and passed through a fully connected layer producing a feature vector.

    Args:
        input_channels: Number of input channels.  For EEG/EOG/EMG signals
            recorded as single‑channel time series, this is typically 1.
        output_features: Dimensionality of the final dense layer.  In the
            original Keras model this was 1024.
        kernel_size: Size of the convolutional kernels.  Defaults to 3.
    """

    def __init__(self,
                 input_channels: int = 1,
                 output_features: int = 1024,
                 kernel_size: int = 3) -> None:
        super().__init__()
        # Convolutional layers are defined as lists so they can be iterated
        # during weight initialization.
        self.conv1_1 = nn.Conv1d(input_channels, 32, kernel_size,
                                 padding='same')
        self.conv1_2 = nn.Conv1d(32, 32, kernel_size, padding='same')
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv1d(32, 64, kernel_size, padding='same')
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size, padding='same')
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv1d(64, 128, kernel_size, padding='same')
        self.conv3_2 = nn.Conv1d(128, 128, kernel_size, padding='same')
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv1d(128, 256, kernel_size, padding='same')
        self.conv4_2 = nn.Conv1d(256, 256, kernel_size, padding='same')
        self.dropout4 = nn.Dropout(p=0.5)
        self.pool4 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv1d(256, 512, kernel_size, padding='same')
        self.conv5_2 = nn.Conv1d(512, 512, kernel_size, padding='same')
        self.dropout5 = nn.Dropout(p=0.5)
        self.pool5 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.conv6_1 = nn.Conv1d(512, 1024, kernel_size, padding='same')
        self.conv6_2 = nn.Conv1d(1024, 1024, kernel_size, padding='same')
        self.dropout6 = nn.Dropout(p=0.5)
        self.pool6 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.conv7_1 = nn.Conv1d(1024, 2048, kernel_size, padding='same')
        self.conv7_2 = nn.Conv1d(2048, 2048, kernel_size, padding='same')
        self.dropout7 = nn.Dropout(p=0.5)

        # The output size after flatten depends on input length.  We don't
        # calculate it here; instead we infer it during the first forward
        # pass using ``self._output_dim``.
        self._output_dim: Optional[int] = None
        self.fc = nn.Linear(1, output_features)  # placeholder shape; will be
        # reset in forward if needed

        # Initialize weights with Kaiming normal initialization, similar to
        # ``kernel_initializer = 'he_normal'`` in Keras.
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        # Block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        # Block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(x)
        # Block 4
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = self.dropout4(x)
        x = self.pool4(x)
        # Block 5
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = self.dropout5(x)
        x = self.pool5(x)
        # Block 6
        x = F.relu(self.conv6_1(x))
        x = F.relu(self.conv6_2(x))
        x = self.dropout6(x)
        x = self.pool6(x)
        # Block 7
        x = F.relu(self.conv7_1(x))
        x = F.relu(self.conv7_2(x))
        x = self.dropout7(x)
        # Flatten
        x = torch.flatten(x, start_dim=1)
        # Infer and adjust fc layer on first forward
        # Infer and adjust fc layer on first forward, and put it on the proper device
        if self._output_dim is None:
            self._output_dim = x.size(1)
            # build a new fc with the right input dim
            new_fc = nn.Linear(self._output_dim, self.fc.out_features)
            # initialize its weights
            nn.init.kaiming_normal_(new_fc.weight, nonlinearity='linear')
            if new_fc.bias is not None:
                nn.init.zeros_(new_fc.bias)
            # move it to the same device as x
            self.fc = new_fc.to(x.device)

        # Dense output
        x = self.fc(x)
        return x


def train_model(model: nn.Module,
                dataset: Dataset,
                batch_size: int = 64,
                epochs: int = 10,
                lr: float = 1e-3,
                device: Optional[str] = None,
                validation_split: float = 0.1) -> Tuple[nn.Module, float]:
    """Train the given model on the provided dataset.

    This helper function demonstrates how to train ``NovelCNN`` on
    supervised data.  It takes a dataset of input/target pairs and
    performs a simple training loop using mean squared error loss.  A
    portion of the data can be held out for validation.  Training
    statistics (loss and optional validation loss) are printed at each
    epoch.

    Args:
        model: The neural network to train.
        dataset: A ``torch.utils.data.Dataset`` yielding tuples
            ``(X, y)`` where ``X`` is the input tensor with shape
            ``(channels, length)`` and ``y`` is the target tensor.  For
            regression or autoencoder tasks ``y`` could have the same
            dimensionality as the model's output; for classification it
            could be a class label vector.
        batch_size: Batch size for training.
        epochs: Number of epochs to train.
        lr: Learning rate for the Adam optimizer.
        device: The device to run on (``'cpu'``, ``'cuda'``, etc.).  If
            None, defaults to CUDA if available.
        validation_split: Fraction of the dataset to reserve for
            validation.  Set to 0.0 to disable.

    Returns:
        A tuple ``(model, best_val_loss)``.  If no validation is used,
        ``best_val_loss`` will be ``float('inf')``.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Split dataset into training and validation sets if requested.
    if validation_split > 0.0:
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size])
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False)
    else:
        train_dataset = dataset
        val_loader = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        avg_loss = running_loss / len(train_loader.dataset)

        val_loss = float('inf')
        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.to(device)
                    y_val = y_val.to(device)
                    outputs = model(X_val)
                    loss = criterion(outputs, y_val)
                    val_running_loss += loss.item() * X_val.size(0)
            val_loss = val_running_loss / len(val_loader.dataset)
            best_val_loss = min(best_val_loss, val_loss)
            print(f"Epoch {epoch}/{epochs} - train loss: {avg_loss:.6f}, "
                  f"val loss: {val_loss:.6f}")
        else:
            print(f"Epoch {epoch}/{epochs} - train loss: {avg_loss:.6f}")

    return model, best_val_loss


if __name__ == '__main__':
    # Example demonstration when run as a script.  This section will not
    # execute during import.  If actual EEG/EOG/EMG tensors are available
    # under the paths described in the prompt (e.g. loaded by your own
    # script), you can replace the dummy data here with the real data.
    import argparse
    parser = argparse.ArgumentParser(
        description="Train NovelCNN on dummy data.")
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='Batch size')
    parser.add_argument('--device', type=str,
                        default=None, help='Device to use')
    args = parser.parse_args()

    # Create dummy dataset: 500 samples of length 512, with single channel
    # Input shape: (batch_size, channels, length)
    X_dummy = torch.randn(500, 1, 512)
    # Target shape matches dense output: 1024 features
    y_dummy = torch.randn(500, 1024)
    dummy_dataset = torch.utils.data.TensorDataset(X_dummy, y_dummy)

    model = NovelCNN(input_channels=1, output_features=1024)
    train_model(model, dummy_dataset, batch_size=args.batch_size,
                epochs=args.epochs, device=args.device or None)
