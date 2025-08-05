import torch
import torch.nn as nn


import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # ahora input_size=1 porque cada paso recibe un valor
        self.blstm = nn.LSTM(
            input_size=1,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(256*2, 1)
        self.lrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [B, 1, 512] → [B, 512, 1]
        x = x.permute(0, 2, 1)
        out, _ = self.blstm(x)          # [B, 512, 512]
        out = self.fc(out)              # [B, 512,   1]
        out = self.lrelu(out)
        out = self.dropout(out)
        # de vuelta a [B, 1, 512]
        return out.permute(0, 2, 1)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=8,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 128, 256),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
