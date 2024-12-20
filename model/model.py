from dataclasses import dataclass
import json

import torch.nn as nn

@dataclass
class ModelParameters:
    input_dim: int
    encoded_dim: int = 64
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    output_dim: int = 1

    @staticmethod
    def from_json(json_path: str):
        with open(json_path, 'r') as f:
            params_dict = json.load(f)
        return ModelParameters(**params_dict)

class AutoencoderLSTM(nn.Module):
    def __init__(self, params: ModelParameters):

        if type(params.input_dim) == str:
            print("Please Specify the input_dim parameter.")
            return
        
        super(AutoencoderLSTM, self).__init__()
        # Encoder: Dimensionality reduction
        self.encoder = nn.Sequential(
            nn.Linear(params.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, params.encoded_dim),
            nn.ReLU()
        )
        
        # LSTM: Sequential modeling
        self.lstm = nn.LSTM(params.encoded_dim, params.lstm_hidden_dim, params.lstm_num_layers, batch_first=True)
        
        # Fully connected layer for prediction
        self.fc = nn.Linear(params.lstm_hidden_dim, params.output_dim)

        # Optional Decoder for reconstruction (can be skipped)
        self.decoder = nn.Sequential(
            nn.Linear(params.encoded_dim, 128),
            nn.ReLU(),
            nn.Linear(128, params.input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Ensure x has a sequence dimension
        if len(x.size()) == 2:
            x = x.unsqueeze(1)  # Add a sequence dimension if missing
        batch_size, seq_len, feature_dim = x.size()
        
        # Encoder
        x = x.view(batch_size * seq_len, feature_dim)  # Flatten time steps for encoder
        encoded = self.encoder(x)
        encoded = encoded.view(batch_size, seq_len, -1)  # Reshape for LSTM
        
        # LSTM
        lstm_out, _ = self.lstm(encoded)
        lstm_out = lstm_out[:, -1, :]  # Take the last time step
        
        # Prediction
        prediction = self.fc(lstm_out).squeeze(-1)  # Ensure shape is (batch_size)

        # Decoder
        decoded = self.decoder(encoded.view(batch_size * seq_len, -1))
        decoded = decoded.view(batch_size, seq_len, feature_dim).squeeze(1)  # Align with inputs

        return prediction, decoded
