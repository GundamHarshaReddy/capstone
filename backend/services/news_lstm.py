import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np

class NewsLSTMService:
    def __init__(self, model_path="models/news_lstm.pt"):
        self.model_path = model_path
        self.model = None
        self._is_loaded = False

    def _ensure_loaded(self):
        if self._is_loaded:
            return
        try:
            import torch
            import torch.nn as nn

            class NewsSentimentLSTM(nn.Module):
                def __init__(self, input_dim=6, hidden_dim=32, num_layers=2, dropout=0.2):
                    super(NewsSentimentLSTM, self).__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_dim,
                        hidden_size=hidden_dim,
                        num_layers=num_layers,
                        batch_first=True,
                        dropout=dropout if num_layers > 1 else 0.0
                    )
                    self.dropout = nn.Dropout(dropout)
                    self.fc = nn.Linear(hidden_dim, 1)

                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    last_step = lstm_out[:, -1, :]
                    out = self.dropout(last_step)
                    logits = self.fc(out)
                    return logits

            self.model = NewsSentimentLSTM(input_dim=6, hidden_dim=32, num_layers=2, dropout=0.2)
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
                print("NewsLSTMService: PyTorch LSTM weights loaded successfully.")
            self.model.eval()
            self._is_loaded = True
        except Exception as e:
            print(f"Error loading LSTM model weights: {e}")
            self.model = None
            self._is_loaded = False

    def predict_sequence(self, sequence: np.ndarray) -> dict:
        self._ensure_loaded()
        if sequence.ndim == 2:
            sequence = np.expand_dims(sequence, axis=0)

        # Ensure sequence has 6 features
        if sequence.shape[2] < 6:
            # Pad news_available column if 5 features provided
            padding = np.ones((sequence.shape[0], sequence.shape[1], 6 - sequence.shape[2]), dtype=np.float32)
            sequence = np.concatenate([sequence, padding], axis=2)

        if self.model is None:
            return {
                "news_risk_probability": 0.5,
                "news_risk_score": 50.0,
                "temporal_window_days": sequence.shape[1],
                "model_architecture": "PyTorch 2-layer LSTM (Fallback)"
            }

        import torch
        tensor_seq = torch.tensor(sequence, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(tensor_seq)
            prob = float(torch.sigmoid(logits).item())

        news_risk_score = round(float(np.clip(prob * 100.0, 0.0, 100.0)), 2)

        return {
            "news_risk_probability": round(prob, 4),
            "news_risk_score": news_risk_score,
            "temporal_window_days": sequence.shape[1],
            "model_architecture": "PyTorch 2-layer LSTM (Trained on S&P 500 Sequences)"
        }
