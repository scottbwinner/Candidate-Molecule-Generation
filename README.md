### Models

**char_lstm_0804_160242**:
Train Loss: 0.5287
Validation Loss: 0.5383
Parameters:
  - Char Tokenization
  - num_epochs: 50
  - lr: 1e-3
  - embed_dim: 128
  - hidden_dim: 256
  - num_layers: 2
  - batch_size: 32
  - max_norm: 1

  Generation Results (More in results folder):
    - Temperature: 0.7
    - Validity Rate: 0.9830
    - Uniqueness Rate: 1.0
    - Novelty Rate: .995931
    - Scaffold Novelty Rate: 0.092065