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
  - dropout: 0.2
  - batch_size: 32
  - max_norm: 1

Generation Results (More in results folder):
  - Temperature: 0.7
  - Validity Rate: 0.9802
  - Uniqueness Rate: 0.999592
  - Novelty Rate: 0.996938
  - Scaffold Novelty Rate: 0.477648
  - Scaffold Holdout Rate: 0.089406
  - Mean QED: 0.762936
  - Median QED: 0.789328

**atom_lstm_0806_152355**:
Train Loss: 0.6004
Validation Loss: 0.6229
Parameters:
  - Atom Tokenization
  - num_epochs: 50
  - lr: 1e-3
  - embed_dim: 128
  - hidden_dim: 256
  - num_layers: 2
  - dropout: 0.2
  - batch_size: 32
  - max_norm: 1

Generation Results (More in results folder)
  - Temperature: 0.7
  - Validity Rate: 0.9822
  - Uniqueness Rate: 0.999186
  - Novelty Rate: 0.996536
  - Scaffold Novelty Rate: 0.463623
  - Scaffold Holdout Rate: 0.089260
  - Mean QED: 0.764234
  - Median QED: 0.791075

To try:
- Learning Rate decay
- bigger hidden_dim
- bigger layers?
- atom level tokenization