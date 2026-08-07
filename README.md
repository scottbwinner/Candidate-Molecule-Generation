## Project Structure

```
Candidate-Molecule-Generation/
├── README.md
├── .gitignore
├── conftest.py
│
├── data/
│   ├── zinc250k.csv
│   └── processed/
│       ├── holdout_scaffolds.json
│       ├── char_tokenized/
│       │   ├── training_tensor.pt
│       │   ├── holdout_tensor.pt
|       |   ├── canonical_training_smiles.json
|       |   ├── train_scaffolds.json
│       │   └── metadata.json
│       └── atom_tokenized/
│           ├── training_tensor.pt
│           ├── holdout_tensor.pt
|           ├── canonical_training_smiles.json
|           ├── train_scaffolds.json
│           └── metadata.json
│
├── notebooks/
│   ├── data_import.ipynb
│   ├── data_exploration.ipynb
│   ├── data_processing.ipynb
│   └── molecule_generation.ipynb
│
├── src/
│   ├── tokenizer.py
│   ├── split.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── sample.py
│   └── evaluate.py
│
├── tests/
│   ├── test_tokenizer.py
│   ├── test_split.py
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_train.py
│
├── models/
│   └── *.th          (saved checkpoints, one per training run)
│
├── logs/
│   └── (TensorBoard run logs, one folder per training run)
│
└── results/
    └── (metrics.csv, QED comparison plots, example generated molecules — per model)

```

### `data/`

| File / folder | Purpose |
|---|---|
| `zinc250k.csv` | Raw dataset: ~250k drug-like molecules as SMILES strings, plus logP/QED/SAS columns. |
| `processed/holdout_scaffolds.json` | Scaffolds deliberately held out of training entirely. Used for the scaffold holdout-retrieval metric. |
| `processed/{char,atom}_tokenized/training_tensor.pt` | Pre-encoded, padded training sequences for a given tokenization strategy. Loaded directly by `train.py`. |
| `processed/{char,atom}_tokenized/holdout_tensor.pt` |  Pre-encoded, padded holdout sequences for a given tokenization strategy. Loaded directly by `train.py`. |
| `processed/{char,atom}_tokenized/metadata.json` | `token2idx`, `max_len`, and the tokenization strategy used to build the tensors above — kept alongside them so nothing downstream has to guess. |
| `processed/{char,atom}_tokenized/canonical_training_smiles.json` | Canonicalized SMILES for the training split for each respective tokenization strategy. Needed for both tokenization strategies since they have different maximum token lengths. Used for the novelty metric. |
| `processed/{char,atom}_tokenized/train_scaffolds.json` | Bemis-Murcko scaffolds present in the training split for each respective tokenization strategy. Needed for both tokenization strategies since they have different maximum token lengths. Used to distinguish "reproduced known chemistry" from genuinely novel output. |

### `notebooks/`

| File | Purpose |
|---|---|
|`data_import.ipynb`| Runs initial import of the database |
|`data_exploration.ipynb`| Basic exploration of ZINC250k data. Analysis used for maximum token length selections. |
| `data_processing.ipynb` | Builds the vocabulary, scaffold split, and encoded tensors under a given tokenization strategy. |
| `generate_molecules.ipynb` | Loads a trained checkpoint, generates molecules across a temperature sweep, and computes validity/uniqueness/novelty/scaffold metrics/QED. |


### `src/`

| File | Purpose |
|---|---|
| `tokenizer.py` | Character and atom based SMILES tokenization, vocabulary construction, encode/decode. |
| `split.py` | Scaffold computation (Bemis-Murcko) and the scaffold-aware train/holdout split, including the random-split carve-out for acyclic (no-ring) molecules. |
| `dataset.py` | `Zinc250kDataset` — wraps a pre-encoded tensor for use with a PyTorch `DataLoader`. |
| `model.py` | The `LSTMModel` architecture: embedding → LSTM → linear projection to vocabulary logits. |
| `train.py` | Training loop (`train_one_epoch`, `evaluate_loss`) and orchestration (`main`) — optimizer, gradient clipping, optional LR scheduler, checkpointing on best validation loss. Run via `python -m src.train`. |
| `sample.py` | Loads a trained checkpoint and generates new molecules via autoregressive, temperature-controlled sampling. |
| `evaluate.py` | Validity, uniqueness, novelty, QED, and scaffold-breakdown metrics — all pure functions operating on plain SMILES strings, no model or file I/O involved. |

### `tests/`

| File | Purpose |
|---|---|
| `test_tokenizer.py` | Round-trip encode/decode correctness, vocab determinism, edge cases (missing `<END>`, over-length sequences). |
| `test_split.py` | Scaffold computation against a known example, and the train/holdout partition/leakage guarantees. |
| `test_dataset.py` | `Zinc250kDataset`'s indexing and `DataLoader` integration. |
| `test_model.py` | Forward-pass shapes, hidden-state threading, gradient flow, the dropout/`num_layers=1` guard. |
| `test_train.py` | Confirms `evaluate_loss` never updates weights, `train_one_epoch` does, and the training loop genuinely reduces loss on a trivially memorizable dataset. |

### `models/`, `logs/`, `results/`, `reports/`

| Folder | Purpose |
|---|---|
| `models/` | Saved checkpoints (`.th`), one per training run, self-describing (architecture + vocab bundled in, not just weights). |
| `logs/` | TensorBoard logs per run — train/eval loss, gradient-clipped rate, learning rate over time. |
| `results/` | Evaluation outputs per model: metrics CSVs, QED comparison plots, example generated molecules. |





## Models
 
All models share the following fixed hyperparameters: `lr=1e-3`, `embed_dim=128`, `num_layers=2`, `dropout=0.2`, `batch_size=32`, `max_norm=1`.
 
#### Training Configuration
 
| Model | Tokenization | Epochs | Hidden Dim | LR Decay Factor | LR Patience | Train Loss | Val Loss |
|---|---|---|---|---|---|---|---|
| `char_lstm_0804_160242` | Character | 50 | 256 | None | — | 0.5287 | 0.5383 |
| `atom_lstm_0806_152355` | Atom | 50 | 256 | None | — | 0.6004 | 0.6229 |
| `char_lstm_v2_0806_182319` | Character | 100 | 256 | 0.5 | 5 | 0.4842 | 0.5206 |
| `char_lstm_v3_0806_220850` | Character | 100 | 512 | 0.5 | 5 | 0.4588 | 0.5135 |
 
#### Generation Results
 
*Full results in `results/` folder for each model.*
 
| Model | Temp | Validity | Uniqueness | Novelty | Scaffold Novelty | Scaffold Holdout | Mean QED | Median QED |
|---|---|---|---|---|---|---|---|---|
| `char_lstm_0804_160242` | 0.7 | 0.9802 | 0.999592 | 0.996938 | 0.477648 | 0.089406 | 0.762936 | 0.789328 |
| `atom_lstm_0806_152355` | 0.7 | 0.9822 | 0.999186 | 0.996536 | 0.463623 | 0.089260 | 0.764234 | 0.791075 |
| `char_lstm_v2_0806_182319` | 0.7 | 0.9894 | 0.999394 | 0.996157 | 0.438714 | 0.079086 | 0.762419 | 0.790997 |
| `char_lstm_v3_0806_220850` | 1.0 | 0.9728 | 0.995888 | 0.995888 | 0.557360 | 0.066201 | 0.732084 | 0.762720 |
 
> Note: all models were tested against temperatures 0.7, 1.0, and 1.3. Only one result was selected for each model for this chart, but all results can be found in the results/ folder for each model

> Note: The actual Mean QED for the ZINC250k dataset is 0.728264