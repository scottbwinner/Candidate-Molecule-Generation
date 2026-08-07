### Models
 
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