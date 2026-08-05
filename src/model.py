import torch

class LSTMModel(torch.nn.Module):
    """
    Class that builds LSTM Model
    Parameters:
        vocab_size: number of tokens in vocabulary
        embed_dim: dimension size of embeddings
        pad_idx: padding index to freeze padding tokens at zero and keep them out of gradient updates
        hidden_dim: Number of dimensions the LSTM recurrent layer builds up to
        num_layers: number of recurrent layers
        dropout: dropout that is only applied between stacked LSTM layers, forced to 0 when num_layers=1
    """
    def __init__(self, vocab_size, embed_dim, pad_idx, hidden_dim, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)
                
    def forward(self, x, hidden=None):
        # Getting each tokens embedding
        embedded = self.embedding(x)                     # (batch, seq_len) -> (batch, seq_len, embed_dim)
        # The recurrent layer that processes the sequence step by step and decides what to forget or add to cell state and what to expose as hidden state.
        lstm_out, hidden = self.lstm(embedded, hidden)   # -> (batch, seq_len, hidden_dim), (h_n, c_n)
        # Linear layer turns hidden_dim dimension size into vocab_size, generating logit scores over all vocab tokens for each token in the sequence.
        logits = self.fc(lstm_out)                       # -> (batch, seq_len, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        """
        This function initializes the cell state and hidden state tensors with zeros for the start of generation
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)