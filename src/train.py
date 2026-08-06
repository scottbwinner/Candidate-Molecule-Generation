import torch
import json
import torch.utils.tensorboard as tb
from datetime import datetime
import copy
from torch.utils.data import DataLoader
import argparse

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/train.py -> src/ -> project root
data_dir = PROJECT_ROOT / "data" / "processed"

from src.dataset import Zinc250kDataset
from src.model import LSTMModel
from src.tokenizer import PAD_TOKEN

def train_one_epoch(model, train_loader, optimizer, criterion, device, max_norm=1.0):
    """
    This training helper function trains one epoch of the LSTM model.

    Parameters:
        model: LSTM model
        train_loader: DataLoader object for training set
        optimizer: Optimizer function
        criterion: Loss function
        device: Device to store data on
        max_norm: The maximum allowed norm of the combined gradient across all parameters

    Returns:
        (average_loss, grad_norm_clipped_rate)
        average_loss: Average training loss across epoch
        grad_norm_clipped_rate: Fraction of batches whose combined gradient norm exceed max_norm
    """

    model.train()
    losses = []
    grad_norm_clips = []
    for batch in train_loader:
        batch = batch.to(device)
        input_seq = batch[:, :-1]   # everything except the last token
        target_seq = batch[:, 1:]   # everything except the first token

        optimizer.zero_grad()

        # Forward pass calculates logit predictions for tokens
        logits, hidden = model(input_seq) 
        vocab_size = logits.shape[-1]

        # Calculate loss
        loss = criterion(logits.reshape(-1, vocab_size), target_seq.reshape(-1))
        losses.append(loss.item())

        # Backward pass computes gradients for each parameter
        loss.backward() 

        # Cap gradients to avoid exploding gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm) 

        # Keeping track of number of gradients that are clipped
        if grad_norm.item() > max_norm:
            grad_norm_clips.append(1)
        else:
            grad_norm_clips.append(0)

        # update weights using the computed gradients
        optimizer.step() 

    average_loss = sum(losses) / len(losses)
    grad_norm_clipped_rate = sum(grad_norm_clips) / len(grad_norm_clips)

    return average_loss, grad_norm_clipped_rate



def evaluate_loss(model, val_loader, criterion, device):
    """
    This function calculates loss for inputted model against the validation set.

    Parameters:
        model: LSTM Model
        val_loader: DataLoader object for validation set
        criterion: Loss Function
        device: Device to store data on
        
    Returns:
        average_loss: Average loss of inputted model against the validation set
    """

    # disable gradient computation and switch to evaluation mode
    with torch.inference_mode():
        model.eval()
        losses = []
        for batch in val_loader:
            batch = batch.to(device)
            input_seq = batch[:, :-1]   # everything except the last token
            target_seq = batch[:, 1:]   # everything except the first token


             # Forward pass calculates logit predictions for tokens
            logits, hidden = model(input_seq)
            vocab_size = logits.shape[-1]

            # Calculate loss
            loss = criterion(logits.reshape(-1, vocab_size), target_seq.reshape(-1))
            losses.append(loss.item())

        average_loss = sum(losses) / len(losses)
        return average_loss



def main(num_epochs, model_name, lr, embed_dim, hidden_dim, num_layers, dropout, batch_size, max_norm, tokenization):
    """
    This function orchestrates the entire training process for an LSTM model.

    Parameters:
        num_epochs: Number of epochs to train over
        model_name: Name for your model, to be used for the file name for the saved model
        lr: Learning Rate for training
        embed_dim: dimension size of embeddings
        hidden_dim: Number of dimensions the LSTM recurrent layer builds up to
        num_layers: Number of layers for LSTM model
        dropout: dropout that is only applied between stacked LSTM layers, forced to 0 when num_layers=1
        batch_size: Batch size to parse through DataLoader objects.
        max_norm: The maximum allowed norm of the combined gradient across all parameters
        tokenization: Select either character tokenization or atom tokenization ["char", "atom"]


    Run via: python -m src.train --parameter1 value1 --parameter2 value2 ...
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    if tokenization == "char":
        output_subdir = "char_tokenization"
        max_len = 68
    else:
        output_subdir = "atom_tokenization"
        max_len = 59

    # Directory with timestamp to save tensorboard logs and model checkpoints
    full_model_name = f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    log_dir = PROJECT_ROOT / "logs" / full_model_name
    exp_dir = PROJECT_ROOT / "models"
    logger = tb.SummaryWriter(log_dir)

    training_tensor = torch.load(data_dir / output_subdir / "training_tensor.pt")
    holdout_tensor = torch.load(data_dir / output_subdir / "holdout_tensor.pt")

    with open(data_dir / output_subdir / "metadata.json", 'r', encoding='utf-8') as file:
        metadata = json.load(file)

    token2idx = metadata['token2idx']
    vocab_size = len(token2idx)
    pad_idx = token2idx[PAD_TOKEN]

    train_dataset = Zinc250kDataset(training_tensor)
    val_dataset = Zinc250kDataset(holdout_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = LSTMModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        pad_idx=pad_idx,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_state_dict = {}
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        average_train_loss, gradient_clipped_rate = train_one_epoch(
            model=model, 
            train_loader=train_loader, 
            optimizer=optimizer, 
            criterion=criterion, 
            device=device, 
            max_norm=max_norm
        )
        average_val_loss = evaluate_loss(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )

        logger.add_scalar("train_loss", average_train_loss, epoch)
        logger.add_scalar("gradient_clipped_rate", gradient_clipped_rate, epoch)
        logger.add_scalar("val_loss", average_val_loss, epoch)

        print(
            f"Epoch {epoch + 1:2d} / {num_epochs:2d}: "
            f"train_loss={average_train_loss:.4f} "
            f"gradient_clipped_rate={gradient_clipped_rate:.4f} "
            f"val_loss={average_val_loss:.4f} "
        )

        if (average_val_loss < best_val_loss):
            best_val_loss = average_val_loss
            best_state_dict = copy.deepcopy(model.state_dict())


    # Save best checkpoint of the model in the exp_dir directory
    best_checkpoint = {
        "model_state_dict": best_state_dict,
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "pad_idx": pad_idx,
        "val_loss": best_val_loss,
        "token2idx": token2idx,
        "max_len": max_len,
        "tokenization": tokenization
    }
    exp_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_checkpoint, exp_dir / f"{full_model_name}.th")
    print(f"Model saved to {exp_dir / f'{full_model_name}.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--model_name", type=str, default="char_lstm")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--tokenization", type=str, default="char", choices=["char", "atom"])
    args = parser.parse_args()

    main(
        num_epochs=args.num_epochs, 
        model_name=args.model_name, 
        lr=args.lr,
        embed_dim=args.embed_dim, 
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, 
        dropout=args.dropout, 
        batch_size=args.batch_size, 
        max_norm=args.max_norm
    )