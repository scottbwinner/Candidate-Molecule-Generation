import torch
import json
import torch.utils.tensorboard as tb
from datetime import datetime
import copy
from torch.utils.data import DataLoader
import argparse

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/train.py -> src/ -> project root
data_dir = PROJECT_ROOT / "data" / "processed" / "char_tokenized"

from src.dataset import Zinc250kDataset
from src.model import LSTMModel
from src.tokenizer import PAD_TOKEN

def train_one_epoch(model, train_loader, optimizer, criterion, device, max_norm=1.0):

    model.train()
    losses = []
    grad_norm_clips = []
    for batch in train_loader:
        batch = batch.to(device)
        input_seq = batch[:, :-1]   # everything except the last token
        target_seq = batch[:, 1:]   # everything except the first token

        optimizer.zero_grad()

        logits, hidden = model(input_seq) # forward pass calculates logit predictions for tokens
        vocab_size = logits.shape[-1]
        loss = criterion(logits.reshape(-1, vocab_size), target_seq.reshape(-1))
        losses.append(loss.item())

        loss.backward() # backward pass computes gradients for each parameter
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm) # Cap gradients to avoid exploding gradients
        if grad_norm.item() > max_norm:
            grad_norm_clips.append(1)
        else:
            grad_norm_clips.append(0)
        optimizer.step() # update weights using the computed gradients

    average_loss = sum(losses) / len(losses)
    grad_norm_clipped_rate = sum(grad_norm_clips) / len(grad_norm_clips)
    return average_loss, grad_norm_clipped_rate



def evaluate_loss(model, eval_loader, criterion, device):
    # disable gradient computation and switch to evaluation mode
    with torch.inference_mode():
        model.eval()
        losses = []
        for batch in eval_loader:
            batch = batch.to(device)
            input_seq = batch[:, :-1]   # everything except the last token
            target_seq = batch[:, 1:]   # everything except the first token


            logits, hidden = model(input_seq) # forward pass calculates logit predictions for tokens
            vocab_size = logits.shape[-1]
            loss = criterion(logits.reshape(-1, vocab_size), target_seq.reshape(-1))
            losses.append(loss.item())

        average_loss = sum(losses) / len(losses)
        return average_loss

def main(num_epochs, model_name, lr, embed_dim, hidden_dim, num_layers, dropout, batch_size, max_norm):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # directory with timestamp to save tensorboard logs and model checkpoints
    full_model_name = f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    log_dir = PROJECT_ROOT / "logs" / full_model_name
    exp_dir = PROJECT_ROOT / "models"
    logger = tb.SummaryWriter(log_dir)

    training_tensor = torch.load(data_dir / "training_tensor.pt")
    holdout_tensor = torch.load(data_dir / "holdout_tensor.pt")

    with open(data_dir / "metadata.json", 'r', encoding='utf-8') as file:
        metadata = json.load(file)

    token2idx = metadata['token2idx']
    vocab_size = len(token2idx)
    pad_idx = token2idx[PAD_TOKEN]

    train_dataset = Zinc250kDataset(training_tensor)
    eval_dataset = Zinc250kDataset(holdout_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

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
    best_eval_loss = float("inf")
    for epoch in range(num_epochs):
        average_train_loss, gradient_clipped_rate = train_one_epoch(
            model=model, 
            train_loader=train_loader, 
            optimizer=optimizer, 
            criterion=criterion, 
            device=device, 
            max_norm=max_norm
        )
        average_eval_loss = evaluate_loss(
            model=model,
            eval_loader=eval_loader,
            criterion=criterion,
            device=device
        )

        logger.add_scalar("train_loss", average_train_loss, epoch)
        logger.add_scalar("gradient_clipped_rate", gradient_clipped_rate, epoch)
        logger.add_scalar("eval_loss", average_eval_loss, epoch)

        print(
            f"Epoch {epoch + 1:2d} / {num_epochs:2d}: "
            f"train_loss={average_train_loss:.4f} "
            f"gradient_clipped_rate={gradient_clipped_rate:.4f} "
            f"eval_loss={average_eval_loss:.4f} "
        )

        if (average_eval_loss < best_eval_loss):
            best_eval_loss = average_eval_loss
            best_state_dict = copy.deepcopy(model.state_dict())


    # save a copy of model weights in the log directory
    best_checkpoint = {
        "model_state_dict": best_state_dict,
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "pad_idx": pad_idx,
        "eval_loss": best_eval_loss
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
    args = parser.parse_args()

    main(
        num_epochs=args.num_epochs, model_name=args.model_name, lr=args.lr,
        embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, dropout=args.dropout, batch_size=args.batch_size, max_norm=args.max_norm
    )