import torch

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