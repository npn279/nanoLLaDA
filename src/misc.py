import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.nn import functional as F
from typing import List


def get_batch(
    data_dir: str,
    split: str,
    block_size: int,
    batch_size: int = 64,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Load a batch of data from memory-mapped numpy array.

    Args:
        data_dir: Directory containing the data files
        split: Whether to load 'train' or 'val' data
        block_size: Number of tokens in each sequence
        batch_size: Number of sequences to load
        device: Device to load the data onto (cuda or cpu)

    Returns:
        Tensor of input sequences
    """

    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )

    if device == "cuda":
        # pin arrays x, which allows moving to GPU asynchronously
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)

    return x


def forward_process(
    input_ids: torch.Tensor, mask_token: int, eps: float = 1e-3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply masking to input sequences with probabilistic masking.

    Args:
        input_ids: Input token sequences
        mask_token: Token ID used for masking
        eps: Probability of masking (small constant)

    Returns:
        Tuple of (noisy batch, masked indices, mask probabilities)
    """
    # Get batch and sequence dimensions
    b, l = input_ids.shape

    # Generate random mask probabilities
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    # Determine which tokens to mask
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    # Create noisy batch by replacing selected tokens with mask token
    noisy_batch = torch.where(masked_indices, mask_token, input_ids)

    return noisy_batch, masked_indices, p_mask


def evaluate_batch(
    model: torch.nn.Module, inputs: torch.Tensor, mask_id: int
) -> torch.Tensor:
    """
    Evaluate model performance on a batch of inputs.

    Args:
        model: Neural network model
        inputs: Input token sequences
        mask_id: Token ID used for masking

    Returns:
        Computed loss for the batch
    """
    # Occasionally truncate input sequence to test model robustness
    if torch.rand(1) < 0.01:
        random_length = torch.randint(1, inputs.shape[1] + 1, (1,))
        inputs = inputs[:, :random_length]

    # Apply masking process
    noisy_batch, masked_indices, p_mask = forward_process(inputs, mask_id)

    # Get model predictions
    logits = model(noisy_batch)

    # Compute cross-entropy loss with probability-weighted correction
    token_loss = (
        F.cross_entropy(
            logits[masked_indices], inputs[masked_indices], reduction="none"
        )
        / p_mask[masked_indices]
    )

    # Normalize loss across batch and sequence length
    loss = token_loss.sum() / (inputs.shape[0] * inputs.shape[1])
    return loss


def train_model(
    model: torch.nn.Module,
    data_dir: str,
    batch_size: int,
    max_iters: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    mask_id: int,
    save_every: int = 100,
    device: str = "cpu",
    save_filename = None
) -> tuple[torch.nn.Module, List[float], List[float]]:
    """
    Train the neural network model.

    Args:
        model: Neural network to train
        data_dir: Directory containing training data
        batch_size: Number of sequences per batch
        max_iters: Total training iterations
        lr: Learning rate
        weight_decay: L2 regularization strength
        grad_clip: Gradient clipping threshold
        mask_id: Token ID used for masking
        save_every: Frequency of model checkpointing
        device: Training device

    Returns:
        Tuple of (trained model, training losses, validation losses)
    """
    if save_filename is None : 
        save_filename = str(model.__class__.__name__).lower()
    # Initialize progress bar
    pbar = tqdm(range(max_iters))
    losses, val_losses = [], []

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    for i in pbar:
        # Training
        model.train()
        inputs = get_batch(data_dir, "train", model.block_size, batch_size, device)
        loss = evaluate_batch(model, inputs, mask_id)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        opt.step()
        opt.zero_grad()

        # Validation
        model.eval()
        with torch.no_grad():
            inputs = get_batch(data_dir, "val", model.block_size, batch_size, device)
            val_loss = evaluate_batch(model, inputs, mask_id)

        # Update progress bar
        pbar.set_description(
            "Iter :{}/{} Train Loss {:.3e}, Val Loss {:.3e}".format(
                i, max_iters, loss.item(), val_loss.item()
            )
        )

        losses.append(loss.item())
        val_losses.append(val_loss.item())

        if i % save_every == 0 or i == max_iters - 1:
            plt.plot(losses, label="train")
            plt.plot(val_losses, "--", label="val")
            plt.savefig("loss_last.png")
            plt.close()
            torch.save(
                model.state_dict(), save_filename + "_last.pt"
            )

    return model, losses, val_losses
