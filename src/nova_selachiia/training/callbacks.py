"""
Callback system for training loop.

Callbacks are objects that execute custom logic at specific points during
training (e.g., end of epoch). They enable:
- Model checkpointing (save best/latest weights)
- Early stopping (stop if no improvement)
- Learning rate scheduling
- Custom logging
- Weight & Biases integration

All callbacks inherit from the base Callback class and implement
on_epoch_end() method.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
import json
import numpy as np


class Callback:
    """Base class for training callbacks."""

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        history: Dict,
    ):
        """Called at the end of each epoch."""
        pass


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.

    Features
    --------
    - Save best model (based on validation loss)
    - Save latest model (every epoch or every N epochs)
    - Save optimizer state (for resuming training)
    - Save training history

    Parameters
    ----------
    save_dir : Path
        Directory to save checkpoints
    monitor : str, default='val_loss'
        Metric to monitor ('val_loss', 'train_loss', etc.)
    mode : str, default='min'
        'min' = save when monitored metric decreases
        'max' = save when monitored metric increases
    save_best_only : bool, default=True
        Only save when model improves
    save_freq : int, default=1
        Save every N epochs (if save_best_only=False)
    verbose : bool, default=True
        Print when saving checkpoints

    Saved Files
    -----------
    best_model.pt: Best model weights
    latest_model.pt: Most recent model weights
    checkpoint_epoch_{N}.pt: Periodic checkpoints
    training_history.json: Full training history

    Example
    -------
    >>> checkpoint = ModelCheckpoint(save_dir='checkpoints', save_best_only=True)
    >>> trainer = Trainer(model, optimizer, callbacks=[checkpoint])
    """

    def __init__(
        self,
        save_dir: Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_freq: int = 1,
        verbose: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.verbose = verbose

        # Track best metric
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = -1

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        history: Dict,
    ):
        """Save checkpoint if conditions are met."""
        # Determine current metric value
        if self.monitor == "val_loss":
            current_metric = val_loss
        elif self.monitor == "train_loss":
            current_metric = train_loss
        else:
            # Extract from metrics dict
            current_metric = history.get(self.monitor, [np.nan])[-1]

        # Check if this is the best model so far
        is_best = False
        if self.mode == "min" and current_metric < self.best_metric:
            is_best = True
        elif self.mode == "max" and current_metric > self.best_metric:
            is_best = True

        # Update best metric
        if is_best:
            self.best_metric = current_metric
            self.best_epoch = epoch

        # Save checkpoint
        should_save = (not self.save_best_only) or is_best
        should_save = should_save and (epoch % self.save_freq == 0)

        if should_save:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_metric": self.best_metric,
                "best_epoch": self.best_epoch,
            }

            # Save best model
            if is_best:
                best_path = self.save_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
                if self.verbose:
                    print(
                        f"✓ Saved best model: {self.monitor}={current_metric:.6f} (epoch {epoch+1})"
                    )

            # Save latest model
            if not self.save_best_only:
                latest_path = self.save_dir / "latest_model.pt"
                torch.save(checkpoint, latest_path)

            # Save periodic checkpoint
            if epoch % (self.save_freq * 10) == 0:
                periodic_path = self.save_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
                torch.save(checkpoint, periodic_path)

        # Save training history (every epoch)
        history_path = self.save_dir / "training_history.json"
        with open(history_path, "w") as f:
            # Convert numpy types to native Python types
            serializable_history = {}
            for key, values in history.items():
                if key.endswith("_metrics"):
                    # Metrics are dicts of floats
                    serializable_history[key] = [
                        {k: float(v) if not np.isnan(v) else None for k, v in m.items()}
                        for m in values
                    ]
                else:
                    # Losses and LR are simple lists
                    serializable_history[key] = [float(v) for v in values]

            json.dump(serializable_history, f, indent=2)


class EarlyStopping(Callback):
    """
    Stop training when monitored metric stops improving.

    Algorithm
    ---------
    If validation loss hasn't improved for `patience` epochs:
        → Stop training (set `should_stop = True`)

    This prevents overfitting and saves computation time.

    Parameters
    ----------
    patience : int, default=20
        Number of epochs with no improvement before stopping
    min_delta : float, default=1e-6
        Minimum change to qualify as improvement
    monitor : str, default='val_loss'
        Metric to monitor
    mode : str, default='min'
        'min' = lower is better, 'max' = higher is better
    restore_best_weights : bool, default=True
        Restore model weights from best epoch when stopping
    verbose : bool, default=True
        Print when stopping

    Example
    -------
    >>> early_stop = EarlyStopping(patience=20, min_delta=1e-4)
    >>> trainer = Trainer(model, optimizer, callbacks=[early_stop])
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-6,
        monitor: str = "val_loss",
        mode: str = "min",
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.best_weights = None
        self.wait = 0
        self.should_stop = False

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        history: Dict,
    ):
        """Check if training should stop."""
        # Get current metric
        if self.monitor == "val_loss":
            current_metric = val_loss
        elif self.monitor == "train_loss":
            current_metric = train_loss
        else:
            current_metric = history.get(self.monitor, [np.nan])[-1]

        # Check if improved
        improved = False
        if self.mode == "min":
            improved = (self.best_metric - current_metric) > self.min_delta
        else:
            improved = (current_metric - self.best_metric) > self.min_delta

        if improved:
            self.best_metric = current_metric
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(
                        f"\n⚠️  Early stopping: no improvement for {self.patience} epochs"
                    )
                    print(f"    Best {self.monitor}: {self.best_metric:.6f}")

                # Restore best weights
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(
                        {
                            k: v.to(next(model.parameters()).device)
                            for k, v in self.best_weights.items()
                        }
                    )
                    if self.verbose:
                        print(f"    Restored best model weights")


class ReduceLROnPlateau(Callback):
    """
    Reduce learning rate when metric plateaus.

    Algorithm
    ---------
    If metric hasn't improved for `patience` epochs:
        → Multiply learning rate by `factor`
        → Continue until LR reaches `min_lr`

    This helps escape local minima and fine-tune at the end of training.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to adjust
    patience : int, default=10
        Epochs with no improvement before reducing LR
    factor : float, default=0.5
        Multiply LR by this factor (e.g., 0.5 = halve the LR)
    min_lr : float, default=1e-6
        Minimum learning rate
    monitor : str, default='val_loss'
        Metric to monitor
    mode : str, default='min'
        'min' or 'max'
    min_delta : float, default=1e-6
        Minimum change to qualify as improvement
    verbose : bool, default=True
        Print when reducing LR

    Example
    -------
    >>> lr_scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    >>> trainer = Trainer(model, optimizer, callbacks=[lr_scheduler])
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        patience: int = 10,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        monitor: str = "val_loss",
        mode: str = "min",
        min_delta: float = 1e-6,
        verbose: bool = True,
    ):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.wait = 0

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        history: Dict,
    ):
        """Reduce learning rate if needed."""
        # Get current metric
        if self.monitor == "val_loss":
            current_metric = val_loss
        elif self.monitor == "train_loss":
            current_metric = train_loss
        else:
            current_metric = history.get(self.monitor, [np.nan])[-1]

        # Check if improved
        improved = False
        if self.mode == "min":
            improved = (self.best_metric - current_metric) > self.min_delta
        else:
            improved = (current_metric - self.best_metric) > self.min_delta

        if improved:
            self.best_metric = current_metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Reduce learning rate
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group["lr"]
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    param_group["lr"] = new_lr

                    if self.verbose and new_lr < old_lr:
                        print(f"    Reduced LR: {old_lr:.2e} → {new_lr:.2e}")

                self.wait = 0  # Reset counter after reducing LR
