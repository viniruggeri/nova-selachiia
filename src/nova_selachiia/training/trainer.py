"""
Generic Training Loop for State Space Models.

This module provides a flexible Trainer class that handles:
- Training/validation loops
- Metrics computation (MSE, MAE, R², F1, AUC)
- Gradient clipping
- Learning rate scheduling
- Progress logging
- Model checkpointing (handled by callbacks)

The trainer is model-agnostic and works with any PyTorch nn.Module
that implements a `compute_loss` method.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Optional, Callable, List
from pathlib import Path
import time
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc


class MetricsCalculator:
    """
    Compute evaluation metrics for regression and classification tasks.

    Metrics
    -------
    Regression (continuous predictions):
        - MSE: Mean Squared Error
        - RMSE: Root Mean Squared Error
        - MAE: Mean Absolute Error
        - R²: Coefficient of determination

    Classification (binary predictions with threshold=0.5):
        - F1: Harmonic mean of precision and recall
        - AUC-ROC: Area under ROC curve
        - AUC-PR: Area under Precision-Recall curve
    """

    @staticmethod
    def compute_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute regression metrics.

        Parameters
        ----------
        y_true : np.ndarray, shape (n_samples,)
            True values
        y_pred : np.ndarray, shape (n_samples,)
            Predicted values
        mask : np.ndarray, shape (n_samples,), optional
            Binary mask (1=valid, 0=ignore)

        Returns
        -------
        metrics : dict
            {'mse': ..., 'rmse': ..., 'mae': ..., 'r2': ...}
        """
        if mask is not None:
            mask = mask.astype(bool)
            y_true = y_true[mask]
            y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {"mse": np.nan, "rmse": np.nan, "mae": np.nan, "r2": np.nan}

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        # R² can be negative for very poor fits
        try:
            r2 = r2_score(y_true, y_pred)
        except:
            r2 = np.nan

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
        }

    @staticmethod
    def compute_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute binary classification metrics.

        Parameters
        ----------
        y_true : np.ndarray, shape (n_samples,)
            True binary labels (0 or 1)
        y_pred : np.ndarray, shape (n_samples,)
            Predicted continuous values (will be thresholded)
        y_prob : np.ndarray, shape (n_samples,), optional
            Predicted probabilities for AUC computation
            If None, uses y_pred as probabilities
        threshold : float, default=0.5
            Classification threshold
        mask : np.ndarray, shape (n_samples,), optional
            Binary mask (1=valid, 0=ignore)

        Returns
        -------
        metrics : dict
            {'f1': ..., 'auc_roc': ..., 'auc_pr': ...}
        """
        if mask is not None:
            mask = mask.astype(bool)
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            if y_prob is not None:
                y_prob = y_prob[mask]

        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            return {"f1": np.nan, "auc_roc": np.nan, "auc_pr": np.nan}

        # Binarize predictions
        y_pred_binary = (y_pred >= threshold).astype(int)

        # F1 score
        try:
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        except:
            f1 = np.nan

        # AUC metrics (require probabilities)
        if y_prob is None:
            y_prob = y_pred

        try:
            auc_roc = roc_auc_score(y_true, y_prob)
        except:
            auc_roc = np.nan

        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            auc_pr = auc(recall, precision)
        except:
            auc_pr = np.nan

        return {
            "f1": float(f1),
            "auc_roc": float(auc_roc),
            "auc_pr": float(auc_pr),
        }


class Trainer:
    """
    Generic training loop for state space models.

    Features
    --------
    - Automatic mixed precision (AMP) for faster training
    - Gradient clipping for stability
    - Learning rate scheduling
    - Metrics logging (train/val)
    - Callback system (checkpointing, early stopping, etc.)
    - Progress bars with tqdm

    Parameters
    ----------
    model : nn.Module
        PyTorch model (NSSM, DMM, etc.)
    optimizer : Optimizer
        PyTorch optimizer (Adam, AdamW, etc.)
    device : torch.device
        Device for training (cuda/cpu)
    scheduler : _LRScheduler, optional
        Learning rate scheduler
    grad_clip : float, default=1.0
        Gradient clipping threshold (prevents exploding gradients)
    use_amp : bool, default=False
        Use automatic mixed precision (faster on modern GPUs)
    callbacks : list of Callback, optional
        List of callback objects

    Example
    -------
    >>> model = NSSM(input_dim=5, hidden_dim=64)
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> trainer = Trainer(model, optimizer, device='cuda')
    >>> history = trainer.fit(train_loader, val_loader, epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        grad_clip: float = 1.0,
        use_amp: bool = False,
        callbacks: Optional[List] = None,
        loss_type: str = "focal",  # NEW: "mse", "bce", "focal"
        focal_alpha: float = 0.25,  # NEW: Focal loss alpha
        focal_gamma: float = 2.0,  # NEW: Focal loss gamma
        pos_weight: Optional[float] = None,  # NEW: BCE pos_weight
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.callbacks = callbacks or []

        # Loss configuration (NEW)
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight

        # Automatic mixed precision setup
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # Training state
        self.current_epoch = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
            "lr": [],
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> Dict:
        """
        Train model for specified number of epochs.

        Algorithm
        ---------
        For each epoch:
            1. Train phase: forward pass, compute loss, backprop, update weights
            2. Validation phase: evaluate on validation set
            3. Compute metrics (MSE, MAE, R², F1, AUC)
            4. Update learning rate (if scheduler provided)
            5. Execute callbacks (checkpointing, early stopping, etc.)
            6. Log progress

        Parameters
        ----------
        train_loader : DataLoader
            Training data
        val_loader : DataLoader
            Validation data
        epochs : int
            Number of training epochs

        Returns
        -------
        history : dict
            Training history with losses and metrics
        """
        print(f"\n{'='*80}")
        print(f"TRAINING STARTED")
        print(f"{'='*80}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print(f"Epochs: {epochs}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"AMP enabled: {self.use_amp}")
        print(f"{'='*80}\n")

        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # TRAINING PHASE
            train_loss, train_metrics = self._train_epoch(train_loader)

            # VALIDATION PHASE
            val_loss, val_metrics = self._validate(val_loader)

            # LEARNING RATE UPDATE
            if self.scheduler is not None:
                self.scheduler.step(val_loss)  # ReduceLROnPlateau

            current_lr = self.optimizer.param_groups[0]["lr"]

            # STORE HISTORY
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_metrics"].append(train_metrics)
            self.history["val_metrics"].append(val_metrics)
            self.history["lr"].append(current_lr)

            # EXECUTE CALLBACKS
            for callback in self.callbacks:
                callback.on_epoch_end(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    model=self.model,
                    optimizer=self.optimizer,
                    history=self.history,
                )

            # LOG PROGRESS
            epoch_time = time.time() - epoch_start_time
            self._log_epoch(
                epoch,
                epochs,
                train_loss,
                val_loss,
                train_metrics,
                val_metrics,
                current_lr,
                epoch_time,
            )

            # CHECK EARLY STOPPING
            if any(
                hasattr(cb, "should_stop") and cb.should_stop for cb in self.callbacks
            ):
                print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                break

        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*80}")
        print(f"Total epochs: {self.current_epoch + 1}")
        print(f"Best val loss: {min(self.history['val_loss']):.6f}")
        print(f"{'='*80}\n")

        return self.history

    def _train_epoch(self, train_loader: DataLoader) -> tuple[float, Dict]:
        """Execute one training epoch."""
        self.model.train()

        epoch_loss = 0.0
        all_preds = []
        all_targets = []
        all_masks = []

        pbar = tqdm(
            train_loader, desc=f"Epoch {self.current_epoch+1} [TRAIN]", leave=False
        )

        for batch_idx, (X, Y, mask) in enumerate(pbar):
            try:
                # Move to device
                X = X.to(self.device)
                Y = Y.to(self.device)
                mask = mask.to(self.device)

                # ========================================================
                # RUNTIME SHAPE VALIDATION AND AUTO-CORRECTION
                # ========================================================
                batch_size, seq_len = X.shape[0], X.shape[1]

                # Validate X shape (batch, seq_len, features)
                assert (
                    X.ndim == 3
                ), f"X must be 3D (batch, seq, feat), got {X.ndim}D: {X.shape}"

                # Validate Y shape (batch, seq_len, 1) or auto-correct
                if Y.ndim == 2 and Y.shape[1] != seq_len:
                    # (batch, 1) → (batch, seq_len, 1) broadcast
                    Y = Y.unsqueeze(1).expand(batch_size, seq_len, 1)
                elif Y.ndim == 2 and Y.shape[1] == seq_len:
                    # (batch, seq_len) → (batch, seq_len, 1)
                    Y = Y.unsqueeze(-1)
                elif Y.ndim == 3 and Y.shape[1] != seq_len:
                    # Shape mismatch - try to fix
                    if Y.shape[1] == 1:
                        # (batch, 1, feat) → (batch, seq_len, feat) broadcast
                        Y = Y.expand(batch_size, seq_len, Y.shape[2])

                # Validate mask shape
                if mask.ndim == 2 and mask.shape[1] != seq_len:
                    mask = mask.unsqueeze(1).expand(batch_size, seq_len, 1)
                elif mask.ndim == 2 and mask.shape[1] == seq_len:
                    mask = mask.unsqueeze(-1)
                elif mask.ndim == 3 and mask.shape[1] != seq_len:
                    if mask.shape[1] == 1:
                        mask = mask.expand(batch_size, seq_len, mask.shape[2])

                # Final assertion (should match now)
                assert (
                    Y.shape[0] == batch_size and Y.shape[1] == seq_len
                ), f"Y shape mismatch after correction: expected (batch={batch_size}, seq={seq_len}, 1), got {Y.shape}"
                assert (
                    mask.shape[0] == batch_size and mask.shape[1] == seq_len
                ), f"Mask shape mismatch after correction: expected (batch={batch_size}, seq={seq_len}, 1), got {mask.shape}"

            except (AssertionError, RuntimeError) as e:
                print(f"\n⚠️  Batch {batch_idx} shape error: {e}")
                print(f"   X: {X.shape}, Y: {Y.shape}, mask: {mask.shape}")
                print(f"   Skipping this batch...")
                continue  # Skip problematic batch

            # Forward pass (with AMP if enabled)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                Y_pred = self.model(X)
                loss = self.model.compute_loss(
                    Y_pred,
                    Y,
                    mask,
                    loss_type=self.loss_type,
                    focal_alpha=self.focal_alpha,
                    focal_gamma=self.focal_gamma,
                    pos_weight=self.pos_weight,
                )

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            # Accumulate metrics
            epoch_loss += loss.item()

            # Convert logits to probabilities for metrics (if using focal/bce loss)
            if self.loss_type in ["focal", "bce"]:
                Y_pred_probs = torch.sigmoid(Y_pred)
                all_preds.append(Y_pred_probs.detach().cpu().numpy())
            else:
                all_preds.append(Y_pred.detach().cpu().numpy())

            all_targets.append(Y.detach().cpu().numpy())
            all_masks.append(mask.detach().cpu().numpy())

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

        # Compute epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        all_preds = np.concatenate(all_preds, axis=0).ravel()
        all_targets = np.concatenate(all_targets, axis=0).ravel()
        all_masks = np.concatenate(
            all_masks, axis=0
        ).ravel()  # Must match preds/targets shape

        # For classification tasks (focal/bce), use classification metrics
        # For regression (mse), use regression metrics
        if self.loss_type in ["focal", "bce"]:
            # all_preds are probabilities [0,1]
            # Compute binary classification metrics with threshold=0.5
            metrics = MetricsCalculator.compute_classification_metrics(
                all_targets, all_preds, threshold=0.5, mask=all_masks
            )
        else:
            # Standard regression metrics (MSE, R², etc.)
            metrics = MetricsCalculator.compute_regression_metrics(
                all_targets, all_preds, all_masks
            )

        return avg_loss, metrics

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> tuple[float, Dict]:
        """Execute validation."""
        self.model.eval()

        epoch_loss = 0.0
        all_preds = []
        all_targets = []
        all_masks = []

        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch+1} [VAL]", leave=False)

        for batch_idx, (X, Y, mask) in enumerate(pbar):
            try:
                X = X.to(self.device)
                Y = Y.to(self.device)
                mask = mask.to(self.device)

                # ========================================================
                # RUNTIME SHAPE VALIDATION AND AUTO-CORRECTION (VALIDATION)
                # ========================================================
                batch_size, seq_len = X.shape[0], X.shape[1]

                # Auto-correct Y shape
                if Y.ndim == 2 and Y.shape[1] != seq_len:
                    Y = Y.unsqueeze(1).expand(batch_size, seq_len, 1)
                elif Y.ndim == 2 and Y.shape[1] == seq_len:
                    Y = Y.unsqueeze(-1)
                elif Y.ndim == 3 and Y.shape[1] != seq_len:
                    if Y.shape[1] == 1:
                        Y = Y.expand(batch_size, seq_len, Y.shape[2])

                # Auto-correct mask shape
                if mask.ndim == 2 and mask.shape[1] != seq_len:
                    mask = mask.unsqueeze(1).expand(batch_size, seq_len, 1)
                elif mask.ndim == 2 and mask.shape[1] == seq_len:
                    mask = mask.unsqueeze(-1)
                elif mask.ndim == 3 and mask.shape[1] != seq_len:
                    if mask.shape[1] == 1:
                        mask = mask.expand(batch_size, seq_len, mask.shape[2])

                # Final validation
                assert (
                    Y.shape[0] == batch_size and Y.shape[1] == seq_len
                ), f"Val Y shape mismatch: expected (batch={batch_size}, seq={seq_len}, 1), got {Y.shape}"
                assert (
                    mask.shape[0] == batch_size and mask.shape[1] == seq_len
                ), f"Val mask shape mismatch: expected (batch={batch_size}, seq={seq_len}, 1), got {mask.shape}"

            except (AssertionError, RuntimeError) as e:
                print(f"\n⚠️  Val batch {batch_idx} shape error: {e}")
                print(f"   X: {X.shape}, Y: {Y.shape}, mask: {mask.shape}")
                print(f"   Skipping this batch...")
                continue

            Y_pred = self.model(X)
            loss = self.model.compute_loss(
                Y_pred,
                Y,
                mask,
                loss_type=self.loss_type,
                focal_alpha=self.focal_alpha,
                focal_gamma=self.focal_gamma,
                pos_weight=self.pos_weight,
            )

            epoch_loss += loss.item()

            # Convert logits to probabilities for metrics (if using focal/bce loss)
            if self.loss_type in ["focal", "bce"]:
                Y_pred_probs = torch.sigmoid(Y_pred)
                all_preds.append(Y_pred_probs.cpu().numpy())
            else:
                all_preds.append(Y_pred.cpu().numpy())

            all_targets.append(Y.cpu().numpy())
            all_masks.append(mask.cpu().numpy())

            pbar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(val_loader)
        all_preds = np.concatenate(all_preds, axis=0).ravel()
        all_targets = np.concatenate(all_targets, axis=0).ravel()
        all_masks = np.concatenate(
            all_masks, axis=0
        ).ravel()  # Must match preds/targets shape

        # For classification tasks (focal/bce), use classification metrics
        # For regression (mse), use regression metrics
        if self.loss_type in ["focal", "bce"]:
            # all_preds are probabilities [0,1]
            # Compute binary classification metrics with threshold=0.5
            metrics = MetricsCalculator.compute_classification_metrics(
                all_targets, all_preds, threshold=0.5, mask=all_masks
            )
        else:
            # Standard regression metrics (MSE, R², etc.)
            metrics = MetricsCalculator.compute_regression_metrics(
                all_targets, all_preds, all_masks
            )

        return avg_loss, metrics

    def _log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: float,
        train_metrics: Dict,
        val_metrics: Dict,
        lr: float,
        epoch_time: float,
    ):
        """Print epoch summary."""
        # For classification (focal/bce), show F1 and AUC
        # For regression (mse), show R²
        if self.loss_type in ["focal", "bce"]:
            print(
                f"Epoch [{epoch+1:3d}/{total_epochs}] | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Train F1: {train_metrics.get('f1', 0):.4f} | "
                f"Val F1: {val_metrics.get('f1', 0):.4f} | "
                f"Val AUC: {val_metrics.get('auc_roc', 0):.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {epoch_time:.2f}s"
            )
        else:
            print(
                f"Epoch [{epoch+1:3d}/{total_epochs}] | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Train R²: {train_metrics.get('r2', 0):.4f} | "
                f"Val R²: {val_metrics.get('r2', 0):.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {epoch_time:.2f}s"
            )

    @torch.no_grad()
    def predict(self, data_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions on a dataset.

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Model predictions (probabilities if using focal/bce, raw values if mse)
        targets : np.ndarray, shape (n_samples,)
            True targets
        """
        self.model.eval()

        all_preds = []
        all_targets = []

        for X, Y, mask in tqdm(data_loader, desc="Predicting"):
            X = X.to(self.device)
            Y_pred = self.model(X)

            # CRITICAL FIX: Apply sigmoid for classification tasks
            # (same logic as _validate_epoch to ensure consistency)
            if self.loss_type in ["focal", "bce"]:
                Y_pred = torch.sigmoid(Y_pred)  # Convert logits → probabilities

            all_preds.append(Y_pred.cpu().numpy())
            all_targets.append(Y.cpu().numpy())

        predictions = np.concatenate(all_preds, axis=0).ravel()
        targets = np.concatenate(all_targets, axis=0).ravel()

        return predictions, targets
