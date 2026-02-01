"""
Visualization utilities for model training and evaluation.

This module provides plotting functions for:
- Training curves (loss, metrics over epochs)
- Model predictions vs true values
- Latent state trajectories
- Counterfactual scenarios
- Spatial-temporal heatmaps
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import torch


def plot_training_history(
    history: Dict,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10),
):
    """
    Plot training and validation losses/metrics over epochs.

    Creates a 2×2 subplot grid:
    - Top-left: Loss curves (train/val)
    - Top-right: R² score (train/val)
    - Bottom-left: MAE (train/val)
    - Bottom-right: Learning rate schedule

    Parameters
    ----------
    history : dict
        Training history from Trainer.fit()
        Expected keys: 'train_loss', 'val_loss', 'train_metrics', 'val_metrics', 'lr'
    save_path : Path, optional
        Path to save figure
    figsize : tuple, default=(15, 10)
        Figure size

    Example
    -------
    >>> history = trainer.fit(train_loader, val_loader, epochs=100)
    >>> plot_training_history(history, save_path='figures/training.png')
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    epochs = range(1, len(history["train_loss"]) + 1)

    # PLOT 1: Loss curves
    ax = axes[0, 0]
    ax.plot(
        epochs,
        history["train_loss"],
        label="Train",
        marker="o",
        markersize=3,
        alpha=0.7,
    )
    ax.plot(
        epochs,
        history["val_loss"],
        label="Validation",
        marker="s",
        markersize=3,
        alpha=0.7,
    )
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Loss (MSE)", fontweight="bold")
    ax.set_title("Training & Validation Loss", fontweight="bold", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    # Mark best epoch
    best_epoch = np.argmin(history["val_loss"]) + 1
    best_loss = min(history["val_loss"])
    ax.axvline(
        best_epoch,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Best (epoch {best_epoch})",
    )
    ax.scatter([best_epoch], [best_loss], color="red", s=100, zorder=5, marker="*")

    # PLOT 2: R² score
    ax = axes[0, 1]
    train_r2 = [m["r2"] for m in history["train_metrics"]]
    val_r2 = [m["r2"] for m in history["val_metrics"]]
    ax.plot(epochs, train_r2, label="Train", marker="o", markersize=3, alpha=0.7)
    ax.plot(epochs, val_r2, label="Validation", marker="s", markersize=3, alpha=0.7)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("R² Score", fontweight="bold")
    ax.set_title("Coefficient of Determination (R²)", fontweight="bold", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", linestyle="--", alpha=0.3)

    # PLOT 3: MAE
    ax = axes[1, 0]
    train_mae = [m["mae"] for m in history["train_metrics"]]
    val_mae = [m["mae"] for m in history["val_metrics"]]
    ax.plot(epochs, train_mae, label="Train", marker="o", markersize=3, alpha=0.7)
    ax.plot(epochs, val_mae, label="Validation", marker="s", markersize=3, alpha=0.7)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("MAE", fontweight="bold")
    ax.set_title("Mean Absolute Error", fontweight="bold", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    # PLOT 4: Learning rate schedule
    ax = axes[1, 1]
    ax.plot(epochs, history["lr"], color="purple", marker="o", markersize=3)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Learning Rate", fontweight="bold")
    ax.set_title("Learning Rate Schedule", fontweight="bold", fontsize=14)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved training history plot: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str = "Test",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5),
):
    """
    Plot predictions vs true values.

    Creates 2 subplots:
    - Left: Scatter plot (predicted vs true)
    - Right: Time series (first 500 timesteps)

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        True values
    y_pred : np.ndarray, shape (n_samples,)
        Predicted values
    split_name : str, default='Test'
        Dataset split name (for title)
    save_path : Path, optional
        Path to save figure
    figsize : tuple, default=(12, 5)
        Figure size

    Example
    -------
    >>> y_pred, y_true = trainer.predict(test_loader)
    >>> plot_predictions(y_true, y_pred, split_name='Test', save_path='figures/predictions.png')
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # PLOT 1: Scatter plot (predicted vs true)
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, edgecolor="none")

    # Add diagonal line (perfect predictions)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot(
        [min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect fit"
    )

    # Compute R²
    from sklearn.metrics import r2_score, mean_squared_error

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    ax.set_xlabel("True Values", fontweight="bold")
    ax.set_ylabel("Predicted Values", fontweight="bold")
    ax.set_title(
        f"{split_name} Set: Predictions vs True\nR²={r2:.4f}, RMSE={rmse:.4f}",
        fontweight="bold",
        fontsize=12,
    )
    ax.legend()
    ax.grid(alpha=0.3)

    # PLOT 2: Time series (first 500 samples)
    ax = axes[1]
    n_display = min(500, len(y_true))
    timesteps = np.arange(n_display)

    ax.plot(timesteps, y_true[:n_display], label="True", alpha=0.7, linewidth=1.5)
    ax.plot(timesteps, y_pred[:n_display], label="Predicted", alpha=0.7, linewidth=1.5)
    ax.fill_between(timesteps, y_true[:n_display], y_pred[:n_display], alpha=0.2)

    ax.set_xlabel("Timestep", fontweight="bold")
    ax.set_ylabel("Value", fontweight="bold")
    ax.set_title(
        f"{split_name} Set: Time Series (first {n_display} points)",
        fontweight="bold",
        fontsize=12,
    )
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved predictions plot: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5),
):
    """
    Plot residual diagnostics.

    Creates 2 subplots:
    - Left: Residuals vs predicted values
    - Right: Residual histogram with normal distribution overlay

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    save_path : Path, optional
        Path to save figure
    figsize : tuple, default=(12, 5)
        Figure size
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # PLOT 1: Residuals vs predicted
    ax = axes[0]
    ax.scatter(y_pred, residuals, alpha=0.3, s=10, edgecolor="none")
    ax.axhline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted Values", fontweight="bold")
    ax.set_ylabel("Residuals (True - Predicted)", fontweight="bold")
    ax.set_title(
        "Residual Plot\n(should be randomly scattered around 0)", fontweight="bold"
    )
    ax.grid(alpha=0.3)

    # PLOT 2: Residual distribution
    ax = axes[1]
    ax.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor="black")

    # Overlay normal distribution
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    from scipy.stats import norm

    ax.plot(
        x,
        norm.pdf(x, mu, sigma),
        "r-",
        linewidth=2,
        label=f"Normal(μ={mu:.3f}, σ={sigma:.3f})",
    )

    ax.set_xlabel("Residual Value", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title(
        "Residual Distribution\n(should be approximately normal)", fontweight="bold"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved residuals plot: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot feature importance (from permutation importance or similar).

    Parameters
    ----------
    feature_names : list of str
        Names of features
    importances : np.ndarray, shape (n_features,)
        Importance scores
    save_path : Path, optional
        Path to save figure
    figsize : tuple, default=(10, 6)
        Figure size
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_features)))

    ax.barh(
        range(len(sorted_features)), sorted_importances, color=colors, edgecolor="black"
    )
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel("Importance Score", fontweight="bold")
    ax.set_title("Feature Importance", fontweight="bold", fontsize=14)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved feature importance plot: {save_path}")
    else:
        plt.show()

    plt.close()


def set_plot_style():
    """Set consistent matplotlib style for all plots."""
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
