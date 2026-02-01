"""
Nonlinear State Space Model (NSSM) for Shark Population Dynamics

Mathematical Formulation
========================

State Transition (Stochastic):
    z_t = f_θ(z_t-1, X_t) + ε_t    where ε_t ~ N(0, σ²_z)

Observation Model:
    Y_t = g_φ(z_t, X_t) + η_t      where η_t ~ N(0, σ²_y)

Architecture:
    - f_θ: LSTM + ResNet encoder (temporal + residual connections)
    - g_φ: ResNet decoder (skip connections for gradient flow)
    - X_t: External covariates (SST, fishing effort, prey density)
    - z_t: Latent state (ecological dynamics)
    - Y_t: Observations (shark presence/density)

Training Objective (MSE + Regularization):
    L(θ, φ) = (1/N) Σ_t ||Y_t - Ŷ_t||² + λ₁||θ||² + λ₂KL(q||p)

    where Ŷ_t = g_φ(f_θ(z_t-1, X_t), X_t)

Optimizations:
    - SiLU (Swish) activation: Better for continuous stochastic regimes
    - ResNet blocks: Skip connections preserve weak pre-collapse signals
    - Learnable noise injection: Gaussian noise with trainable σ
    - Monte Carlo rollout: N forward passes → trajectory ensemble
    - Gradient stability: Anomaly detection, NaN/Inf hooks, global clipping

Reference: Durbin & Koopman (2012) - Time Series Analysis by State Space Methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import math
import warnings


class ResNetBlock(nn.Module):
    """
    Residual block with SiLU activation and skip connection.

    Design rationale:
    - Shallower blocks for ecological data (not ImageNet scale)
    - Skip connections preserve weak pre-collapse signals
    - SiLU (Swish) better for continuous + noisy regimes
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),  # Smooth activation, fewer dead neurons
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        return self.norm(x + self.block(x))


class LearnableGaussianNoise(nn.Module):
    """
    Learnable Gaussian noise injection.

    σ is a trainable parameter, allowing the model to learn
    optimal noise levels for different regimes (persistence vs collapse).
    """

    def __init__(self, initial_sigma: float = 0.01):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(math.log(initial_sigma)))

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Inject Gaussian noise during training."""
        if not training:
            return x
        sigma = torch.exp(self.log_sigma).clamp(min=1e-6, max=0.5)  # Stability
        noise = torch.randn_like(x) * sigma
        return x + noise

    @property
    def sigma(self) -> float:
        """Current noise level."""
        return torch.exp(self.log_sigma).item()


class NaNInfHook:
    """Hook to detect NaN/Inf in forward and backward passes."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, module, input, output):
        """Check for NaN/Inf in output."""
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                raise RuntimeError(f"NaN detected in {self.name} forward pass")
            if torch.isinf(output).any():
                raise RuntimeError(f"Inf detected in {self.name} forward pass")
        elif isinstance(output, tuple):
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    if torch.isnan(o).any():
                        raise RuntimeError(
                            f"NaN detected in {self.name} forward pass (output {i})"
                        )
                    if torch.isinf(o).any():
                        raise RuntimeError(
                            f"Inf detected in {self.name} forward pass (output {i})"
                        )


class NSSM(nn.Module):
    """
    Nonlinear State Space Model with ResNet backbone and stochastic features.

    Enhanced Architecture:
    - ResNet blocks with SiLU activation (smoother gradients)
    - Learnable Gaussian noise injection (controlled stochasticity)
    - Skip connections (preserve weak signals)
    - Monte Carlo rollout capability (trajectory ensembles)
    - NaN/Inf detection hooks (stability)

    Parameters
    ----------
    input_dim : int
        Number of input features (covariates X_t)
    hidden_dim : int
        Dimension of latent state z_t
    output_dim : int
        Dimension of observations Y_t (typically 1 for binary/count)
    num_layers : int, default=2
        Number of LSTM layers in state transition
    num_resnet_blocks : int, default=2
        Number of ResNet blocks in decoder
    dropout : float, default=0.1
        Dropout probability (only in latent layers, not output)
    noise_sigma : float, default=0.01
        Initial noise injection level (learnable)
    use_skip_connections : bool, default=True
        Enable long skip connections in decoder
    bidirectional : bool, default=False
        Whether to use bidirectional LSTM (not recommended for forecasting)
    enable_nan_hooks : bool, default=False
        Enable NaN/Inf detection hooks (slow, use for debugging)

    Attributes
    ----------
    f_theta : nn.LSTM
        State transition function (recurrent network)
    noise_injector : LearnableGaussianNoise
        Trainable noise injection module
    g_phi_resnet : nn.ModuleList
        ResNet blocks for observation decoder
    g_phi_output : nn.Linear
        Final output projection
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        num_layers: int = 2,
        num_resnet_blocks: int = 2,
        dropout: float = 0.1,
        noise_sigma: float = 0.01,
        use_skip_connections: bool = True,
        bidirectional: bool = False,
        enable_nan_hooks: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_resnet_blocks = num_resnet_blocks
        self.bidirectional = bidirectional
        self.use_skip_connections = use_skip_connections
        self.enable_nan_hooks = enable_nan_hooks

        # State transition function: z_t = f_θ(z_{t-1}, X_t)
        self.f_theta = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Dimension adjustment for bidirectional LSTM
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Learnable Gaussian noise injection
        self.noise_injector = LearnableGaussianNoise(initial_sigma=noise_sigma)

        # Observation function: Y_t = g_φ(z_t, X_t) with ResNet blocks
        # Input projection: concatenate latent state with covariates
        decoder_input_dim = lstm_output_dim + input_dim
        self.g_phi_input_proj = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.SiLU(),
        )

        # ResNet blocks (shallower for ecological data)
        self.g_phi_resnet = nn.ModuleList(
            [ResNetBlock(hidden_dim, dropout=dropout) for _ in range(num_resnet_blocks)]
        )

        # Output projection (no dropout on output layer)
        self.g_phi_output = nn.Linear(hidden_dim, output_dim)

        # Long skip connection from input to output (optional)
        if use_skip_connections:
            self.skip_projection = nn.Linear(decoder_input_dim, output_dim)
        else:
            self.skip_projection = None

        # Initialize weights using Xavier/Glorot initialization
        self._init_weights()

        # Register NaN/Inf hooks if enabled
        if enable_nan_hooks:
            self._register_nan_hooks()

    def _init_weights(self):
        """Initialize network weights for stable training."""
        for name, param in self.named_parameters():
            if param.dim() < 2:
                # Skip 1D tensors (biases, log_sigma, etc.)
                # Xavier/Orthogonal only work with 2D+ tensors
                if "bias" in name:
                    nn.init.constant_(param.data, 0)
                # log_sigma keeps its initialization from LearnableGaussianNoise
                continue
            elif "weight_ih" in name:  # Input-to-hidden weights (LSTM)
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:  # Hidden-to-hidden weights (LSTM)
                nn.init.orthogonal_(param.data)
            elif "weight" in name:  # Fully connected layers (2D+)
                nn.init.xavier_uniform_(param.data)

    def _register_nan_hooks(self):
        """Register forward hooks to detect NaN/Inf."""
        self.f_theta.register_forward_hook(NaNInfHook("f_theta"))
        for i, block in enumerate(self.g_phi_resnet):
            block.register_forward_hook(NaNInfHook(f"resnet_block_{i}"))
        self.g_phi_output.register_forward_hook(NaNInfHook("g_phi_output"))

    def forward(
        self,
        X: torch.Tensor,
        h0: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_hidden: bool = False,
        inject_noise: bool = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through enhanced NSSM.

        Parameters
        ----------
        X : torch.Tensor, shape (batch, seq_len, input_dim)
            Input covariates (SST, fishing, prey density, etc.)
        h0 : tuple of torch.Tensor, optional
            Initial hidden state (h_0, c_0) for LSTM
            If None, initialized to zeros
        return_hidden : bool, default=False
            Whether to return final hidden state (for multi-step forecasting)
        inject_noise : bool, optional
            Whether to inject learnable noise. If None, uses self.training

        Returns
        -------
        Y_pred : torch.Tensor, shape (batch, seq_len, output_dim)
            Predicted observations
        hidden : tuple of torch.Tensor, optional
            Final hidden state (h_T, c_T) if return_hidden=True

        Mathematical Operations
        -----------------------
        1. Noise injection (training only):
           X̃_t = X_t + ε_t,  ε_t ~ N(0, σ²)

        2. State transition:
           h_t, c_t = LSTM(X̃_t, (h_{t-1}, c_{t-1}))

        3. Observation prediction (ResNet decoder):
           h̃ = InputProj([h_t; X_t])
           h̃ = ResNetBlock₁(h̃) + h̃  (skip)
           h̃ = ResNetBlock₂(h̃) + h̃  (skip)
           Ŷ_t = OutputProj(h̃) + SkipProj([h_t; X_t])  (long skip)
        """
        batch_size, seq_len, _ = X.shape

        # Physical constraint validation
        self._validate_input(X)

        # Step 1: Learnable noise injection (training only)
        if inject_noise is None:
            inject_noise = self.training
        X_noisy = self.noise_injector(X, training=inject_noise)

        # Step 2: State transition via LSTM
        # z_t = f_θ(z_{t-1}, X̃_t)
        z, hidden = self.f_theta(X_noisy, h0)  # z: (batch, seq_len, hidden_dim)

        # Step 3: Concatenate latent states with inputs (original, not noisy)
        # This allows the decoder to use both learned dynamics and direct covariate info
        z_with_context = torch.cat(
            [z, X], dim=-1
        )  # (batch, seq_len, hidden_dim + input_dim)

        # Step 4: Observation prediction via ResNet decoder
        # Input projection
        h = self.g_phi_input_proj(z_with_context)

        # ResNet blocks with skip connections
        for block in self.g_phi_resnet:
            h = block(h)

        # Output projection
        Y_pred = self.g_phi_output(h)

        # Long skip connection from input to output (preserves weak signals)
        if self.skip_projection is not None:
            skip = self.skip_projection(z_with_context)
            Y_pred = Y_pred + skip

        if return_hidden:
            return Y_pred, hidden
        return Y_pred

    def _validate_input(self, X: torch.Tensor):
        """Validate input tensor for physical plausibility."""
        # Check shape
        assert X.ndim == 3, f"Expected 3D tensor (batch, seq, features), got {X.shape}"
        assert (
            X.shape[-1] == self.input_dim
        ), f"Expected {self.input_dim} features, got {X.shape[-1]}"

        # Check dtype
        assert X.dtype in [
            torch.float32,
            torch.float64,
        ], f"Expected float tensor, got {X.dtype}"

        # Check for NaN/Inf
        if torch.isnan(X).any():
            warnings.warn("NaN detected in input X (will be masked during training)")
        if torch.isinf(X).any():
            raise ValueError("Inf detected in input X - check normalization")

        # Physical range check (assuming normalized features)
        # Extreme values indicate normalization issues
        if X.abs().max() > 10:
            warnings.warn(
                f"Extreme input values detected: max={X.abs().max():.2f} (check normalization)"
            )

    def predict_step(
        self,
        X_t: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        inject_noise: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single-step prediction (for autoregressive forecasting).

        Parameters
        ----------
        X_t : torch.Tensor, shape (batch, 1, input_dim)
            Covariates at time t
        hidden : tuple of torch.Tensor
            Previous hidden state (h_{t-1}, c_{t-1})
        inject_noise : bool, default=False
            Whether to inject noise during forecasting (for Monte Carlo)

        Returns
        -------
        Y_t : torch.Tensor, shape (batch, 1, output_dim)
            Prediction at time t
        hidden_t : tuple of torch.Tensor
            Updated hidden state (h_t, c_t)
        """
        return self.forward(
            X_t, h0=hidden, return_hidden=True, inject_noise=inject_noise
        )

    def monte_carlo_forecast(
        self,
        X: torch.Tensor,
        n_steps: int,
        n_samples: int = 100,
        initial_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        inject_noise: bool = True,
        apply_sigmoid: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Monte Carlo forecasting for trajectory ensembles.

        This is the core method for counterfactual analysis: generate
        multiple stochastic trajectories to quantify uncertainty.

        Parameters
        ----------
        X : torch.Tensor, shape (batch, n_steps, input_dim)
            Future covariates (SST scenarios, fishing policies, etc.)
        n_steps : int
            Number of steps to forecast
        n_samples : int, default=100
            Number of Monte Carlo samples (trajectories)
        initial_hidden : tuple of torch.Tensor, optional
            Initial hidden state from historical context
        inject_noise : bool, default=True
            Whether to inject learnable noise (should be True for MC)
        apply_sigmoid : bool, default=True
            Whether to apply sigmoid activation for classification tasks.
            Set to True when using Focal Loss or BCE (model outputs logits).
            Set to False for regression tasks or if output is already in [0,1].

        Returns
        -------
        ensemble : torch.Tensor, shape (n_samples, batch, n_steps, output_dim)
            Ensemble of forecasted trajectories (probabilities if apply_sigmoid=True)
        stats : dict
            Statistical summaries:
            - 'mean': Mean trajectory
            - 'std': Standard deviation (uncertainty)
            - 'quantile_05': 5th percentile
            - 'quantile_25': 25th percentile (Q1)
            - 'quantile_50': 50th percentile (median)
            - 'quantile_75': 75th percentile (Q3)
            - 'quantile_95': 95th percentile
            - 'collapse_prob': P(Y_t < threshold)
            - 'entropy': Temporal entropy (regime uncertainty)

        Algorithm
        ---------
        For each sample s = 1 to N:
            1. Sample noise: ε ~ N(0, σ²)
            2. Forecast: Ŷ_t^(s) = g_φ(f_θ(z_{t-1}, X_t + ε_t), X_t)
            3. Apply σ(·) if classification: p_t = σ(Ŷ_t)
            4. Store trajectory

        Compute statistics across samples:
            - Mean: E[p_t]
            - Variance: Var[p_t]
            - Quantiles: Q_α[p_t]

        Use Cases
        ---------
        - Output N1: Trajectory ensemble visualization (fan charts)
        - Output N2: P_surv = P(Y_t > threshold)
        - Output N3: T_collapse = min{t : Y_t < threshold}
        - Output N4: Entropy = H(Y_t) over time
        - Output N5: Sensitivity = dY/dX via finite differences

        Notes
        -----
        CRITICAL: When training with Focal Loss or BCE, the model outputs
        logits z_t ∈ (-∞, +∞). For inference and Monte Carlo forecasting,
        these must be converted to probabilities p_t = σ(z_t) ∈ [0, 1].

        This transformation ensures:
        - Valid probability distributions (sum to 1)
        - Correct collapse probability P(Y_t < τ)
        - Proper entropy calculation H(p_t)
        - Realistic mean predictions (e.g., 0.04 for 0.05% prevalence)

        Without sigmoid, logits yield:
        - Negative mean predictions (e.g., -3.01)
        - Saturated collapse probabilities (e.g., 1.00)
        - Zero entropy (no uncertainty)
        """
        batch_size = X.shape[0]
        device = X.device

        # Storage for trajectories
        ensemble = []

        # Monte Carlo loop
        for _ in range(n_samples):
            # Single stochastic forecast
            Y_forecast = self.forecast(
                X=X,
                n_steps=n_steps,
                initial_hidden=initial_hidden,
                inject_noise=inject_noise,
            )

            # CRITICAL FIX: Apply sigmoid for classification tasks
            # When using Focal Loss or BCE, model outputs logits that must
            # be converted to probabilities [0,1] for inference
            if apply_sigmoid and self.output_dim == 1:
                Y_forecast = torch.sigmoid(Y_forecast)

            ensemble.append(Y_forecast)

        # Stack: (n_samples, batch, n_steps, output_dim)
        ensemble = torch.stack(ensemble, dim=0)

        # Compute statistics across samples (dim=0)
        stats = {
            "mean": ensemble.mean(dim=0),  # (batch, n_steps, output_dim)
            "std": ensemble.std(dim=0),  # Uncertainty
            "quantile_05": torch.quantile(ensemble, 0.05, dim=0),
            "quantile_25": torch.quantile(ensemble, 0.25, dim=0),
            "quantile_50": torch.quantile(ensemble, 0.50, dim=0),  # Median
            "quantile_75": torch.quantile(ensemble, 0.75, dim=0),
            "quantile_95": torch.quantile(ensemble, 0.95, dim=0),
        }

        # Collapse probability (assuming binary outcome, threshold=0.5)
        # P(collapse) = P(Y_t < 0.5)
        threshold = 0.5
        stats["collapse_prob"] = (ensemble < threshold).float().mean(dim=0)
        stats["persistence_prob"] = (ensemble >= threshold).float().mean(dim=0)

        # Temporal entropy (uncertainty about regime at each timestep)
        # H = -p*log(p) - (1-p)*log(1-p)
        p = stats["persistence_prob"]
        p_safe = p.clamp(min=1e-6, max=1 - 1e-6)  # Avoid log(0)
        entropy = -(p_safe * torch.log(p_safe) + (1 - p_safe) * torch.log(1 - p_safe))
        stats["entropy"] = entropy

        return ensemble, stats

    def forecast(
        self,
        X: torch.Tensor,
        n_steps: int,
        initial_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        inject_noise: bool = False,
    ) -> torch.Tensor:
        """
        Multi-step ahead forecasting.

        Parameters
        ----------
        X : torch.Tensor, shape (batch, n_steps, input_dim)
            Future covariates (SST scenarios, fishing policies, etc.)
        n_steps : int
            Number of steps to forecast
        initial_hidden : tuple of torch.Tensor, optional
            Initial hidden state from historical context
        inject_noise : bool, default=False
            Whether to inject noise during forecasting

        Returns
        -------
        Y_forecast : torch.Tensor, shape (batch, n_steps, output_dim)
            Forecasted observations

        Algorithm
        ---------
        For t = 1 to T:
            1. Predict: Ŷ_t = g_φ(z_{t-1}, X_t)
            2. Update state: z_t = f_θ(z_{t-1}, X_t)
            3. Store prediction

        This is the foundation for counterfactual analysis:
        - Modify X (e.g., reduce fishing effort by 50%)
        - Run forecast to see impact on shark dynamics
        """
        batch_size = X.shape[0]
        predictions = []

        # Initialize hidden state
        hidden = initial_hidden
        if hidden is None:
            h0 = torch.zeros(
                self.num_layers * (2 if self.bidirectional else 1),
                batch_size,
                self.hidden_dim,
                device=X.device,
            )
            c0 = torch.zeros_like(h0)
            hidden = (h0, c0)

        # Autoregressive forecasting
        for t in range(n_steps):
            X_t = X[:, t : t + 1, :]  # Current covariates (batch, 1, input_dim)
            Y_t, hidden = self.predict_step(X_t, hidden, inject_noise=inject_noise)
            predictions.append(Y_t)

        return torch.cat(predictions, dim=1)  # (batch, n_steps, output_dim)

    def compute_loss(
        self,
        Y_pred: torch.Tensor,
        Y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_decomposed: bool = False,
        loss_type: str = "focal",  # NEW: "mse", "bce", "focal"
        focal_alpha: float = 0.25,  # NEW: Focal loss weight for positive class
        focal_gamma: float = 2.0,  # NEW: Focal loss focusing parameter
        pos_weight: Optional[float] = None,  # NEW: Weight for positive class (BCE)
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Compute decomposed loss with optional masking and configurable loss function.

        Parameters
        ----------
        Y_pred : torch.Tensor, shape (batch, seq_len, output_dim)
            Predicted observations (logits for BCE/Focal, probabilities for MSE)
        Y_true : torch.Tensor, shape (batch, seq_len, output_dim)
            True observations (binary labels)
        mask : torch.Tensor, shape (batch, seq_len), optional
            Binary mask for missing data (1=valid, 0=missing)
        return_decomposed : bool, default=False
            Whether to return decomposed loss components
        loss_type : str, default="focal"
            Loss function type: "mse", "bce", "focal"
        focal_alpha : float, default=0.25
            Focal loss alpha (weight for positive class)
        focal_gamma : float, default=2.0
            Focal loss gamma (focusing parameter, higher = focus more on hard examples)
        pos_weight : float, optional
            Weight multiplier for positive class in BCE loss

        Returns
        -------
        loss : torch.Tensor or dict
            If return_decomposed=False: Total loss (scalar)
            If return_decomposed=True: Dictionary with components

        Loss Functions
        --------------
        MSE: L = (Y_pred - Y_true)²
            - Good for: Regression, balanced data
            - Bad for: Extreme class imbalance (learns to predict majority class)

        BCE: L = -[Y*log(p) + (1-Y)*log(1-p)] * pos_weight
            - Good for: Binary classification, moderate imbalance
            - Bad for: Extreme imbalance (still biased toward majority)

        Focal Loss: L = -α*(1-p_t)^γ * log(p_t)
            - Good for: Extreme class imbalance (10,000:1)
            - Focuses on hard-to-classify examples
            - α controls class balance, γ controls focus on hard examples
        """
        if loss_type == "mse":
            # Original MSE loss (keep for backward compatibility)
            squared_error = (Y_pred - Y_true) ** 2
            if mask is not None:
                # Ensure mask has same shape as squared_error
                if mask.dim() == 2:
                    mask = mask.unsqueeze(-1)  # (batch, seq) → (batch, seq, 1)
                # mask is now (batch, seq, 1)
                masked_error = squared_error * mask
                loss_dynamics = masked_error.sum() / mask.sum().clamp(min=1)
            else:
                loss_dynamics = squared_error.mean()

        elif loss_type == "bce":
            # Weighted Binary Cross-Entropy Loss
            # Expects Y_pred as logits (NOT probabilities)
            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                Y_pred, Y_true, reduction="none"
            )
            if pos_weight is not None:
                # Apply pos_weight to positive examples
                weight_tensor = torch.where(Y_true == 1, pos_weight, 1.0)
                bce_loss = bce_loss * weight_tensor

            if mask is not None:
                # Ensure mask has same shape as bce_loss
                if mask.dim() == 2:
                    mask = mask.unsqueeze(-1)  # (batch, seq) → (batch, seq, 1)
                # mask is now (batch, seq, 1)
                masked_loss = bce_loss * mask
                loss_dynamics = masked_loss.sum() / mask.sum().clamp(min=1)
            else:
                loss_dynamics = bce_loss.mean()

        elif loss_type == "focal":
            # Focal Loss for extreme class imbalance
            # Formula: FL = -α_t * (1 - p_t)^γ * log(p_t)
            # where p_t = p if y=1, else 1-p

            # Get probabilities from logits (or use directly if already probs)
            if Y_pred.min() < 0 or Y_pred.max() > 1:
                # Logits detected, convert to probabilities
                Y_prob = torch.sigmoid(Y_pred)
            else:
                Y_prob = Y_pred

            # Compute p_t: probability of true class
            p_t = Y_prob * Y_true + (1 - Y_prob) * (1 - Y_true)

            # Focal modulation factor: (1 - p_t)^γ
            # This down-weights easy examples (p_t close to 1)
            focal_weight = (1 - p_t) ** focal_gamma

            # Alpha balancing factor
            # α for positive class, (1-α) for negative class
            alpha_t = focal_alpha * Y_true + (1 - focal_alpha) * (1 - Y_true)

            # Compute focal loss
            # FL = -α_t * (1-p_t)^γ * log(p_t)
            focal_loss = -alpha_t * focal_weight * torch.log(p_t.clamp(min=1e-7))

            # Apply mask (mask is already (batch, seq, 1), no need to unsqueeze)
            if mask is not None:
                # Ensure mask has same shape as focal_loss
                if mask.dim() == 2:
                    # (batch, seq) → (batch, seq, 1)
                    mask = mask.unsqueeze(-1)
                # mask is now (batch, seq, 1), same as focal_loss
                masked_loss = focal_loss * mask
                loss_dynamics = masked_loss.sum() / mask.sum().clamp(min=1)
            else:
                loss_dynamics = focal_loss.mean()

        else:
            raise ValueError(
                f"Unknown loss_type: {loss_type}. Choose 'mse', 'bce', or 'focal'"
            )

        if not return_decomposed:
            return loss_dynamics

        # Decomposed loss components
        loss_dict = {
            "loss_dynamics": loss_dynamics,
            "loss_regularization": torch.tensor(
                0.0, device=Y_pred.device
            ),  # Handled by optimizer
            "loss_stochastic": torch.tensor(
                0.0, device=Y_pred.device
            ),  # For future DMM
        }
        loss_dict["loss_total"] = sum(loss_dict.values())

        return loss_dict

    def get_latent_states(
        self,
        X: torch.Tensor,
        h0: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Extract learned latent states z_t for visualization/analysis.

        Parameters
        ----------
        X : torch.Tensor, shape (batch, seq_len, input_dim)
            Input covariates
        h0 : tuple of torch.Tensor, optional
            Initial hidden state

        Returns
        -------
        z : torch.Tensor, shape (batch, seq_len, hidden_dim)
            Latent state trajectories

        Use Case
        --------
        Analyze what ecological patterns the model has learned:
        - PCA/t-SNE visualization of z_t
        - Correlation with environmental drivers
        - Regime detection (clustering of z_t)
        """
        with torch.no_grad():
            z, _ = self.f_theta(X, h0)
        return z


class NSSMConfig:
    """
    Enhanced configuration container for NSSM hyperparameters.

    Includes ResNet, stochastic features, and stability controls.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 2,
        num_resnet_blocks: int = 2,
        dropout: float = 0.1,
        noise_sigma: float = 0.01,
        use_skip_connections: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 32,
        max_epochs: int = 200,
        patience: int = 20,
        grad_clip: float = 1.0,
        enable_anomaly_detection: bool = False,
        enable_nan_hooks: bool = False,
        mc_samples: int = 100,
    ):
        """
        Parameters
        ----------
        input_dim : int
            Number of input features
        hidden_dim : int
            Latent state dimension
        output_dim : int
            Output dimension (1 for binary shark presence)
        num_layers : int
            LSTM layers
        num_resnet_blocks : int
            Number of ResNet blocks in decoder (2-3 recommended)
        dropout : float
            Dropout probability (latent layers only)
        noise_sigma : float
            Initial Gaussian noise level (learnable)
        use_skip_connections : bool
            Enable long skip connections (preserves weak signals)
        learning_rate : float
            Adam learning rate
        weight_decay : float
            L2 regularization strength
        batch_size : int
            Training batch size
        max_epochs : int
            Maximum training epochs
        patience : int
            Early stopping patience
        grad_clip : float
            Gradient clipping threshold (global norm)
        enable_anomaly_detection : bool
            Enable torch.autograd.set_detect_anomaly (slow, for debugging)
        enable_nan_hooks : bool
            Enable NaN/Inf detection hooks (slow, for debugging)
        mc_samples : int
            Number of Monte Carlo samples for ensemble forecasting
        """
        # Model architecture
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_resnet_blocks = num_resnet_blocks
        self.dropout = dropout
        self.noise_sigma = noise_sigma
        self.use_skip_connections = use_skip_connections

        # Training
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip

        # Stability
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_nan_hooks = enable_nan_hooks

        # Monte Carlo
        self.mc_samples = mc_samples

    def to_dict(self) -> Dict:
        """Convert config to dictionary for logging."""
        return {k: v for k, v in self.__dict__.items()}

    def validate(self):
        """Validate configuration parameters."""
        assert self.input_dim > 0, "input_dim must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.output_dim > 0, "output_dim must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert self.num_resnet_blocks >= 0, "num_resnet_blocks must be non-negative"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        assert self.noise_sigma > 0, "noise_sigma must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.max_epochs > 0, "max_epochs must be positive"
        assert self.patience > 0, "patience must be positive"
        assert self.grad_clip > 0, "grad_clip must be positive"
        assert self.mc_samples > 0, "mc_samples must be positive"

        # Warnings for suboptimal configs
        if self.num_resnet_blocks > 4:
            warnings.warn("num_resnet_blocks > 4 may be excessive for ecological data")
        if self.noise_sigma > 0.1:
            warnings.warn("noise_sigma > 0.1 may inject too much noise")
        if self.mc_samples < 50:
            warnings.warn("mc_samples < 50 may underestimate uncertainty")
