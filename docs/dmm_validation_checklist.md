# DMM Validation Checklist: "À Prova de Revisor Chato"

**Project**: Nova Selachiia - Counterfactual Analysis of Shark Population Collapse  
**Phase**: Pre-DMM + DMM Implementation  
**Purpose**: Surgical validation to ensure robustness before and after Deep Markov Model

---

## Part 1: Pre-DMM NSSM Robustness (Before Touching DMM Code)

### 1.1. Threshold Replacement ⚠️ **CRITICAL**

**Problem**: Fixed $\tau = 0.5$ is absurd for 0.05% prevalence

**Solution**: Prevalence-based adaptive threshold

```python
# Compute adaptive threshold from baseline predictions
tau_adaptive = np.quantile(y_pred_baseline, q=0.05)  # 5th percentile

# Recompute metrics at adaptive threshold
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true=y_test,
    y_pred=(y_pred > tau_adaptive).astype(int),
    average='binary',
    zero_division=0
)
```

**Expected**:
- `tau_adaptive` ≈ 0.01-0.05 (not 0.5)
- F1 > 0.01 (was 0.0076 at $\tau=0.5$)
- Precision/Recall balanced (not all zeros)

**Checklist**:
- [ ] Compute `tau_adaptive` from baseline model
- [ ] Recompute F1, Precision, Recall
- [ ] Update notebook cell "7.3. Classification Metrics"
- [ ] Add visualization: ROC curve + optimal threshold marker

---

### 1.2. Risk Decile Analysis (Not Global Accuracy)

**Problem**: Global accuracy is meaningless for 0.05% prevalence

**Solution**: Stratified analysis by risk decile

```python
# Bin predictions into deciles
deciles = pd.qcut(y_pred, q=10, labels=False, duplicates='drop')

# Compute per-decile statistics
decile_stats = []
for d in range(10):
    mask = (deciles == d)
    decile_stats.append({
        'decile': d,
        'n_samples': mask.sum(),
        'observed_rate': y_test[mask].mean(),
        'predicted_mean': y_pred[mask].mean(),
        'brier_score': ((y_pred[mask] - y_test[mask])**2).mean(),
    })

df_deciles = pd.DataFrame(decile_stats)

# Expected: Top decile has >>10x prevalence vs bottom
discrimination_ratio = df_deciles.loc[9, 'observed_rate'] / df_deciles.loc[0, 'observed_rate']
assert discrimination_ratio > 10, f"Poor discrimination: {discrimination_ratio:.1f}x"
```

**Expected**:
- Decile 0 (lowest risk): Observed rate ≈ 0.001% (near zero)
- Decile 9 (highest risk): Observed rate ≈ 0.1-1% (10-100x higher)
- Monotonic trend: Higher decile → higher observed rate

**Checklist**:
- [ ] Create decile analysis function
- [ ] Plot: Decile bar chart (observed vs predicted)
- [ ] Compute discrimination ratio (top/bottom)
- [ ] Add to notebook: "7.4. Risk Stratification"

---

### 1.3. Normalization by Baseline

**Problem**: MSE/Log-likelihood alone don't show improvement over trivial model

**Solution**: Normalize by "predict prevalence everywhere" baseline

```python
# Baseline model: predict mean prevalence
y_baseline = np.full_like(y_test, y_test.mean())

# Compute baseline metrics
mse_baseline = np.mean((y_baseline - y_test)**2)
ll_baseline = np.mean(y_test * np.log(y_baseline + 1e-8) + 
                      (1 - y_test) * np.log(1 - y_baseline + 1e-8))

# Compute NSSM metrics
mse_nssm = np.mean((y_pred - y_test)**2)
ll_nssm = np.mean(y_test * np.log(y_pred + 1e-8) + 
                  (1 - y_test) * np.log(1 - y_pred + 1e-8))

# Compute ratios (lower is better)
mse_ratio = mse_nssm / mse_baseline
ll_ratio = ll_nssm / ll_baseline  # Higher is better for LL

print(f"MSE Improvement: {(1 - mse_ratio)*100:.1f}%")
print(f"Log-Likelihood Improvement: {(ll_ratio - 1)*100:.1f}%")
```

**Expected**:
- `mse_ratio` < 0.9 (at least 10% better than baseline)
- `ll_ratio` > 1.1 (at least 10% better log-likelihood)
- If ratios ≈ 1.0: Model is no better than trivial baseline (red flag!)

**Checklist**:
- [ ] Implement baseline model
- [ ] Compute MSE/LL ratios
- [ ] Add to notebook: "7.5. Baseline Comparison"
- [ ] Table: NSSM vs Baseline (MSE, LL, ECE, Brier)

---

### 1.4. Monte Carlo Validation (Entropy & Quantile Collapse)

**Problem**: Monte Carlo might be degenerate (no uncertainty)

**Solution**: Validate ensemble spread and entropy

```python
# From Monte Carlo results (already computed)
# ensemble: (100, batch, 12, 1)
# stats: dict with 'mean', 'std', 'quantile_05', 'quantile_95', 'entropy'

# CHECK 1: Entropy > 0 (non-trivial uncertainty)
H = safe_extract(stats['entropy'], "entropy")
assert H.min() > 0, f"Zero entropy detected: {H.min()}"
assert H.mean() > 0.01, f"Too low entropy: {H.mean():.4f} bits"

# CHECK 2: Quantiles not collapsed (ensemble has spread)
q05 = safe_extract(stats['quantile_05'], "q05")
q95 = safe_extract(stats['quantile_95'], "q95")
spread = q95 - q05
assert spread.mean() > 0.05, f"Quantiles collapsed: {spread.mean():.4f}"

# CHECK 3: Mean stability across seeds
# Re-run Monte Carlo with different random seed
ensemble_v2, stats_v2 = model.monte_carlo_forecast(X_test, n_steps=12, n_samples=100)
mean_v1 = safe_extract(stats['mean'], "mean")
mean_v2 = safe_extract(stats_v2['mean'], "mean")
delta_mean = np.abs(mean_v1 - mean_v2).mean()
assert delta_mean < 0.01, f"Unstable mean: {delta_mean:.4f}"

print(f"✅ Entropy range: [{H.min():.3f}, {H.max():.3f}] bits")
print(f"✅ Quantile spread: {spread.mean():.3f} (mean)")
print(f"✅ Mean stability: Δ = {delta_mean:.4f}")
```

**Expected**:
- Entropy: $H \in [0.01, 0.5]$ bits (non-trivial, not saturated)
- Quantile spread: $Q_{0.95} - Q_{0.05} > 0.05$ (visible uncertainty)
- Mean stability: $|\Delta \bar{Y}| < 0.01$ across seeds

**Checklist**:
- [ ] Add entropy check to notebook
- [ ] Add quantile collapse check
- [ ] Re-run Monte Carlo with 3 different seeds
- [ ] Plot: Entropy time series, quantile spread over time

---

### 1.5. Stress Tests (Model Panic Detection)

**Problem**: Model might be overconfident or ignore covariates

**Solution**: Break inputs, watch model freak out

#### Test 1: Zero Covariates

```python
# Create zero covariate scenario
X_zeros = torch.zeros_like(X_test)

# Forecast with no information
ensemble_zeros, stats_zeros = model.monte_carlo_forecast(
    X_zeros, n_steps=12, n_samples=100
)

# Extract uncertainty
std_zeros = safe_extract(stats_zeros['std'], "std")
std_normal = safe_extract(stats['std'], "std")

# Expected: Higher uncertainty when covariates are missing
assert std_zeros.mean() > std_normal.mean(), \
    "Model doesn't panic when covariates are zero!"

print(f"Normal σ: {std_normal.mean():.3f}")
print(f"Zero-X σ: {std_zeros.mean():.3f} (should be higher)")
```

**Expected**: $\sigma_{\text{zero}} > 2 \times \sigma_{\text{normal}}$ (model panics)

#### Test 2: Noise Injection

```python
# Add 10% Gaussian noise to all covariates
X_noisy = X_test + 0.1 * X_test.std(dim=(0, 1)) * torch.randn_like(X_test)

# Forecast with noisy inputs
ensemble_noisy, stats_noisy = model.monte_carlo_forecast(
    X_noisy, n_steps=12, n_samples=100
)

# Compare predictions
mean_normal = safe_extract(stats['mean'], "mean")
mean_noisy = safe_extract(stats_noisy['mean'], "mean")

delta_pred = np.abs(mean_normal - mean_noisy).mean()
assert delta_pred > 0.01, "Model ignores noise in covariates!"

print(f"Prediction shift from noise: {delta_pred:.3f}")
```

**Expected**: $|\Delta \hat{Y}| > 0.01$ (model responds to noise)

#### Test 3: Extreme Values

```python
# Apply +10°C SST anomaly (climate catastrophe)
X_extreme = X_test.clone()
X_extreme[:, :, sst_idx] += 10.0  # 10°C warming

ensemble_extreme, stats_extreme = model.monte_carlo_forecast(
    X_extreme, n_steps=12, n_samples=100
)

# Extract predictions
mean_extreme = safe_extract(stats_extreme['mean'], "mean")
entropy_extreme = safe_extract(stats_extreme['entropy'], "entropy")

# Expected: Predictions saturate (near 0 or 1), entropy increases
assert (mean_extreme < 0.01).sum() > 0.5 * len(mean_extreme) or \
       (mean_extreme > 0.99).sum() > 0.5 * len(mean_extreme), \
    "Model doesn't saturate under extreme conditions!"

print(f"Extreme predictions: min={mean_extreme.min():.3f}, max={mean_extreme.max():.3f}")
print(f"Extreme entropy: {entropy_extreme.mean():.3f} bits")
```

**Expected**:
- Predictions saturate: $\hat{Y} < 0.01$ or $\hat{Y} > 0.99$ for majority
- Entropy spikes: $H_{\text{extreme}} > H_{\text{normal}}$

**Checklist**:
- [ ] Implement 3 stress tests
- [ ] Add to notebook: "8.3. Stress Tests"
- [ ] Plot: Normal vs Zero vs Noisy vs Extreme (4-panel)
- [ ] Assert: Model shows **visible panic** under stress

---

## Part 2: DMM-Specific Validations (During Implementation)

### 2.1. Posterior Collapse Detection ⚠️ **CRITICAL**

**Problem**: DMM without monitoring KL → fancy LSTM (latent ignored)

**Solution**: Track KL per timestep, per batch, with alerts

```python
class DMM(nn.Module):
    def forward(self, X, Y):
        # ... encoding logic ...
        
        # Compute KL per timestep
        kl_per_t = []
        for t in range(T):
            # KL between posterior q(z_t | Y, X) and prior p(z_t | z_{t-1}, X)
            kl_t = 0.5 * (
                (sigma_post[t]**2 / sigma_prior[t]**2) +
                ((mu_post[t] - mu_prior[t])**2 / sigma_prior[t]**2) -
                1 + torch.log(sigma_prior[t]**2 / sigma_post[t]**2)
            ).sum()
            kl_per_t.append(kl_t)
        
        kl_per_t = torch.stack(kl_per_t)
        kl_mean = kl_per_t.mean()
        
        # ALERT: Posterior collapse
        if kl_mean < 0.1:
            warnings.warn(f"⚠️  KL = {kl_mean:.3f} nats (possible collapse!)")
        
        return reconstruction, kl_per_t

# In training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        recon, kl_per_t = model(X, Y)
        
        # Track KL statistics
        kl_mean = kl_per_t.mean().item()
        kl_min = kl_per_t.min().item()
        kl_max = kl_per_t.max().item()
        
        history['kl_mean'].append(kl_mean)
        history['kl_min'].append(kl_min)
        history['kl_max'].append(kl_max)
        
        # Check for collapse (5 epochs in a row)
        if len(history['kl_mean']) > 50 and \
           np.mean(history['kl_mean'][-50:]) < 0.1:
            raise ValueError("POSTERIOR COLLAPSE DETECTED! Increase β schedule.")
```

**Detection Heuristics**:
- $\overline{\text{KL}} < 0.1$ nats for 5 consecutive epochs → **COLLAPSE**
- $\sigma_t^{\text{post}} / \sigma_t^{\text{prior}} < 0.01$ → posterior too confident
- $\text{KL}_t \to 0$ for all $t$ → latent ignored

**Prevention Strategies**:
1. **KL Floor**: $\text{KL}_{\text{clamped}} = \max(\text{KL}, 0.01)$
2. **Free Bits** (Kingma et al., 2016): 
   $$\mathcal{L}_{\text{KL}} = \sum_t \max(\text{KL}_t, \lambda_{\text{fb}})$$
   - $\lambda_{\text{fb}} \in [0.1, 0.5]$ nats
3. **Aggressive $\beta$ annealing**: Start $\beta = 0.001$, anneal over 30 epochs (not 20)

**Checklist**:
- [ ] Add KL tracking to DMM forward pass
- [ ] Plot KL per timestep (heatmap: epoch × time)
- [ ] Alert system in training loop
- [ ] Implement Free Bits regularization
- [ ] Test: Force collapse (β=0) → verify alert triggers

---

### 2.2. Uncertainty Decomposition (Output N6)

**Purpose**: Separate aleatoric (irreducible) from epistemic (reducible) uncertainty

```python
def decompose_uncertainty(model, X, n_samples=100):
    """
    Decompose total variance into 3 components:
    - Aleatoric (process noise): E[σ²_prior]
    - Epistemic (parameter uncertainty): Var[μ_post]
    - Observational (measurement noise): σ²_y
    """
    # Monte Carlo over latent states
    ensemble = []
    sigma_prior_samples = []
    mu_post_samples = []
    
    for _ in range(n_samples):
        # Forward pass with different noise realizations
        z, mu_post, sigma_prior, Y_pred = model.sample(X)
        ensemble.append(Y_pred)
        mu_post_samples.append(mu_post)
        sigma_prior_samples.append(sigma_prior)
    
    # Stack samples
    ensemble = torch.stack(ensemble, dim=0)  # (n_samples, batch, T, 1)
    mu_post = torch.stack(mu_post_samples, dim=0)
    sigma_prior = torch.stack(sigma_prior_samples, dim=0)
    
    # Compute variance components
    var_aleatoric = (sigma_prior**2).mean(dim=0)  # E[σ²_prior]
    var_epistemic = mu_post.var(dim=0)  # Var[μ_post]
    var_observational = model.decoder.sigma_y**2  # σ²_y (if heteroscedastic)
    
    var_total = ensemble.var(dim=0)
    
    # Verify decomposition (approximately)
    var_reconstructed = var_aleatoric + var_epistemic + var_observational
    assert torch.allclose(var_total, var_reconstructed, atol=0.01), \
        "Variance decomposition doesn't match!"
    
    return {
        'var_total': var_total,
        'var_aleatoric': var_aleatoric,
        'var_epistemic': var_epistemic,
        'var_observational': var_observational,
        'percent_aleatoric': (var_aleatoric / var_total * 100).mean().item(),
        'percent_epistemic': (var_epistemic / var_total * 100).mean().item(),
        'percent_observational': (var_observational / var_total * 100).mean().item(),
    }

# Usage
uncertainty = decompose_uncertainty(dmm_model, X_test, n_samples=100)

print(f"Uncertainty Breakdown:")
print(f"  Aleatoric (process):      {uncertainty['percent_aleatoric']:.1f}%")
print(f"  Epistemic (knowledge):    {uncertainty['percent_epistemic']:.1f}%")
print(f"  Observational (measure):  {uncertainty['percent_observational']:.1f}%")
```

**Expected**:
- Aleatoric: 50-70% (ecological systems are inherently noisy)
- Epistemic: 20-40% (limited data, model uncertainty)
- Observational: 5-15% (detection probability, sampling)

**Checklist**:
- [ ] Implement `decompose_uncertainty()` function
- [ ] Add variance outputs to DMM decoder
- [ ] Create visualization: Stacked bar chart (3 components)
- [ ] Add to notebook: "Output N6: Uncertainty Decomposition"

---

### 2.3. Temporal Counterfactual Alignment

**Problem**: Comparing S1 vs S2 without fixing initial state = "different worlds"

**Solution**: Fix historical encoding, only alter future

```python
def temporal_counterfactual(model, X_hist, Y_hist, X_future_cf, t0):
    """
    Two-phase counterfactual:
    1. Encode history (t=1 to t0) → fix z_t0
    2. Rollout future (t=t0+1 to T) with modified X_cf
    
    Args:
        X_hist: Historical covariates (batch, t0, input_dim)
        Y_hist: Historical observations (batch, t0, output_dim)
        X_future_cf: Counterfactual future (batch, T-t0, input_dim)
        t0: Intervention start time (e.g., Jan 2024)
    
    Returns:
        Y_cf: Counterfactual trajectory (batch, T, output_dim)
        z_t0: Fixed initial state (same for all scenarios)
    """
    # Phase 1: Historical encoding (FIXED for all scenarios)
    with torch.no_grad():  # Don't update encoder
        z_t0, mu_post, sigma_post = model.encode(X_hist, Y_hist)
    
    # Phase 2: Counterfactual rollout (SCENARIO-SPECIFIC)
    z_t = z_t0
    Y_cf = []
    
    for t in range(X_future_cf.shape[1]):
        # Prior transition: p(z_t | z_{t-1}, X_t^cf)
        mu_prior, sigma_prior = model.prior(z_t, X_future_cf[:, t])
        z_t = mu_prior + sigma_prior * torch.randn_like(sigma_prior)
        
        # Decode: p(Y_t | z_t)
        Y_t = model.decode(z_t)
        Y_cf.append(Y_t)
    
    Y_cf = torch.stack(Y_cf, dim=1)
    
    return Y_cf, z_t0

# Usage: Compare scenarios with SAME initial state
z_t0_shared = None

for scenario in ['S0_baseline', 'S1_fishing', 'S2_sst', 'S3_combined']:
    X_future = get_scenario_covariates(scenario, t_start=t0)
    
    Y_cf, z_t0 = temporal_counterfactual(
        model, X_hist, Y_hist, X_future, t0=t0
    )
    
    # Verify: Same initial state across scenarios
    if z_t0_shared is None:
        z_t0_shared = z_t0
    else:
        assert torch.allclose(z_t0, z_t0_shared, atol=1e-6), \
            "Initial states differ across scenarios!"
    
    results[scenario] = Y_cf
```

**Why This Matters**:
- **Without alignment**: S1 vs S2 compares "parallel universes" (no causal meaning)
- **With alignment**: "What if policy changed on Jan 2024?" (true counterfactual)

**Checklist**:
- [ ] Implement `temporal_counterfactual()` function
- [ ] Assert: Same $z_{t_0}$ for all scenarios (within 1e-6)
- [ ] Plot: Divergence point (t=t0) in fan charts
- [ ] Add to notebook: "Temporal Alignment Validation"

---

### 2.4. Counterfactual Plausibility Checks

**Problem**: No ground truth for counterfactuals → need sanity checks

**Solution**: Physics-based + statistical validation

#### Check 1: Directional Sanity

```python
def check_directional_sanity(results):
    """
    Verify:
    - ↓ Fishing ⇒ ↑ P_surv (less mortality)
    - ↑ SST ⇒ ↓ P_surv (stress, if literature supports)
    """
    P_surv_baseline = results['S0_baseline']['P_surv'].mean()
    P_surv_fishing = results['S1_fishing']['P_surv'].mean()
    P_surv_sst = results['S2_sst']['P_surv'].mean()
    
    # Check 1: Fishing reduction increases survival
    assert P_surv_fishing > P_surv_baseline, \
        f"❌ S1 (fishing -50%) DECREASES survival! ({P_surv_fishing:.3f} < {P_surv_baseline:.3f})"
    
    # Check 2: SST warming decreases survival (if ecologically plausible)
    # NOTE: Skip this if literature suggests SST doesn't harm sharks
    # assert P_surv_sst < P_surv_baseline, \
    #     f"❌ S2 (SST +2°C) INCREASES survival! ({P_surv_sst:.3f} > {P_surv_baseline:.3f})"
    
    print(f"✅ Directional sanity passed:")
    print(f"   S0 (baseline): {P_surv_baseline:.3f}")
    print(f"   S1 (fishing -50%): {P_surv_fishing:.3f} ({((P_surv_fishing/P_surv_baseline - 1)*100):.1f}%)")
    print(f"   S2 (SST +2°C): {P_surv_sst:.3f} ({((P_surv_sst/P_surv_baseline - 1)*100):.1f}%)")
```

#### Check 2: Elasticity Bounds

```python
def check_elasticity_bounds(results, X_baseline, X_cf, delta_X_percent):
    """
    Verify: No absurd responses (e.g., 1% fishing cut → 1000x sharks)
    
    Elasticity: E = (ΔY / Y) / (ΔX / X)
    Reasonable: |E| ∈ [0.1, 10]
    """
    Y_baseline = results['S0_baseline']['mean']
    Y_cf = results['S1_fishing']['mean']
    
    delta_Y_percent = (Y_cf - Y_baseline) / (Y_baseline + 1e-8)
    
    # Elasticity per timestep
    E = delta_Y_percent / (delta_X_percent + 1e-8)
    
    # Flag absurdities
    absurd = (np.abs(E) > 100)
    if absurd.sum() > 0:
        print(f"⚠️  {absurd.sum()} timesteps with |E| > 100 (absurd elasticity):")
        print(f"   Max |E| = {np.abs(E).max():.1f}")
    
    # Reasonable range
    reasonable = (np.abs(E) > 0.1) & (np.abs(E) < 10)
    print(f"✅ {reasonable.sum()} / {len(E)} timesteps have reasonable elasticity")
    
    return E
```

#### Check 3: Monotonicity (Mean)

```python
def check_monotonicity(sensitivity_matrix):
    """
    Verify: On average, ∂Y/∂X_fishing < 0
    
    Not required for EVERY timestep (non-linearities OK),
    but mean should hold.
    """
    mean_sensitivity = sensitivity_matrix.mean(axis=1)  # Mean over time
    
    # Fishing sensitivity should be negative (↓ fishing → ↑ sharks)
    assert mean_sensitivity['fishing'] < 0, \
        f"❌ Mean ∂Y/∂X_fishing = {mean_sensitivity['fishing']:.3f} (should be < 0)"
    
    print(f"✅ Monotonicity check passed:")
    print(f"   Mean ∂Y/∂X_fishing = {mean_sensitivity['fishing']:.3f} (negative)")
```

**Checklist**:
- [ ] Implement 3 plausibility checks
- [ ] Add to notebook: "Counterfactual Validation"
- [ ] Plot: Elasticity heatmap (flag |E| > 100)
- [ ] Assert: All directional checks pass

---

### 2.5. Adaptive Threshold Implementation

**Problem**: Fixed $\tau = 0.5$ for 0.05% prevalence is nonsense

**Solution**: Quantile-based threshold (5th percentile)

```python
# Compute adaptive threshold from baseline predictions
tau_adaptive = np.quantile(y_pred_baseline, q=0.05)

print(f"Adaptive threshold: τ = {tau_adaptive:.4f}")
print(f"  (vs fixed τ = 0.5, which is absurd for 0.05% prevalence)")

# Use for ALL outputs (N2, N3)
# Output N2: Survival probability
P_surv = (Y_cf > tau_adaptive).float().mean(dim=0)

# Output N3: First hitting time
T_collapse = []
for i in range(Y_cf.shape[0]):  # Loop over samples
    hits = (Y_cf[i] < tau_adaptive).nonzero(as_tuple=True)[0]
    if len(hits) > 0:
        T_collapse.append(hits[0].item())
    else:
        T_collapse.append(np.inf)  # No collapse observed
```

**Checklist**:
- [ ] Compute `tau_adaptive` from baseline S0
- [ ] Replace all instances of `tau=0.5` with `tau_adaptive`
- [ ] Update N2 (survival) and N3 (hitting time) calculations
- [ ] Add horizontal line in fan charts: $\tau_{\text{adaptive}}$

---

### 2.6. Policy Optimization with Temporal Regularization (S4)

**Problem**: Unconstrained optimization → "reduce fishing 90% for 1 month, then explode"

**Solution**: Add temporal smoothness constraint

```python
def optimize_policy(model, X_baseline, t0, target_P_surv=0.9, lambda_smooth=0.1):
    """
    Find optimal intervention:
    
    min δX: ||δX||² + λ Σ_t ||δX_t - δX_{t+1}||²
    s.t.   P(Y_T > τ) ≥ 0.9
           δF ∈ [-50%, 0]
           δT ∈ [-2, +5]
    """
    # Initialize δX (learnable)
    delta_X = torch.zeros(X_baseline.shape, requires_grad=True)
    
    optimizer = torch.optim.Adam([delta_X], lr=0.01)
    
    for step in range(1000):
        # Apply intervention
        X_cf = X_baseline + delta_X
        
        # Forecast with DMM
        Y_cf, _ = temporal_counterfactual(model, X_hist, Y_hist, X_cf, t0)
        
        # Constraint: P_surv ≥ target
        P_surv = (Y_cf[:, -1] > tau_adaptive).float().mean()
        
        # Objective: Minimize ||δX||² + temporal smoothness
        magnitude_loss = (delta_X**2).sum()
        smoothness_loss = ((delta_X[:, 1:] - delta_X[:, :-1])**2).sum()
        
        loss = magnitude_loss + lambda_smooth * smoothness_loss
        
        # Add penalty if P_surv constraint violated
        if P_surv < target_P_surv:
            loss += 1000 * (target_P_surv - P_surv)**2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Project to box constraints
        with torch.no_grad():
            delta_X[:, :, fishing_idx].clamp_(-0.5 * X_baseline[:, :, fishing_idx], 0)
            delta_X[:, :, sst_idx].clamp_(-2, 5)
        
        if step % 100 == 0:
            print(f"Step {step}: Loss={loss.item():.3f}, P_surv={P_surv.item():.3f}")
    
    return delta_X.detach()
```

**Expected**:
- Policy is **smooth**: No wild oscillations
- Policy is **monotonic**: Gradual reduction, not pulse
- Policy is **feasible**: $\delta F \in [-50\%, 0]$ (politically realistic)

**Checklist**:
- [ ] Implement `optimize_policy()` function
- [ ] Plot: Optimal $\delta X_t$ over time (smooth curve)
- [ ] Verify: $\|\delta X_t - \delta X_{t+1}\|$ is small
- [ ] Compare: Gradient-based vs. grid search (sanity)

---

## Part 3: Output Generation & Narrative

### 3.1. Automated Narrative Generation

**Purpose**: Translate results to policy language (not just plots)

```python
def generate_narrative(results):
    """
    Auto-generate policy summaries from counterfactual results.
    """
    # Baseline
    P_surv_baseline = results['S0_baseline']['P_surv'].mean()
    T_collapse_baseline = np.median(results['S0_baseline']['T_collapse'])
    
    narratives = []
    
    # S1: Fishing reduction
    P_surv_S1 = results['S1_fishing']['P_surv'].mean()
    delta_P_S1 = (P_surv_S1 / P_surv_baseline - 1) * 100
    narratives.append(
        f"**S1 (Fishing -50%)**: Mean survival probability increases by {delta_P_S1:.1f}% "
        f"(from {P_surv_baseline:.3f} to {P_surv_S1:.3f})."
    )
    
    # S2: SST warming
    T_collapse_S2 = np.median(results['S2_sst']['T_collapse'])
    delta_T_S2 = T_collapse_baseline - T_collapse_S2
    narratives.append(
        f"**S2 (SST +2°C)**: Median time-to-collapse advances by {delta_T_S2:.0f} months "
        f"(from {T_collapse_baseline:.0f} to {T_collapse_S2:.0f})."
    )
    
    # S3: Non-linearity test
    P_surv_S3 = results['S3_combined']['P_surv'].mean()
    P_surv_additive = P_surv_S1 + (results['S2_sst']['P_surv'].mean() - P_surv_baseline)
    nonlinearity = (P_surv_S3 - P_surv_additive) / P_surv_additive * 100
    
    if abs(nonlinearity) > 5:
        narratives.append(
            f"**S3 (Combined)**: Shows {abs(nonlinearity):.1f}% non-additivity "
            f"({'synergy' if nonlinearity > 0 else 'antagonism'})."
        )
    else:
        narratives.append(
            f"**S3 (Combined)**: Effects are approximately additive "
            f"(non-linearity < 5%)."
        )
    
    # S4: Optimal policy
    delta_X_opt = results['S4_optimal']['delta_X']
    delta_F_mean = delta_X_opt[:, :, fishing_idx].mean().item() * 100
    narratives.append(
        f"**S4 (Optimal)**: Minimal intervention requires {abs(delta_F_mean):.1f}% "
        f"fishing reduction to achieve 90% survival target."
    )
    
    # Combine
    full_narrative = "\n\n".join(narratives)
    
    return full_narrative

# Usage
narrative = generate_narrative(results)
print(narrative)

# Save to markdown
with open('results/counterfactual_narrative.md', 'w') as f:
    f.write("# Counterfactual Analysis: Policy Summaries\n\n")
    f.write(narrative)
```

**Expected Output**:
```
**S1 (Fishing -50%)**: Mean survival probability increases by 12.3% 
(from 0.042 to 0.047).

**S2 (SST +2°C)**: Median time-to-collapse advances by 3 months 
(from 9 to 6).

**S3 (Combined)**: Shows 8.5% non-additivity (antagonism).

**S4 (Optimal)**: Minimal intervention requires 35.2% fishing reduction 
to achieve 90% survival target.
```

**Checklist**:
- [ ] Implement `generate_narrative()` function
- [ ] Add to notebook: "Output N7: Policy Narrative"
- [ ] Export: Markdown report for policymakers
- [ ] Test: Verify percentages are correct

---

## Summary Checklist

### Pre-DMM (Do First)
- [ ] Replace threshold: $\tau = 0.5 \to \tau_{\text{adaptive}}$
- [ ] Risk decile analysis (discrimination ratio > 10x)
- [ ] Baseline normalization (MSE/LL ratios)
- [ ] Monte Carlo validation (entropy, quantiles, stability)
- [ ] Stress tests (zero-X, noise, extremes)

### During DMM Implementation
- [ ] Posterior collapse detection (KL tracking, alerts)
- [ ] Uncertainty decomposition (Output N6)
- [ ] Temporal alignment (fix $z_{t_0}$ across scenarios)
- [ ] Adaptive threshold (use in N2, N3)
- [ ] S4 regularization (smooth policies)

### After DMM Training
- [ ] Counterfactual plausibility (directional, elasticity, monotonicity)
- [ ] Narrative generation (auto-summaries)
- [ ] Validation plots (calibration, deciles, variance breakdown)

---

**Expected Timeline**:
- Pre-DMM validation: 2-3 hours
- DMM implementation: 1-2 weeks
- Counterfactual generation: 3-5 days
- Total: ~2-3 weeks to "à prova de bala"
