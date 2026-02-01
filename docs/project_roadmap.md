# Nova Selachiia Project Roadmap

**Project**: Counterfactual Analysis of Shark Population Collapse under Climate and Anthropogenic Stressors

**Mathematical Framework**: Deep Markov Model (DMM) with Variational Inference for Stochastic Ecological Dynamics

---

## 1. Completed Milestones ✅

### Phase 1: Data Infrastructure (Completed)
- ✅ **GBIF Data Pipeline**: Downloaded and cleaned 530K shark occurrence records
- ✅ **Environmental Covariates**: 
  - SST (Sea Surface Temperature): NOAA ERSST v5, monthly, 1854-2024
  - Fishing Effort: Global Fishing Watch (10km resolution, 2020-2024)
  - Prey Density: GBIF occurrences for 7 families (Carangidae, Clupeidae, etc.)
- ✅ **Data Cleaning**: 
  - Removed invalid coordinates, duplicates, terrestrial records
  - Temporal alignment: Monthly aggregation (1980-2024)
  - Spatial alignment: 0.25° grid cells
- ✅ **Format Optimization**: 
  - Converted CSV → Parquet (10x faster I/O, 50% smaller)
  - Partitioned by year for efficient queries
  - Schema validation with PyArrow

### Phase 2: NSSM Baseline (Completed)
- ✅ **Model Architecture**:
  - LSTM (2 layers, 128 hidden units) + ResNet decoder (2 blocks, skip connections)
  - SiLU (Swish) activation: $\text{SiLU}(x) = x \cdot \sigma(x)$
  - Learnable Gaussian noise: $\mathbf{z}_t = f_\theta(\mathbf{z}_{t-1}, \mathbf{X}_t) + \mathcal{N}(0, \sigma_z^2)$
  - Global gradient clipping: $\|\nabla_\theta \mathcal{L}\|_2 \leq 1.0$
  
- ✅ **Class Imbalance Solution**:
  - **Focal Loss** (Lin et al., 2017): $\mathcal{L}_{\text{FL}} = -\alpha_t (1-p_t)^\gamma \log(p_t)$
    - $\alpha = 0.25$ (up-weight minority class)
    - $\gamma = 2.0$ (down-weight easy examples)
  - **SMOTE Oversampling** (Chawla et al., 2002): Synthetic minority samples via k-NN ($k=5$)
  - **Balanced Training**: 50/50 split (10,603 samples → 21,206 with SMOTE)
  - **Realistic Testing**: Test set retains original 0.05% prevalence

- ✅ **Training Results** (49 epochs, early stopping at epoch 38):
  - Training: MSE=0.002, F1=0.99 (on balanced data)
  - Validation: AUC-ROC=0.89, MSE=0.009
  - Test: AUC-ROC=0.74, MSE=0.010, Median Prediction=0.0002 ✅
  - **Interpretation**: Model correctly learned "sharks are rare" (not overfitting)

- ✅ **Monte Carlo Forecasting** (100 samples):
  - Fan charts with 90% confidence intervals
  - Collapse probability: $P(Y_t < \tau)$ over time
  - Temporal entropy: $H_t = -p_t \log p_t - (1-p_t) \log(1-p_t)$
  - **Critical Bug Fixed**: Applied sigmoid to convert logits → probabilities [0,1]

- ✅ **Artifacts Saved**:
  - `models/nssm/best_model.pt`: Trained weights (11.2 MB)
  - `models/nssm/nssm_config.json`: Hyperparameters
  - `models/nssm/training_history.json`: Loss curves (49 epochs)
  - `data/processed/test_predictions.csv`: Predictions + targets (1,473 samples)
  - `data/figures/nssm/`: Training curves, Monte Carlo ensembles

---

## 2. Active Development 🚧

### Phase 3: Deep Markov Model (DMM) - In Progress

**Objective**: Upgrade NSSM to probabilistic latent dynamics for counterfactual inference

**Mathematical Framework** (Krishnan et al., 2017):

**Generative Model**:
$$p_\theta(\mathbf{z}_{1:T}, \mathbf{Y}_{1:T} | \mathbf{X}_{1:T}) = p(\mathbf{z}_1) \prod_{t=1}^T p_\theta(\mathbf{z}_t | \mathbf{z}_{t-1}, \mathbf{X}_t) p_\theta(\mathbf{Y}_t | \mathbf{z}_t)$$

**Inference Model** (recognition network):
$$q_\phi(\mathbf{z}_{1:T} | \mathbf{Y}_{1:T}, \mathbf{X}_{1:T}) = \prod_{t=1}^T q_\phi(\mathbf{z}_t | \mathbf{z}_{t-1}, \mathbf{Y}_{1:t}, \mathbf{X}_{1:t})$$

**ELBO Objective** (Evidence Lower Bound):
$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi}\left[ \log p_\theta(\mathbf{Y}_{1:T} | \mathbf{z}_{1:T}) \right] - \text{KL}\left( q_\phi(\mathbf{z}_{1:T}) \| p_\theta(\mathbf{z}_{1:T}) \right)$$

**Implementation Tasks**:

#### Task 3.1: DMM Architecture (Week 1-2)
- [ ] **Encoder Network** $q_\phi(\mathbf{z}_t | \cdot)$:
  - Bi-directional LSTM (backward pass through $\mathbf{Y}_{1:T}$)
  - MLP → $\boldsymbol{\mu}_t, \log \boldsymbol{\sigma}_t^2$ (mean and log-variance)
  - Input: $[\mathbf{Y}_{1:t}, \mathbf{X}_{1:t}, \mathbf{z}_{t-1}]$ (concatenated)
  
- [ ] **Reparameterization Trick**:
  - $\mathbf{z}_t = \boldsymbol{\mu}_t + \boldsymbol{\sigma}_t \odot \boldsymbol{\epsilon}$, where $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$
  - Enables backpropagation through stochastic sampling
  
- [ ] **Prior Network** $p_\theta(\mathbf{z}_t | \mathbf{z}_{t-1}, \mathbf{X}_t)$:
  - LSTM transition model (reuse NSSM architecture)
  - Output: $\boldsymbol{\mu}_t^{\text{prior}}, \log \boldsymbol{\sigma}_t^{2,\text{prior}}$
  
- [ ] **Decoder** $p_\theta(\mathbf{Y}_t | \mathbf{z}_t)$:
  - ResNet decoder (reuse NSSM architecture)
  - Output: Bernoulli logits for shark presence

#### Task 3.2: ELBO Training (Week 2-3)
- [ ] **Reconstruction Loss** (negative log-likelihood):
  - Binary: $\mathcal{L}_{\text{recon}} = -\sum_t y_t \log p_t + (1-y_t) \log(1-p_t)$
  - Alternative: Focal Loss (maintain class imbalance handling)
  
- [ ] **KL Divergence** (regularization):
  - Closed-form KL between Gaussians:
    $$\text{KL}(q_\phi || p_\theta) = \frac{1}{2} \sum_t \left[ \frac{\sigma_t^2}{\sigma_t^{2,\text{prior}}} + \frac{(\mu_t - \mu_t^{\text{prior}})^2}{\sigma_t^{2,\text{prior}}} - 1 + \log\frac{\sigma_t^{2,\text{prior}}}{\sigma_t^2} \right]$$
  
- [ ] **KL Annealing** ($\beta$-VAE schedule):
  - Start: $\beta = 0.01$ (prioritize reconstruction)
  - Anneal to: $\beta = 1.0$ over 20 epochs (linear or cosine)
  - Prevents posterior collapse ($q_\phi \to p_\theta$)
  
- [ ] **Optimizer**:
  - AdamW (weight decay = 0.01)
  - Learning rate: 1e-4 (start), 1e-5 (end) with cosine annealing
  - Gradient clipping: $\|\nabla\|_2 \leq 1.0$

#### Task 3.3: Validation Metrics (Week 3)
- [ ] **ELBO Decomposition**:
  - Track: $\mathcal{L}_{\text{recon}}$, $\mathcal{L}_{\text{KL}}$, $\mathcal{L}_{\text{ELBO}}$ separately
  - Monitor KL collapse: If $\text{KL} < 0.1$, increase $\beta$
  
- [ ] **Reconstruction Quality**:
  - MSE, AUC-ROC, F1 (same as NSSM baseline)
  - Expected: AUC-ROC ≥ 0.74 (match or beat NSSM)
  
- [ ] **Latent Space Analysis**:
  - t-SNE/UMAP of $\mathbf{z}_t$ (check for regime clustering)
  - KL per timestep (identify anomalies)
  - Prior vs. posterior divergence (should be moderate)

#### Task 3.4: Counterfactual Inference (Week 4)
- [ ] **Ancestral Sampling** (forward mode):
  - Start: $\mathbf{z}_1 \sim p(\mathbf{z}_1)$
  - Loop: $\mathbf{z}_t \sim p_\theta(\mathbf{z}_t | \mathbf{z}_{t-1}, \mathbf{X}_t^{\text{cf}})$
  - Generate: $\mathbf{Y}_t \sim p_\theta(\mathbf{Y}_t | \mathbf{z}_t)$
  - **Input**: Counterfactual covariates $\mathbf{X}^{\text{cf}}$ (modified fishing/SST)
  
- [ ] **Scenario Implementations**:
  
  | Scenario | Intervention | Implementation |
  |----------|--------------|----------------|
  | **S1: Fishing Reduction** | -50% effort | `X[:, :, fishing_idx] *= 0.5` |
  | **S2: Climate Warming** | +2°C SST | `X[:, :, sst_idx] += 2.0` |
  | **S3: Combined** | S1 + S2 | Apply both modifications |
  | **S4: Optimal Policy** | Minimize $\|\delta \mathbf{X}\|$ | Gradient descent on $\delta \mathbf{X}$ |
  
- [ ] **Monte Carlo Ensemble** (100 samples per scenario):
  - Generate 100 trajectories for each of 4 scenarios
  - Store: Ensemble, mean, quantiles, collapse prob, entropy
  - Compare: Baseline vs. S1, S2, S3, S4

#### Task 3.5: Output Generation (Week 5)
- [ ] **Output N1: Trajectory Ensembles**:
  - Fan charts with 50%/90% confidence intervals
  - Spaghetti plots (100 thin lines + bold mean)
  - Compare scenarios side-by-side (2x2 grid)
  
- [ ] **Output N2: Survival Probability**:
  - $P_{\text{surv}}(t) = P(Y_t > \tau | \mathbf{X}^{\text{cf}})$
  - Kaplan-Meier style curves (one per scenario)
  - Statistical test: Log-rank test (baseline vs. interventions)
  
- [ ] **Output N3: First Hitting Time**:
  - $T_{\text{collapse}} = \min\{t : Y_t < \tau\}$
  - Distribution: Histogram + KDE for each scenario
  - Metrics: Median, IQR, hazard ratios
  
- [ ] **Output N4: Temporal Entropy**:
  - $H_t = -p_t \log p_t - (1-p_t) \log(1-p_t)$
  - Time series: Entropy over 12 months
  - Interpretation: High entropy = regime uncertainty
  
- [ ] **Output N5: Sensitivity Analysis**:
  - Finite difference: $\frac{\partial Y_t}{\partial X_j} \approx \frac{Y_t(X_j + \epsilon) - Y_t(X_j)}{\epsilon}$
  - Heatmap: Covariates (rows) × Time (columns)
  - Identify: Critical intervention windows (e.g., SST impact in summer)

---

## 3. Planned Features 🔮

### Phase 4: UI Development (Month 2)

#### Task 4.1: Streamlit Dashboard
- [ ] **Interactive Sliders**:
  - Fishing effort: $\delta F \in [-100\%, +50\%]$
  - SST anomaly: $\delta T \in [-2°C, +5°C]$
  - Prey density: $\delta P \in [-50\%, +100\%]$
  
- [ ] **Real-Time Forecasting**:
  - User sets $\delta \mathbf{X}$ → Click "Generate" → See trajectories in <2s
  - Cache DMM weights (ONNX export for CPU inference)
  - Parallel sampling (multiprocessing for 100 trajectories)
  
- [ ] **Policy Dashboard**:
  - Compare 4 scenarios simultaneously (parallel coordinates plot)
  - Show: $P_{\text{surv}}(T)$, $T_{\text{collapse}}$, intervention cost
  - Export: PDF report, CSV results

#### Task 4.2: Visualization Enhancements
- [ ] **Interactive Fan Charts** (Plotly):
  - Zoom, pan, select time ranges
  - Hover: Show exact values ($\mu_t$, $\sigma_t$, quantiles)
  - Toggle: Show/hide individual trajectories
  
- [ ] **Spatial Maps** (Folium/Plotly):
  - Collapse risk by region (0.25° grid cells)
  - Fishing effort overlay (heatmap)
  - SST anomaly contours

### Phase 5: Causal Inference (Month 3)

#### Task 5.1: Instrumental Variables (IV)
- [ ] **Identify Instruments**:
  - Candidate: El Niño-Southern Oscillation (ENSO) index
  - Justification: Affects SST (relevant) but not directly sharks (exclusion)
  - Data: NOAA ENSO indices (monthly, 1950-2024)
  
- [ ] **Two-Stage Least Squares (2SLS)**:
  - Stage 1: $\text{SST}_t = \alpha + \beta \cdot \text{ENSO}_t + \epsilon_t$
  - Stage 2: $Y_t = \gamma + \delta \cdot \widehat{\text{SST}}_t + \eta_t$
  - Compare: DMM counterfactual vs. 2SLS causal estimate

#### Task 5.2: Structural Causal Models (SCM)
- [ ] **Causal Graph Learning**:
  - PC algorithm (constraint-based)
  - GES algorithm (score-based)
  - Ground truth: Domain knowledge (e.g., SST → Prey → Sharks)
  
- [ ] **do-Calculus** (Pearl, 2009):
  - Query: $P(Y_t | do(\text{Fishing} = f))$
  - Compute via backdoor adjustment or front-door criterion
  - Validate DMM counterfactuals against causal estimands

### Phase 6: Extensions (Month 4+)

#### Task 6.1: Spatial DMM
- [ ] **Convolutional Encoder**:
  - Input: $\mathbf{X}_t \in \mathbb{R}^{H \times W \times C}$ (spatial grid)
  - Architecture: ResNet18 encoder → latent $\mathbf{z}_t$
  - Dynamics: ConvLSTM (spatial + temporal)
  
- [ ] **Spatial Counterfactuals**:
  - Localized interventions (e.g., fishing ban in specific region)
  - Spillover effects (do adjacent regions benefit?)

#### Task 6.2: Multi-Species Modeling
- [ ] **Joint VAE**:
  - Shared latent: $\mathbf{z}_t$ (ecosystem state)
  - Multiple decoders: $p(\text{Sharks}_t | \mathbf{z}_t)$, $p(\text{Prey}_t | \mathbf{z}_t)$
  - Objective: Multi-task ELBO
  
- [ ] **Trophic Cascades**:
  - Model interactions: Sharks ← Prey ← Primary producers
  - Counterfactual: What if prey density drops 30%?

#### Task 6.3: Reinforcement Learning (RL) for Policy Optimization
- [ ] **Markov Decision Process (MDP)**:
  - State: $\mathbf{s}_t = [\mathbf{z}_t, \mathbf{X}_t]$ (latent + covariates)
  - Action: $\mathbf{a}_t = \delta \mathbf{X}_t$ (policy intervention)
  - Reward: $r_t = Y_t - \lambda \|\mathbf{a}_t\|_2$ (shark survival - intervention cost)
  
- [ ] **Actor-Critic Algorithm** (PPO or SAC):
  - Actor: $\pi_\psi(\mathbf{a}_t | \mathbf{s}_t)$ (policy network)
  - Critic: $V_\omega(\mathbf{s}_t)$ (value function)
  - Train: Maximize $\mathbb{E}[\sum_t \gamma^t r_t]$
  
- [ ] **Constraints**:
  - Fishing reduction: $\delta F \in [-50\%, 0\%]$ (politically feasible)
  - Budget: $\sum_t \|\mathbf{a}_t\|_1 \leq B$ (limited resources)

---

## 4. Technical Debt & Refactoring

### Code Quality
- [ ] **Type Annotations**: Add to all functions (mypy strict mode)
- [ ] **Docstrings**: NumPy style for all public methods
- [ ] **Unit Tests**: pytest for data loaders, model components
- [ ] **Integration Tests**: End-to-end training + inference
- [ ] **CI/CD**: GitHub Actions (linting, tests, coverage)

### Performance Optimization
- [ ] **torch.compile()**: Re-enable with `mode="reduce-overhead"` (requires profiling)
- [ ] **Mixed Precision** (AMP): FP16 training (2x speedup, 50% memory reduction)
- [ ] **Gradient Checkpointing**: Trade compute for memory (enable for large batches)
- [ ] **Data Loading**: Prefetch with `num_workers=4`, `pin_memory=True`

### Deployment
- [ ] **ONNX Export**: Convert DMM to ONNX for CPU inference
- [ ] **Docker**: Containerize training + UI (Dockerfile + docker-compose)
- [ ] **Cloud**: Deploy to Azure ML / AWS SageMaker (optional)
- [ ] **API**: FastAPI endpoint for counterfactual queries

---

## 5. Publications & Dissemination

### Academic Outputs
- [ ] **Paper 1**: *"Deep Markov Models for Counterfactual Analysis of Shark Population Collapse"*
  - Target: Ecological Modelling, Ecology Letters, or PLOS ONE
  - Sections: Methods (DMM), Results (4 scenarios), Discussion (policy implications)
  
- [ ] **Paper 2**: *"Focal Loss and SMOTE for Extreme Class Imbalance in Ecological Time Series"*
  - Target: Methods in Ecology and Evolution
  - Contribution: Methodological comparison (NSSM baseline)

### Conference Presentations
- [ ] **ESA Annual Meeting** (Ecological Society of America)
- [ ] **NeurIPS Workshop**: Climate Change AI
- [ ] **ICML Workshop**: AI for Science

### Open-Source Contributions
- [ ] **GitHub Release**: v1.0 with documentation
- [ ] **Zenodo DOI**: Archive code + data + models
- [ ] **HuggingFace Space**: Interactive demo (Streamlit app)

---

## 6. Timeline & Milestones

| Month | Phase | Deliverables |
|-------|-------|-------------|
| **Month 1** ✅ | NSSM Baseline | Trained model, Monte Carlo, bug fixes |
| **Month 2** 🚧 | DMM Implementation | Architecture, ELBO training, validation |
| **Month 2-3** | Counterfactuals | 4 scenarios, outputs N1-N5 |
| **Month 3** | UI Development | Streamlit dashboard, interactive plots |
| **Month 4** | Causal Inference | IV analysis, SCM, validation |
| **Month 5+** | Extensions | Spatial DMM, multi-species, RL |
| **Month 6** | Publication | Write paper, submit to journal |

---

## 7. Resources & References

### Key Papers
1. **Krishnan et al. (2017)**: "Structured Inference Networks for Nonlinear State Space Models" (AAAI)
   - DMM architecture, ELBO training, sequential VAE
   
2. **Lin et al. (2017)**: "Focal Loss for Dense Object Detection" (ICCV)
   - Class imbalance solution (used in NSSM baseline)
   
3. **Chawla et al. (2002)**: "SMOTE: Synthetic Minority Over-sampling Technique" (JAIR)
   - Oversampling for imbalanced data
   
4. **Pearl (2009)**: "Causality: Models, Reasoning, and Inference"
   - do-calculus, SCM, counterfactual inference

### Datasets
- **GBIF**: Global Biodiversity Information Facility (shark occurrences)
- **NOAA ERSST**: Extended Reconstructed Sea Surface Temperature
- **Global Fishing Watch**: AIS-based fishing effort (2020-2024)
- **NOAA ENSO**: El Niño-Southern Oscillation indices (for IV analysis)

### Software Stack
- **PyTorch**: 2.5.1 (DMM implementation)
- **Poetry**: Dependency management
- **Streamlit**: UI development
- **Plotly**: Interactive visualizations
- **ONNX**: Model export for deployment

---

## 8. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| **DMM doesn't converge** | High | Start with pretrained NSSM weights, tune KL annealing |
| **Posterior collapse** | High | Increase $\beta$ gradually, monitor KL divergence |
| **Overfitting to rare events** | Medium | Use validation set, early stopping, dropout |
| **Computational cost** | Medium | Use mixed precision, gradient checkpointing, cloud GPUs |
| **Data quality issues** | Low | Already cleaned, but add validation checks |
| **Causal identification fails** | Medium | Fall back to associational analysis, sensitivity tests |

---

## 9. Success Metrics

### Technical Metrics
- ✅ **NSSM Baseline**: AUC-ROC ≥ 0.74, MSE ≤ 0.01
- 🎯 **DMM Performance**: AUC-ROC ≥ 0.80, MSE ≤ 0.008
- 🎯 **Counterfactual Validity**: Statistical significance (p < 0.05) for interventions
- 🎯 **UI Responsiveness**: <2s latency for 100-sample Monte Carlo

### Scientific Metrics
- 🎯 **Causal Insights**: Identify at least 2 significant intervention points
- 🎯 **Policy Relevance**: Quantify fishing reduction needed for 90% survival
- 🎯 **Robustness**: Results hold across 10-fold cross-validation

### Dissemination Metrics
- 🎯 **Publication**: Accepted in peer-reviewed journal (IF > 3.0)
- 🎯 **Open-Source Impact**: >50 GitHub stars, >10 forks
- 🎯 **Community Engagement**: Presented at ≥1 conference

---

## 10. Contact & Collaboration

**Project Lead**: [Your Name]  
**Repository**: https://github.com/[username]/nova-selachiia  
**Documentation**: https://nova-selachiia.readthedocs.io  
**Email**: [your.email@domain.com]

**Collaborators Welcome**:
- Ecologists (domain expertise, validation)
- ML researchers (DMM improvements, causal inference)
- Conservation practitioners (policy translation)

---

**Last Updated**: 2026-02-01  
**Version**: 1.0  
**Status**: Phase 2 Complete ✅, Phase 3 In Progress 🚧
