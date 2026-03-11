# Embedding Factor Analysis

Factor analysis of model response embeddings to discover latent behavioral dimensions.

## Quick Start

```python
from scripts.factor_analysis.preprocessing import load_embeddings, deduplicate_by_group, residualize, pca_reduce
from scripts.factor_analysis.parallel_analysis import parallel_analysis
from scripts.factor_analysis.factor_analysis import run_factor_analysis, adequacy_tests
from scripts.factor_analysis.interpretation import (
    factor_extremes, rank_by_factor_purity,
    analytical_factor_embedding, corpus_nearest_neighbor,
    optimize_factor_embedding, prompt_effects,
)

# Load and preprocess
embeddings, metadata = load_embeddings("path/to/embeddings.npy", "path/to/metadata.jsonl")
embeddings, metadata = deduplicate_by_group(embeddings, metadata, max_per_group=50)
corpus_embeddings = embeddings.copy()  # keep for nearest-neighbor lookups later
residuals, group_means, group_inv = residualize(embeddings, metadata)
global_mean = embeddings.mean(axis=0)
reduced, pca_model, scaler = pca_reduce(residuals, n_components=100)

# Determine number of factors
pa = parallel_analysis(reduced)  # Horn's method
n_factors = pa["n_recommended"]  # or pick manually from scree plot

# Run factor analysis
fa = run_factor_analysis(reduced, n_factors=15, method="principal", rotation="varimax")
scores, loadings = fa["scores"], fa["loadings"]
```

## Dataset

Current test dataset: `qwen4embeddings/stage123-240x50-singleturn-v2/`
- 12,000 embeddings (239 prompts x 50 responses + 1 duplicate prompt)
- Dimension 2560, L2-normalized, from Qwen3-Embedding-4B
- Responses from Llama-3.1-8B-Instruct
- After dedup: 11,950 samples (cap 50 per prompt)

## Pipeline Steps

1. **Load + deduplicate** — cap samples per prompt group
2. **Residualize** — subtract per-prompt mean to isolate style variation from topic variation
3. **PCA pre-reduce** — 2560d is too large for factor analysis correlation matrix; reduce to ~100d
4. **Parallel analysis** — determine how many factors have more structure than noise
5. **Factor analysis** — extract factors with rotation (varimax default)
6. **Interpret** — four methods to understand what factors mean (see below)

## Interpretation Methods

### Method 2: Analytical + corpus nearest neighbor
Back-project factor direction to embedding space, find closest real response.
```python
target, direction = analytical_factor_embedding(0, loadings, pca_model, scaler, global_mean, scale=3.0)
neighbors = corpus_nearest_neighbor(target, corpus_embeddings, metadata, top_k=10)
```

### Method 3: Score-based purity ranking
Find responses that score high on one factor and low on all others.
```python
purity = rank_by_factor_purity(scores, metadata, factor_idx=0, penalty_weight=1.0, top_n=20)
```

### Simple extremes
Top/bottom responses per factor by raw score.
```python
extremes = factor_extremes(scores, metadata, top_n=20)
```

### Method 1: Gradient descent through embedding model (GPU required)
Optimize continuous token embeddings to maximize a factor score.
```python
opt = optimize_factor_embedding(
    factor_idx=0, n_factors=15, loadings=loadings,
    pca_model=pca_model, scaler=scaler, global_mean=global_mean,
    model_name="Qwen/Qwen3-Embedding-4B", n_steps=500,
)
# Method 4: combine with corpus lookup
neighbors = corpus_nearest_neighbor(opt["optimized_embedding"], corpus_embeddings, metadata)
```

## Key Questions to Explore

### Immediate priorities
- **How many factors?** Parallel analysis says 50 on this dataset (all PCA dims beat random). Try scree plot visually — look for an elbow. Also try fewer factors (5, 10, 15, 20) and compare interpretability.
- **Do the factors mean anything?** Run `factor_extremes` and `rank_by_factor_purity` on the top few factors. Read the extreme responses. Are there coherent themes (verbose vs terse, formal vs casual, hedging vs assertive)?
- **Residualized vs non-residualized**: Residualization removes between-prompt variance. Factors on residuals capture *style* variation (how differently the model responds to the same prompt). Without residualization, factors will mix topic and style. Compare both.
- **Communalities are low (~0.15)**: This means the 15 factors only explain 15% of the variance in the PCA-reduced space. This could mean: (a) need more factors, (b) the PCA pre-reduction is lossy (only 47% variance retained), or (c) the embedding space is genuinely high-dimensional with no low-rank factor structure. Try increasing PCA_PRE_N to 200 or 300 and see if communalities improve.

### Deeper questions
- **Rotation matters**: Varimax forces orthogonal simple structure. Try promax (oblique) — if factors are correlated, promax will give cleaner loadings. Compare with `rotation="promax"`.
- **Method comparison**: Do Methods 1-4 converge on similar interpretations for the same factor? If not, why? Method 1 (gradient descent) explores the model's embedding manifold; Methods 2-3 are constrained to real corpus responses.
- **Stability**: Run factor analysis on random 80% subsets. Do the same factors appear? Use factor congruence coefficients to check.
- **Connection to OCEAN traits**: Once factors are interpretable, do any correspond to personality dimensions? This is the ultimate research question.

## Known Issues

- `factor_analyzer` 0.5.1 has a sklearn 1.8 compatibility bug (`force_all_finite` renamed to `ensure_all_finite`). Monkey-patched in `factor_analysis.py`.
- Adequacy tests (Bartlett, KMO) give uninformative results on PCA-reduced data because PCA scores are uncorrelated by construction. Run on original standardized data if you need them.
- Eta-squared on residualized factor scores is ~0 by construction. The experiment script also projects full embeddings through the factor model for a meaningful prompt-effect check.

## Experiment Script

`scripts/experiments/factor_analysis/embedding_factor_exploration.py` — cell-based script with all steps + plotly visualizations. Run interactively or as a script. Outputs to `scratch/factor_analysis/`.
