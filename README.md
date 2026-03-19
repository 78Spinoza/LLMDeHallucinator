<!--
  SEO Metadata — LLMDeHallucinator
  ─────────────────────────────────────────────────────────────────────────────
  description: LLMDeHallucinator is an open-source Python research pipeline for
    automated detection, visualization, and surgical suppression of
    hallucination-associated neurons (H-Neurons) in large language models (LLMs).
    Built on mechanistic interpretability techniques using TransformerLens,
    PaCMAP, and sparse L1 probing, it provides a GUI-driven workflow for
    neuron-level model editing without retraining.

  keywords: LLM hallucination detection, mechanistic interpretability,
    H-Neurons, hallucination suppression, neuron weight editing, TransformerLens,
    LLM interpretability, sparse probing, PaCMAP visualization, activation
    extraction, CETT metric, TruthfulQA, HaluEval, TriviaQA, MMLU benchmark,
    large language model research, AI safety, model editing, feed-forward neurons,
    layer analysis, logistic regression probe, Llama 3.1, GPT-2, Mistral,
    open-source LLM, Plotly Dash, neural network interpretability, LLM alignment,
    hallucination mitigation, LLM-as-judge, safetensors, HuggingFace

  og:title: LLMDeHallucinator — Surgical Hallucination Suppression for Open-Source LLMs
  og:description: Interactive pipeline for finding and suppressing hallucination-associated
    neurons in LLMs using mechanistic interpretability. No retraining required.
  og:type: website

  twitter:card: summary_large_image
  twitter:title: LLMDeHallucinator — LLM Mechanistic Interpretability Research Tool
  twitter:description: Detect, visualize and surgically suppress hallucination neurons
    in open-source LLMs using TransformerLens, PaCMAP and sparse L1 probing.
-->

# LLMDeHallucinator

> **Automated detection, visualization and suppression of hallucination-associated neurons in open-source LLMs**

[![Status: Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-orange.svg)](#roadmap)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Dash](https://img.shields.io/badge/Dash-2.x-darkblue.svg)](https://dash.plotly.com/)
[![TransformerLens](https://img.shields.io/badge/TransformerLens-latest-orange.svg)](https://github.com/TransformerLensOrg/TransformerLens)
[![PaCMAP](https://img.shields.io/badge/PaCMAP-latest-green.svg)](https://github.com/YingfanWang/PaCMAP)
[![arXiv](https://img.shields.io/badge/arXiv-2512.01797-red.svg)](https://arxiv.org/abs/2512.01797)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [!WARNING]
> **This project is actively under development.** The architecture and documentation are in place, but the implementation is ongoing. Expect breaking changes. Contributions and feedback are welcome — see the [Roadmap](#roadmap) for current status.

![LLMDeHallucinator Dashboard](docs/LLMDeHallucinator_dashboard.svg)

---

**Topics:**
`llm-hallucination` · `mechanistic-interpretability` · `h-neurons` · `hallucination-detection` · `hallucination-suppression` · `neuron-editing` · `transformerlens` · `pacmap` · `sparse-probing` · `cett-metric` · `activation-extraction` · `model-editing` · `ai-safety` · `llm-alignment` · `truthfulqa` · `halu-eval` · `mmlu` · `open-source-llm` · `llama` · `gpt2` · `plotly-dash` · `interpretability-research` · `weight-suppression` · `feed-forward-neurons`

---

## Research Background

### H-Neurons — Gao et al., arXiv:2512.01797 *(primary reference)*

The paper that directly motivates this project. Key findings:

- Fewer than **0.1% of neurons** reliably predict hallucination events across six LLMs (Mistral, Gemma, Llama families)
- H-Neurons concentrate in **middle layers** and originate during **pre-training** — instruction tuning leaves them largely untouched ("parameter inertia")
- Detection AUROC exceeds **86%** on Mistral family models using only 2 features per neuron
- Causal validation: amplifying H-Neuron activations increases hallucination and sycophancy rates; ablating them reduces them

**Their exact method:** Consistency filtering (10 samples per question, keep only always-correct / always-wrong), compute CETT scores per neuron, train L1 logistic regression on `[CETT_answer, CETT_other]`. That is the complete input feature set.

LLMDeHallucinator extends this with richer features (distribution comparison, weight statistics, LightGBM, Boruta) and adds the suppression + evaluation pipeline the paper stopped short of building.

---

### Inference-Time Intervention — Li et al., NeurIPS 2023

Found that truthfulness is **linearly encoded in attention head activations**. Key contribution: the **mass mean shift** — the vector from the centroid of false-response activations to the centroid of true-response activations — is a more reliable intervention direction than trained probe weights. Shifting activations along this direction at inference time (no weight editing) improves TruthfulQA scores by 13–42 percentage points depending on model.

*Relevance: confirms that correct vs. hallucinating distribution comparison is meaningful and causal. Informs our baseline distribution feature design.*

---

### Towards Monosemanticity — Bricken et al., Anthropic 2023

MLP neurons are **polysemantic** — one neuron fires for multiple unrelated concepts due to superposition. Sparse Autoencoders (SAEs) fix this: training a two-layer autoencoder with L1 sparsity on MLP activations (e.g. 512 neurons → 4096 features) recovers monosemantic features, ~70% of which are judged interpretable by human raters. Causal validation: steering a single SAE feature shifts model output in the predicted direction.

*Relevance: motivates SAE as the advanced detection mode. Raw neurons are noisy; SAE features give cleaner signal.*

---

### Do I Know This Entity? — Ferrando et al., ICLR 2025

Applied pre-trained SAEs (Gemma Scope) to the entity knowledge problem. Introduced the **separation score** per SAE latent:

```
score = fraction_fires_on_known − fraction_fires_on_unknown
```

Selected the latent that maximises this score across all entity types without any supervised training. Achieved AUROC 73.2 detecting hallucination. Causal steering of the identified latent induced near-100% refusal rate on unknown entities.

*Relevance: separation score is the unsupervised analogue of our Cohen's d feature. Confirms that distribution-level comparison between known and unknown responses is causally meaningful.*

---

### Key Hypothesis

> H-Neurons concentrated in layers 8–20 are sufficient for hallucination detection and suppression, with a MMLU accuracy delta below 2% at a 20% suppression factor.

This hypothesis is testable within the LLMDeHallucinator pipeline and represents a concrete contribution beyond the original H-Neurons paper, which stopped short of building a production suppression tool.

---

## Citation

If you use LLMDeHallucinator in your research, please cite the H-Neurons paper that motivated it:

```bibtex
@misc{gao2025hneurons,
  title={H-Neurons: On the Existence, Impact, and Origin of
         Hallucination-Associated Neurons in LLMs},
  author={Cheng Gao et al.},
  year={2025},
  eprint={2512.01797},
  archivePrefix={arXiv}
}
```

---

## Overview

LLMDeHallucinator is an interactive research pipeline built with Python and Plotly Dash that takes an open-source LLM as input and produces a de-hallucinated model as output — along with a full audit report detailing exactly which neurons were modified and by how much.

The pipeline is grounded in the **H-Neurons** research (Gao et al., 2025, arXiv:2512.01797), which demonstrated that fewer than 0.1% of neurons in a model reliably predict hallucination events. LLMDeHallucinator operationalizes this finding into an end-to-end, GUI-driven workflow accessible to any researcher — no custom scripts required.

```
model_in.safetensors
        │
        ▼
┌───────────────────────────────────────┐
│         LLMDeHallucinator             │
│                                       │
│  1. Generate hallucination dataset    │
│  2. Extract layer activations         │
│  3. Identify H-Neurons (ML/AI)        │
│  4. Visualize with PaCMAP             │
│  5. Suppress neuron weights           │
│  6. Evaluate before/after             │
│  7. Generate audit report             │
└───────────────────────────────────────┘
        │
        ▼
model_out.safetensors + report.pdf
```

---

## Motivation

Large language models hallucinate. Existing mitigation strategies — RAG, RLHF, prompt engineering — treat the symptom rather than the cause. Recent mechanistic interpretability research has shown that hallucination is not diffuse across the entire model; it is concentrated in a small, identifiable subset of feed-forward neurons that encode an **over-compliance bias**: the tendency to produce a confident-sounding answer even when the model has no reliable knowledge.

LLMDeHallucinator makes it possible to:

- **See** exactly where hallucinations live inside the model, layer by layer
- **Understand** which neurons are responsible and why, using AI-assisted classification
- **Fix** the problem by surgically reducing neuron weights rather than retraining
- **Verify** that the edit improved hallucination rates without degrading general capability

---

## Key Features

### Interactive Dash Application
- Load any HuggingFace-compatible model directly from the UI
- Choose between running a fresh hallucination dataset or loading a pre-generated one
- Real-time progress feedback during pipeline execution
- Full GPU/CPU support with automatic device detection

### Hallucination Dataset Generation
- Built-in support for **TruthfulQA**, **HaluEval**, and **TriviaQA**
- Automatic answer labeling via exact match and semantic similarity
- Optional **LLM-as-judge** mode using an external model (e.g. GPT-4o, Gemini) for richer, non-binary labeling beyond simple factual QA
- Export generated dataset for reuse across runs

### H-Neuron Detection
- **Three-tier detection pipeline** — fast baseline to frontier-grade analysis:
  - **L1 Logistic Regression** — sparse linear probe, fast, good for quick runs (baseline)
  - **LightGBM** — gradient-boosted trees, captures non-linear neuron interactions, built-in SHAP feature importance
  - **Sparse Autoencoders (SAE)** — frontier approach; decomposes polysemantic neurons into monosemantic features for the cleanest possible signal *(planned)*
- CETT metric (Contribution to Each Token's representation) for neuron activation quantification
- Focus on **middle layers (8–20)** where hallucination signal peaks — configurable
- Confidence scoring per neuron across all detection methods

### PaCMAP Visualization (2D and 3D)

PaCMAP projects the same 120-dimensional space (40 H-Neuron candidates × 3 features) that LightGBM trained on down to 2D or 3D. Each point is one prompt. It answers two questions simultaneously:

**1 — Are the two sets separable?**
If the detected H-Neurons genuinely explain hallucination, the red and blue points should form distinct clusters. Clean separation = the neuron selection is good. Poor separation = go back and refine.

```
Good separation          Poor separation — refine neuron selection

  ● ● ●                    ● ○ ● ○
    ● ●    ○ ○ ○           ○ ● ○ ●
  ● ● ●      ○ ○             ● ○ ●
```

**2 — Are there sub-clusters within the hallucination group?**
All red points are hallucinations — but they may not all be the same *type*. Sub-cluster A might be hallucinations about dates, sub-cluster B about names, each driven by a different neuron. The statistical pipeline treats them as one group. PaCMAP reveals the structure inside.

```
  ● ●          ← sub-cluster A (neuron 847 drives this)
 ● ● ●

        ● ●   ← sub-cluster B (neuron 1203 drives this —
       ● ●       missed by LightGBM, found by researcher)

○ ○ ○ ○ ○    ← correct prompts
```

The researcher lasso-selects sub-cluster B, the UI shows which neurons fire specifically on those prompts, and neuron 1203 is manually added to the H-Neuron list.

- Interactive scatter — zoom, pan, hover, lasso-select
- Hover reveals: original prompt, model response, which neurons fired
- Separation index shown per layer — quantifies cluster quality numerically
- Layer-by-layer animation to see how separation evolves across the network
- Small-multiples view: all layers simultaneously in a grid

### Neuron Weight Suppression

Suppression multiplies `W_out[j, :]` by a factor < 1 (e.g. 0.8 = 20% reduction). The neuron still exists — it just writes less strongly into the residual stream.

**Which neurons to suppress — SHAP score, not weight magnitude**

SHAP determines which H-Neurons are uniquely responsible for hallucination. Target those first. Neurons with near-zero SHAP are redundant — correlated with a higher-ranked neuron — and can be skipped entirely.

```
Rank 1  Neuron 847   SHAP = 0.041  w_out_norm = 8.2  → suppress
Rank 2  Neuron 2341  SHAP = 0.029  w_out_norm = 1.1  → suppress
Rank 3  Neuron 1203  SHAP = 0.003  w_out_norm = 0.4  → skip (redundant with 847)
```

**Note:** The CETT metric already incorporates `w_out_norm` — a neuron with large output weights naturally scores higher on CETT. So weight magnitude was already part of what made a neuron look like an H-Neuron in the first place. It does not need to be re-applied as a selection criterion.

**How much to suppress — calibrate by w_out_norm**

The same suppression factor produces very different absolute weight changes depending on the neuron's output magnitude:

```
Neuron 847   w_out_norm = 8.2  →  small factor (e.g. 10%) achieves large effect
Neuron 2341  w_out_norm = 1.1  →  larger factor needed for equivalent effect
```

High `w_out_norm` neurons need a smaller suppression factor to achieve the same hallucination reduction — and carry more collateral risk per unit of suppression. Start conservatively and increase iteratively.

**Suppression strategy:**

| Step | Action |
|---|---|
| 1 | Sort H-Neurons by SHAP score descending |
| 2 | Skip neurons with SHAP near zero (redundant) |
| 3 | Apply initial suppression factor scaled inversely to w_out_norm |
| 4 | Re-evaluate hallucination rate and MMLU after each neuron |
| 5 | Stop when hallucination rate plateaus or MMLU drops below threshold |

- Configurable suppression factor per neuron (10–50%)
- Iterative mode — suppress one neuron at a time, re-evaluate, continue
- Safety clamp — automatic stop if MMLU degrades beyond user-defined threshold
- All suppression factors saved to `h_neurons.json` and fully reversible

### Before / After Evaluation
- Re-runs the full hallucination dataset on the edited model
- Side-by-side comparison: hallucination rate, separation index, PaCMAP overlay
- General capability benchmark (MMLU subset) before and after — ensures neurons edited did not damage the model
- Delta report: exactly what changed, how much, where

### Audit Report Generation
- Auto-generated PDF report per run
- Lists every edited neuron: layer, index, original weight, new weight, suppression factor
- Before/after metrics: TruthfulQA score, HaluEval score, MMLU delta
- PaCMAP visualizations embedded in report
- Reproducibility section: random seeds, model version, dataset used

---

## CETT — The Core Neuron Metric

**CETT (Contribution-based Efficacy Token Test)** is the metric introduced by the H-Neurons paper for measuring how much a single feedforward neuron contributes to the model's information flow at a specific token position.

For neuron `j` at token position `t`:

```
CETT(j, t) = ‖ z_t^(j) · W_out[j, :] ‖₂  /  ‖ h_t ‖₂
```

Where:
- `z_t^(j)` — the post-activation scalar value of neuron `j` at token `t` (after GELU/ReLU)
- `W_out[j, :]` — the `j`-th row of the MLP down-projection matrix (neuron `j`'s "write vector" into the residual stream)
- `z_t^(j) · W_out[j, :]` — the rank-1 vector that neuron `j` contributes to the hidden state
- `h_t` — the full MLP output at token `t`

The ratio is a number between 0 and 1 expressing what **fraction of the total MLP output magnitude** at token `t` is attributable to neuron `j` alone.

### Why CETT over raw activations

Raw activation values (`z_t^(j)`) are not comparable across neurons — a neuron with small activations but large output weights can dominate the residual stream, while a neuron with large activations but tiny output weights may be irrelevant. CETT accounts for both, making it a true measure of influence rather than mere activity.

### The two CETT features per neuron (H-Neurons paper)

The paper aggregates CETT into exactly two features per neuron:

| Feature | Description |
|---|---|
| `CETT_answer` | Mean CETT across the **answer token span** — where hallucination happens |
| `CETT_other` | Mean CETT across all **non-answer tokens** — the baseline contribution |

A neuron with high `CETT_answer` and low `CETT_other` is specifically active during answer generation — a strong hallucination signal.

### Computing CETT with TransformerLens

TransformerLens exposes everything needed natively — no additional libraries required:

```python
# Hook into per-neuron post-activation values
z = model.hook("blocks.{layer}.mlp.hook_post")  # [batch, seq, d_mlp]

# Get the down-projection write vectors
W_out = model.W_out[layer]                       # [d_mlp, d_model]

# Compute neuron j's rank-1 contribution at token t
contribution = z[0, t, j] * W_out[j, :]         # [d_model]

# CETT for neuron j at token t
cett = contribution.norm() / h_t.norm()          # scalar
```

`W_out` norms can be precomputed once per model — making CETT efficient to compute across thousands of neurons and prompts.

---

## Feature Engineering — What Goes Into Detection

### Training Data Construction — Consistency Filtering
Following the H-Neurons paper: for each question, **10 responses are sampled** at non-zero temperature. Only the extremes are kept:
- **Always correct** — all 10 responses right
- **Always wrong** — all 10 responses wrong (confirmed hallucination, not refusal)

This removes ambiguous middle-ground examples and produces clean binary labels.

### The Three Per-Prompt Features

The pipeline computes three values **per neuron per prompt**. These are the only ML features — they vary per prompt so they carry actual discriminative information. Static properties like weight norms are the same for a neuron on every prompt, so they carry zero discriminative power and are excluded from all ML stages.

The correct-prompt baseline (mean and std of `CETT_answer` per neuron) is computed first across all consistently-correct responses. This baseline is what makes `CETT_zscore` meaningful.

| Feature | What it is | Varies per prompt? |
|---|---|---|
| `CETT_answer` | CETT score over answer tokens for this prompt | Yes |
| `CETT_other` | CETT score over non-answer tokens for this prompt | Yes |
| `CETT_zscore` | `(CETT_answer − mean_correct) / std_correct` — how unusual is this vs the neuron's normal behaviour | Yes |

> `CETT_zscore` is the most powerful of the three. A neuron firing at 0.031 is unremarkable if it always fires at 0.029. The same value when the correct baseline is 0.003 ± 0.002 is a 14σ event — that is an H-Neuron.

---

### Worked Example — 1,000 prompts, 4,096 neurons

> In practice numbers are much larger — tens of thousands of prompts, thousands of neurons per layer across multiple layers. The example uses small numbers for clarity.

**The raw matrix — computed once, cached to HDF5**

Each row is one prompt. Each neuron occupies 3 consecutive columns. Label = 1 (hallucinated) or 0 (correct).

```
         │←─────────── neuron 0 ───────────→│←─────────── neuron 1 ───────────→│ ... │←──────── neuron 4095 ───────────→│
         │ CETT_ans  CETT_other  CETT_zscore │ CETT_ans  CETT_other  CETT_zscore │     │ CETT_ans  CETT_other  CETT_zscore │ label
─────────┼──────────────────────────────────┼──────────────────────────────────┼─────┼──────────────────────────────────┼──────
prompt_1 │  0.031      0.012       3.21     │  0.002      0.001       0.11     │ ... │  0.001      0.000       0.08     │   1
prompt_2 │  0.028      0.011       2.98     │  0.003      0.001       0.15     │ ... │  0.001      0.000       0.07     │   1
prompt_3 │  0.004      0.009      -0.21     │  0.002      0.001       0.09     │ ... │  0.044      0.031       8.92     │   0
  ...
```

```
Total: 1,000 rows  ×  (4,096 neurons × 3 features)  =  1,000 × 12,288 matrix
```

---

### Scalability — Pre-filtering Before Boruta

In practice the feature space is far larger than the worked example. Boruta trains a Random Forest 100+ times — on a large model that becomes computationally prohibitive:

```
Llama 3.1 8B:
  14,336 neurons per layer × 12 middle layers × 3 features = 516,096 columns
  Boruta on 516,096 columns → days, not hours
```

The solution is a tiered pre-filter that reduces the space to something Boruta can handle using cheap arithmetic operations first, before any ML runs.

**Tier 1 — Delta pre-filter** *(instant — just arithmetic)*

Compute `CETT_answer_delta = mean(CETT_answer on halluc) − mean(CETT_answer on correct)` for every neuron. Keep only neurons with a meaningful positive delta — those that actually fire more during hallucination. Neurons with delta ≤ 0 cannot be H-Neurons by definition.

```
~172,000 neurons (14,336 × 12 layers)  →  ~17,000  (top 10% positive delta)
```

**Tier 2 — CETT_zscore spike threshold** *(instant — single pass)*

Of those, keep only neurons where at least one prompt produced a `CETT_zscore` above a threshold (e.g. > 3σ). A neuron that never spiked unusually on any individual prompt — even if its mean is slightly elevated — is unlikely to be an H-Neuron.

```
~17,000  →  ~3,000  (ever exceeded 3σ on at least one hallucinating prompt)
```

**Tier 3 — Boruta** *(now feasible)*

```
~3,000 neurons × 3 features = ~9,000 columns  →  tractable
Output: ~80–200 confirmed neurons
```

**Tier 4 — Fallback: LightGBM feature importance** *(if Boruta still times out)*

Skip Boruta and use LightGBM's built-in feature importance directly after Tier 2. Less statistically rigorous — it does not formally reject noise the way Boruta does — but far faster and still much better than no filtering at all. The UI warns the researcher when this fallback is active.

**Full pre-filter funnel:**

```
172,000 neurons  (14,336 per layer × 12 layers)
      ↓  Tier 1: delta pre-filter        (instant)
 ~17,000
      ↓  Tier 2: CETT_zscore threshold   (instant)
  ~3,000
      ↓  Tier 3: Boruta                  (feasible — minutes not days)
   80–200 confirmed
      ↓  Delta direction filter
   ~40 H-Neuron candidates
      ↓  LightGBM + SHAP
   Final ranked H-Neuron list
```

The UI shows estimated runtime before each stage and allows the researcher to configure thresholds (delta cutoff, zscore threshold, number of Boruta iterations) or switch to the LightGBM fallback if Boruta is too slow for their hardware.

---

### Pipeline Stages

| Stage | Rows | Columns | What it does | Why |
|---|---|---|---|---|
| **Boruta** | 1,000 prompts — each labeled hallucinated (1) or correct (0) | 4,096 neurons × 3 features = **12,288 columns** | Creates shuffled shadow copies of every column. Trains Random Forest on real + shadow together. Rejects any neuron whose 3 features cannot beat their own shuffled copies at predicting the label | Eliminates pure noise neurons before any expensive ML. Reduces 4,096 neurons to ~80 confirmed — those that carry any real hallucination signal |
| **Delta filter** | One value per confirmed neuron — no matrix | `mean(CETT_answer on halluc rows) − mean(CETT_answer on correct rows)` | Directly computes direction for each of the ~80 confirmed neurons. Keeps only neurons where delta > 0 | We have the labels so we can see directly which neurons fire more during hallucination. Neurons with delta < 0 fire more during correct responses — suppressing them would make hallucination worse |
| **LightGBM** | 1,000 prompts — same rows and labels as Boruta | ~40 H-Neuron candidates × 3 features = **~120 columns** | Trains gradient-boosted classifier to predict hallucination label. SHAP values give signed importance per neuron | Delta filter gives direction but not importance or interactions. LightGBM ranks H-Neurons by unique contribution and detects which neurons always fire together (see below) |
| **PaCMAP** | 1,000 prompts — same rows as LightGBM | ~40 H-Neuron candidates × 3 features = **~120 columns → compressed to 2D** | Projects the same 120-dimensional space into a 2D scatter plot. Each point = one prompt, red = hallucinated, blue = correct | Statistics treat all hallucinations as one group. PaCMAP reveals sub-clusters — specific types of hallucination driven by different neurons — that the researcher can lasso-select and investigate |

---

### Correlated Neurons — Suppress One, Not Both

LightGBM detects when two H-Neurons always fire together — whenever neuron 847 fires, neuron 1203 fires too, and vice versa. SHAP assigns near-full importance to one and near-zero to the other, because they carry redundant signal.

```
Neuron 847   SHAP = 0.041   ← carries the signal
Neuron 1203  SHAP = 0.003   ← redundant — fires with 847 but adds nothing unique
```

In this case suppressing **only neuron 847** achieves the same reduction in hallucination as suppressing both. This is important — every unnecessary weight edit risks degrading general capability. LightGBM's redundancy detection minimises the number of suppressions needed.

---

### Static features — display only

These have zero variance across prompts — the same value for a neuron regardless of which prompt is being processed. They cannot help any ML stage split hallucinated from correct prompts and are excluded from all training. They are shown in the neuron inspector UI only.

| Feature | Used for |
|---|---|
| `w_out_norm` | How large is this neuron's write vector into the residual stream |
| `weight_rank_in_layer` | Tie-breaking when two neurons score similarly |
| `weight_zscore_in_layer` | Highlighting structurally outlier neurons |
| `layer_index` | Layer heatmap display |
| `layer_relative_position` | Showing where in the network the neuron lives |

---

## AI + Human — Not a Black Box

LLMDeHallucinator is designed as a **collaborative tool**, not a fully automated pipeline. The L1 probe detects H-Neurons at scale; the researcher stays in control.

The automatic detector finds neurons that statistically predict hallucination. The PaCMAP visualization reveals the geometry — and geometry often tells you what statistics miss. A neuron with a confidence score of 0.61 might fire precisely on a tight sub-cluster of hallucination points that the top-ranked neuron never touches. Only a human looking at the visualization catches that.

```
Auto-detected neurons (L1 probe)
         +
Researcher inspects PaCMAP clusters
         +
Lasso-select → "which neurons fire here?"
         +
Manual add / remove candidates
         +
Re-run suppression with curated selection
```

This is **human-in-the-loop mechanistic interpretability** — ML detection at scale, human pattern recognition where it matters. The researcher can question the probe, catch false positives, and build genuine understanding of why specific neurons matter. Neither alone is as good as both together.

---

## Architecture

```
LLMDeHallucinator/
│
├── app.py                    # Dash entry point
│
├── pipeline/
│   ├── loader.py             # Model loading (HuggingFace + TransformerLens)
│   ├── dataset.py            # TruthfulQA / HaluEval / TriviaQA loading and labeling
│   ├── activations.py        # TransformerLens activation extraction
│   ├── detection.py          # H-Neuron identification (L1 probe + CETT)
│   ├── judge.py              # Optional LLM-as-judge labeling
│   ├── suppression.py        # Weight editing and model export
│   └── evaluation.py         # Before/after benchmark runner
│
├── visualization/
│   ├── pacmap_view.py        # PaCMAP projection + Plotly scatter
│   ├── layer_grid.py         # Small-multiples layer view
│   └── neuron_inspector.py   # Ranked neuron list with activation heatmap
│
├── report/
│   └── generator.py          # PDF report generation
│
├── ui/
│   ├── layout.py             # Dash layout definition
│   └── callbacks.py          # All Dash callbacks
│
├── notebooks/
│   └── colab_quickstart.ipynb  # Google Colab notebook for GPU runs
│
├── data/
│   └── pregenerated/         # Pre-labeled hallucination datasets
│
├── cache/
│   └── {model_id}/           # Top-level directory per model (e.g. llama-3.1-8b)
│       └── {session_id}/     # One sub-directory per run (dataset + date)
│           ├── session.json      # Session metadata: model, dataset, config, timestamps
│       ├── dataset.parquet       # Labeled prompts and responses (correct / hallucinating)
│       ├── cett/
│       │   ├── correct.h5        # CETT scores — correct prompts  [n_neurons × n_correct]
│       │   └── halluc.h5         # CETT scores — hallucinating prompts [n_neurons × n_halluc]
│       ├── features.parquet      # Full feature set per neuron (all contrast + static features)
│       ├── detection/
│       │   ├── boruta.json       # Boruta confirmed / rejected neuron lists
│       │   ├── lightgbm.json     # LightGBM scores + SHAP values per neuron
│       │   └── rankings.parquet  # Final ranked neuron list with all scores
│       ├── pacmap/
│       │   ├── coords_2d.npy     # PaCMAP 2D projection coordinates
│       │   └── coords_3d.npy     # PaCMAP 3D projection coordinates
│       ├── h_neurons.json        # H-Neuron selection state (see below)
│       └── evaluation/
│           ├── before.json       # Pre-suppression benchmark metrics
│           └── after.json        # Post-suppression benchmark metrics
│
├── tests/
│   └── test_pipeline.py      # Unit tests (GPT-2 Small, CPU)
│
└── requirements.txt
```

---

## Session & Cache Management

Computing CETT scores across thousands of neurons and hundreds of prompts is expensive. Running it every time a researcher adjusts the suppression factor or tweaks the neuron selection would make the tool unusable. Every computed artifact is therefore persisted to disk under `cache/{session_id}/` and reloaded on subsequent runs — no recomputation unless the source data or model changes.

### What is cached and when

| Artifact | File | Computed once when |
|---|---|---|
| Labeled dataset (correct / halluc prompts) | `dataset.parquet` | Dataset step completes |
| CETT scores — correct prompts | `cett/correct.h5` | Activation extraction completes |
| CETT scores — hallucinating prompts | `cett/halluc.h5` | Activation extraction completes |
| Full feature set per neuron | `features.parquet` | Feature engineering completes |
| Boruta results | `detection/boruta.json` | Boruta run completes |
| LightGBM scores + SHAP values | `detection/lightgbm.json` | LightGBM run completes |
| Ranked neuron list | `detection/rankings.parquet` | Detection completes |
| PaCMAP 2D / 3D coordinates | `pacmap/coords_2d.npy` | PaCMAP projection completes |
| H-Neuron selection state | `h_neurons.json` | Updated on every change |
| Evaluation results | `evaluation/before.json` / `after.json` | Evaluation runs complete |

On startup the UI shows which cache artifacts already exist for the current session and skips straight to the first uncompleted step.

### H-Neuron selection state — `h_neurons.json`

This file tracks the full history of the neuron selection, including manual researcher overrides:

```json
{
  "session_id": "llama-3.1-8b_2025-01-15",
  "model": "meta-llama/Llama-3.1-8B",
  "auto_detected": [
    {"layer": 12, "neuron": 847, "score": 0.94, "cohen_d": 3.21}
  ],
  "manually_added": [
    {"layer": 12, "neuron": 1203, "score": 0.61, "reason": "fires on halluc sub-cluster in PaCMAP"}
  ],
  "manually_removed": [
    {"layer": 15, "neuron": 302, "score": 0.88, "reason": "false positive — fires on code tokens"}
  ],
  "final_selection": [847, 1203, ...],
  "suppression_factors": {
    "12_847": 0.20,
    "12_1203": 0.15
  },
  "last_modified": "2025-01-15T14:23:11Z"
}
```

Every manual add, remove, and suppression factor change is written immediately. The researcher can undo any change, reload a previous selection state, or export the selection for use in a different session.

### Cache structure — per model, per session

```
cache/
├── llama-3.1-8b/
│   ├── truthfulqa_2025-01-15/    ← session 1
│   └── halu-eval_2025-01-22/     ← session 2 (different dataset, same model)
├── mistral-7b/
│   └── truthfulqa_2025-01-18/
└── gpt2/
    └── truthfulqa_2025-01-10/    ← dev run
```

CETT scores computed for a given model are stored under that model's directory and **reused across sessions** that use the same model — even with a different dataset. This means if you run TruthfulQA and then HaluEval on the same model, the `w_out_norm` and weight statistics are computed only once. Only the prompt-level CETT scores (which depend on the dataset) are recomputed.

### Cache invalidation rules

| What changed | What is invalidated | What is reused |
|---|---|---|
| Different model | Everything | Nothing |
| Same model, different dataset | Dataset, CETT scores, features, detection, PaCMAP | Weight statistics (`w_out_norm`, ranks) |
| Same model + dataset, different detection params | Detection results, PaCMAP, evaluation | CETT scores, features |
| Suppression factor changed | Evaluation only | Everything upstream |
| Neuron selection changed | Evaluation only | Everything upstream |

### Force Recalculate

Every pipeline step in the UI shows its current cache status and a **↺ Recalculate** button. Clicking it discards the cached result for that step and all steps downstream, then reruns from scratch — no config editing or CLI required. This is useful when:

- A new version of the dataset is available and CETT scores need refreshing
- Detection parameters were changed and a clean rerun is preferred over incremental updates
- The researcher suspects a cached result is stale or corrupted

```
[ ✓ CETT cached — loaded ]  [ ↺ Recalculate ]
[ ✓ Detection cached ]       [ ↺ Recalculate ]
[ ✓ PaCMAP cached ]          [ ↺ Recalculate ]
[ – Evaluation not run ]     [ ▶ Run ]
```

Recalculating a step automatically invalidates all downstream steps but leaves upstream cache intact. Recalculating CETT discards detection, PaCMAP, and evaluation — but not the dataset or weight statistics. The previous cached result is moved to a timestamped backup folder rather than deleted, so it can be restored if needed.

### Storage format rationale

| Format | Used for | Why |
|---|---|---|
| HDF5 (`.h5`) | CETT score matrices | Efficient columnar access to large float arrays; supports partial reads by neuron or by prompt |
| Parquet | Tabular data (dataset, features, rankings) | Columnar, compressed, fast pandas load |
| NumPy (`.npy`) | PaCMAP coordinates | Simple, fast, small |
| JSON | Selection state, detection results, evaluation metrics | Human-readable, easy to inspect and version |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/LLMDeHallucinator.git
cd LLMDeHallucinator
pip install -r requirements.txt
python app.py
# Open http://localhost:8050 in your browser
```

### Requirements

```
dash>=2.14
plotly>=5.18
transformer-lens>=2.0
pacmap>=0.7
torch>=2.1
transformers>=4.40
datasets>=2.18
scikit-learn>=1.4
pandas>=2.0
numpy>=1.26
reportlab>=4.0       # PDF report generation
```

---

## Quickstart

### Local development (CPU, GPT-2 Small)

```python
# GPT-2 Small (117M) runs on CPU in seconds — perfect for development
# Switch to Llama 3.1 8B on GPU for real research runs

python app.py --model gpt2 --device cpu
```

### Google Colab (A100 GPU, Llama 3.1 8B)

Open `notebooks/colab_quickstart.ipynb` in Google Colab.  
Select **Runtime → Change runtime type → A100 GPU**.  
Run all cells — the Dash app exposes itself via an ngrok tunnel.

### Vast.ai / RunPod (recommended for iterative research)

```bash
# ~$0.30/h on RTX 4090, ~$1.00/h on A100
# Clone repo, pip install, run app.py
# Saves model_out.safetensors to /outputs for download
```

---

## Workflow Walkthrough

### Step 1 — Load Model
Select a model from the dropdown (GPT-2, Llama 3.1 8B, Mistral 7B, Phi-3 Mini, or any HuggingFace path). GPU memory requirement is shown automatically.

**As soon as a model is selected, the UI checks `cache/{model_id}/` for existing sessions.** If previous runs exist, they are listed with their dataset, date, and completed steps. The researcher can resume any prior session — jumping directly to the first incomplete step — or start a fresh run. No recomputation of already-cached artifacts.

```
Model selected: Llama 3.1 8B
  ┌─────────────────────────────────────────────┐
  │ Existing sessions found:                    │
  │                                             │
  │ ● truthfulqa_2025-01-15   [complete]        │
  │   → Resume: go to suppression / export      │
  │                                             │
  │ ● halu-eval_2025-01-22    [CETT cached]     │
  │   → Resume: start from detection step       │
  │                                             │
  │ + Start new session                         │
  └─────────────────────────────────────────────┘
```

### Step 2 — Prepare Dataset
Choose a pre-packaged dataset (TruthfulQA, HaluEval, TriviaQA) or upload your own CSV. Optionally enable **LLM-as-judge** mode for richer labeling — recommended for non-factual hallucination detection. If a cached dataset exists for this model + dataset combination, it is loaded instantly.

### Step 3 — Run Activation Extraction & CETT
The pipeline runs the dataset through the model via TransformerLens, extracts per-neuron post-activation values, and computes CETT scores for every neuron across correct and hallucinating prompts. Results are written to `cett/correct.h5` and `cett/halluc.h5` immediately. On any subsequent run this step is skipped entirely and the cached HDF5 files are loaded directly.

### Step 4 — H-Neuron Detection
Choose your detection mode: **L1 probe** (fast baseline), **LightGBM + SHAP** (non-linear, captures neuron interactions), or **SAE** (frontier, monosemantic feature decomposition — planned). Boruta runs first to eliminate uninformative neurons. Results are cached in `detection/`. Changing detection parameters re-runs only this step — CETT scores upstream are untouched.

### Step 5 — PaCMAP Visualization
The CETT activation space is projected to 2D or 3D with PaCMAP. Hallucination responses cluster separately from correct responses — most clearly in middle layers 8–20. Projections are cached in `pacmap/` and reloaded instantly on revisit. Use the layer slider to animate through the network and observe cluster formation.

### Step 6 — Select Neurons for Suppression
Neurons are pre-selected based on detection confidence. Review and adjust in the neuron inspector — manually add or remove neurons via the ranked list or by lasso-selecting PaCMAP clusters. Every change is saved immediately to `h_neurons.json`, including the reason for manual overrides. The full history of additions and removals is preserved and reversible.

### Step 7 — Suppress Weights
Set suppression factor per neuron (recommended: start at 20%). Click **Apply**. The model weights are edited in memory. Suppression factors are saved to `h_neurons.json`.

### Step 8 — Before / After Evaluation
The dataset is re-run on the edited model. PaCMAP overlay shows the shift in cluster separation. Hallucination rate and MMLU delta are displayed side by side. Results saved to `evaluation/before.json` and `evaluation/after.json`. Adjusting only the suppression factor and re-evaluating reuses all upstream cache — only the evaluation step reruns.

### Step 9 — Export
Save the edited model as `.safetensors`. Download the PDF audit report. Optionally push to HuggingFace Hub.

---

## Limitations and Known Challenges

**Labeling quality**  
Simple exact-match labeling against TriviaQA/TruthfulQA only captures factual hallucinations. LLM-as-judge mode extends coverage but adds latency and cost. Non-factual hallucinations (false reasoning, confabulation) remain harder to label automatically.

**Suppression trade-off**  
The original H-Neurons paper notes that simple weight suppression can reduce model helpfulness. LLMDeHallucinator mitigates this with gradual iterative suppression and MMLU monitoring, but the fundamental tension between compliance and truthfulness is an open research problem.

**Model coverage**  
Tested on GPT-2 (development) and Llama 3.1 8B (primary target). H-Neuron patterns may differ across architectures. Contributions testing on Mistral, Phi, Gemma welcome.

**GPU requirement**  
Models above 3B parameters require a GPU with 16GB+ VRAM for comfortable use. Google Colab Pro (A100) or Vast.ai are recommended for 7B/8B models.

---

## Roadmap

- [x] Project architecture design
- [ ] Core pipeline (TransformerLens + TruthfulQA + L1 probe baseline)
- [ ] LightGBM detection mode with SHAP neuron importance
- [ ] Dash UI skeleton with model loader
- [ ] PaCMAP visualization with hover/zoom and manual lasso selection
- [ ] Neuron weight suppression and model export
- [ ] Before/after evaluation with MMLU
- [ ] PDF report generation
- [ ] LLM-as-judge labeling mode
- [ ] 3D PaCMAP view
- [ ] Small-multiples layer grid
- [ ] Google Colab quickstart notebook
- [ ] HuggingFace Hub integration
- [ ] Support for Mistral, Phi-3, Gemma
- [ ] SAE-based detection (monosemantic feature decomposition)

---

## Contributing

Contributions welcome. This is an early-stage research project — the most valuable contributions right now are:

1. Testing the pipeline on models other than GPT-2 and Llama 3.1 8B
2. Improving the LLM-as-judge labeling pipeline
3. Experimenting with alternative suppression strategies (e.g. LoRA-based rather than direct weight editing)
4. Adding new benchmark datasets

Please open an issue before starting significant work.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built with Python, Dash, TransformerLens, PaCMAP, and a genuine interest in making LLMs more honest.*
