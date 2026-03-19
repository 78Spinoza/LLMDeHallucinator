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

This project directly builds on and extends:

- **H-Neurons** (Gao et al., Dec 2025) — arXiv:2512.01797
  Identified that <0.1% of neurons reliably predict hallucination events. Established the CETT metric and L1-probe methodology. Found that H-Neurons originate during pre-training, not alignment.

- **Inference-Time Intervention** (Li et al., 2023)
  Demonstrated that targeted activation editing at inference time can improve truthfulness.

- **Sparse Autoencoders for Interpretability** (Bricken et al., Anthropic 2023)
  Showed that sparse, interpretable features can be extracted from LLM activations.

- **Entity Recognition via SAE Latents** (Ferrando et al., ICLR 2025)
  Found linear directions encoding model self-knowledge about whether it knows a fact.

### Key Hypothesis

> H-Neurons concentrated in layers 8–20 are sufficient for hallucination detection and suppression, with a MMLU accuracy delta below 2% at a 20% suppression factor.

This hypothesis is testable and measurable within the LLMDeHallucinator pipeline — and represents a concrete contribution beyond the original H-Neurons paper, which stopped short of building a production suppression tool.

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
- Interactive scatter plot — zoom, pan, hover, lasso-select
- Each point = one model response, colored by hallucination/correct
- Hover reveals: original prompt, model response, layer, contributing neurons
- Layer-by-layer animation to see how clusters form across the network
- Small-multiples view: all layers simultaneously in a grid
- Separation index per layer — quantifies cluster quality numerically

### Neuron Weight Suppression
- Select neurons for editing directly from the visualization or from the ranked list
- Configurable suppression factor (e.g. reduce weight by 10–50%)
- Iterative mode: suppress gradually and re-evaluate after each step
- Safety clamp: MMLU accuracy monitored in parallel — automatic stop if general capability degrades beyond threshold

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

## Feature Engineering — What Goes Into Detection

H-Neuron detection is only as good as its features. Raw activations alone are not enough — a neuron firing at 2.3 means nothing without knowing what *normal* looks like for that neuron. The pipeline computes a rich feature set by running the dataset twice: once on **correct (non-hallucinating) prompts** to establish a per-neuron baseline, and once on **hallucinating prompts** to measure the deviation.

### Baseline Statistics — Correct Prompts
Computed per neuron across all non-hallucinating responses:

| Feature | Description |
|---|---|
| `mean_activation_correct` | Average firing level during correct responses |
| `std_activation_correct` | Spread — how consistently does this neuron fire on correct responses? |
| `p25 / p75 / p95_correct` | Percentile distribution of correct activations |
| `cett_mean_correct` | Average CETT score during correct responses |

### Hallucination Statistics — Hallucinating Prompts
Computed per neuron across all hallucinating responses:

| Feature | Description |
|---|---|
| `mean_activation_halluc` | Average firing level during hallucinations |
| `std_activation_halluc` | Spread during hallucinations |
| `p25 / p75 / p95_halluc` | Percentile distribution of hallucination activations |
| `cett_mean_halluc` | Average CETT score during hallucinations |

### Contrast Features — The Signal
Derived by comparing the two distributions:

| Feature | Description |
|---|---|
| `activation_delta` | `mean_halluc − mean_correct` — raw lift during hallucination |
| `cohen_d` | Effect size between the two distributions — how separable are they? |
| `activation_zscore` | How many std above the correct baseline is this neuron firing right now? |
| `kl_divergence` | How different are the two activation distributions overall? |

### Static Neuron Features — Structural Properties
Fixed per neuron, independent of any prompt:

| Feature | Description |
|---|---|
| `weight_l2_norm` | Total weight magnitude — larger = more downstream influence |
| `weight_rank_in_layer` | Rank by weight magnitude among all neurons in the same layer |
| `weight_zscore_in_layer` | How far above average is this neuron's weight magnitude? |
| `layer_index` | Which layer this neuron belongs to |
| `layer_relative_position` | `layer / total_layers` — where in the network (0 = early, 1 = late) |

### Per-Prompt Dynamic Features
Computed fresh for every prompt, combined with the above at inference time:

| Feature | Description |
|---|---|
| `raw_activation` | Neuron activation value for this specific prompt |
| `activation_zscore_vs_correct` | `(raw − mean_correct) / std_correct` — deviation from normal |
| `activation_percentile_vs_correct` | Where does this activation fall in the correct distribution? |
| `cett_score` | CETT contribution for this prompt |

### Why Both Distributions Matter

> A neuron that fires at 2.3 on a hallucinating prompt is unremarkable if it always fires at 2.3. The same neuron firing at 2.3 when its correct-prompt mean is 0.4 with std 0.2 is a 9.5σ event — that is an H-Neuron.

Boruta uses these features to reject neurons with no discriminative power. LightGBM then learns the interaction patterns — e.g. high `cohen_d` AND high `weight_rank_in_layer` AND spiking `cett_score` — that a linear probe could never capture.

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
├── tests/
│   └── test_pipeline.py      # Unit tests (GPT-2 Small, CPU)
│
└── requirements.txt
```

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

### Step 2 — Prepare Dataset
Choose a pre-packaged dataset (TruthfulQA, HaluEval, TriviaQA) or upload your own CSV. Optionally enable **LLM-as-judge** mode for richer labeling — recommended for non-factual hallucination detection.

### Step 3 — Run Activation Extraction
The pipeline runs the dataset through the model via TransformerLens and caches activations for all (or selected) layers. Estimated runtime shown before starting.

### Step 4 — H-Neuron Detection
Choose your detection mode: **L1 probe** (fast baseline), **LightGBM** (non-linear, captures neuron interactions with SHAP importance scores), or **SAE** (frontier, monosemantic feature decomposition — planned). Results shown as a ranked neuron list with confidence scores. Layer heatmap shows where H-Neurons concentrate across the network.

### Step 5 — PaCMAP Visualization
The activation space is projected to 2D or 3D with PaCMAP. Hallucination responses cluster separately from correct responses — most clearly in middle layers 8–20. Use the layer slider to animate through the network and observe cluster formation.

### Step 6 — Select Neurons for Suppression
Neurons are pre-selected based on detection confidence. Review and adjust in the neuron inspector. The UI shows for each neuron: which layer, what its activation pattern looks like, and which prompts it fires on.

### Step 7 — Suppress Weights
Set suppression factor (recommended: start at 20%). Click **Apply**. The model weights are edited in memory.

### Step 8 — Before / After Evaluation
The dataset is re-run on the edited model. PaCMAP overlay shows the shift in cluster separation. Hallucination rate and MMLU delta are displayed side by side. If MMLU drops beyond threshold, a warning is shown and you can reduce the suppression factor.

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
