# Competence Collapse in Code-Mixed Generation

**Spectral Evidence and Mechanistic Recovery via Cross-Lingual Activation Steering**

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](reports/CLAS_600_Report.md)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.XXXX%2Fxxxxxx-blue)](https://doi.org/10.XXXX/xxxxxx)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

> **Author**: Tanushree Ravindra Pratap Yadav  
> **Institution**: Indian Institute of Science Education and Research (IISER) Bhopal  
> **Contact**: yadav23@iiserb.ac.in  
> **ORCID**: [0009-0004-0411-255X](https://orcid.org/0009-0004-0411-255X)

---

## Overview

This repository contains the code, data, and experimental results for our paper on **Competence Collapse**, a distinct pathology where Large Language Models (LLMs) capable of complex reasoning in English exhibit severe utility degradation when prompted in code-mixed languages like Hinglish (Hindi-English).

### Key Findings

- **Service Gap Quantified**: Statistically significant decline in instructional quality (∆D ≈ -11.3%, p < 0.001) across 9 diverse architectures
- **CLAS Recovery**: Cross-Lingual Activation Steering recovers utility by ∆D = +2.22 (Cohen's d = 0.60) using only N = 50 calibration pairs
- **Safety Preservation**: Jailbreak rate reduced by 52.5% while maintaining code-mixed fidelity (CMI ≈ 0.4)
- **Human Validation**: 600 prompts with 100% human annotation (inter-rater ICC = 0.89, Human-LLM correlation ρ = 0.87)

## Repository Structure

```
ablations/
├── README.md                          # This file
├── reports/                          # Analysis reports and documentation
│   ├── CLAS_600_Report.md           # Main experimental report
│   ├── clas_summary.md              # Summary of findings
│   ├── README_figures.md            # Figure descriptions
│   └── figures/                     # Generated plots and visualizations
├── figures/                          # Summary statistics and metadata
│   ├── ablations_manifest.json
│   ├── delta_by_alpha_*.json
│   ├── delta_by_layer_*.json
│   ├── delta_by_model_*.json
│   └── diagnostics_summary.json
├── vectors/                          # Pre-computed steering vectors
│   ├── pc1_layer_*.pt               # Principal components per layer
│   ├── comilingua_vector.pt         # Domain-specific vectors
│   ├── spanglish_vector.pt
│   └── tech_vector.pt
├── abalation model wise/             # Model-specific ablation results
│   ├── llama31_8b/
│   ├── nous_hermes_mistral/
│   ├── openhermes_mistral/
│   ├── phi35_mini/
│   ├── qwen2_7b/
│   ├── qwen25_7b/
│   └── zephyr_7b/
├── outputs_local/                    # Generated outputs and judgments
│   └── judged_outputs_llamaguard.jsonl
├── clas_ablation_results.jsonl       # Main experimental results
├── clas_multi_model_results_*.jsonl  # Multi-model comparison data
└── PromptPersona_Full_600-human annotated  # Human evaluation dataset
```

## Key Contributions

### 1. Competence Collapse Identification

We identify and quantify a novel failure mode where models exhibit **categorical utility loss** in code-mixed contexts despite maintaining semantic understanding. This is distinct from:
- Capacity dilution
- Catastrophic forgetting
- Simple translation failure

### 2. Spectral Analysis Framework

We introduce a geometric interpretation of the service gap:
- **Steerability Window**: Layers 12-16 show peak spectral concentration (PC1 explained variance ≈ 26%)
- **Coherence Cliff**: Cosine similarity drops below τ ≈ 0.98 at mid-layers
- **Entropy Phase**: High-dimensional divergence in late layers (16+)

### 3. Cross-Lingual Activation Steering (CLAS)

A lightweight, inference-time intervention that:
- Requires only **N = 50 calibration pairs** (one-time)
- Injects a "Competence Gap Vector" into residual stream
- **No weight updates or fine-tuning required**
- Preserves safety protocols while recovering utility

### 4. Comprehensive Human Validation

- **600 prompts** across 10 high-stakes domains
- **3 annotators per prompt** (native Hindi-English bilinguals)
- **Inter-rater reliability**: ICC(3,1) = 0.89 (Excellent)
- **Human-LLM correlation**: Spearman ρ = 0.87

## Quick Links

-  **[Full Paper](reports/CLAS_600_Report.md)** - Detailed experimental report
-  **[Reproduction Guide](REPRODUCE.md)** - Step-by-step instructions
-  **[Data Documentation](DATA.md)** - Data availability and usage
-  **[Citation](CITATION.cff)** - BibTeX and citation metadata
-  **[License](LICENSE)** - MIT License

## Installation

```bash
# Clone the repository
git clone https://github.com/SinUbyCosU/competence-collapse.git
cd competence-collapse

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**System Requirements**:
- Python 3.9+
- 16GB RAM (32GB recommended)
- GPU with 16GB+ VRAM (for model inference)
- 20GB storage

**Data Access**:
-  **This Repository**: Documentation, code, and analysis scripts
-  **Complete Dataset**: [github.com/SinUbyCosU/competence-collapse](https://github.com/SinUbyCosU/competence-collapse) (11 MB)
  - Includes: steering vectors (.pt), activation data (.npy), full results (.jsonl)

## Quick Start

### 1. Reproduce Spectral Analysis

```python
import torch
import numpy as np
from pathlib import Path

# Load pre-computed steering vectors
vectors_dir = Path("vectors")
layer_16_vector = torch.load(vectors_dir / "pc1_layer_16.pt")

# Examine spectral properties
print(f"Vector shape: {layer_16_vector.shape}")
print(f"L2 norm: {torch.norm(layer_16_vector).item():.4f}")
```

### 2. Examine Model-Specific Results

```python
import json

# Load diagnostics for a specific model
with open("phi35_mini/diagnostics.json") as f:
    diagnostics = json.load(f)

print(f"Baseline Instructional Density: {diagnostics['baseline_id']}")
print(f"CLAS Recovery: +{diagnostics['delta_id']:.2f}")
print(f"Effect Size (Cohen's d): {diagnostics['cohens_d']:.2f}")
```

### 3. View Summary Statistics

```python
# Load overall summary
with open("figures/delta_overall_summary.json") as f:
    summary = json.load(f)

for model, stats in summary.items():
    print(f"{model}: ΔD = {stats['mean_delta']:.2f} ± {stats['std']:.2f}")
```

## Evaluation Metrics

### Instructional Density (D)

Quantifies actionability via imperative verbs and structural elements:

```
D(y) = Σ I_imp(s) + λ Σ I_struct(b)
```

where:
- `I_imp(s) = 1` if sentence contains imperative verb
- `I_struct(b) = 1` if block contains structural markers
- `λ = 0.5` (structural weight)

### Code-Mixing Index (CMI)

Measures language fidelity:

```
CMI(y) = 1 - max(p_eng, p_hin)
```

Higher CMI indicates better code-mixing preservation.

## Models Evaluated

### Service Gap Audit (9 models)
- Gemini 1.5 Flash
- Llama 3.1 8B
- Yi 1.5 6B
- Qwen 2.5 7B
- Qwen 2 7B
- Phi-3.5 Mini
- Zephyr 7B
- OpenHermes 2.5
- Nous Hermes 2

### CLAS Intervention (6 open-weight models)
- Phi-3.5 Mini (Best: ΔD = +5.61, d = 2.89)
- Qwen 2 7B (ΔD = +2.38, d = 0.57)
- OpenHermes 2.5 (ΔD = +2.31, d = 0.62)
- Zephyr 7B (ΔD = +2.04, d = 0.59)
- Nous Hermes 2 (ΔD = +1.53, d = 0.52)
- Qwen 2.5 7B (ΔD = -0.71, "Alignment Ceiling")

## Key Results

### Service Gap (Baseline)

| Model | Human ID | LLM ID | Correlation |
|-------|----------|--------|-------------|
| Gemini 1.5 Flash | 7.62 | 7.46 | 0.89 |
| Qwen 2.5 7B | 7.41 | 7.29 | 0.85 |
| Llama 3.1 8B | 5.74 | 5.61 | 0.86 |
| **Phi-3.5 Mini** | **4.39** | **4.27** | **0.81** |

### CLAS Recovery (Human-Validated)

| Model | ΔD | 95% CI | Cohen's d | Outcome |
|-------|-----|--------|-----------|---------|
| Phi-3.5 Mini | +5.61 | [+4.92, +6.30] | 2.89 | Latent Recovery |
| Qwen 2 7B | +2.38 | [+1.24, +3.52] | 0.57 | Robust Gain |
| OpenHermes 2.5 | +2.31 | [+0.78, +3.84] | 0.62 | Robust Gain |
| Zephyr 7B | +2.04 | [+1.09, +2.99] | 0.59 | Robust Gain |

### Safety Reinforcement

| Metric | Baseline | CLAS |
|--------|----------|------|
| Jailbreak Rate ↓ | 43.3% | 20.3% |
| False Refusal Rate ↓ | 12.5% | 5.0% |
| Cohen's κ (Inter-rater) | 0.93 | 0.93 |

## Reproducing Paper Results

### 1. Service Gap Audit

```bash
# Results are pre-computed in:
# - clas_ablation_results.jsonl
# - figures/diagnostics_summary.json

# View summary
python -c "
import json
with open('figures/diagnostics_summary.json') as f:
    data = json.load(f)
    print(json.dumps(data, indent=2))
"
```

### 2. Spectral Analysis

```bash
# View layer-wise spectral properties
python -c "
import torch
from pathlib import Path

for layer in range(0, 33):
    try:
        vec = torch.load(f'vectors/pc1_layer_{layer}.pt')
        print(f'Layer {layer:2d}: shape={vec.shape}, norm={torch.norm(vec):.4f}')
    except FileNotFoundError:
        continue
"
```

### 3. Alpha Sensitivity Analysis

```bash
# Results available in:
# - figures/delta_by_alpha_summary.json
# - figures/delta_by_alpha_layer_summary.json
```

## Domain Coverage

Our evaluation spans 10 high-stakes domains:
1. **Medical Triage** (100 prompts)
2. **Legal Advice** (60 prompts)
3. **Financial Planning** (60 prompts)
4. **Technical Support** (80 prompts)
5. **Educational Tutoring** (60 prompts)
6. **Career Counseling** (60 prompts)
7. **Mental Health** (60 prompts)
8. **Travel Planning** (40 prompts)
9. **Home Maintenance** (40 prompts)
10. **Parenting Advice** (40 prompts)

## Limitations

1. **Language Scope**: Primary validation on Hinglish; preliminary Spanglish results show transferability
2. **Lexical Bottleneck**: CLAS recovers reasoning but cannot add missing vocabulary
3. **Calibration Sensitivity**: Domain-specific calibration recommended (see cross-domain transfer in paper)
4. **Emotional Distance Trade-off**: Higher α increases utility but reduces conversational warmth
5. **Threat Model**: Safety evaluation assumes non-adaptive adversaries
Reproducing Results

For detailed reproduction instructions, see [REPRODUCE.md](REPRODUCE.md).

**Quick verification**:
```python
import json
import pandas as pd

# Verify main results
with open('clas_ablation_results.jsonl', 'r') as f:
    results = [json.loads(line) for line in f]
df = pd.DataFrame(results)

print(f"Average ΔD: {df['delta_id'].mean():.2f}")  # Expected: 2.22
```

## Data Availability

All experimental data, human annotations, and pre-computed steering vectors are publicly available. See [DATA.md](DATA.md) for:
- Data structure and formats
- Download instructions
- Ethical considerations
- Version information

**Repository Structure**:
-  This repository: Publication-ready code and documentation (~500 KB)
-  [Data repository](https://github.com/SinUbyCosU/competence-collapse): Complete experimental data (~11 MB)

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add YourFeature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## Contact

**Tanushree Ravindra Pratap Yadav**  
Indian Institute of Science Education and Research (IISER) Bhopal  
 Email: yadav23@iiserb.ac.in  
 ORCID: [0009-0004-0411-255X](https://orcid.org/0009-0004-0411-255X)

## License

- **Code**: MIT License (see [LICENSE](LICENSE))
- **Data**: CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **Paper**: arXiv license
}
```

## Contact

**Tanushree Ravindra Pratap Yadav**  
Indian Institute of Science Education and Research (IISER) Bhopal  
Email: yadav23@iiserb.ac.in  
ORCID: 0009-0004-0411-255X

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Human annotators for comprehensive evaluation (N = 600 prompts)
- OpenAI, Meta, Microsoft, and Qwen teams for open-weight models
- COMI-LINGUA dataset creators for evaluation framework

## Related Work

- **Activation Addition**: Turner et al. (2023) - Foundation for steering methods
- **Contrastive Activation Addition**: Rimsky et al. (2024) - Safety steering
- **Representation Engineering**: Zou et al. (2023) - Theoretical framework
- **Code-Switching Safety**: Deng et al. (2023) - Adversarial vectors
- **GLUECoS**: Khanuja et al. (2020) - Evaluation benchmark

---

**Note**: This is research code. Medical domain examples are for linguistic analysis only and should not be treated as professional medical advice. Safety evaluation assumes non-adaptive threat models; production deployment requires additional safeguards.
