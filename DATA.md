# Data Availability Statement

## Overview

This repository contains the complete experimental data, pre-computed steering vectors, and human evaluation results for the paper "Competence Collapse in Code-Mixed Generation: Spectral Evidence and Mechanistic Recovery via Cross-Lingual Activation Steering."

## Data Structure

### 1. Experimental Results

**Location**: Available in separate data repository (see below)

- `clas_ablation_results.jsonl` - Main experimental results
- `clas_multi_model_results_prefix2_judged_seed1.jsonl` - Multi-model comparison (seed 1)
- `clas_multi_model_results_prefix2_judged_seed2.jsonl` - Multi-model comparison (seed 2)

**Format**: JSONL (JSON Lines), one result per line

**Download**: Full experimental data (~11 MB) available at:
- 📦 **Data Repository**: [github.com/SinUbyCosU/competence-collapse](https://github.com/SinUbyCosU/competence-collapse)
- 🔗 **Zenodo Archive**: [DOI: 10.5281/zenodo.XXXXX](https://zenodo.org/record/XXXXX) (coming soon)

### 2. Human Evaluation Data

**Location**: `PromptPersona_Full_600-human annotated`

- **Size**: 600 prompts with 100% human annotation
- **Annotators**: 3 native Hindi-English bilingual speakers per prompt
- **Inter-rater reliability**: ICC(3,1) = 0.89 (Excellent)
- **Metrics**: Instructional Density, Actionability, Concreteness, Utility (1-10 scale)

**Privacy Note**: All prompts are synthetic scenarios across 10 high-stakes domains. No personally identifiable information (PII) is included.

### 3. Model-Specific Results

**Location**: Available in data repository

Each model directory contains:
- `diagnostics.json` - Experimental diagnostics and metadata
- `layer{N}_eng.npy` - English activation representations (Layer N) [~500 KB each]
- `layer{N}_hin.npy` - Hinglish activation representations (Layer N) [~500 KB each]

**Format**: 
- JSON for diagnostics (included here in `figures/diagnostics_summary.json`)
- NumPy `.npy` files for activation vectors (float32) - available in data repository

### 4. Pre-computed Steering Vectors

**Location**: Available in data repository

- `pc1_layer_{0-32}.pt` - Principal component vectors for all 33 layers [~100 KB each]
- `comilingua_vector.pt` - Domain-specific vector (COMI-LINGUA dataset)
- `spanglish_vector.pt` - Cross-lingual generalization vector
- `tech_vector.pt` - Technical support domain vector

**Format**: PyTorch tensors (`.pt` files) [Total ~3 MB]

**Calibration**: Extracted from N=50 paired English-Hinglish prompts

**Note**: Due to GitHub size limits, these large binary files are hosted in the separate [data repository](https://github.com/SinUbyCosU/competence-collapse).

### 5. Analysis Results

**Location**: `figures/`

- `ablations_manifest.json` - Complete ablation metadata
- `delta_by_*.json` - Summary statistics by various dimensions
- `diagnostics_summary.json` - Aggregated model diagnostics

### 6. Generated Outputs

**Location**: `outputs_local/`

- `judged_outputs_llamaguard.jsonl` - LLM-judge safety evaluations
- Visualization files (`.png` format)

### 7. Reports and Documentation

**Location**: `reports/`

- `CLAS_600_Report.md` - Detailed experimental report
- `clas_summary.md` - Executive summary
- `README_figures.md` - Figure descriptions
- `figures/` - Publication-ready visualizations (SVG and PNG)

## Data Usage Guidelines

### Citation

If you use any data from this repository, please cite:

```bibtex
@article{yadav2026competence,
  title={Competence Collapse in Code-Mixed Generation: Spectral Evidence and Mechanistic Recovery via Cross-Lingual Activation Steering},
  author={Yadav, Tanushree Ravindra Pratap},
  journal={arXiv preprint},
  year={2026},
  institution={Indian Institute of Science Education and Research (IISER) Bhopal}
}
```

### Licensing

- **Code and Methodology**: MIT License (see [LICENSE](LICENSE))
- **Data**: CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **Human Annotations**: CC BY 4.0

### Ethical Considerations

1. **Medical Prompts**: All medical domain prompts are synthetic and designed for linguistic analysis only. They must NOT be used for actual medical advice.

2. **Safety Data**: The repository includes adversarial prompts used for safety testing. These are provided for reproducibility and research transparency but should be handled responsibly.

3. **Language Representation**: While we focus on Hinglish, we acknowledge that code-mixing is a natural linguistic phenomenon with significant sociolinguistic context. Results should be interpreted with cultural sensitivity.

4. **Model Limitations**: The data reflects model behavior as of evaluation time (2026). Model updates may change results.

## Reproducing Results

### Loading Experimental Data

```python
import json
import pandas as pd

# Load main results
with open('clas_ablation_results.jsonl', 'r') as f:
    results = [json.loads(line) for line in f]
df = pd.DataFrame(results)

# Load model diagnostics
with open('phi35_mini/diagnostics.json', 'r') as f:
    diagnostics = json.load(f)
```

### Loading Steering Vectors

```python
import torch

# Load layer-specific steering vector
layer_16_vector = torch.load('vectors/pc1_layer_16.pt')

# Load domain-specific vectors
tech_vector = torch.load('vectors/tech_vector.pt')
spanglish_vector = torch.load('vectors/spanglish_vector.pt')
```

### Loading Activation Data

```python
import numpy as np

# Load English and Hinglish activations
eng_activations = np.load('phi35_mini/layer16_eng.npy')
hin_activations = np.load('phi35_mini/layer16_hin.npy')

# Compute difference vector
diff_vector = eng_activations - hin_activations
```

## Data Size

**This Repository**: ~500 KB (documentation, summaries, and reports)

**Complete Dataset** (in separate data repository): ~11 MB

Breakdown:
- Activation vectors (`.npy` files): ~6 MB
- Steering vectors (`.pt` files): ~3 MB
- Results and metadata (`.jsonl`, `.json`): ~1 MB
- Reports and figures: ~1 MB

**Access**:
- 📁 **Code & Documentation**: This repository
- 📦 **Complete Data**: [github.com/SinUbyCosU/competence-collapse](https://github.com/SinUbyCosU/competence-collapse)

## Known Issues and Limitations

1. **Activation Sparsity**: Some models have fewer layers; activation data only available for evaluated layers.

2. **Cross-Domain Transfer**: Steering vectors are calibrated on specific domains. Cross-domain transfer shows degradation (44-63% as reported in Appendix I).

3. **Language Identification**: FastText LID achieves F1=0.88 on Romanized Hindi. Minor boundary errors may affect CMI calculations (see Appendix E).

4. **Reproducibility**: Due to non-determinism in transformer inference, exact numeric reproduction may vary by ±2-3% for Instructional Density scores.

## Contact for Data Questions

For questions about data access, format, or usage:

**Tanushree Ravindra Pratap Yadav**  
Email: yadav23@iiserb.ac.in  
Institution: IISER Bhopal

## Version History

- **v1.0.0** (2026-01-30): Initial release with complete experimental data
