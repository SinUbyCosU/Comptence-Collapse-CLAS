# Reproduction Guide

This document provides step-by-step instructions to reproduce the key results from "Competence Collapse in Code-Mixed Generation: Spectral Evidence and Mechanistic Recovery via Cross-Lingual Activation Steering."

## Prerequisites

### System Requirements

- **OS**: Linux, macOS, or Windows 10+
- **RAM**: 16 GB minimum (32 GB recommended)
- **GPU**: NVIDIA GPU with 16GB+ VRAM (for model inference)
- **Storage**: 20 GB free space
- **Python**: 3.9 or higher

### Software Dependencies

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

## Quick Start: Verifying Pre-computed Results

### 1. Load and Inspect Main Results

```python
import json
import pandas as pd
import numpy as np

# Load experimental results
with open('clas_ablation_results.jsonl', 'r') as f:
    results = [json.loads(line) for line in f]

df = pd.DataFrame(results)

# Verify key statistics
print(f"Total experiments: {len(df)}")
print(f"Models evaluated: {df['model'].nunique()}")
print(f"Average ΔD: {df['delta_id'].mean():.2f}")

# Expected output:
# Total experiments: 2184
# Models evaluated: 6
# Average ΔD: 2.22
```

### 2. Verify Human Evaluation Agreement

```python
import json

# Load diagnostics summary
with open('figures/diagnostics_summary.json', 'r') as f:
    diagnostics = json.load(f)

# Check inter-rater reliability
icc = diagnostics.get('inter_rater_icc', None)
spearman_rho = diagnostics.get('human_llm_correlation', None)

print(f"Inter-rater ICC(3,1): {icc}")
print(f"Human-LLM Spearman ρ: {spearman_rho}")

# Expected output:
# Inter-rater ICC(3,1): 0.89
# Human-LLM Spearman ρ: 0.87
```

### 3. Verify Spectral Analysis Results

```python
import torch
import numpy as np

# Load steering vectors for multiple layers
layers_to_check = [5, 13, 16, 28]
norms = {}

for layer in layers_to_check:
    try:
        vec = torch.load(f'vectors/pc1_layer_{layer}.pt')
        norms[layer] = torch.norm(vec).item()
        print(f"Layer {layer:2d}: norm = {norms[layer]:.4f}, shape = {vec.shape}")
    except FileNotFoundError:
        print(f"Layer {layer} not found")

# Verify steerability window (layers 12-16 should have higher norms)
# Expected: Layer 13 and 16 have higher norms than 5 or 28
```

## Reproducing Core Results

### Table 1: CLAS Steering Results (Human-Validated)

```python
import pandas as pd
import json

# Load model-specific diagnostics
models = ['phi35_mini', 'qwen2_7b', 'openhermes_mistral', 
          'zephyr_7b', 'nous_hermes_mistral', 'qwen25_7b']

results = []
for model in models:
    try:
        with open(f'{model}/diagnostics.json', 'r') as f:
            diag = json.load(f)
        results.append({
            'Model': model.replace('_', ' ').title(),
            'ΔD': diag['delta_id'],
            'Effect Size (d)': diag['cohens_d'],
            'Baseline': diag['baseline_id'],
            'Post-CLAS': diag['baseline_id'] + diag['delta_id']
        })
    except FileNotFoundError:
        print(f"Diagnostics not found for {model}")

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# Expected top result: Phi-3.5 Mini with ΔD ≈ +5.61, d = 2.89
```

### Figure 2: Spectral Fingerprinting

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load steering vectors across all layers
layers = range(0, 33)
norms = []
exists = []

for layer in layers:
    try:
        vec = torch.load(f'vectors/pc1_layer_{layer}.pt')
        norms.append(torch.norm(vec).item())
        exists.append(True)
    except:
        norms.append(0)
        exists.append(False)

# Plot norm distribution
plt.figure(figsize=(10, 5))
plt.plot([i for i, e in enumerate(exists) if e], 
         [n for n, e in zip(norms, exists) if e], 
         marker='o', linewidth=2, markersize=6)
plt.axvspan(12, 16, alpha=0.2, color='green', label='Steerability Window')
plt.xlabel('Layer')
plt.ylabel('Vector Norm')
plt.title('Steering Vector Magnitude Across Layers')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('reproduced_spectral_fingerprint.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Peak norm at layer: {np.argmax(norms)}")
# Expected: Peak around layer 13-16
```

### Safety Evaluation (Table 5)

```python
import json

# Load safety evaluation results
with open('outputs_local/judged_outputs_llamaguard.jsonl', 'r') as f:
    safety_data = [json.loads(line) for line in f]

df_safety = pd.DataFrame(safety_data)

# Calculate jailbreak rates
baseline_jailbreak = df_safety[df_safety['condition'] == 'baseline']['is_jailbreak'].mean() * 100
clas_jailbreak = df_safety[df_safety['condition'] == 'clas']['is_jailbreak'].mean() * 100

print(f"Baseline Jailbreak Rate: {baseline_jailbreak:.1f}%")
print(f"CLAS Jailbreak Rate: {clas_jailbreak:.1f}%")
print(f"Reduction: {baseline_jailbreak - clas_jailbreak:.1f}%")

# Expected: ~43.3% → 20.3% (52.5% reduction)
```

## Advanced: Re-running CLAS Inference

**Note**: This requires access to the models and significant compute resources.

### Step 1: Load a Pre-trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
```

### Step 2: Load Steering Vector

```python
# Load pre-computed steering vector for layer 16
steering_vector = torch.load('vectors/pc1_layer_16.pt').to(model.device)

# Normalize
steering_vector = steering_vector / torch.norm(steering_vector)

print(f"Steering vector shape: {steering_vector.shape}")
print(f"Steering vector device: {steering_vector.device}")
```

### Step 3: Apply CLAS Intervention

```python
def apply_clas_steering(model, layer_idx, steering_vector, alpha=1.0):
    """
    Apply CLAS steering to a specific layer during inference.
    
    Args:
        model: HuggingFace model
        layer_idx: Target layer index (e.g., 16 for L/2 in 32-layer model)
        steering_vector: Pre-computed competence gap vector
        alpha: Steering coefficient
    """
    
    def steering_hook(module, input, output):
        # output is (batch_size, seq_len, hidden_dim)
        hidden_states = output[0] if isinstance(output, tuple) else output
        
        # Apply prompt-bound injection (only to prompt tokens)
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Compute norm-scaled steering
        hidden_norms = torch.norm(hidden_states, dim=-1, keepdim=True)
        steering_component = alpha * steering_vector.unsqueeze(0).unsqueeze(0)
        steering_component = steering_component * hidden_norms / torch.norm(steering_vector)
        
        # Inject to all prompt tokens
        hidden_states = hidden_states + steering_component
        
        return (hidden_states,) if isinstance(output, tuple) else hidden_states
    
    # Register hook on target layer
    target_layer = model.model.layers[layer_idx]
    hook_handle = target_layer.register_forward_hook(steering_hook)
    
    return hook_handle

# Example usage
hook = apply_clas_steering(model, layer_idx=16, steering_vector=steering_vector, alpha=1.0)

# Generate with steering
prompt = "mere pet mein tez dard ho raha hai, kya karun?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)

# Remove hook when done
hook.remove()
```

### Step 4: Compute Instructional Density

```python
import spacy

# Load SpaCy for imperative verb detection
nlp = spacy.load("en_core_web_sm")

def compute_instructional_density(text, lambda_weight=0.5):
    """
    Compute Instructional Density (D) as defined in Equation 1.
    
    D(y) = Σ I_imp(s) + λ Σ I_struct(b)
    """
    doc = nlp(text)
    
    # Count imperative sentences
    imperative_count = 0
    for sent in doc.sents:
        # Simple heuristic: sentence starts with verb in base form
        if sent[0].pos_ == "VERB" and sent[0].tag_ == "VB":
            imperative_count += 1
    
    # Count structural elements
    structural_count = 0
    # Lists (bullet points, numbers)
    structural_count += text.count('\n- ')
    structural_count += text.count('\n* ')
    structural_count += sum(1 for line in text.split('\n') if line.strip() and line.strip()[0].isdigit())
    
    # Code blocks
    structural_count += text.count('```')
    
    # Headings
    structural_count += text.count('\n#')
    
    # Compute total density
    density = imperative_count + lambda_weight * structural_count
    
    return {
        'density': density,
        'imperative_count': imperative_count,
        'structural_count': structural_count
    }

# Example
baseline_response = "doctor ko dikhao"  # Low utility
clas_response = response  # Your CLAS output

baseline_d = compute_instructional_density(baseline_response)
clas_d = compute_instructional_density(clas_response)

print(f"Baseline D: {baseline_d['density']:.2f}")
print(f"CLAS D: {clas_d['density']:.2f}")
print(f"ΔD: {clas_d['density'] - baseline_d['density']:.2f}")
```

## Validation Checklist

Use this checklist to verify your reproduction:

- [ ] ✅ Installed all dependencies from `requirements.txt`
- [ ] ✅ Loaded main experimental results (N=2184)
- [ ] ✅ Verified global average ΔD ≈ +2.22
- [ ] ✅ Confirmed Phi-3.5 Mini achieves ΔD ≈ +5.61
- [ ] ✅ Verified inter-rater reliability ICC = 0.89
- [ ] ✅ Confirmed Human-LLM correlation ρ = 0.87
- [ ] ✅ Loaded steering vectors for layers 0-32
- [ ] ✅ Identified steerability window at layers 12-16
- [ ] ✅ Verified safety improvements (jailbreak reduction)
- [ ] ✅ (Optional) Re-ran inference with CLAS on sample prompts

## Common Issues and Solutions

### Issue 1: GPU Memory Error

**Solution**: Use smaller batch sizes or quantized models

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # Use 8-bit quantization
    device_map="auto"
)
```

### Issue 2: Steering Vector Shape Mismatch

**Solution**: Verify model architecture matches

```python
# Check hidden dimension
print(f"Model hidden size: {model.config.hidden_size}")
print(f"Steering vector size: {steering_vector.shape}")

# They should match
assert model.config.hidden_size == steering_vector.shape[0]
```

### Issue 3: Slightly Different Numerical Results

**Expected**: Due to floating-point precision and non-deterministic operations, results may vary by ±2-3%.

**Solution**: Set random seeds

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
```

## Computing Infrastructure Used

For reference, the original experiments were conducted on:

- **GPU**: NVIDIA A100 (40GB) × 2
- **CPU**: AMD EPYC 7742 (64 cores)
- **RAM**: 512 GB
- **OS**: Ubuntu 22.04 LTS
- **CUDA**: 12.1
- **PyTorch**: 2.1.0

## Timeline for Full Reproduction

- **Data loading and verification**: 10 minutes
- **Reproducing figures and tables**: 30 minutes
- **Re-running CLAS inference (1 model, 100 prompts)**: 2-4 hours
- **Full experimental suite (6 models, 600 prompts)**: 24-48 hours

## Contact for Reproduction Issues

If you encounter issues reproducing results:

1. **Check existing issues**: https://github.com/SinUbyCosU/competence-collapse/issues
2. **Open a new issue** with:
   - Python version
   - Dependency versions (`pip freeze`)
   - Error message and traceback
   - Minimal reproducible example

**Maintainer**: Tanushree Ravindra Pratap Yadav (yadav23@iiserb.ac.in)

## Citation

If you use this reproduction guide or code:

```bibtex
@misc{yadav2026competence_code,
  author = {Yadav, Tanushree Ravindra Pratap},
  title = {Competence Collapse in Code-Mixed Generation: Reproduction Code},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/SinUbyCosU/competence-collapse}
}
```
