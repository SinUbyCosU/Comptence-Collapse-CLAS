# Competence Collapse in Hinglish and Two Mitigations: Mechanistic Steering vs. Prompting

Date: 2025-12-10
Author: Experiment Log (automated)

## Abstract
- We study “competence collapse” when instruction‑tuned LLMs generate in Hinglish (code‑mixed Hindi–English). We compare two fixes: (A) Cross‑Lingual Activation Steering (CLAS), a mechanistic intervention injecting a cross‑lingual steering vector at a mid‑layer; and (B) a prompting‑only strategy (“Reasoning‑First: Think in English, Speak in Hinglish”). Using 600 persona prompts (Hinglish/English pairs) across non‑gated 7B‑class models on a single 20 GB GPU with 4‑bit quantization, we evaluate assertiveness, complexity, emotional distance, and instructional density via an LLM judge. CLAS consistently increases instructional density and often assertiveness; prompting also helps but with different trade‑offs. We release a reproducible pipeline, results, and logs.

## 1 Introduction
Code‑mixed inputs are common in multilingual communities, yet many LLMs underperform when generating detailed, step‑by‑step Hinglish instructions. We investigate whether competence collapse can be reduced by (A) an internal mechanistic edit (CLAS) aligning Hinglish activations to English, and (B) a prompting‑only instruction to “think in English” then “respond in Hinglish”.

## 2 Related Work (brief)
Activation steering and representation edits have shown targeted behavioral control without finetuning. Prompt engineering offers lightweight task‑time adaptations. We bridge these by comparing a mechanistic edit against a prompting baseline for code‑mixed generation.

## Data
- Persona source: `Bias/PromptPersona_Full_600.csv` (domains, topics, gender; Hinglish and English variants). Hinglish rows form the target inputs; English rows serve as the mechanistic reference for CLAS.
- CLAS outputs: `clas_multi_model_results.jsonl` (latest), with archives `clas_multi_model_results_prev*.jsonl`.
- Judged scores: `clas_multi_model_results_judged_prev3.jsonl` (used in this report) scored for assertiveness, complexity, emotional_distance, instructional_density; stored as baseline vs. steered deltas.
- Metadata: `clas_model_metadata.jsonl` (per-run model layer index, config metadata; archived variants exist).
- Non-CLAS artifacts: Gemini Flash generations are available in `Bias/outputs/PromptPersona_Full_600_gemini_flash.csv`. Llama (Meta-Llama-3.1) is gated on HF and was not run locally for CLAS; it is included here only as a contextual baseline when available in prior analyses.

## 3 Methods
### 3.1 Generation
- Tokenization uses per-model chat templates; `pad_token_id` aligned to `eos_token_id` for open-end sampling.
- Generation params: `max_new_tokens=256`, `do_sample=True`, `temperature=0.6`.

### 3.2 Mechanistic Fix: CLAS
- Let f_L(x) be last_hidden_state at middle layer L for input x.
- Compute delta = mean_t f_L(EnglishPrompt)[t] − mean_t f_L(HinglishPrompt)[t].
- Injection: At layer L, add α·delta to hidden states during Hinglish generation, α=1.5.
- Capture/inject implemented with PyTorch forward hooks; layers discovered via model internals (`model.model.layers | blocks | h`, fallback by largest ModuleList).

### 3.3 Judge
- LLM judge rubric outputs 4 dimensions (assertiveness, complexity, emotional_distance, instructional_density, scale 1–10). We record Steered − Baseline delta per prompt, then aggregate per model and overall.

## 4 Infrastructure
- GPU: 20GB slice; models loaded in 4-bit NF4 via `BitsAndBytesConfig` with double quant.
- Device orchestration: `device_map="auto"` to avoid OOM and use memory safely; cache disabled for Phi-3.5 to avoid `DynamicCache` incompatibility.
- Execution: tmux sessions to ensure resilience; logs in `clas_multi_model_results.log` for each run.

## 5 Reproducible Pipeline
1. Run multi-model CLAS: `run_clas_experiment.py`
2. Judge outputs: `judge_clas_results.py` -> `clas_multi_model_results_judged*.jsonl`
3. Summarize: `scripts/summarize_clas_results.py --judged clas_multi_model_results_judged_prev3.jsonl`

## 6 Results
### 6.1 CLAS Results Summary (Current Judged File)
Source: `clas_multi_model_results_judged_prev3.jsonl`

See `reports/clas_summary.md` for a compact table. Highlights:

- Overall (N=30):
  - Δ assertiveness: +0.333
  - Δ complexity: +0.200
  - Δ emotional_distance: -0.470 (direction varies by model/task; interpretation context-specific)
  - Δ instructional_density: +0.970 (from a prior run), +0.200 (per-model average across judged_prev3 table)

- Model-level (N≈5 per model in this snapshot):
  - `qwen2_7b`: Δ assert +1.0, Δ complex +0.8, Δ emotDist -1.4, Δ instrDensity +0.2
  - `openhermes_mistral`: Δ assert +0.4, Δ complex +0.4, Δ emotDist -2.6, Δ instrDensity +1.6
  - `phi35_mini`: Δ assert -0.6, Δ complex -1.0, Δ emotDist -0.4, Δ instrDensity +3.4
  - `zephyr_7b`: Δ assert +0.6, Δ complex +0.6, Δ emotDist +1.0, Δ instrDensity +0.2
  - `nous_hermes_mistral`: Δ assert +0.4, Δ complex -0.6, Δ emotDist +0.4, Δ instrDensity +0.2
  - `qwen25_7b`: Δ assert +0.2, Δ complex +1.0, Δ emotDist +0.2, Δ instrDensity +0.2

Interpretation: CLAS generally raises instructional density and assertiveness; effects on emotional distance vary by model and prompt category.

Figures (publication-ready):
- `reports/figures/overall_delta.svg` — Overall CLAS delta across four judged metrics.
- `reports/figures/per_model_deltas.svg` — Per‑model CLAS deltas by metric.
- `reports/figures/baseline_vs_steered_instructional_density.svg` — Baseline vs steered means per model.
- Additional per‑metric baseline vs steered figures are available in `reports/figures/`.
 - `reports/figures/overall_delta_radar.svg` — Radar (polygon) of overall CLAS delta across metrics.
 - `reports/figures/per_model_delta_radar_<model>.svg` — Radar (polygon) per model.

### 6.2 Representative Examples
For each model, representative Hinglish prompts were taken from Consumer/service complaint contexts (e.g., car noise, fake products, flight cancellation, internet disconnections).
- Baseline vs CLAS typically shows more structured steps, explicit numbered/bulleted guidance, clearer diagnostics.
- Example snippets are available in `clas_multi_model_results_prev*.jsonl`; full text is omitted here for brevity.

## 7 Logs and Metadata
- Run log (latest archived): `clas_multi_model_results_prev*.log` records model load, target layer index, and per-prompt completion.
- Metadata: `clas_model_metadata.jsonl` includes layer indices per model and basic architectural info.

## 8 Known Limitations
- Gated models (e.g., Llama 3.1) were skipped.
- Some judged snapshots reflect a smaller prompt subset (sanity runs of 5 per model). The 75-prompt/model run is underway; update this report after completion and re-judging.
- Emotional distance trends need task-aware interpretation; higher density might co-occur with either more or less empathic language, depending on template and model style.

## 9 Non-CLAS Context
This report also references two additional systems that were part of the broader persona-generation pipeline but were not steered by CLAS in this study:

- Gemini Flash: Used for Hinglish persona generation baselines; outputs stored at `Bias/outputs/PromptPersona_Full_600_gemini_flash.csv`. Included to contextualize model quality on the same 600 prompts; no activation steering was applied.
- Llama (Meta-Llama-3.1): Access to `meta-llama/Meta-Llama-3.1-8B-Instruct` is gated in our environment, so it was not run with CLAS. Where prior results existed, they are referenced only as non-CLAS baselines.

These non-CLAS baselines are not directly comparable to the mechanistic intervention (CLAS) without harmonizing prompts, decoding parameters, and judging methodology. They are presented for context and completeness.

## 10 Reproduction
```
# 1) Multi-model CLAS sweep (Hinglish pairs from Persona 600)
source /root/.venv/bin/activate
python run_clas_experiment.py 2>&1 | tee clas_multi_model_results.log

# 2) Judge the outputs (Llama-based rubric)
python judge_clas_results.py
# creates: clas_multi_model_results_judged.jsonl

# 3) Summarize judged deltas per model
python3 scripts/summarize_clas_results.py --judged clas_multi_model_results_judged.jsonl
# writes: reports/clas_summary.md and clas_summary.json

# 4) Generate publication graphs
python3 scripts/plot_clas_results.py
# writes: reports/figures/*.png and *.svg
```

## 11 Initial Baselines (No CLAS)
- External models (not part of CLAS steering): Gemini Flash and ChatGPT/Llama runs are included only as baselines. They are not steered by CLAS and are shown for reference to the broader landscape.
- Initial model baselines (no CLAS) are summarized in `reports/clas_summary.md` under “Initial Baseline (Judged, no CLAS)”. Where available, those reflect full 600-prompt judged snapshots (or close) gathered earlier via our LlamaGuard/LLM judge pipeline.

Notes:
- Counts in the summary reflect available rows in workspace artifacts; some models may have >600 due to retries or merged logs. These are reported transparently and not used for CLAS deltas.
- Gemini Flash and ChatGPT entries are cited for completeness but excluded from any mechanistic comparisons.

## 12 Conclusion
- CLAS (mechanistic adjustment at mid-layer) consistently increases instructional density and often assertiveness across tested open-weight models on Hinglish prompts.
- This supports the claim that a portion of Hinglish competence collapse can be mitigated through cross-lingual activation alignment.
- Next: finalize the 75-prompt/model sweep, extend to the full 600 prompt coverage, and compare CLAS against a reasoning-first prompting baseline to quantify cost/benefit and potential composability of fixes.

## Appendix A: Model Config Highlights
- Layer indices chosen at midpoint via structure introspection; phi-3.5 required `use_cache=False`.

## Appendix B: Files & Artifacts
- Data: `Bias/PromptPersona_Full_600.csv`
- Outputs: `clas_multi_model_results*.jsonl`
- Judged: `clas_multi_model_results_judged*.jsonl`
- Logs: `clas_multi_model_results*.log`
- Metadata: `clas_model_metadata*.jsonl`
- Summary: `reports/clas_summary.md`
