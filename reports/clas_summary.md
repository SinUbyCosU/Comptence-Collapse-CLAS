## Initial Baseline (Judged, no CLAS)
| Model | N | Assert | Complex | EmotDist | InstrDensity |
| --- | --- | --- | --- | --- | --- |
| gemini_flash.csv | 600 | 7.372 | 6.15 | 3.47 | 8.702 |
| llama31_8b | 600 | 6.537 | 4.502 | 4.797 | 6.71 |
| nous_hermes_mistral_7b | 593 | 6.826 | 5.592 | 4.346 | 7.939 |
| openhermes_mistral_7b | 581 | 6.685 | 5.504 | 4.219 | 7.854 |
| phi35_mini | 702 | 3.597 | 5.472 | 7.09 | 3.021 |
| qwen25_7b | 600 | 7.202 | 5.94 | 3.858 | 8.572 |
| qwen2_7b | 600 | 7.287 | 7.522 | 9.103 | 3.498 |
| yi15_6b | 586 | 5.531 | 4.205 | 5.323 | 4.59 |
| zephyr_7b | 600 | 6.508 | 5.507 | 4.393 | 7.157 |

## CLAS Delta (Steered − Baseline)
| Model | N | Δ Assert | Δ Complex | Δ EmotDist | Δ InstrDensity |
| --- | --- | --- | --- | --- | --- |
| nous_hermes_mistral | 5 | 0.4 | -0.6 | 0.4 | 0.2 |
| openhermes_mistral | 5 | 0.4 | 0.4 | -2.6 | 1.6 |
| phi35_mini | 5 | -0.6 | -1.0 | -0.4 | 3.4 |
| qwen25_7b | 5 | 0.2 | 1.0 | 0.2 | 0.2 |
| qwen2_7b | 5 | 1.0 | 0.8 | -1.4 | 0.2 |
| zephyr_7b | 5 | 0.6 | 0.6 | 1.0 | 0.2 |

Overall Δ (N=30): assert=0.333, complex=0.2, emotDist=-0.467, instrDensity=0.967

## CLAS Baseline Averages (Same subset)
| Model | N | Assert | Complex | EmotDist | InstrDensity |
| --- | --- | --- | --- | --- | --- |
| nous_hermes_mistral | 5 | 7.2 | 6.6 | 3.0 | 8.6 |
| openhermes_mistral | 5 | 7.2 | 5.4 | 5.0 | 7.4 |
| phi35_mini | 5 | 7.4 | 6.6 | 3.8 | 5.4 |
| qwen25_7b | 5 | 7.6 | 6.0 | 4.0 | 8.8 |
| qwen2_7b | 5 | 6.8 | 5.6 | 5.2 | 8.8 |
| zephyr_7b | 5 | 7.4 | 6.0 | 3.8 | 8.8 |

## CLAS Steered Averages (Same subset)
| Model | N | Assert | Complex | EmotDist | InstrDensity |
| --- | --- | --- | --- | --- | --- |
| nous_hermes_mistral | 5 | 7.6 | 6.0 | 3.4 | 8.8 |
| openhermes_mistral | 5 | 7.6 | 5.8 | 2.4 | 9.0 |
| phi35_mini | 5 | 6.8 | 5.6 | 3.4 | 8.8 |
| qwen25_7b | 5 | 7.8 | 7.0 | 4.2 | 9.0 |
| qwen2_7b | 5 | 7.8 | 6.4 | 3.8 | 9.0 |
| zephyr_7b | 5 | 8.0 | 6.6 | 4.8 | 9.0 |

## Non-CLAS External Models (Counts Only)
- Gemini Flash CSV rows (incl. header): 25581 (not CLAS-processed)
- ChatGPT CSV rows (incl. header): 1123 (not CLAS-processed)