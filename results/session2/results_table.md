backend: torch:cuda
model: checkpoints/run1/cloakbreak_best.pt
baseline: Vansh180/deepfake-audio-wav2vec2
| Eval Slice | Clips (Real/Spoof) | Baseline EER (%) | CloakBreak EER (%) | CloakBreak AUROC | p50 / p95 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| test_clean | 590 (295/295) | 40.00 | 3.73 | 0.9905 | 20.7 / 23.6 |
| test_opus_16k | 590 (295/295) | 41.36 | 5.76 | 0.9866 | 26.8 / 30.0 |
| test_phone_8k | 590 (295/295) | 46.10 | 9.15 | 0.9603 | 20.8 / 23.2 |
| test_noise_10db | 590 (295/295) | 63.39 | 16.61 | 0.8730 | 21.1 / 23.8 |
| test_noise_5db | 590 (295/295) | 68.14 | 21.02 | 0.8554 | 20.7 / 23.2 |
| test_short_0.5s | 590 (295/295) | 55.93 | 52.54 | 0.4843 | 20.6 / 23.1 |
| test_short_0.8s | 590 (295/295) | 59.32 | 35.59 | 0.6938 | 20.6 / 22.0 |
| unseen_kokoro_clean | 600 (300/300) | 84.33 | 2.00 | 0.9982 | 20.6 / 24.5 |
| unseen_kokoro_opus_16k | 600 (300/300) | 88.00 | 3.33 | 0.9958 | 26.8 / 29.9 |
| clean_real_vs_kokoro_clean | 600 (300/300) | 59.00 | 6.33 | 0.9822 | 19.9 / 20.5 |
| clean_real_vs_kokoro_opus_16k | 600 (300/300) | 58.17 | 5.67 | 0.9837 | 26.4 / 28.8 |
| rolloff_matched_clean | 590 (295/295) | 31.53 | 6.78 | 0.9790 | 20.0 / 22.4 |
| rolloff_matched_unseen_kokoro_clean | 590 (295/295) | 71.19 | 4.07 | 0.9904 | 20.1 / 22.2 |
| Floor: trivial 4-feature logistic (dur/RMS/SNR/rolloff) | 600 | N/A | 36.03 | 0.7084 | N/A |

Clean human speech FPR@0.5 (600 clips): CloakBreak = 18.17% (mean prob 0.1871, p95 0.9456) | Baseline FPR@0.5: 0.17%
