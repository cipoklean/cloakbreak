# CloakBreak — Vast.ai Compute & Submission Log

Tracks every Vast.ai GPU session to verify the contest >=1 active GPU-hour rule
and enforce the strict **$5.00** budget cap. Every instance is destroyed
immediately after checkpoints/artifacts are pushed (no stopped instances incurring
idle storage charges).

## Budget Cap
- **Target max spend:** `$5.00`
- **Hard rule:** Pause and get confirmation before launching any run that would push cumulative spend over `$5.00`.

## GPU Session Ledger

| Session | Date (WAT) | GPU Type | Vast Instance ID | Purpose | Active Hours | Rate ($/hr) | Disk/Net ($) | Session Cost ($) | Cumulative ($) | Artifacts Pushed | Destroyed? |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| 0 (Prep) | 2026-11-24 | Oracle ARM CPU | N/A | Day 0 scaffold & CPU unit tests | 0.00 | $0.00 | $0.00 | $0.00 | $0.00 | Local repo scaffold | N/A |
| 1 | TBD | 1x RTX 3090 | TBD | Dataset download + Chatterbox/Kokoro spoof generation + smoke test | TBD | TBD | TBD | TBD | TBD | HF dataset manifest | TBD |
| 2 | TBD | 1x RTX 3090 | TBD | Baseline evals + WavLM training + INT8 ONNX export | TBD | TBD | TBD | TBD | TBD | HF checkpoint + ONNX | TBD |
| 3 | TBD | 1x RTX 3090 | TBD | Held-out & honest-limits evals + live Arena test | TBD | TBD | TBD | TBD | TBD | `eval/results.json` | TBD |
| 4 | TBD | 1x RTX 3090 | TBD | Unedited live demo video capture | TBD | TBD | TBD | TBD | TBD | Final demo video | TBD |

## Verified Hugging Face Dependencies (Checked 2026-11-24)

| Role | Primary HF Repo | Gated | License | Fallback HF Repo |
| :--- | :--- | :---: | :--- | :--- |
| Detector backbone | `microsoft/wavlm-base-plus` | No | MIT (UniLM) | `facebook/wav2vec2-base` (Apache-2.0) |
| Legacy ASVspoof baseline | `Vansh180/deepfake-audio-wav2vec2` | No | MIT | `garystafford/wav2vec2-deepfake-voice-detector` (Apache-2.0) |
| 2026 voice cloner (train/demo) | `ResembleAI/chatterbox` (Turbo 350M / 500M) | No | MIT | `SWivid/F5-TTS` (`CC-BY-NC-4.0`, non-commercial — disclose in README if used) |
| Unseen held-out TTS | `hexgrad/Kokoro-82M` | No | Apache-2.0 | `facebook/mms-tts-eng` (`CC-BY-NC-4.0`, non-commercial — disclose in README if used) |
| Real/In-The-Wild dataset | `SpeechAntiSpoofingBenchmarks/InTheWild` | No | Apache-2.0 | `mueller91/In-The-Wild` (CC-BY-SA-4.0) |
| Diverse English sentence pool | `agentlans/high-quality-english-sentences` | No | ODC-BY | `databricks/databricks-dolly-15k` (CC-BY-SA-3.0) |
| Optional neural-codec dataset | `abdulahh35/ANC-Spoof` *(cut first if tight)* | No | CC-BY-4.0 | On-the-fly Opus/8kHz augmentation |

## Submission Checklist
- [ ] GitHub repo created (`cipoklean`) & commits authored solely as `cipoklean`
- [ ] Vast.ai account funded ($5.00) & entry form submitted at `https://vast.ai/showcase`
- [ ] >=1 active GPU-hour logged above with instance IDs
- [ ] All published metrics measured directly from `eval/results.json`
- [ ] Voice-cloning endpoint disabled (`ENABLE_CLONE_ENDPOINT=false`) on public demo
- [ ] Feature freeze hit Sat Nov 28, 22:00 WAT
- [ ] Final release commit tagged before Mon Nov 30, 08:59 WAT (Sun Nov 29, 11:59 PM PT)
- [ ] Public X post with `@vast_ai`, `#BuiltOnVast`, video, GPU info, reused components
