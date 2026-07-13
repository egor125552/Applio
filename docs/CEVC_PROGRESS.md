# CEVC implementation progress

Branch: `agent/compact-expressive-vc-architecture`

This checklist records completed engineering steps separately from the long-form architecture document. A step is closed only after code, tests, and the user-facing Colab path exist.

## Milestone 0 — Reproducible baseline

- [x] Create the research branch from current `main`.
- [x] Add the architecture document.
- [x] Prepare a stable Applio baseline notebook with downloadable diagnostics.
- [x] Run one real Colab conversion on a Tesla T4 and confirm the stable baseline starts and converts audio.
- [x] Capture environment, inference time, application output, and GPU process information.
- [ ] Add a standalone benchmark command that produces a machine-readable JSON report.
- [ ] Add fixed reference WAV files or checksums for automated audio regression testing.

## Experiment 1 — Generator registry

- [x] Add `rvc/lib/algorithm/generators/registry.py`.
- [x] Move generator selection out of `Synthesizer`.
- [x] Preserve legacy checkpoint names and unknown-name fallback to `HiFi-GAN`.
- [x] Preserve the existing no-F0 behaviour for unsupported vocoders.
- [x] Preserve the exact constructor arguments for HiFi-GAN, MRF HiFi-GAN, and RefineGAN.
- [x] Add import and alias tests.
- [x] Add state-dict equivalence tests against the legacy constructor path.
- [x] Add CPU shape, forward, backward, finite-value, amplitude, and optimizer-step tests.
- [x] Add deterministic-seed and checkpoint save/load tests.
- [x] Add a conditional CUDA smoke test.
- [x] Add a clean Colab notebook for the registry branch; no tests are embedded in it.
- [x] Add separate notebook JSON, syntax, branch, log-button, and clean-output tests.
- [x] Add GitHub Actions checks for the registry and notebook.
- [ ] Confirm a real conversion in Colab using the registry branch and compare it with the previously confirmed stable baseline.

## Experiment 2 — Roughness Adapter

- [ ] Freeze the baseline model.
- [ ] Implement the first small `RoughnessAdapter`.
- [ ] Keep trainable parameters below 1.5 million.
- [ ] Add `roughness = 0.0 / 0.5 / 1.0` controls.
- [ ] Add paired clean/rough voice data preparation.
- [ ] Add audio and metric comparison against the baseline.

## Current gate

Experiment 1 is code-complete after CI passes. It is acoustically closed only after one real Colab conversion from the registry notebook succeeds and its output is compared with the previously confirmed stable baseline.
