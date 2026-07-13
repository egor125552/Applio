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
- [x] Pass the GitHub Actions registry and notebook job on the real repository.
- [x] Fix silent Colab startup: unbuffered app output, live progress, port detection, and Colab proxy fallback.
- [x] Confirm real Colab inference on the registry branch with the standard HiFi-GAN path.

### Real Colab registry result

- Environment: Tesla T4, PyTorch 2.7.1+cu128, CUDA 12.8.
- Research commit tested: `e4f3ba9bfdfc1fcb9e239b871f025349029e6048`.
- Standard checkpoint detected: RVC v2, F0 enabled, 40 kHz, HiFi-GAN, 27,537,346 parameters.
- Cold conversion: 54.68 seconds.
- Warm conversion with the model already loaded: 16.65 seconds.
- Peak observed GPU memory: 5,259 MiB.
- Idle/loaded application GPU memory: about 1,176–1,400 MiB.
- Installation commands exited successfully; no traceback or conversion failure was recorded.

## Experiment 2 — Roughness Adapter

- [x] Add a disabled-by-default `RoughnessAdapter` extension point to `Synthesizer`.
- [x] Preserve the legacy state-dict surface while the adapter is disabled.
- [x] Make an untrained adapter and `roughness = 0` exact identity paths.
- [x] Keep the production adapter below 1.5 million trainable parameters.
- [x] Freeze the baseline model and construct an optimizer from adapter parameters only.
- [x] Save the adapter as a separate `.cevc.pth` checkpoint tied to the base checkpoint hash.
- [x] Add scalar and time-varying `roughness` controls to the model API.
- [x] Extend preprocessing to accept iPhone M4A/AAC inputs and preserve source filenames/labels.
- [x] Ignore hidden macOS metadata files and make CEVC data validation require a real `G_*.pth` base checkpoint.
- [x] Extract energy, spectral tilt, HNR, band aperiodicity, and F0 instability from the existing Extract step.
- [x] Add automatic clean/rough/mixed filename hints and a per-frame roughness manifest.
- [x] Validate the uploaded Russian clean/rough/mixed iPhone filenames and confirm ordered roughness ranges.
- [x] Reuse the current Train UI model name, epochs, batch size, GPU, sample rate, vocoder, save interval, and checkpointing settings.
- [x] Add data validation and adapter-only training buttons without a second model selector.
- [x] Add a clean CEVC Adapter Colab notebook and standalone module/integration/notebook tests.
- [x] Pass the expanded CEVC GitHub Actions job (`29281872700`).
- [ ] Run the first real adapter training on the uploaded clean/rough/mixed recordings.
- [ ] Expose adapter loading and `roughness = 0.0 / 0.5 / 1.0` A/B controls in inference.
- [ ] Add audio and metric comparison against the baseline.

## Current gate

Experiment 1 is closed. Experiment 2 now has a repository-tested engineering path for source-label preservation, expressive extraction, frozen-base adapter training, separate checkpoints, and a Train UI that reuses the current experiment settings. The expanded GitHub Actions suite passes. The next gate is one real Tesla T4 extraction/training run before inference controls and acoustic claims are closed.
