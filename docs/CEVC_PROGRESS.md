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

## Experiment 2 — Roughness Adapter v1

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
- [x] Pass the expanded CEVC GitHub Actions job (`29283738664`).
- [x] Run the first real adapter extraction/training on the uploaded clean/rough/mixed recordings.
- [x] Expose adapter loading and `roughness = 0.0 / 0.5 / 1.0` deterministic A/B controls in inference.
- [x] Run and listen to the first real three-output A/B conversion on Tesla T4.
- [x] Compare output duration, RMS, peak, clipping and endpoint continuity.
- [x] Record that v1 failed the acoustic quality gate.

### First real adapter training result

- Dataset: 170 slices from clean, rough and mixed recordings; 7 minutes 20 seconds total source audio.
- Environment: Google Colab Tesla T4.
- Adapter: 60,160 trainable parameters; the base RVC network remained frozen.
- Settings: 20 epochs, batch size 32, 6 batches per epoch, 120 optimizer steps.
- Runtime: 2 minutes 7 seconds.
- Best checkpoint: epoch 18, loss `0.52470`.
- Export: `logs/egor/egor.cevc.pth`, with an identical copy at `logs/egor/cevc/roughness_adapter_best.pth`.

### First real A/B result — acoustic gate failed

The fixed external test input was `Новая запись 278.mp3` from the user's Google Drive root. The private audio remains outside the repository.

| Variant | Duration | RMS |
|---|---:|---:|
| Input | 27.336 s | -15.71 dB |
| Roughness 0.0 | 27.135 s | -15.62 dB |
| Roughness 0.5 | 27.320 s | -16.13 dB |
| Roughness 1.0 | 26.623 s | -16.26 dB |

- `roughness = 1.0` lost about 0.7 seconds relative to `0.5`.
- Its final sample was about `-0.1806`, proving that the WAV ended during an active waveform rather than near silence.
- Increasing roughness mainly reduced level and produced a darker/muddier result.
- `0.5` and `1.0` were not clearly separated as natural roughness levels.
- No clipping was detected.
- The v1 checkpoint remains useful as an engineering artifact, but it is not accepted as a usable acoustic model.

### Root cause hypothesis

- Training reconstructs target audio from a posterior latent, while real inference uses a prior/reverse-flow latent.
- The v1 loss contains mel L1, waveform L1 and small weight regularization, but no explicit roughness direction, monotonicity, loudness preservation, content preservation or duration requirement.
- Increasing epochs or adapter capacity before changing the objective would likely strengthen the wrong behaviour.

## Experiment 2B — revised adapter, no new recordings

Detailed plan: [`CEVC_EXPERIMENT_2B_PLAN.md`](CEVC_EXPERIMENT_2B_PLAN.md).

Dataset decision:

- [x] Freeze the existing three source recordings and 170 slices.
- [x] Do not request additional recordings from the user for Experiment 2B.
- [x] Keep `Новая запись 278.mp3` as an external fixed test and exclude it from training.

Inference harness:

- [x] Compute ContentVec, F0, index features and latent once per A/B request.
- [x] Decode `0.0`, `0.5` and `1.0` from the same latent.
- [x] Require identical output lengths and fail loudly on mismatch.
- [x] Add endpoint-discontinuity and duration regression tests.
- [x] Save a machine-readable audio comparison report.

Data reuse and real-only supervision:

- [x] Add a separate `CEVC 2B Lab` tab so standard Train and Inference remain uncluttered.
- [x] Implement train/validation splitting by contiguous source-time tails rather than random neighbouring slices.
- [x] Remove synthetic noisy WAV files as critic or adapter training targets.
- [x] Build a real roughness reference profile from actual clean, mixed and rough slices.
- [x] Save median clean/mixed/rough normalized spectra and the real rough-minus-clean spectral delta as NPZ and JSON.
- [x] Reuse extracted energy, spectral tilt, HNR, band aperiodicity and F0 instability to save real rough-minus-clean expressive deltas.
- [x] Provide clean, spectral-only diagnostic, real mixed and real rough preview WAVs.
- [x] Mark the spectral-only preview explicitly as diagnostic and never a training target.
- [x] Implement a differentiable waveform Roughness Critic with scalar score and clean/mixed/rough classification heads.
- [x] Anchor the scalar axis only with real clean and real rough; keep mixed as a real classification class without a fabricated scalar target.
- [x] Apply the same random gain, EQ, low-level noise and polarity augmentation across every class to reduce simple recording/noise shortcuts.
- [x] Save best/final critic checkpoints, full history and a short human-readable JSON summary.
- [x] Replace raw per-epoch console dictionaries with clear staged Colab messages and final verdicts.
- [x] Copy only UI artifacts to `/tmp`; persistent profile, manifest and checkpoints remain on Google Drive.
- [x] Add full data checks for every source slice, class coverage, split validity, profile artifacts, preview length and RMS preservation.
- [x] Add real-profile numerical determinism, critic forward/backward and one-epoch real-only checkpoint tests.
- [x] Reproduce the Google Drive symlink path and run Stage 1 and Stage 2 outputs through Gradio `async_move_files_to_cache`.
- [x] Generate the real profile and split for `logs/egor` on Colab.
- [x] Train and validate the clean/rough-anchor Roughness Critic on Tesla T4.
- [x] Accept and freeze the best critic checkpoint from epoch 78 for Adapter v2 supervision.
- [x] Pass CEVC checks run `29345099758` and full Colab/Gradio smoke run `29345099496`.

### Accepted real critic result

- Best epoch: 78.
- Anchor MAE: about `0.1269`.
- Three-class validation accuracy: about `76.47%`.
- Clean mean score: about `0.150`.
- Rough mean score: about `0.814`.
- Clean-to-rough margin: about `0.664`.
- Mixed is intentionally diagnostic on the scalar axis because the source contains time-varying transitions.

Adapter v2 objective and interface:

- [x] Generate random low/high controls from one shared prior latent that follows the inference path.
- [x] Use the accepted frozen critic for directional roughness supervision.
- [x] Train only from real clean slices; never use spectral-only or noisy synthetic audio as a target.
- [x] Freeze the base RVC model, decoder and critic; optimize only the 60,160-parameter adapter.
- [x] Preserve exact identity at `roughness = 0` by architecture and validate it automatically.
- [x] Add loudness, temporal-envelope, normalized-spectrum, residual-size and clipping safeguards.
- [x] Require ordered critic scores for baseline, `0.5` and `1.0` during validation.
- [x] Prefer gate-passing epochs when selecting the best Adapter v2 checkpoint.
- [x] Preserve the training random stream across deterministic validation passes.
- [x] Add a lightweight preflight that verifies the accepted critic, full `G_*.pth`, filelist and clean train/validation coverage before loading the heavy trainer.
- [x] Add Stage 3 controls and human-readable results to `CEVC 2B Lab`.
- [x] Save a working `.cevc.pth`, full history and a short JSON report for sharing.
- [x] Add objective gradient, resampling, gate and rejected-critic tests.
- [ ] Run the real Adapter v2 training on `logs/egor` with Tesla T4.
- [ ] Pass the automatic Adapter v2 gate on the real validation clean slices.
- [ ] Run the fixed one-latent acoustic A/B using `Новая запись 278.mp3`.
- [ ] Accept or reject Adapter v2 by listening and machine-readable diagnostics.
- [ ] Increase capacity to 200–300k only if the 60k direction is correct but too weak.
- [ ] Use the approximately 1.03M-parameter configuration only after the smaller model passes the acoustic gate.

## Current gate

Experiment 1 is closed. Experiment 2 v1 failed its acoustic gate. Experiment 2B Stage 1 and Stage 2 have completed real Colab runs, and the epoch-78 clean/rough-anchor critic is accepted. Adapter v2 code, safeguards, UI, console reporting, preflight and repository tests are implemented. The next gate is the first real Adapter v2 training on the user's Tesla T4, followed by the existing one-latent A/B harness. No new recordings are required.
