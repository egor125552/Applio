# Mandatory user-path validation rule

This project does not treat compilation, imports, isolated unit tests, mocked calls, or direct Python function calls as sufficient proof that a user-facing feature works.

## Required validation before reporting a feature as ready

For every new or materially changed user-facing workflow:

1. Prepare a small but structurally realistic dataset or project fixture.
2. Start the actual application interface used by the user.
3. Operate the workflow through a real browser whenever the feature is exposed through Gradio.
4. Change settings through the interface rather than calling private functions directly.
5. Click the same buttons the user clicks, in the same order.
6. Exercise the real production code path, including file loading, model construction, forward pass, backward pass, optimizer step, validation, checkpoint saving, JSON reports, and Gradio file postprocessing when applicable.
7. Run at least 20 real optimizer steps for training features. A one-step mock or shape-only backward test is not an end-to-end training validation.
8. Reject hidden tracebacks, missing output files, non-finite losses, silently shortened runs, and reports that do not match the requested settings.
9. Inspect the generated artifacts and machine-readable report after the browser run.
10. Fix every discovered failure and rerun the complete path from a clean fixture. Do not rerun only the previously failing isolated step.

## Evidence required in CI

A completed user-path test must retain enough evidence to diagnose failures:

- server log;
- fixture/setup log;
- final browser screenshot;
- training history JSON;
- short summary JSON;
- best and final checkpoints;
- a browser E2E report containing completed epochs, batch size, optimizer-step count, and confirmation that the production path was used.

## Status language

A feature may be called **implemented** after code and unit tests exist.

A feature may be called **ready for the user** only after its complete browser user path has passed.

If a required environment cannot be reproduced in CI, the limitation must be stated explicitly and the corresponding real-device or Colab smoke run remains mandatory.

## CEVC reference implementation

The workflow `.github/workflows/cevc-adapter-v2-browser-e2e.yml` is the reference implementation for this rule. It uses public real speech, builds a miniature but real RVC checkpoint, launches the actual CEVC 2B Gradio tab, clicks Stage 1, trains the real critic, checks the gate, trains Adapter v2 for at least 20 optimizer steps, and verifies every generated artifact.
