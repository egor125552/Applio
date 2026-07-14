# CEVC Adapter v2 browser E2E result

Successful workflow run: `29370926394`

Tested commit: `9928772b1a2bd392ee58c8d6fa5666289c95b8ce`

## Complete user path

The workflow installed the full Colab dependencies and Chromium, prepared a miniature but structurally valid RVC experiment from public speech, launched the actual CEVC 2B Gradio tab, and operated the accessible browser controls in user order.

It completed Stage 1, trained and accepted the critic, checked Adapter v2 readiness, trained Adapter v2, saved all checkpoints and JSON reports, processed the returned files through Gradio, and finished with no hidden server traceback.

## Training completed

Critic:

- 100 epochs
- batch 8
- 300 real optimizer steps
- best epoch 23
- accepted by the normal production gate

Adapter v2:

- 30 epochs
- batch 4
- 60 real optimizer steps
- best epoch 18
- production prior-latent path used
- shared latent for low and high controls used
- base model and critic frozen
- exact zero identity, loudness, spectrum and clipping checks passed
- best, final and exported checkpoints created

The acoustic margin gate did not pass on the randomly initialized miniature RVC checkpoint. Its baseline waveform already scored about 0.962 on the fixture critic, leaving no useful score headroom. This is expected for a random engineering checkpoint and is not acoustic evidence about the real `logs/egor` model. The E2E proves that the complete training path runs and reports a failed acoustic gate safely.

## Bugs found by the full browser path

- missing `mel_processing` import path in the heavy trainer
- script entrypoints unable to import repository modules
- fixture configuration used before construction
- obsolete report verification fields
- insufficiently distinct critic fixture classes
- hidden Gradio Accordion settings
- incorrect slider locators
- Gradio numeric controls requiring their accessible `spinbutton` names

Every fix was followed by a clean rerun of the complete workflow.

## Required evidence retained

The workflow artifact contains fixture, browser and server logs; the final browser screenshot; Stage 1 manifest; critic and Adapter summaries and histories; best/final/export checkpoints; batch report; and the machine-readable browser E2E report.

## Remaining gate

The engineering execution gate is closed. The next separate gate is the real Tesla T4 run on `logs/egor`, then the one-latent acoustic comparison at controls 0.0, 0.5 and 1.0.
