"""Batch-size fallback policy for CEVC Adapter v2."""

from __future__ import annotations


AUTO_BATCH_CANDIDATES = (32, 24, 16, 12, 8, 6, 4, 2, 1)


def is_cuda_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return "out of memory" in text and ("cuda" in text or "cudnn" in text)


def choose_largest_fitting_batch(probe_one, candidates=AUTO_BATCH_CANDIDATES):
    """Return the first successful candidate and a machine-readable trial list."""

    trials = []
    for candidate in candidates:
        candidate = int(candidate)
        try:
            details = dict(probe_one(candidate) or {})
            trials.append({"batch_size": candidate, "fits": True, **details})
            return candidate, trials
        except Exception as error:
            if not is_cuda_oom(error):
                raise
            trials.append(
                {
                    "batch_size": candidate,
                    "fits": False,
                    "reason": "cuda_out_of_memory",
                }
            )
    raise RuntimeError("Adapter v2 did not fit even with batch size 1")
