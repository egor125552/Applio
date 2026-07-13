"""Generator registry for RVC/Applio synthesizers.

The registry centralizes vocoder selection without changing checkpoint names or
legacy fallback behaviour. New CEVC generators can be added here without
adding another conditional tree to ``Synthesizer``.
"""

from __future__ import annotations

from typing import Optional

import torch

DEFAULT_VOCODER = "HiFi-GAN"
MRF_VOCODER = "MRF HiFi-GAN"
REFINEGAN_VOCODER = "RefineGAN"


def _alias_key(name: object) -> str:
    value = "" if name is None else str(name)
    value = value.strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(value.split())


_ALIASES = {
    "": DEFAULT_VOCODER,
    "hifi gan": DEFAULT_VOCODER,
    "hifigan": DEFAULT_VOCODER,
    "nsf hifi gan": DEFAULT_VOCODER,
    "nsf hifigan": DEFAULT_VOCODER,
    "nsf": DEFAULT_VOCODER,
    "mrf hifi gan": MRF_VOCODER,
    "hifi gan mrf": MRF_VOCODER,
    "mrf hifigan": MRF_VOCODER,
    "mrf": MRF_VOCODER,
    "refinegan": REFINEGAN_VOCODER,
    "refine gan": REFINEGAN_VOCODER,
}


def normalize_vocoder_name(vocoder: object) -> str:
    """Return a canonical checkpoint-compatible vocoder name.

    Unknown names intentionally fall back to the historical HiFi-GAN path.
    The previous ``Synthesizer`` implementation used the same fallback, so old
    or imperfect checkpoints keep loading instead of failing hard.
    """

    return _ALIASES.get(_alias_key(vocoder), DEFAULT_VOCODER)


def available_vocoders(*, use_f0: Optional[bool] = None) -> tuple[str, ...]:
    """List canonical vocoders supported by the current registry."""

    if use_f0 is False:
        return (DEFAULT_VOCODER,)
    return (DEFAULT_VOCODER, MRF_VOCODER, REFINEGAN_VOCODER)


def build_generator(
    vocoder: object,
    *,
    use_f0: bool,
    initial_channel: int,
    resblock_kernel_sizes: list,
    resblock_dilation_sizes: list,
    upsample_rates: list,
    upsample_initial_channel: int,
    upsample_kernel_sizes: list,
    gin_channels: int,
    sample_rate: int,
    checkpointing: bool = False,
) -> Optional[torch.nn.Module]:
    """Construct the selected generator while preserving legacy behaviour."""

    name = normalize_vocoder_name(vocoder)

    if not use_f0:
        if name != DEFAULT_VOCODER:
            print(f"{name} does not support training without pitch guidance.")
            return None

        from rvc.lib.algorithm.generators.hifigan import HiFiGANGenerator

        return HiFiGANGenerator(
            initial_channel,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )

    if name == MRF_VOCODER:
        from rvc.lib.algorithm.generators.hifigan_mrf import HiFiGANMRFGenerator

        return HiFiGANMRFGenerator(
            in_channel=initial_channel,
            upsample_initial_channel=upsample_initial_channel,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilations=resblock_dilation_sizes,
            gin_channels=gin_channels,
            sample_rate=sample_rate,
            harmonic_num=8,
            checkpointing=checkpointing,
        )

    if name == REFINEGAN_VOCODER:
        from rvc.lib.algorithm.generators.refinegan import RefineGANGenerator

        return RefineGANGenerator(
            sample_rate=sample_rate,
            downsample_rates=upsample_rates[::-1],
            upsample_rates=upsample_rates,
            start_channels=16,
            num_mels=initial_channel,
            checkpointing=checkpointing,
        )

    from rvc.lib.algorithm.generators.hifigan_nsf import HiFiGANNSFGenerator

    return HiFiGANNSFGenerator(
        initial_channel,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=gin_channels,
        sr=sample_rate,
        checkpointing=checkpointing,
    )
