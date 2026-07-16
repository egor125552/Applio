import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from rvc.lib.algorithm.generators.hifigan import HiFiGANGenerator
from rvc.lib.algorithm.generators.hifigan_nsf import HiFiGANNSFGenerator
from rvc.lib.algorithm.generators.registry import (
    DEFAULT_VOCODER,
    MRF_VOCODER,
    REFINEGAN_VOCODER,
    available_vocoders,
    build_generator,
    normalize_vocoder_name,
)


SMALL_CONFIG = {
    "initial_channel": 4,
    "resblock_kernel_sizes": [3],
    "resblock_dilation_sizes": [[1, 3, 5]],
    "upsample_rates": [2, 2],
    "upsample_initial_channel": 16,
    "upsample_kernel_sizes": [4, 4],
    "gin_channels": 4,
    "sample_rate": 16000,
    "checkpointing": False,
}


class GeneratorRegistryTests(unittest.TestCase):
    def test_import_and_alias_normalization(self):
        self.assertEqual(normalize_vocoder_name(None), DEFAULT_VOCODER)
        self.assertEqual(normalize_vocoder_name("nsf_hifigan"), DEFAULT_VOCODER)
        self.assertEqual(normalize_vocoder_name("MRF HiFi-GAN"), MRF_VOCODER)
        self.assertEqual(normalize_vocoder_name("refine gan"), REFINEGAN_VOCODER)
        self.assertEqual(normalize_vocoder_name("old-unknown-name"), DEFAULT_VOCODER)
        self.assertEqual(available_vocoders(use_f0=False), (DEFAULT_VOCODER,))

    def test_default_f0_matches_legacy_constructor_exactly(self):
        torch.manual_seed(1234)
        legacy = HiFiGANNSFGenerator(
            SMALL_CONFIG["initial_channel"],
            SMALL_CONFIG["resblock_kernel_sizes"],
            SMALL_CONFIG["resblock_dilation_sizes"],
            SMALL_CONFIG["upsample_rates"],
            SMALL_CONFIG["upsample_initial_channel"],
            SMALL_CONFIG["upsample_kernel_sizes"],
            gin_channels=SMALL_CONFIG["gin_channels"],
            sr=SMALL_CONFIG["sample_rate"],
            checkpointing=False,
        )
        torch.manual_seed(1234)
        registered = build_generator(DEFAULT_VOCODER, use_f0=True, **SMALL_CONFIG)

        self.assertIsInstance(registered, HiFiGANNSFGenerator)
        self.assertEqual(tuple(legacy.state_dict()), tuple(registered.state_dict()))
        for key, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, registered.state_dict()[key]), key)

    def test_default_non_f0_matches_legacy_constructor_exactly(self):
        torch.manual_seed(4321)
        legacy = HiFiGANGenerator(
            SMALL_CONFIG["initial_channel"],
            SMALL_CONFIG["resblock_kernel_sizes"],
            SMALL_CONFIG["resblock_dilation_sizes"],
            SMALL_CONFIG["upsample_rates"],
            SMALL_CONFIG["upsample_initial_channel"],
            SMALL_CONFIG["upsample_kernel_sizes"],
            gin_channels=SMALL_CONFIG["gin_channels"],
        )
        torch.manual_seed(4321)
        registered = build_generator(DEFAULT_VOCODER, use_f0=False, **SMALL_CONFIG)

        self.assertIsInstance(registered, HiFiGANGenerator)
        for key, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, registered.state_dict()[key]), key)

    def test_forward_backward_shape_range_nan_and_training_step(self):
        torch.manual_seed(7)
        generator = build_generator(DEFAULT_VOCODER, use_f0=True, **SMALL_CONFIG)
        generator.train()
        optimizer = torch.optim.AdamW(generator.parameters(), lr=1e-4)

        latent = torch.randn(1, 4, 8)
        f0 = torch.full((1, 8), 180.0)
        conditioning = torch.randn(1, 4, 1)

        torch.manual_seed(99)
        output = generator(latent, f0, g=conditioning)
        self.assertEqual(output.shape, (1, 1, 32))
        self.assertTrue(torch.isfinite(output).all())
        self.assertLessEqual(float(output.detach().abs().max()), 1.0 + 1e-6)

        loss = output.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [p.grad for p in generator.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
        optimizer.step()

    def test_seeded_forward_is_reproducible(self):
        torch.manual_seed(8)
        generator = build_generator(DEFAULT_VOCODER, use_f0=True, **SMALL_CONFIG)
        generator.eval()
        latent = torch.randn(1, 4, 8)
        f0 = torch.full((1, 8), 150.0)
        conditioning = torch.randn(1, 4, 1)

        torch.manual_seed(55)
        first = generator(latent, f0, g=conditioning)
        torch.manual_seed(55)
        second = generator(latent, f0, g=conditioning)
        self.assertTrue(torch.equal(first, second))

    def test_checkpoint_round_trip(self):
        torch.manual_seed(12)
        source = build_generator(DEFAULT_VOCODER, use_f0=True, **SMALL_CONFIG)
        with tempfile.NamedTemporaryFile(suffix=".pth") as checkpoint:
            torch.save(source.state_dict(), checkpoint.name)
            target = build_generator(DEFAULT_VOCODER, use_f0=True, **SMALL_CONFIG)
            state = torch.load(checkpoint.name, map_location="cpu", weights_only=True)
            target.load_state_dict(state)
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)

    def test_unsupported_non_f0_preserves_legacy_none_behaviour(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = build_generator(MRF_VOCODER, use_f0=False, **SMALL_CONFIG)
        self.assertIsNone(result)
        self.assertIn("does not support training without pitch guidance", buffer.getvalue())

    def test_refinegan_route_is_lazy_and_receives_legacy_arguments(self):
        fake_module = types.ModuleType("rvc.lib.algorithm.generators.refinegan")
        dummy_instance = torch.nn.Identity()
        constructor = mock.Mock(return_value=dummy_instance)
        fake_module.RefineGANGenerator = constructor

        with mock.patch.dict(sys.modules, {fake_module.__name__: fake_module}):
            result = build_generator(REFINEGAN_VOCODER, use_f0=True, **SMALL_CONFIG)

        self.assertIs(result, dummy_instance)
        constructor.assert_called_once_with(
            sample_rate=16000,
            downsample_rates=[2, 2],
            upsample_rates=[2, 2],
            start_channels=16,
            num_mels=4,
            checkpointing=False,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_smoke(self):
        generator = build_generator(DEFAULT_VOCODER, use_f0=True, **SMALL_CONFIG).cuda()
        latent = torch.randn(1, 4, 8, device="cuda")
        f0 = torch.full((1, 8), 180.0, device="cuda")
        conditioning = torch.randn(1, 4, 1, device="cuda")
        output = generator(latent, f0, g=conditioning)
        self.assertTrue(torch.isfinite(output).all())


class SynthesizerRegistryIntegrationTests(unittest.TestCase):
    def test_synthesizer_delegates_to_registry(self):
        source = Path("rvc/lib/algorithm/synthesizers.py").read_text(encoding="utf-8")
        self.assertIn(
            "from rvc.lib.algorithm.generators.registry import build_generator", source
        )
        self.assertIn("self.dec = build_generator(", source)
        self.assertNotIn("HiFiGANNSFGenerator", source)
        self.assertNotIn("RefineGANGenerator", source)


if __name__ == "__main__":
    unittest.main()
