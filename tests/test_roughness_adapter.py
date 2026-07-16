import tempfile
import unittest
from pathlib import Path

import torch

from rvc.lib.algorithm.cevc.checkpoint import (
    CEVC_CHECKPOINT_FORMAT,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)
from rvc.lib.algorithm.cevc.roughness_adapter import (
    RoughnessAdapter,
    RoughnessAdapterConfig,
)


class RoughnessAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.config = RoughnessAdapterConfig(
            channels=16,
            feature_dim=5,
            hidden_channels=8,
            num_blocks=2,
        )
        self.adapter = RoughnessAdapter(self.config)
        self.latent = torch.randn(2, 16, 12)
        self.features = torch.randn(2, 5, 15)

    def test_parameter_budget(self):
        production = RoughnessAdapter(RoughnessAdapterConfig())
        self.assertGreater(production.trainable_parameter_count, 0)
        self.assertLess(production.trainable_parameter_count, 1_500_000)

    def test_untrained_adapter_is_exact_identity(self):
        output = self.adapter(self.latent, self.features, 1.0)
        self.assertTrue(torch.equal(output, self.latent))

    def test_missing_features_or_control_is_identity(self):
        self.assertIs(self.adapter(self.latent), self.latent)
        self.assertIs(self.adapter(self.latent, self.features, None), self.latent)

    def test_zero_control_is_exact_identity_after_training_like_weights(self):
        torch.nn.init.normal_(self.adapter.output_projection.weight, std=0.02)
        torch.nn.init.normal_(self.adapter.output_projection.bias, std=0.02)
        controls = [0.0, torch.zeros(2), torch.zeros(2, 12)]
        for control in controls:
            with self.subTest(control_type=type(control).__name__):
                output = self.adapter(self.latent, self.features, control)
                self.assertTrue(torch.equal(output, self.latent))

    def test_nonzero_control_changes_output_and_interpolates_time(self):
        torch.nn.init.normal_(self.adapter.output_projection.weight, std=0.02)
        torch.nn.init.normal_(self.adapter.output_projection.bias, std=0.02)
        control = torch.linspace(0.0, 1.0, 7).repeat(2, 1)
        output = self.adapter(self.latent, self.features, control)
        self.assertEqual(output.shape, self.latent.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(torch.equal(output, self.latent))

    def test_forward_backward_and_optimizer_step(self):
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=1e-3)
        output = self.adapter(self.latent, self.features, 0.75)
        loss = (output - self.latent).square().mean() + output.abs().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in self.adapter.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        before = self.adapter.output_projection.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, self.adapter.output_projection.weight))

    def test_checkpoint_round_trip(self):
        torch.nn.init.normal_(self.adapter.output_projection.weight, std=0.01)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "G_100.pth"
            base.write_bytes(b"base-checkpoint")
            adapter_path = Path(directory) / "voice.cevc.pth"
            save_adapter_checkpoint(
                str(adapter_path),
                adapter=self.adapter,
                model_name="voice",
                base_checkpoint=str(base),
                sample_rate=40000,
                epoch=12,
                feature_stats={"feature_names": ["energy_db"]},
            )
            restored, payload = load_adapter_checkpoint(str(adapter_path))

        self.assertEqual(payload["format"], CEVC_CHECKPOINT_FORMAT)
        self.assertEqual(payload["epoch"], 12)
        self.assertEqual(restored.config, self.adapter.config)
        for key, value in self.adapter.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)


class SynthesizerCEVCIntegrationTests(unittest.TestCase):
    def test_disabled_adapter_preserves_legacy_state_dict_surface(self):
        source = Path("rvc/lib/algorithm/synthesizers.py").read_text(encoding="utf-8")
        self.assertIn("self.cevc_adapter = DisabledRoughnessAdapter()", source)
        self.assertIn("def enable_cevc_adapter(", source)
        self.assertIn("z_for_decoder = self._apply_cevc_adapter", source)
        self.assertNotIn("register_buffer(\"cevc", source)


if __name__ == "__main__":
    unittest.main()
